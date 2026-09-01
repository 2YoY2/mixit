#!/usr/bin/env python3
"""Pose from ORACLE-NAMED slots: fixed semantic input layout.

Per receiver, slots are named per recording using GT (the oracle):
  pelvis  slot whose envelope best explains the absolute root (probe 54)
  5 limbs Hungarian of slot envelopes vs residualized limb envelopes
Input per window: 6 named channels x [logE, 3 band energies, sin phi,
cos phi, concentration] x 3 rx = 126 dims.  phi included — probe 55
showed the pelvis slot's bearing tracks the root at |corr| ~ 0.6.
Model: small GRU -> 15 joints (MOVING GT), L1 + VELW velocity matching.
This is the named-slot CEILING (naming uses GT at train AND test);
deployment needs a naming module later.

  TRSC=1,2,3 TESC=4 HOURS=1 VELW=2 python3 train/train_pose_named.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/cluster/tok/pa-v1"))
ABSD = os.path.expanduser(os.environ.get(
    "ABSD", "~/zerdani/buffer/cluster/tok/pa-v1-absgt/pose_abs"))
LIMBD = os.path.expanduser(os.environ.get(
    "LIMBD", "~/zerdani/buffer/cluster/tok/pa-v1/limbenv"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/cluster/runs/clusterer/limbtok.pa-v1.r12"))
OUTD = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/cluster/runs/downstream/posenamed.r1"))
TRSC = [int(v) for v in os.environ.get("TRSC", "1,2,3").split(",")]
TESC = int(os.environ.get("TESC", "4"))
HOURS = float(os.environ.get("HOURS", "1.0"))
STEPS = int(os.environ.get("STEPS", "8000"))
B = int(os.environ.get("B", "32"))
LR = float(os.environ.get("LR", "1e-3"))
H = int(os.environ.get("H", "256"))
VELW = float(os.environ.get("VELW", "2"))
SEED = int(os.environ.get("SEED", "0"))
NJ = 15
HOPF, WINF = 128, 256
BANDS = [(2, 10), (10, 40), (40, 150)]
NCH, NF = 6, 7                        # named channels x feats per channel
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

ck = torch.load(f"{RUNS}/best.pt", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x):
        return torch.softmax(self.head(self.enc(self.inp(x))), -1)

sep = SetSep(); sep.load_state_dict(ck["model"])
sep = sep.to(dev).eval()

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def cv_r2(E, Y):
    n = len(E)
    if n < 12: return -9.0
    ev = np.arange(n) % 2 == 0
    A = np.c_[E, np.ones(n)]
    beta, *_ = np.linalg.lstsq(A[ev], Y[ev], rcond=None)
    sst = ((Y[~ev] - Y[ev].mean(0)) ** 2).sum()
    return float(1 - ((Y[~ev] - A[~ev] @ beta) ** 2).sum()
                 / max(sst, 1e-12))

def rec_named(rid, root, G):
    """-> (nw, NCH*NF) named-channel features for one rx, or None."""
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    if len(t) < 16: return None
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int)
    f = t[:, 1]; phi = t[:, 2]
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    hard = a.argmax(1)
    env = np.zeros((nw, M)); zs = np.zeros((nw, M)); zc = np.zeros((nw, M))
    bnd = np.zeros((nw, M, 3))
    for m in range(M):
        s_ = hard == m
        np.add.at(env[:, m], w[s_], e[s_])
        np.add.at(zs[:, m], w[s_], e[s_] * np.sin(phi[s_]))
        np.add.at(zc[:, m], w[s_], e[s_] * np.cos(phi[s_]))
        for bi, (lo, hi) in enumerate(BANDS):
            sb = s_ & (f >= lo) & (f < hi)
            np.add.at(bnd[:, m, bi], w[sb], e[sb])
    nww = min(nw, len(root), len(G))
    fin = np.isfinite(root[:nww]).all(1)
    if fin.sum() < 12: return None
    # name pelvis: best CV-R2 for absolute root trajectory
    rs = [cv_r2(np.log10(env[:nww][fin, m:m + 1] + 1e-9), root[:nww][fin])
          for m in range(M)]
    pel = int(np.nanargmax(rs))
    # name limbs: Hungarian slot-envelope vs residualized limb envelopes
    C = np.zeros((M, 5))
    for m in range(M):
        for j in range(5):
            C[m, j] = corr(env[:nww, m], G[:nww, j])
    C[pel] = -9                      # pelvis slot not reusable as a limb
    r_, c_ = linear_sum_assignment(-C)
    order = [pel] + [int(r_[list(c_).index(j)]) for j in range(5)]
    F = np.zeros((nw, NCH, NF), np.float32)
    for k, m in enumerate(order):
        F[:, k, 0] = np.log10(env[:, m] + 1e-9)
        F[:, k, 1:4] = np.log10(bnd[:, m] + 1e-9)
        ph = np.arctan2(zs[:, m], zc[:, m])
        F[:, k, 4] = np.sin(ph); F[:, k, 5] = np.cos(ph)
        F[:, k, 6] = np.sqrt(zs[:, m] ** 2 + zc[:, m] ** 2) / (env[:, m]
                                                               + 1e-12)
    return F.reshape(nw, NCH * NF)

def residual_limbs(gi, nw):
    g2 = gi.copy()
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    return np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                     for w in range(nw)])

def build(scene):
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene == scene].copy()
    man["ckl"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for _, g in man.groupby("ckl"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        pf = f"{ABSD}/{rids[0]:06d}.npy"
        gf = f"{LIMBD}/{rids[0]:06d}.npy"
        if not (os.path.exists(pf) and os.path.exists(gf)): continue
        P = np.asarray(np.load(pf), np.float32)
        gi = np.asarray(np.load(gf), np.float32)
        nw0 = len(P)
        G = residual_limbs(gi, nw0)
        root = P[:, 8]
        fs = [rec_named(r, root, G) for r in rids]
        if any(f is None for f in fs): continue
        nw = min(min(len(f) for f in fs), nw0)
        out.append((np.concatenate([f[:nw] for f in fs], 1), P[:nw]))
        if len(out) % 500 == 0: print(f"  s{scene}: {len(out)}", flush=True)
    return out

class Head(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.gru = nn.GRU(fin, H, 2, batch_first=True)
        self.out = nn.Linear(H, NJ * 3)
    def forward(self, x):
        return self.out(self.gru(x)[0]).view(x.shape[0], -1, NJ, 3)

def mpjpe(pred, gt):
    m = np.isfinite(gt).all(-1); m[:, 8] = False
    if not m.any(): return np.nan
    return float(np.linalg.norm(np.nan_to_num(pred - gt),
                                axis=-1)[m].mean() * 100)

def pck2(pred, gt):
    m = np.isfinite(gt).all(-1); m[:, 8] = False
    if not m.any(): return np.nan, np.nan
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)[m]
    return float((d < 0.02).mean() * 100), float((d < 0.05).mean() * 100)

def main():
    print(f"building (train {TRSC} -> test {TESC}); NAMED oracle slots",
          flush=True)
    tr_all = []
    for s in TRSC: tr_all += build(s)
    te = build(TESC)
    rng = np.random.default_rng(SEED)
    ix = rng.permutation(len(tr_all))
    ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
    tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
    print(f"train {len(tr)} ho {len(ho)} test {len(te)}", flush=True)
    fin = tr[0][0].shape[1]
    head = Head(fin).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        bx = rng.choice(len(tr), B)
        nw = max(len(tr[i][1]) for i in bx)
        X = torch.zeros(B, nw, fin)
        Y = torch.full((B, nw, NJ, 3), np.nan)
        for k, i in enumerate(bx):
            F, P = tr[i]
            X[k, :len(F)] = torch.from_numpy(F)
            Y[k, :len(P)] = torch.from_numpy(P)
        X, Y = X.to(dev), Y.to(dev)
        pred = head(X)
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        msk[:, :, 8] = False
        Yn = torch.nan_to_num(Y)
        loss = (torch.where(msk, (pred - Yn).abs(),
                            torch.zeros_like(pred)).sum()
                / msk.sum().clamp(min=1) / 3)
        if VELW > 0:
            mv = msk[:, 1:] & msk[:, :-1]
            dp = pred[:, 1:] - pred[:, :-1]
            dg = Yn[:, 1:] - Yn[:, :-1]
            loss = loss + VELW * (torch.where(mv, (dp - dg).abs(),
                                              torch.zeros_like(dp)).sum()
                                  / mv.sum().clamp(min=1) / 3)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [{step}] L1 {loss.item()*100:.2f} cm "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
    head.eval()
    os.makedirs(OUTD, exist_ok=True)
    torch.save({"model": head.state_dict(), "fin": fin}, f"{OUTD}/named.pt")
    JS = [j for j in range(NJ) if j != 8]
    for tag, ds in (("heldout", ho), (f"scene{TESC}", te)):
        errs, p20s, p50s, sp, sg, tc = [], [], [], [], [], []
        with torch.no_grad():
            for F, P in ds:
                pr = head(torch.from_numpy(F)[None].to(dev)
                          )[0].cpu().numpy()
                errs.append(mpjpe(pr, P))
                a_, b_ = pck2(pr, P)
                p20s.append(a_); p50s.append(b_)
                m = np.isfinite(P).all(-1); m[:, 8] = False
                if not m.any(): continue
                sp.append(float(pr[:, JS].std(0).mean() * 100))
                sg.append(float(np.nanmean(np.nanstd(P[:, JS], 0)) * 100))
                pd_ = (pr - pr.mean(0))[m]
                gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
                den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
                tc.append(float((pd_ * gd).sum() / den))
        print(f"[{tag}] MPJPE {np.nanmedian(errs):.1f} cm  "
              f"PCK@20 {np.nanmean(p20s):.1f}  PCK@50 {np.nanmean(p50s):.1f}"
              f"  pred-std {np.median(sp):.1f} vs GT {np.median(sg):.1f} cm"
              f"  traj-corr {np.median(tc):+.3f}", flush=True)

if __name__ == "__main__":
    main()
