#!/usr/bin/env python3
"""Slot-stream pose decoder — consumes what the clusterer actually gives.

Input: 24 streams (3 rx x 8 slots), each a per-window sequence of
[logE, 3 band energies, sin/cos bearing phi, purity, sin/cos psi]
plus a per-recording ORACLE role one-hot (pelvis-best slot, 5
Hungarian-named limbs, or none) and an rx one-hot -> 19 dims/stream/win.
Architecture: shared stream GRU -> per-window cross-attention of 16
queries (1 root + 15 joints) over the 24 stream embeddings (content-based
correspondence, permutation-safe) -> temporal GRU per query -> FACTORED
output: root anchor trajectory + per-joint offsets; joints = anchor+offset.
Targets: clip-centered moving GT; pelvis fully supervised.
Loss: L1(joints) + VELW*velocity-matching + 0.5*L1(anchor vs GT root).

  TRSC=1,2,3 TESC=4 HOURS=3 python3 train/train_slotstream.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/cluster/tok/pa-v1"))
POSED = os.path.expanduser(os.environ.get(
    "POSED", "~/zerdani/buffer/cluster/tok/pa-v1-relmove/pose_relmove"))
LIMBD = os.path.expanduser(os.environ.get(
    "LIMBD", "~/zerdani/buffer/cluster/tok/pa-v1/limbenv"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/cluster/runs/clusterer/limbtok.pa-v1.r12"))
OUTD = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/cluster/runs/downstream/slotstream.r1"))
TRSC = [int(v) for v in os.environ.get("TRSC", "1,2,3").split(",")]
TESC = int(os.environ.get("TESC", "4"))
HOURS = float(os.environ.get("HOURS", "3.0"))
STEPS = int(os.environ.get("STEPS", "250000"))
EVERY = int(os.environ.get("EVERY", "2000"))
B = int(os.environ.get("B", "24"))
LR = float(os.environ.get("LR", "5e-4"))
DE = int(os.environ.get("DE", "64"))
VELW = float(os.environ.get("VELW", "2"))
SEED = int(os.environ.get("SEED", "0"))
NJ, ROOTJ = 15, 8
NS, NFIN = 24, 19
HOPF, WINF = 128, 256
BANDS = [(2, 10), (10, 40), (40, 150)]
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

def rec_streams(rid, root, G):
    """-> (M, nw, NFIN) streams for one rx incl. oracle roles, or None."""
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
    f = t[:, 1]; phi = t[:, 2]; psi = t[:, 3]
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    hard = a.argmax(1)
    env = np.zeros((nw, M)); zs = np.zeros((nw, M)); zc = np.zeros((nw, M))
    ps_ = np.zeros((nw, M)); pc_ = np.zeros((nw, M))
    bnd = np.zeros((nw, M, 3))
    for m in range(M):
        s_ = hard == m
        np.add.at(env[:, m], w[s_], e[s_])
        np.add.at(zs[:, m], w[s_], e[s_] * np.sin(phi[s_]))
        np.add.at(zc[:, m], w[s_], e[s_] * np.cos(phi[s_]))
        np.add.at(ps_[:, m], w[s_], e[s_] * np.sin(psi[s_]))
        np.add.at(pc_[:, m], w[s_], e[s_] * np.cos(psi[s_]))
        for bi, (lo, hi) in enumerate(BANDS):
            sb = s_ & (f >= lo) & (f < hi)
            np.add.at(bnd[:, m, bi], w[sb], e[sb])
    nww = min(nw, len(root), len(G))
    fin = np.isfinite(root[:nww]).all(1)
    if fin.sum() < 12: return None
    rs = [cv_r2(np.log10(env[:nww][fin, m:m + 1] + 1e-9), root[:nww][fin])
          for m in range(M)]
    pel = int(np.nanargmax(rs))
    C = np.zeros((M, 5))
    for m in range(M):
        for j in range(5):
            C[m, j] = corr(env[:nww, m], G[:nww, j])
    C[pel] = -9
    r_, c_ = linear_sum_assignment(-C)
    role = np.zeros((M, 7), np.float32)
    role[pel, 0] = 1
    for i in range(5):
        role[int(r_[list(c_).index(i)]), 1 + i] = 1
    role[role.sum(1) == 0, 6] = 1
    en = env + 1e-9
    F = np.zeros((M, nw, NFIN), np.float32)
    F[:, :, 0] = np.log10(en).T
    for bi in range(3):
        F[:, :, 1 + bi] = np.log10(bnd[:, :, bi] + 1e-9).T
    amp = np.sqrt(zs ** 2 + zc ** 2) + 1e-12
    F[:, :, 4] = (zs / amp).T
    F[:, :, 5] = (zc / amp).T
    F[:, :, 6] = (amp / en).T                    # purity
    pam = np.sqrt(ps_ ** 2 + pc_ ** 2) + 1e-12
    F[:, :, 7] = (ps_ / pam).T
    F[:, :, 8] = (pc_ / pam).T
    F[:, :, 9:16] = role[:, None, :]
    return F

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
        pf = f"{POSED}/{rids[0]:06d}.npy"
        gf = f"{LIMBD}/{rids[0]:06d}.npy"
        if not (os.path.exists(pf) and os.path.exists(gf)): continue
        P = np.asarray(np.load(pf), np.float32)
        gi = np.asarray(np.load(gf), np.float32)
        G = residual_limbs(gi, len(P))
        root = P[:, ROOTJ]
        fs = []
        for i, r in enumerate(rids):
            F = rec_streams(r, root, G)
            if F is None: break
            F[:, :, 16 + i] = 1.0                # rx one-hot (reserved cols)
            fs.append(F)
        if len(fs) != 3: continue
        nw = min(min(f.shape[1] for f in fs), len(P))
        out.append((np.concatenate([f[:, :nw] for f in fs], 0
                                   ).astype(np.float16), P[:nw]))
        if len(out) % 500 == 0: print(f"  s{scene}: {len(out)}", flush=True)
    return out

class SlotStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.GRU(NFIN, DE, 1, batch_first=True)
        self.q = nn.Parameter(torch.randn(NJ + 1, DE) * 0.1)
        self.att = nn.MultiheadAttention(DE, 4, batch_first=True)
        self.tgru = nn.GRU(DE, DE, 1, batch_first=True)
        self.root = nn.Linear(DE, 3)
        self.off = nn.Linear(DE, 3)
    def forward(self, S):                       # S (B, NS, nw, NFIN)
        Bx, ns, nw, _ = S.shape
        h, _ = self.enc(S.reshape(Bx * ns, nw, NFIN))
        h = h.reshape(Bx, ns, nw, DE).permute(0, 2, 1, 3)   # (B,nw,NS,DE)
        kv = h.reshape(Bx * nw, ns, DE)
        q = self.q[None].expand(Bx * nw, -1, -1)
        o, _ = self.att(q, kv, kv)                          # (B*nw,16,DE)
        o = o.reshape(Bx, nw, NJ + 1, DE).permute(0, 2, 1, 3)
        o2, _ = self.tgru(o.reshape(Bx * (NJ + 1), nw, DE))
        o2 = o2.reshape(Bx, NJ + 1, nw, DE)
        anchor = self.root(o2[:, 0])                        # (B, nw, 3)
        offs = self.off(o2[:, 1:]).permute(0, 2, 1, 3)      # (B,nw,NJ,3)
        return anchor[:, :, None, :] + offs, anchor

def mpjpe_np(pred, gt, exroot=True):
    m = np.isfinite(gt).all(-1)
    if exroot: m[:, ROOTJ] = False
    if not m.any(): return np.nan
    return float(np.linalg.norm(np.nan_to_num(pred - gt),
                                axis=-1)[m].mean() * 1000)

def pck_np(pred, gt):
    m = np.isfinite(gt).all(-1); m[:, ROOTJ] = False
    if not m.any(): return np.nan, np.nan
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)[m]
    return float((d < 0.02).mean() * 100), float((d < 0.05).mean() * 100)

def evalset(net, ds, tag, full=False):
    errs, p20, p50, sp, sg, tc, aerr = [], [], [], [], [], [], []
    JS = list(range(NJ))
    with torch.no_grad():
        for F, P in ds:
            pr, anc = net(torch.from_numpy(
                F.astype(np.float32))[None].to(dev))
            pr = pr[0].cpu().numpy(); anc = anc[0].cpu().numpy()
            errs.append(mpjpe_np(pr, P))
            a_, b_ = pck_np(pr, P)
            p20.append(a_); p50.append(b_)
            rm = np.isfinite(P[:, ROOTJ]).all(-1)
            if rm.any():
                aerr.append(float(np.linalg.norm(
                    anc[rm] - P[rm, ROOTJ], axis=-1).mean() * 1000))
            m = np.isfinite(P).all(-1)
            if not m.any(): continue
            sp.append(float(pr[:, JS].std(0).mean() * 100))
            sg.append(float(np.nanmean(np.nanstd(P[:, JS], 0)) * 100))
            pd_ = (pr - pr.mean(0))[m]
            gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
            den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
            tc.append(float((pd_ * gd).sum() / den))
    line = (f"[{tag}] MPJPE {np.nanmedian(errs):.0f} mm  "
            f"PCK@20 {np.nanmean(p20):.1f}  PCK@50 {np.nanmean(p50):.1f}  "
            f"anchor-err {np.nanmedian(aerr):.0f} mm  "
            f"MR {np.median(sp)/max(np.median(sg),1e-9):.2f}  "
            f"TC {np.median(tc):+.3f}")
    print(line, flush=True)
    return float(np.nanmedian(errs))

def main():
    print(f"slot-stream decoder (train {TRSC} -> test {TESC})", flush=True)
    tr_all = []
    for s in TRSC: tr_all += build(s)
    te = build(TESC)
    rng = np.random.default_rng(SEED)
    ix = rng.permutation(len(tr_all))
    ho = [tr_all[i] for i in ix[int(len(ix) * 0.95):]]
    tr = [tr_all[i] for i in ix[:int(len(ix) * 0.95)]]
    print(f"train {len(tr)} ho {len(ho)} test {len(te)}", flush=True)
    net = SlotStream().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    os.makedirs(OUTD, exist_ok=True)
    best = 1e9
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        bx = rng.choice(len(tr), B)
        nw = max(tr[i][0].shape[1] for i in bx)
        S = torch.zeros(B, NS, nw, NFIN)
        Y = torch.full((B, nw, NJ, 3), np.nan)
        for k, i in enumerate(bx):
            F, P = tr[i]
            S[k, :, :F.shape[1]] = torch.from_numpy(F.astype(np.float32))
            Y[k, :len(P)] = torch.from_numpy(P)
        S, Y = S.to(dev), Y.to(dev)
        pred, anc = net(S)
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        Yn = torch.nan_to_num(Y)
        loss = (torch.where(msk, (pred - Yn).abs(),
                            torch.zeros_like(pred)).sum()
                / msk.sum().clamp(min=1) / 3)
        rmsk = msk[:, :, ROOTJ]
        loss = loss + 0.5 * (torch.where(
            rmsk, (anc - Yn[:, :, ROOTJ]).abs(),
            torch.zeros_like(anc)).sum() / rmsk.sum().clamp(min=1) / 3)
        if VELW > 0:
            mv = msk[:, 1:] & msk[:, :-1]
            dp = pred[:, 1:] - pred[:, :-1]
            dg = Yn[:, 1:] - Yn[:, :-1]
            loss = loss + VELW * (torch.where(mv, (dp - dg).abs(),
                                              torch.zeros_like(dp)).sum()
                                  / mv.sum().clamp(min=1) / 3)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVERY == 0:
            print(f"  [{step}] L1 {loss.item()*100:.2f} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
            net.eval()
            e = evalset(net, ho[:150], f"ho@{step}")
            net.train()
            if e < best:
                best = e
                torch.save({"model": net.state_dict()}, f"{OUTD}/best.pt")
    torch.save({"model": net.state_dict()}, f"{OUTD}/last.pt")
    net.load_state_dict(torch.load(f"{OUTD}/best.pt",
                                   map_location=dev)["model"])
    net.eval()
    print("FINAL (best ckpt):", flush=True)
    evalset(net, ho, "heldout")
    evalset(net, te, f"scene{TESC}")

if __name__ == "__main__":
    main()
