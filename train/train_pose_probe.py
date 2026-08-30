#!/usr/bin/env python3
"""Pose-transfer probe: is the limb-cluster model's OUTPUT a universal
representation? Train a small pose head (root-relative 3D joints, what the
PerceptAlign paper predicts) on SCENE 1 only, test on SCENE 4 (unseen room).

Arms, same head architecture, same budget, only the input differs:
  model  frozen limbtok12 slot features: per window, per slot: log energy
         + 3 Doppler-band log energies (M*4 per rx, 3 rx concat)
  raw    no separator: per window, 8 Doppler-band log energies + total
         (9 per rx, 3 rx concat)
Baseline: scene-1 mean pose (the floor any transfer must beat).
Metric: MPJPE (cm, root-relative, masked) on scene-4 clips + scene-1 heldout.

  HOURS=1 python3 train/train_pose_probe.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
OUTD = os.path.expanduser(os.environ.get("OUT", "~/zerdani/buffer/octonet/pose_probe_runs"))
HOURS = float(os.environ.get("HOURS", "1.0"))
STEPS = int(os.environ.get("STEPS", "8000"))
B = int(os.environ.get("B", "32"))
LR = float(os.environ.get("LR", "1e-3"))
H = int(os.environ.get("H", "256"))
SEED = int(os.environ.get("SEED", "0"))
NJ = 15
BANDS = [(2, 10), (10, 40), (40, 150)]
RAWB = np.linspace(2, 150, 9)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
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

import time as _t
sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        print("gpu retry", flush=True); _t.sleep(60)

def rec_feats(rid):
    """(model_feat (nw, M*4), raw_feat (nw, 9), nw) or None."""
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int); f = t[:, 1]
    mf = np.zeros((nw, M * 4), np.float64)
    for m in range(M):
        np.add.at(mf[:, m * 4], w, a[:, m] * e)
        for bi, (lo, hi) in enumerate(BANDS):
            s = (f >= lo) & (f < hi)
            np.add.at(mf[:, m * 4 + 1 + bi], w[s], a[s, m] * e[s])
    rf = np.zeros((nw, 9), np.float64)
    np.add.at(rf[:, 0], w, e)
    for bi in range(8):
        s = (f >= RAWB[bi]) & (f < RAWB[bi + 1])
        np.add.at(rf[:, 1 + bi], w[s], e[s])
    def nz(v):
        v = np.log10(v + 1e-9)
        return ((v - v.mean(0)) / (v.std(0) + 1e-6)).astype(np.float32)
    return nz(mf), nz(rf), nw

def build(scene):
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene == scene].copy()
    man["ck"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for ckey, g in man.groupby("ck"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        pf = f"{TOK}/pose/{rids[0]:06d}.npy"
        if not os.path.exists(pf): continue
        fs = [rec_feats(r) for r in rids]
        if any(f is None for f in fs): continue
        nw = min(f[2] for f in fs)
        P = np.asarray(np.load(pf), np.float32)[:nw]
        if not np.isfinite(P).any(): continue
        out.append((np.concatenate([f[0][:nw] for f in fs], 1),
                    np.concatenate([f[1][:nw] for f in fs], 1), P))
        if len(out) % 500 == 0: print(f"  scene{scene}: {len(out)}", flush=True)
    return out

class Head(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.gru = nn.GRU(fin, H, 2, batch_first=True)
        self.out = nn.Linear(H, NJ * 3)
    def forward(self, x):
        return self.out(self.gru(x)[0]).view(x.shape[0], -1, NJ, 3)

def mpjpe(pred, gt):
    """masked mean per-joint error (cm), root joint excluded. (nw,NJ,3)."""
    m = np.isfinite(gt).all(-1)
    m[:, 8] = False
    if not m.any(): return np.nan
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)
    return float(d[m].mean() * 100)

def run_arm(name, ai, tr, ho, te):
    fin = tr[0][ai].shape[1]
    head = Head(fin).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS / 2: break
        ix = rng.choice(len(tr), B)
        nw = max(len(tr[i][2]) for i in ix)
        X = torch.zeros(B, nw, fin); Y = torch.full((B, nw, NJ, 3), np.nan)
        for k, i in enumerate(ix):
            F, P = tr[i][ai], tr[i][2]
            X[k, :len(F)] = torch.from_numpy(F)
            Y[k, :len(P)] = torch.from_numpy(P)
        X, Y = X.to(dev), Y.to(dev)
        pred = head(X)
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        msk[:, :, 8] = False
        loss = (torch.where(msk, (pred - torch.nan_to_num(Y)).abs(),
                            torch.zeros_like(pred)).sum()
                / msk.sum().clamp(min=1) / 3)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [{name} {step}] L1 {loss.item()*100:.2f} cm", flush=True)
    def ev(ds):
        errs = []
        with torch.no_grad():
            for F_, R_, P in ds:
                F = F_ if ai == 0 else R_
                pr = head(torch.from_numpy(F)[None].to(dev))[0].cpu().numpy()
                errs.append(mpjpe(pr, P))
        return float(np.nanmedian(errs))
    return ev(ho), ev(te)

print(f"frozen sep: {CKPT} step {ck['step']} | building clip sets", flush=True)
tr_all = build(1)
te = build(4)
rng = np.random.default_rng(SEED)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
print(f"scene1 train {len(tr)} / heldout {len(ho)} | scene4 test {len(te)}",
      flush=True)
mu = np.zeros((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([P[:, j][np.isfinite(P[:, j]).all(-1)]
                         for _, _, P in tr if np.isfinite(P[:, j]).any()])
    mu[j] = vs.mean(0) if len(vs) else 0
base_ho = float(np.nanmedian([mpjpe(np.broadcast_to(mu, P.shape), P)
                           for _, _, P in ho]))
base_te = float(np.nanmedian([mpjpe(np.broadcast_to(mu, P.shape), P)
                           for _, _, P in te]))
print(f"\nmean-pose baseline: scene1-ho {base_ho:.1f} cm | scene4 {base_te:.1f} cm",
      flush=True)
for name, ai in (("model", 0), ("raw", 1)):
    h, t = run_arm(name, ai, tr, ho, te)
    print(f"[{name:5s}] MPJPE scene1-ho {h:.1f} cm | scene4 {t:.1f} cm", flush=True)
print("""
READ: model < raw < baseline on scene 4 = the separator's output transfers
pose information to an unseen room better than raw Doppler statistics --
the output is (that much) universal. model ~ baseline = motion features
alone don't carry posture; expected partial ceiling, see caveat in report.""")
