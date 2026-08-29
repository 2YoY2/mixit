#!/usr/bin/env python3
"""Gate for the learned limb-clustering model (train_limbtok best.pt) on
rooms 4/5 -- the probe-27 battery at the learned level, against the
zero-learning Doppler k-means control scored on the SAME recordings.

Per test recording (two-limb-decorrelated criterion as probes 27/28):
  model:  slot envelopes -> best ordered slot pair vs top-2 GT limbs
          matched | wrong-perm (same pair, swapped) | rolled null
  control: k-means K=2 on Doppler only (probe 28's dopp arm), same scoring
Report: overall + per limb pair + paired model-vs-control wins.

  CKPT=best.pt python3 eval/eval_limbtok.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK  = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
MAXN = int(os.environ.get("MAXN", "0"))          # 0 = all test recordings
MAXCORR = float(os.environ.get("MAXCORR", "0.7"))
HOPF, WINF = 128, 256
DEV5 = ["LW", "RW", "LP", "RP", "HD"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
assert ck.get("limbtok"), "not a train_limbtok checkpoint"
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]
print(f"ckpt {CKPT} step {ck['step']} cfg {ck['cfg']} dev={dev}", flush=True)

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

import time
model = SetSep()
model.load_state_dict(ck["model"])
for attempt in range(10):
    try:
        model = model.to(dev).eval(); break
    except RuntimeError:
        print(f"to({dev}) retry {attempt+1}", flush=True); time.sleep(60)

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def score_envs(E, G, li, lj):
    """E (K,nw): best ordered pair -> (matched, swapped-of-that-pair)."""
    K = len(E)
    C = np.zeros((K, 2))
    for m in range(K):
        C[m, 0] = corr(E[m], G[:, li]); C[m, 1] = corr(E[m], G[:, lj])
    C = np.nan_to_num(C)
    best, bswap = -2, 0.0
    for m1 in range(K):
        for m2 in range(K):
            if m1 == m2: continue
            v = (C[m1, 0] + C[m2, 1]) / 2
            if v > best:
                best, bswap = v, (C[m1, 1] + C[m2, 0]) / 2
    return best, bswap

def kmeans1d(f, wgt):
    c = np.percentile(f, [25, 75]).astype(np.float64)
    for _ in range(25):
        a = (np.abs(f[:, None] - c[None]) ** 2).argmin(1)
        for k in (0, 1):
            m = a == k
            if m.any(): c[k] = (f[m] * wgt[m]).sum() / wgt[m].sum()
    return a

man = pd.read_csv(f"{TOK}/manifest.csv")
te = man[man.split == "test"].reset_index(drop=True)
rng = np.random.default_rng(0)
rows = []
ids = te.rid.values
if MAXN and len(ids) > MAXN:
    ids = rng.permutation(ids)[:MAXN]
for n_, rid in enumerate(ids):
    rid = int(rid)
    tf, gf = f"{TOK}/tokens/{rid:06d}.npz", f"{TOK}/imu/{rid:06d}.npy"
    if not (os.path.exists(tf) and os.path.exists(gf)): continue
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    gi = np.asarray(np.load(gf), np.float32)
    g2 = gi.copy()
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0) for w in range(nw)])
    order = np.argsort(-G.mean(0))
    li, lj = int(order[0]), int(order[1])
    c12 = corr(G[:, li], G[:, lj])
    if not np.isfinite(c12) or abs(c12) > MAXCORR: continue
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    e = (10.0 ** le).astype(np.float32)
    widx = t[:, 0].astype(int)
    with torch.no_grad():
        a = model(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    Em = np.zeros((M, nw))
    for k in range(M):
        np.add.at(Em[k], widx, a[:, k] * e)
    ak = kmeans1d(t[:, 1], np.sqrt(e))
    Ek = np.zeros((2, nw))
    for k in (0, 1):
        np.add.at(Ek[k], widx, (ak == k) * e)
    Gr = np.roll(G, nw // 2, 0)
    m_m, m_x = score_envs(Em, G, li, lj)
    m_n, _ = score_envs(Em, Gr, li, lj)
    k_m, k_x = score_envs(Ek, G, li, lj)
    k_n, _ = score_envs(Ek, Gr, li, lj)
    rows.append(("+".join(sorted((DEV5[li], DEV5[lj]))),
                 m_m, m_x, m_n, k_m, k_x, k_n))
    if (n_ + 1) % 1000 == 0: print(f"  {n_+1}/{len(ids)}", flush=True)

A = np.array([r[1:] for r in rows], float)
pair = np.array([r[0] for r in rows])
print(f"\n{len(rows)} test recordings scored (two-limb criterion)")
for nm, o in (("MODEL  ", 0), ("control", 3)):
    Mm, Xc, N = A[:, o], A[:, o + 1], A[:, o + 2]
    print(f"  [{nm}] matched {np.nanmedian(Mm):+.3f}  wrong-perm "
          f"{np.nanmedian(Xc):+.3f}  (gap {np.nanmedian(Mm-Xc):+.3f})  "
          f"null {np.nanmedian(N):+.3f}  win {np.mean(Mm > N)*100:.0f}%")
print(f"  paired: model beats control on "
      f"{np.mean(A[:, 0] > A[:, 3])*100:.0f}% of recordings")
print("\nper-pair (n>=20): model matched/gap/win | control win")
for p in pd.Series(pair).value_counts().index:
    m = pair == p
    if m.sum() < 20: continue
    print(f"  {p:6s} n={m.sum():4d}: {np.nanmedian(A[m,0]):+.3f}/"
          f"{np.nanmedian(A[m,0]-A[m,1]):+.3f}/"
          f"{np.mean(A[m,0] > A[m,2])*100:3.0f}%"
          f" | {np.mean(A[m,3] > A[m,5])*100:3.0f}%")
print("""
READ: model win >> control win AND per-pair LW+RW gap > 0 -> learned limb
clusters, laterality included, transfer to unseen rooms: the deep-clustering
road delivers. gap ~ 0 -> slots track activity, not limbs.""")
