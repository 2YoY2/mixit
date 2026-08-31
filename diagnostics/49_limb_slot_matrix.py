#!/usr/bin/env python3
"""Probe 49: limbtok12 ALL-LIMB slot matrix — the persontok question for limbs.

The gate (eval_limbtok) only scored the top-2 active limbs.  Here: full
8-slot x 5-limb correlation matrix per rooms-4/5 recording, Hungarian best
assignment over the k ACTIVE limbs (mean envelope >= 0.3 * top limb's),
bucketed by k — does every active limb get its own exclusive slot, and does
it hold at k=4-5 like persontok held at n=5?
  matched   mean corr of Hungarian-assigned (slot, limb) pairs
  wrongperm same slots, cyclically mis-assigned
  null      Hungarian score vs half-rolled GT envelopes
  win       matched > null

  MAXN=400 python3 diagnostics/49_limb_slot_matrix.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
MAXN = int(os.environ.get("MAXN", "400"))
ACTTH = float(os.environ.get("ACTTH", "0.3"))
HOPF, WINF = 128, 256
DEV5 = ["LW", "RW", "LP", "RP", "HD"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]
print(f"ckpt {CKPT} step {ck['step']} M={M} D={D} NL={NL} dev={dev}",
      flush=True)

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

model = SetSep()
model.load_state_dict(ck["model"])
model = model.to(dev).eval()

def corrm(E, G):
    """E (M,nw), G (nw,k) -> (M,k) corr matrix, nan->0."""
    C = np.zeros((len(E), G.shape[1]))
    for m in range(len(E)):
        for j in range(G.shape[1]):
            a, b = E[m], G[:, j]
            if a.std() < 1e-12 or b.std() < 1e-12: continue
            C[m, j] = np.corrcoef(a, b)[0, 1]
    return np.nan_to_num(C)

def hung(C):
    r, c = linear_sum_assignment(-C)
    return r, c, float(C[r, c].mean())

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    te = man[man.split == "test"].reset_index(drop=True)
    rng = np.random.default_rng(49)
    ids = rng.permutation(te.rid.values)[:MAXN]
    res = {k: [] for k in range(1, 6)}
    slotpick = {k: [] for k in range(1, 6)}
    t0 = time.time()
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
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                      for w in range(nw)])
        if len(G) < 8: continue
        mu = G.mean(0)
        act = np.where(mu >= ACTTH * mu.max())[0]
        k = len(act)
        if k < 1: continue
        Gk = G[:, act]
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
        for m in range(M):
            np.add.at(Em[m], widx, a[:, m] * e)
        C = corrm(Em, Gk)
        r, c, matched = hung(C)
        wrong = np.nan
        if k > 1:
            rr = np.roll(r, 1)
            wrong = float(C[rr, c].mean())
        Cn = corrm(Em, np.roll(Gk, nw // 2, 0))
        _, _, null = hung(Cn)
        # classification view: does each matched slot correlate MOST with
        # its own limb (chance 1/k)?  and do limbs' greedy best slots
        # collide?
        acc = float(np.mean([np.argmax(C[r[i]]) == c[i]
                             for i in range(k)])) if k > 1 else np.nan
        greedy = C.argmax(0)                       # best slot per limb
        nocoll = float(len(set(greedy.tolist())) == k)
        res[k].append((matched, wrong, null, acc, nocoll))
        slotpick[k].append(len(set(r)))
        if (n_ + 1) % 100 == 0:
            print(f"  {n_+1}/{len(ids)} {(time.time()-t0)/60:.1f}min",
                  flush=True)

    print(f"\n=== all-limb Hungarian (ACTTH={ACTTH}, rooms 4/5)", flush=True)
    print("  k-active   N   matched  wrongperm   null    win   "
          "cls-acc (chance)  no-collision", flush=True)
    for k in range(1, 6):
        if not res[k]: continue
        a = np.array(res[k], float)
        w = f"{np.nanmedian(a[:,1]):+.3f}" if k > 1 else "   -  "
        acc = f"{np.nanmean(a[:,3]):.3f} ({1/k:.2f})" if k > 1 else "   -   "
        print(f"     {k}     {len(a):4d}  {np.nanmedian(a[:,0]):+.3f}   "
              f"{w}    {np.nanmedian(a[:,2]):+.3f}  "
              f"{np.mean(a[:,0] > a[:,2])*100:3.0f}%   {acc}      "
              f"{np.nanmean(a[:,4]):.2f}", flush=True)
    print("probe 49 done", flush=True)

if __name__ == "__main__":
    main()
