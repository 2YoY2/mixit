#!/usr/bin/env python3
"""Probe 55: does the pelvis slot carry an ANGLE per receiver?

Per clip: per rx, pick the best pelvis-explaining slot (probe 54 rule),
then read that slot's energy-weighted circular-mean token angle phi per
window.  Checks:
  conc     is the slot's angle well-defined (circular concentration 0..1)
  d(rx,rx) are the 3 receivers' mean angles DISTINCT (pairwise circ dist)
  track    does the angle trajectory follow the pelvis: median best
           |corr(dphi_rx(t), pelvis coord)| and CV-R^2 of the pelvis
           trajectory from the 3 dphi(t) curves (4 params)

  N=250 python3 diagnostics/55_pelvis_slot_angle.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/cluster/tok/pa-v1"))
ABSD = os.path.expanduser(os.environ.get(
    "ABSD", "~/zerdani/buffer/cluster/tok/pa-v1-absgt/pose_abs"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/cluster/runs/clusterer/limbtok.pa-v1.r12"))
N = int(os.environ.get("N", "250"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

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

def slot_env_phi(rid):
    """-> (env (nw,M), phi (nw,M), conc (nw,M)) or None."""
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
    phi = t[:, 2]
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    hard = a.argmax(1)
    env = np.zeros((nw, M)); zs = np.zeros((nw, M)); zc = np.zeros((nw, M))
    for m in range(M):
        s_ = hard == m
        np.add.at(env[:, m], w[s_], e[s_])
        np.add.at(zs[:, m], w[s_], e[s_] * np.sin(phi[s_]))
        np.add.at(zc[:, m], w[s_], e[s_] * np.cos(phi[s_]))
    ph = np.arctan2(zs, zc)
    conc = np.sqrt(zs ** 2 + zc ** 2) / (env + 1e-12)
    return np.log10(env + 1e-9), ph, conc

def cv_r2(E, Y):
    n = len(E)
    if n < 12: return np.nan
    ev = np.arange(n) % 2 == 0
    A = np.c_[E, np.ones(n)]
    beta, *_ = np.linalg.lstsq(A[ev], Y[ev], rcond=None)
    sst = ((Y[~ev] - Y[ev].mean(0)) ** 2).sum()
    return float(1 - ((Y[~ev] - A[~ev] @ beta) ** 2).sum()
                 / max(sst, 1e-12))

def circd(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(55)
    for scene in (1, 4):
        ms = man[man.scene == scene].copy()
        ms["ckey"] = ms["name"].str.replace(r"_r\d$", "", regex=True)
        groups = [g for _, g in ms.groupby("ckey")
                  if len(g) == 3 and set(g.node) == {"r1", "r2", "r3"}]
        rng.shuffle(groups)
        concs, pdists, corrs, r2s = [], [], [], []
        t0 = time.time()
        for g in groups:
            if len(concs) >= N: break
            rids = [int(r) for r in g.sort_values("node").rid.values]
            pf = f"{ABSD}/{rids[0]:06d}.npy"
            if not os.path.exists(pf): continue
            root = np.asarray(np.load(pf), np.float32)[:, 8]
            fin = np.isfinite(root).all(1)
            outs = [slot_env_phi(r) for r in rids]
            if any(o is None for o in outs): continue
            nw = min(min(len(o[0]) for o in outs), len(root))
            m_ = fin[:nw]
            if m_.sum() < 16: continue
            Y = root[:nw][m_]
            dphis, mus, cbar = [], [], []
            for env, ph, cc in outs:
                rs = [cv_r2(env[:nw][m_, mm][:, None], Y)
                      for mm in range(M)]
                mb = int(np.nanargmax(rs))
                p_ = ph[:nw][m_, mb]
                zc = np.cos(p_).mean(); zs = np.sin(p_).mean()
                mu = np.arctan2(zs, zc)
                mus.append(mu)
                dphis.append(np.arctan2(np.sin(p_ - mu), np.cos(p_ - mu)))
                cbar.append(float(cc[:nw][m_, mb].mean()))
            concs.append(float(np.mean(cbar)))
            pdists.append(float(np.mean([circd(mus[0], mus[1]),
                                         circd(mus[0], mus[2]),
                                         circd(mus[1], mus[2])])))
            cs = []
            for dp in dphis:
                if dp.std() < 1e-9: continue
                cs.append(max(abs(np.corrcoef(dp, Y[:, k])[0, 1])
                              for k in range(3) if Y[:, k].std() > 1e-9))
            if cs: corrs.append(float(np.median(cs)))
            r2s.append(cv_r2(np.stack(dphis, 1), Y))
        print(f"\n=== SCENE {scene} (N={len(concs)}, "
              f"{(time.time()-t0)/60:.1f}min)", flush=True)
        print(f"  angle concentration of pelvis slot: "
              f"{np.nanmedian(concs):.3f}  (1 = sharp bearing)", flush=True)
        print(f"  pairwise inter-rx angle distance:   "
              f"{np.degrees(np.nanmedian(pdists)):.1f} deg", flush=True)
        print(f"  best |corr(dphi, pelvis coord)|:    "
              f"{np.nanmedian(corrs):.3f}", flush=True)
        print(f"  CV-R2 pelvis traj from 3 dphi(t):   "
              f"{np.nanmedian(r2s):+.3f}", flush=True)
    print("probe 55 done", flush=True)

if __name__ == "__main__":
    main()
