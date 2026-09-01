#!/usr/bin/env python3
"""Probe 56: which slot explains the FEET movement best?

Per clip (3 rx): for each receiver, fit the absolute pelvis trajectory
(moving GT, MidHip j8, (nw,3)) from each single slot's 4 envelope features
(log-energy + 3 Doppler bands); best slot per rx by CV-R^2 (even/odd
windows).  Then the cross-view test: the 3 per-rx best slots TOGETHER
(12 regressors) -> pelvis.  Reference: all 24 slots (96 regressors).
Also same for pelvis SPEED (scalar envelope).

  N=250 python3 diagnostics/54_pelvis_slot.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/cluster/tok/pa-v1"))
ABSD = os.path.expanduser(os.environ.get(
    "ABSD", "~/zerdani/buffer/cluster/tok/pa-v1-relmove/pose_relmove"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/cluster/runs/clusterer/limbtok.pa-v1.r12"))
N = int(os.environ.get("N", "250"))
BANDS = [(2, 10), (10, 40), (40, 150)]
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

def slot_feats(rid):
    """(nw, M, 4) slot envelope features for one recording, or None."""
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
    f = t[:, 1]
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    hard = a.argmax(1)
    F = np.zeros((nw, M, 4))
    for m in range(M):
        sel = hard == m
        acc = np.zeros(nw); np.add.at(acc, w[sel], e[sel])
        F[:, m, 0] = np.log10(acc + 1e-9)
        for bi, (lo, hi) in enumerate(BANDS):
            sb = sel & (f >= lo) & (f < hi)
            acc = np.zeros(nw); np.add.at(acc, w[sb], e[sb])
            F[:, m, bi + 1] = np.log10(acc + 1e-9)
    return F

def cv_r2(E, Y):
    n = len(E)
    if n < 12: return np.nan
    ev = np.arange(n) % 2 == 0
    A = np.c_[E, np.ones(n)]
    beta, *_ = np.linalg.lstsq(A[ev], Y[ev], rcond=None)
    sst = ((Y[~ev] - Y[ev].mean(0)) ** 2).sum()
    return float(1 - ((Y[~ev] - A[~ev] @ beta) ** 2).sum()
                 / max(sst, 1e-12))

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(54)
    for scene in (1, 4):
        ms = man[man.scene == scene].copy()
        ms["ckey"] = ms["name"].str.replace(r"_r\d$", "", regex=True)
        groups = [g for _, g in ms.groupby("ckey")
                  if len(g) == 3 and set(g.node) == {"r1", "r2", "r3"}]
        rng.shuffle(groups)
        best1, comb3, all24, spd1 = [], [], [], []
        bestslots = np.zeros(M)
        t0 = time.time()
        for g in groups:
            if len(best1) >= N: break
            rids = [int(r) for r in g.sort_values("node").rid.values]
            pf = f"{ABSD}/{rids[0]:06d}.npy"
            if not os.path.exists(pf): continue
            P = np.asarray(np.load(pf), np.float32)
            root = P[:, [11, 14]].reshape(len(P), -1)   # ankles (nw, 6)
            fin = np.isfinite(root).all(1)
            Fs = [slot_feats(r) for r in rids]
            if any(F is None for F in Fs): continue
            nw = min(min(len(F) for F in Fs), len(root))
            m_ = fin[:nw]
            if m_.sum() < 16: continue
            Y = root[:nw][m_]
            spdY = np.linalg.norm(np.diff(P[:nw, [11, 14]], axis=1).reshape(nw, -1)[:0], axis=1) if False else np.linalg.norm(np.diff(root[:nw], axis=0), axis=1)
            spdY = np.r_[spdY[:1], spdY][m_][:, None]
            picks = []
            r1s = []
            for F in Fs:
                rs = [cv_r2(F[:nw][m_, m], Y) for m in range(M)]
                mb = int(np.nanargmax(rs))
                picks.append(F[:nw][m_, mb])
                r1s.append(rs[mb])
                bestslots[mb] += 1
            best1.append(float(np.nanmedian(r1s)))
            spd1.append(float(np.nanmedian(
                [max(cv_r2(F[:nw][m_, m], spdY) for m in range(M))
                 for F in Fs])))
            comb3.append(cv_r2(np.concatenate(picks, 1), Y))
            all24.append(cv_r2(np.concatenate(
                [F[:nw][m_].reshape(m_.sum(), -1) for F in Fs], 1), Y))
        A = lambda v: np.nanmedian(v)
        print(f"\n=== SCENE {scene} (N={len(best1)}, "
              f"{(time.time()-t0)/60:.1f}min)  CV-R2 of FEET (2 ankles):", flush=True)
        print(f"  best single slot (per rx, median):  traj {A(best1):+.3f}"
              f"   speed {A(spd1):+.3f}", flush=True)
        print(f"  3 rx-best slots combined (12 reg):  traj {A(comb3):+.3f}",
              flush=True)
        print(f"  all 24 slots (96 reg):              traj {A(all24):+.3f}",
              flush=True)
        print(f"  best-slot histogram s0..s7: "
              f"{(bestslots/max(bestslots.sum(),1)).round(2)}", flush=True)
    print("probe 56 done", flush=True)

if __name__ == "__main__":
    main()
