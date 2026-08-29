#!/usr/bin/env python3
"""Linear-probe information test: is per-limb information PRESENT and linearly
extractable in the signal at all -- independent of any routing loss?

For each representation (raw island energies / CMN-modulation energies):
  X = per-frame per-channel dynamic energy (0.25 s frames), z-scored
  y = residualized limb envelopes (limb-specific motion), per frame
  ridge fit on TRAIN recordings -> r on (a) HELD-OUT recordings of the same
  rooms, (b) rooms 4/5. Null: targets circularly shifted per recording.

READ: heldout r >> null -> limb info exists linearly; a proper objective can
extract it. heldout r ~ null for every representation -> the information is
not extractably present at this aperture; the limb program has a measured
physical boundary, not an engineering one.

  python3 diagnostics/19_limb_linprobe.py
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

TR  = os.path.expanduser(os.environ.get("PREP_TR", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
TE  = os.path.expanduser(os.environ.get("PREP_TE", "~/zerdani/buffer/octonet/prep_pa_xrf400t"))
NTR = int(os.environ.get("NTR", "300"))
NHO = int(os.environ.get("NHO", "120"))
NTE = int(os.environ.get("NTE", "200"))
FRA, HOP = 100, 50
LAM = float(os.environ.get("LAM", "100"))
DEV = ["LW", "RW", "LP", "RP", "HD"]

def load_rec(root, rid, cmn):
    x = np.asarray(np.load(f"{root}/streams/{rid:06d}.npy"), np.float32)
    gi = np.asarray(np.load(f"{root}/imu/{rid:06d}.npy"), np.float32)
    T = min(len(x), len(gi))
    x, gi = x[:T], gi[:T]
    if cmn:
        sa = x.mean(0)
        aa = np.abs(sa[:90]); ga = np.maximum(aa, 0.05 * np.median(aa) + 1e-9)
        zb = sa[90:177] + 1j * sa[177:264]
        gz = np.abs(zb); thr = 0.05 * np.median(gz) + 1e-9
        zb = np.where(gz < thr, thr + 0j, zb)
        vz = (x[:, 90:177] + 1j * x[:, 177:264]) / zb
        x = np.concatenate([x[:, :90] / ga - 1, vz.real - 1, vz.imag], 1)
    hp = x - uniform_filter1d(x, 201, axis=0)
    g2 = gi.copy()
    for i in range(5):
        oth = [j for j in range(5) if j != i]
        A = np.c_[gi[:, oth], np.ones(T, np.float32)]
        b, *_ = np.linalg.lstsq(A, gi[:, i], rcond=None)
        g2[:, i] = np.clip(gi[:, i] - A @ b, 0, None)
    nf = (T - FRA) // HOP + 1
    if nf < 6: return None
    ix = np.arange(nf)[:, None] * HOP + np.arange(FRA)[None, :]
    X = (hp[ix] ** 2).mean(1)                       # (nf, 264)
    Y = g2[ix].mean(1)                              # (nf, 5)
    return X, Y

def frames(root, rids, cmn):
    out = []
    for rid in rids:
        try:
            r = load_rec(root, int(rid), cmn)
            if r: out.append(r)
        except Exception:
            pass
    return out

def fit_ridge(data):
    X = np.concatenate([d[0] for d in data]); Y = np.concatenate([d[1] for d in data])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xz = (X - mu) / sd
    G = Xz.T @ Xz + LAM * np.eye(X.shape[1])
    B = np.linalg.solve(G, Xz.T @ (Y - Y.mean(0)))
    return mu, sd, B, Y.mean(0)

def score(data, mu, sd, B, ym, null=False):
    rs = np.full((len(data), 5), np.nan)
    for k, (X, Y) in enumerate(data):
        P = ((X - mu) / sd) @ B + ym
        Yv = np.roll(Y, len(Y) // 2, axis=0) if null else Y
        for i in range(5):
            if Yv[:, i].std() > 1e-9 and P[:, i].std() > 1e-9:
                rs[k, i] = np.corrcoef(P[:, i], Yv[:, i])[0, 1]
    return np.nanmedian(rs, 0)

mtr = pd.read_csv(f"{TR}/meta.csv")
mtr = mtr[mtr.imu_ok == 1]
mte = pd.read_csv(f"{TE}/meta.csv")
mte = mte[mte.imu_ok == 1]
rng = np.random.default_rng(0)
rids = rng.permutation(mtr.rid.values)
tr_ids, ho_ids = rids[:NTR], rids[NTR:NTR + NHO]
te_ids = rng.permutation(mte.rid.values)[:NTE]
print(f"train {len(tr_ids)} | heldout(same rooms) {len(ho_ids)} | rooms45 {len(te_ids)}")

for cmn, nm in ((0, "raw islands"), (1, "CMN modulation")):
    dtr = frames(TR, tr_ids, cmn)
    dho = frames(TR, ho_ids, cmn)
    dte = frames(TE, te_ids, cmn)
    mu, sd, B, ym = fit_ridge(dtr)
    r_tr = score(dtr, mu, sd, B, ym)
    r_ho = score(dho, mu, sd, B, ym)
    r_te = score(dte, mu, sd, B, ym)
    r_nl = score(dho, mu, sd, B, ym, null=True)
    print(f"\n== {nm} ==")
    print(f"{'limb':5s}{'train':>8s}{'heldout':>9s}{'rooms45':>9s}{'null':>7s}")
    for i in range(5):
        print(f"{DEV[i]:5s}{r_tr[i]:8.3f}{r_ho[i]:9.3f}{r_te[i]:9.3f}{r_nl[i]:7.3f}")
print("""
READ: heldout >> null -> limb-specific info is linearly present in-domain;
rooms45 > null too -> some of it is even room-invariant. all ~ null -> the
information is not extractably present; measured boundary, close the program.
""")
