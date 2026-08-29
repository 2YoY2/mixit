#!/usr/bin/env python3
"""Micro-Doppler probe on the CMN modulation (user's synthesis): divide the
room out FIRST (z/z_bar - 1 on complex islands -- room pattern gone), THEN
STFT -- the spectral structure of the person's modulation, room-normalised.
Probe 19 failed on energy envelopes (motion-blind summaries); this one feeds
the within-frame frequency structure that 400 Hz provably preserves.

Features per 0.64 s window: channel-pooled two-sided modulation spectrum
(|bins| in +-[2..150] Hz), z-scored. Ridge -> residualized limb envelopes.
Rows: CMN-STFT vs raw-STFT (does the normalization matter), scored on
held-out recordings (same rooms) + rooms 4/5 + rolled null.

  python3 diagnostics/20_microdoppler_probe.py
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
WINF, HOPF = 256, 128                      # 0.64 s STFT @400 Hz
LAM = float(os.environ.get("LAM", "100"))
DEV = ["LW", "RW", "LP", "RP", "HD"]
freqs = np.fft.fftfreq(WINF, 1 / 400.0)
FSEL = (np.abs(freqs) >= 2) & (np.abs(freqs) <= 150)

def load_rec(root, rid, cmn):
    x = np.asarray(np.load(f"{root}/streams/{rid:06d}.npy"), np.float32)
    gi = np.asarray(np.load(f"{root}/imu/{rid:06d}.npy"), np.float32)
    T = min(len(x), len(gi))
    if T < WINF + HOPF: return None
    x, gi = x[:T], gi[:T]
    z = x[:, 90:177] + 1j * x[:, 177:264]           # complex islands (T, 87)
    if cmn:
        zb = z.mean(0)
        gz = np.abs(zb); thr = 0.05 * np.median(gz) + 1e-9
        zb = np.where(gz < thr, thr + 0j, zb)
        z = z / zb - 1.0                            # room divided out
    else:
        z = z - z.mean(0)
    g2 = gi.copy()
    for i in range(5):
        oth = [j for j in range(5) if j != i]
        A = np.c_[gi[:, oth], np.ones(T, np.float32)]
        b, *_ = np.linalg.lstsq(A, gi[:, i], rcond=None)
        g2[:, i] = np.clip(gi[:, i] - A @ b, 0, None)
    nw = (T - WINF) // HOPF + 1
    X = np.empty((nw, int(FSEL.sum())), np.float32)
    Y = np.empty((nw, 5), np.float32)
    han = np.hanning(WINF)[:, None]
    for w in range(nw):
        s0 = w * HOPF
        seg = z[s0:s0 + WINF]
        F = np.fft.fft(seg * han, axis=0)
        X[w] = (np.abs(F[FSEL]) ** 2).mean(1)       # channel-pooled spectrum
        Y[w] = g2[s0:s0 + WINF].mean(0)
    X = np.log10(X + 1e-12)                         # log-spectral features
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

mtr = pd.read_csv(f"{TR}/meta.csv"); mtr = mtr[mtr.imu_ok == 1]
mte = pd.read_csv(f"{TE}/meta.csv"); mte = mte[mte.imu_ok == 1]
rng = np.random.default_rng(0)
rids = rng.permutation(mtr.rid.values)
tr_ids, ho_ids = rids[:NTR], rids[NTR:NTR + NHO]
te_ids = rng.permutation(mte.rid.values)[:NTE]
print(f"train {len(tr_ids)} | heldout {len(ho_ids)} | rooms45 {len(te_ids)} | "
      f"{int(FSEL.sum())} spectral bins")

for cmn, nm in ((1, "CMN modulation + STFT"), (0, "raw islands + STFT")):
    dtr = frames(TR, tr_ids, cmn)
    dho = frames(TR, ho_ids, cmn)
    dte = frames(TE, te_ids, cmn)
    mu, sd, B, ym = fit_ridge(dtr)
    r_tr, r_ho = score(dtr, mu, sd, B, ym), score(dho, mu, sd, B, ym)
    r_te, r_nl = score(dte, mu, sd, B, ym), score(dho, mu, sd, B, ym, null=True)
    print(f"\n== {nm} ==")
    print(f"{'limb':5s}{'train':>8s}{'heldout':>9s}{'rooms45':>9s}{'null':>7s}")
    for i in range(5):
        print(f"{DEV[i]:5s}{r_tr[i]:8.3f}{r_ho[i]:9.3f}{r_te[i]:9.3f}{r_nl[i]:7.3f}")
print("""
READ: heldout >> null on the CMN row -> limb info lives in room-normalised
micro-Doppler; the separator must be rebuilt in the STFT basis (masks over
time-frequency bins). CMN row dead but raw row alive -> spectra matter but
normalisation hurts; keep raw-STFT basis. Both dead -> spectral limb info is
also absent; the boundary claim returns, now properly measured.
""")
