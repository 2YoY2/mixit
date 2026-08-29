#!/usr/bin/env python3
"""Probe 27 redone with RAW aperture tokens instead of MUSIC (user's design):
leave the subcarrier x antenna values as they are. Per TF bin the token is
the bin's dominant complex spatial vector across the coherent aperture
(2 antenna-pairs x 57 subcarriers = 114 complex: amplitude AND phase, phase
referenced to the recording's strongest element), plus Doppler. No steering
model, no grid -- the clustering sees the aperture directly (cACGMM-style
spatial clustering, speech separation's multichannel practice).

Three variants clustered per recording, same scoring as probe 27:
  raw   [Re v, Im v, f]  amplitude+phase aperture
  amp   [|v|, f]         amplitude-only aperture
  dopp  [f]              Doppler-only baseline

  NRID=150 NPROC=6 python3 diagnostics/28_limb_clustering_raw.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from scipy.signal.windows import dpss

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
PREPS = [os.path.expanduser(p) for p in os.environ.get(
    "PREPS", "~/zerdani/buffer/octonet/prep_pa_xrf400,"
             "~/zerdani/buffer/octonet/prep_pa_xrf400t").split(",")]
NRID = int(os.environ.get("NRID", "150"))
NPROC = int(os.environ.get("NPROC", "6"))
FS, WINF, HOPF = 400.0, 256, 128
KTAP = 4
MAXCORR = float(os.environ.get("MAXCORR", "0.7"))
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND]
DEV = ["LW", "RW", "LP", "RP", "HD"]
TAPERS = dpss(WINF, 2.5, KTAP).astype(np.float32)

def read_products(path):
    with h5py.File(path, "r") as h:
        c = h["csi/csi"][...]
        ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    x = (c["real"] + 1j * c["imag"]).astype(np.complex64)
    dt = float(np.median(np.diff(ts)))
    rate, t = None, None
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
    if rate is None:
        rate = 810.0; t = np.arange(x.shape[-1]) / rate
    keep = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., keep], t[keep]
    if float(t[-1]) < 2.0: return None
    x = np.moveaxis(x, -1, 0)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    x = x / g
    y = x[:, 1:, :] * np.conj(x[:, :1, :])
    nb = int(float(t[-1]) * FS)
    if nb < WINF + 2 * HOPF: return None
    yf = y.reshape(len(y), -1)
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, yf.shape[1]), np.complex64)
    np.add.at(s.real, idx, yf.real.astype(np.float32))
    np.add.at(s.imag, idx, yf.imag.astype(np.float32))
    m = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(0, len(good) - 1)]
        m[bad] = m[near]
    return m.reshape(nb, 2, 57)

def tokenize(y):
    """per kept TF bin: dominant complex aperture vector (114,), f, energy."""
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 10: return None, None, None, 0
    S = np.empty((KTAP, nw, PBAND.sum(), 2, 57), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))
    floor = np.median(eng)
    V, wf, we = [], [], []
    for w in range(nw):
        for i in np.where(eng[w] >= floor)[0]:
            M = S[:, w, i].reshape(KTAP, -1)              # (K, 114)
            _, _, Vh = np.linalg.svd(M, full_matrices=False)
            V.append(Vh[0]); wf.append((w, FPOS[i])); we.append(eng[w, i])
    V = np.array(V)                                       # (n, 114) unit vecs
    jref = int(np.argmax(np.abs(V).mean(0)))              # common phase ref
    ph = V[:, jref].copy()
    ph = np.where(np.abs(ph) < 1e-9, 1.0, ph / np.abs(ph))
    V = V * np.conj(ph)[:, None]
    return V, np.array(wf), np.array(we), nw

def kmeans2(F, wgt, iters=30, seeds=8):
    best, bl = None, np.inf
    rng = np.random.default_rng(0)
    for s in range(seeds):
        c = F[rng.choice(len(F), 2, replace=False, p=wgt / wgt.sum())]
        for _ in range(iters):
            d = ((F[:, None] - c[None]) ** 2).sum(-1)
            a = d.argmin(1)
            for k in (0, 1):
                m = a == k
                if m.any(): c[k] = (F[m] * wgt[m, None]).sum(0) / wgt[m].sum()
        loss = (wgt * d.min(1)).sum()
        if loss < bl: bl, best = loss, a.copy()
    return best

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def one_rid(job):
    prep, rid, matf = job
    try:
        y = read_products(f"{ROOT}/{matf}")
        if y is None: return None
        V, wf, we, nw = tokenize(y)
        if V is None or len(V) < 8 * nw: return None
        gi = np.asarray(np.load(f"{prep}/imu/{rid:06d}.npy"), np.float32)
        T = min(len(y), len(gi))
        gi = gi[:T]
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(T, np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                      for w in range(nw)])
        order = np.argsort(-G.mean(0))
        li, lj = int(order[0]), int(order[1])
        c12 = corr(G[:, li], G[:, lj])
        if not np.isfinite(c12) or abs(c12) > MAXCORR: return None
        fdim = (wf[:, 1] / 150.0)[:, None]
        variants = (np.c_[V.real, V.imag, fdim],          # amp+phase aperture
                    np.c_[np.abs(V), fdim],               # amplitude-only
                    fdim)                                 # Doppler-only
        wgt = np.sqrt(we)
        out = []
        for F in variants:
            a = kmeans2(F, wgt)
            E = np.zeros((2, nw))
            for k in (0, 1):
                for w, e in zip(wf[a == k, 0].astype(int), we[a == k]):
                    E[k, w] += e
            if E[0].std() < 1e-12 or E[1].std() < 1e-12: return None
            def score(Gm):
                p1 = np.nanmean([corr(E[0], Gm[:, li]), corr(E[1], Gm[:, lj])])
                p2 = np.nanmean([corr(E[0], Gm[:, lj]), corr(E[1], Gm[:, li])])
                return (max(p1, p2), min(p1, p2))
            m, x = score(G)
            mn, _ = score(np.roll(G, nw // 2, 0))
            out += [m, x, mn]
        return (DEV[li], DEV[lj], c12) + tuple(out)
    except Exception:
        return None

rng = np.random.default_rng(0)
for prep in PREPS:
    meta = pd.read_csv(f"{prep}/meta.csv")
    meta = meta[meta.imu_ok == 1]
    cand = meta.iloc[rng.permutation(len(meta))]
    jobs = [(prep, int(r.rid), r.file) for r in cand.itertuples()][:NRID * 4]
    res, tried = [], 0
    with Pool(NPROC) as pool:
        for r in pool.imap_unordered(one_rid, jobs, chunksize=1):
            tried += 1
            if r is not None:
                res.append(r)
                if len(res) >= NRID: break
    print(f"\n== {os.path.basename(prep)}: {len(res)} scored / {tried} tried")
    if not res: continue
    pair = np.array(["+".join(sorted((r[0], r[1]))) for r in res])
    A = np.array([r[3:] for r in res], float)
    for nm, o in (("raw amp+phase", 0), ("amp-only", 3), ("dopp-only", 6)):
        M, Xc, N = A[:, o], A[:, o + 1], A[:, o + 2]
        print(f"  [{nm:13s}] matched {np.nanmedian(M):+.3f}  "
              f"wrong-perm {np.nanmedian(Xc):+.3f}  "
              f"null {np.nanmedian(N):+.3f}  win {np.mean(M > N)*100:.0f}%")
    print("  per-pair (n>=6): raw matched/null/win | amp win | dopp win")
    for p in pd.Series(pair).value_counts().index:
        m = pair == p
        if m.sum() < 6: continue
        print(f"    {p:6s} n={m.sum():3d}: {np.nanmedian(A[m, 0]):+.3f}/"
              f"{np.nanmedian(A[m, 2]):+.3f}/{np.mean(A[m, 0] > A[m, 2])*100:3.0f}%"
              f" | {np.mean(A[m, 3] > A[m, 5])*100:3.0f}%"
              f" | {np.mean(A[m, 6] > A[m, 8])*100:3.0f}%")
print("""
READ: raw >> dopp -> the aperture pattern (amplitude+phase) adds limb
information beyond Doppler trajectories. amp ~ raw -> amplitude carries it,
phase adds nothing. raw ~ dopp -> aperture still mute; trajectories only.""")
