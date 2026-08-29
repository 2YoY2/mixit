#!/usr/bin/env python3
"""THE main-objective probe: does the pipeline end in clusters of the LIMBS?

clean (hardware-AGC removal, cross-antenna coherent rep) -> normalize
(phase-safe CMN) -> Slepian multitaper STFT -> tokenize (per-TF-bin MUSIC:
each bin -> dominant scatterer's angle phi, delay psi, Doppler f, energy)
-> k-means K=2 per recording (DUET-principle, identity-free, no labels)
-> TEST: the 2 clusters' energy envelopes vs the 2 most-active limbs'
keypoint-GT envelopes (PA pose GT, stored under legacy name imu/),
permutation-free matching, against a rolled null.

matched >> null AND matched >> cross (the wrong permutation) on recordings
where two limbs move decorrelated = the pipeline separates limb-coherent
components without identity -> GO for the learned version (TF-GridNet
backbone + Wavesplit-style clustering + MixIT objective).

  NRID=60 NPROC=6 python3 diagnostics/27_limb_clustering.py
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
NRID = int(os.environ.get("NRID", "60"))     # recordings per prep root
NPROC = int(os.environ.get("NPROC", "6"))
FS, WINF, HOPF = 400.0, 256, 128
KTAP = 4                                     # Slepian tapers (NW=2.5)
L, NPH, NPS = 20, 37, 37
MAXCORR = float(os.environ.get("MAXCORR", "0.7"))
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND]
DEV = ["LW", "RW", "LP", "RP", "HD"]
TAPERS = dpss(WINF, 2.5, KTAP).astype(np.float32)      # (K, WINF)

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
    x = x / g                                            # hardware AGC out
    y = x[:, 1:, :] * np.conj(x[:, :1, :])               # coherent rep
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

PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

def tokenize(y):
    """tokens per (window, in-band bin): [w, f, phi, psi, energy]."""
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)           # phase-safe CMN
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 10: return None, 0
    S = np.empty((KTAP, nw, PBAND.sum(), 2, 57), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))          # (nw, nf)
    floor = np.median(eng)
    toks = []
    for w in range(nw):
        for i in np.where(eng[w] >= floor)[0]:
            subs = np.concatenate(
                [S[:, w, i, :, k:k + L].reshape(KTAP, -1)
                 for k in range(57 - L + 1)], 0)         # (K*38, 40)
            R = (subs.conj().T @ subs) / len(subs)
            ew, ev = np.linalg.eigh(R)
            En = ev[:, :2 * L - 1]
            P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
            j = int(np.argmax(P))
            toks.append((w, FPOS[i], PH[IPH[j]], PS[IPS[j]], eng[w, i]))
    return np.array(toks, np.float64), nw

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
        toks, nw = tokenize(y)
        if toks is None or len(toks) < 8 * nw: return None
        gi = np.asarray(np.load(f"{prep}/imu/{rid:06d}.npy"), np.float32)
        T = min(len(y), len(gi))
        gi = gi[:T]
        g2 = gi.copy()                                   # probe-17 residualize
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(T, np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                      for w in range(nw)])               # (nw, 5)
        order = np.argsort(-G.mean(0))
        li, lj = int(order[0]), int(order[1])
        c12 = corr(G[:, li], G[:, lj])
        if not np.isfinite(c12) or abs(c12) > MAXCORR: return None
        F = np.c_[np.cos(toks[:, 2]), np.sin(toks[:, 2]),
                  np.cos(toks[:, 3]), np.sin(toks[:, 3]),
                  toks[:, 1] / 150.0]
        wgt = np.sqrt(toks[:, 4])
        a = kmeans2(F, wgt)
        E = np.zeros((2, nw))
        for k in (0, 1):
            for w, e in zip(toks[a == k, 0].astype(int), toks[a == k, 4]):
                E[k, w] += e
        if E[0].std() < 1e-12 or E[1].std() < 1e-12: return None
        def score(Gm):
            p1 = np.nanmean([corr(E[0], Gm[:, li]), corr(E[1], Gm[:, lj])])
            p2 = np.nanmean([corr(E[0], Gm[:, lj]), corr(E[1], Gm[:, li])])
            return (max(p1, p2), min(p1, p2))
        m, x = score(G)
        mn, _ = score(np.roll(G, nw // 2, 0))
        return DEV[li], DEV[lj], c12, m, x, mn
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
    M = np.array([r[3] for r in res]); Xc = np.array([r[4] for r in res])
    N = np.array([r[5] for r in res])
    pairs = pd.Series([f"{r[0]}+{r[1]}" for r in res]).value_counts()
    print(f"  limb pairs: {dict(pairs.head(6))}")
    print(f"  matched  corr: median {np.nanmedian(M):+.3f}")
    print(f"  wrong-perm   : median {np.nanmedian(Xc):+.3f}   "
          f"(gap {np.nanmedian(M - Xc):+.3f})")
    print(f"  rolled null  : median {np.nanmedian(N):+.3f}")
    print(f"  matched>null on {np.mean(M > N) * 100:.0f}% of recordings")
print("""
READ: matched >> null -> clusters track LIMB MOTION (not noise).
matched-minus-wrong-perm gap >> 0 -> the two clusters are DIFFERENT limbs,
not two copies of total motion. Both -> limb clusters exist: GO learned
version. gap ~ 0 with matched > null -> clusters see the body but not limbs.""")
