#!/usr/bin/env python3
"""WiMANS tokenizer — PA recipe with EMPTY-ROOM normalization (user spec:
clean like PerceptAlign; normalize with the true empty-room print, NOT the
recording's own statistic — so persistent/standing bodies remain in the
DYNAMICS, which is what people-counting needs).

  raw 5300 trace (T pkts, 3rx x 3tx x 30 subc)
  -> per-packet AGC removal (global RMS)
  -> cross-RX conjugate products per TX (CFO/SFO cancel): (T, 2rxpair, 3tx, 30)
  -> 400 Hz uniform binning (PA convention)
  -> pass 1: EMPTY PRINT per (env, band) = mean over 0-user samples of the
     time-mean product; saved to empty_prints.npz
  -> pass 2: normalize (subtract empty print; divide |print| floored),
     Slepian STFT on the VERIFIED overlap grid (WINF=256 HOPF=32),
     per-bin MUSIC tokens [w, f, phi, psi, logE]
     (snapshots = tapers x subband shifts x 3 tx; L=12 over 30 subc).

Jobs shuffled (seed 45) so every (env, band, n_users) cell fills evenly —
correlation probes are valid on partial output.  Resume-safe.

  NPROC=10 python3 phase2/tokenize_wimans.py
"""
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal.windows import dpss

W = os.path.expanduser(os.environ.get("WIMANS", "~/zerdani/buffer/wimans"))
OUT = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/octonet/wimans_tokens"))
NPROC = int(os.environ.get("NPROC", "10"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NEMPTY = int(os.environ.get("NEMPTY", "60"))
FS, WINF, HOPF = 400.0, 256, 32
KTAP, L = 4, 12
NPH, NPS = 37, 37
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND].astype(np.float32)
TAPERS = dpss(WINF, 2.5, KTAP).astype(np.float32)
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))
NSH = 30 - L + 1

def read_products(label):
    f = f"{W}/wifi_csi/mat/{label}.mat"
    if not os.path.exists(f): return None
    try:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        tr = m["trace"]
    except Exception:
        return None
    cs, ts = [], []
    for p in np.atleast_1d(tr):
        c = np.asarray(getattr(p, "csi", None))
        if c is None or c.shape != (3, 3, 30): continue
        cs.append(c.astype(np.complex64))
        ts.append(float(getattr(p, "timestamp_low", 0)))
    if len(cs) < 300: return None
    x = np.stack(cs)                                     # (T, 3rx, 3tx, 30)
    t = np.array(ts)
    t = np.unwrap(t, period=2 ** 32) * 1e-6
    t = t - t[0]
    if not (0.5 < t[-1] < 30):                           # garbage clock
        t = np.arange(len(x)) / (len(x) / 3.0)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-12
    x = x / g                                            # AGC out
    y = x[:, 1:] * np.conj(x[:, :1])                     # (T, 2, 3, 30)
    nb = int(t[-1] * FS)
    if nb < WINF + 2 * HOPF: return None
    yf = y.reshape(len(y), -1)
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, yf.shape[1]), np.complex64)
    np.add.at(s.real, idx, yf.real)
    np.add.at(s.imag, idx, yf.imag)
    mgrid = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(0, len(good) - 1)]
        mgrid[bad] = mgrid[near]
    return mgrid.reshape(nb, 6, 30)                      # 6 = 2rxpair x 3tx

def tokenize(y, emp):
    ga = np.maximum(np.abs(emp), 0.05 * np.median(np.abs(emp)) + 1e-12)
    dyn = ((y - emp[None]) / ga[None]).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None, 0
    nf = int(PBAND.sum())
    S = np.empty((KTAP, nw, nf, 6, 30), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))
    floor = np.median(eng)
    ws, fs = np.where(eng >= floor)
    if len(ws) < 2 * nw: return None, nw
    toks = []
    for b in range(len(ws)):
        M = S[:, ws[b], fs[b]].reshape(KTAP, 2, 3, 30)   # rxpair, tx, subc
        A = np.concatenate(
            [M[:, :, tx, k:k + L].reshape(KTAP, -1)
             for tx in range(3) for k in range(NSH)], 0)  # (KTAP*3*NSH, 2L)
        R = (A.conj().T @ A) / len(A)
        ew, ev = np.linalg.eigh(R)
        vtop = ev[:, -1]
        sv = STEER @ vtop
        P = 1.0 / np.maximum(1.0 - np.abs(sv) ** 2, 1e-6)
        j = int(np.argmax(P))
        toks.append((float(ws[b]), float(FPOS[fs[b]]), float(PH[IPH[j]]),
                     float(PS[IPS[j]]),
                     float(np.log10(eng[ws[b], fs[b]] + 1e-12))))
    return np.array(toks, np.float32), nw

EMPTY = {}

def one(job):
    label, env, band = job
    of = f"{OUT}/tokens/{label}.npz"
    if os.path.exists(of): return 1
    try:
        y = read_products(label)
        if y is None: return 0
        emp = EMPTY[(env, float(band))]
        toks, nw = tokenize(y, emp)
        if toks is None: return 0
        dev = float(np.linalg.norm(y.mean(0) - emp) /
                    (np.linalg.norm(emp) + 1e-12))
        np.savez(of, toks=toks, nw=np.int64(nw), staticdev=np.float32(dev))
        return 1
    except Exception:
        return 0

def main():
    global EMPTY
    os.makedirs(f"{OUT}/tokens", exist_ok=True)
    an = pd.read_csv(f"{W}/annotation.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    an.to_csv(f"{OUT}/manifest.csv", index=False)
    epf = f"{OUT}/empty_prints.npz"
    if os.path.exists(epf):
        z = np.load(epf)
        EMPTY = {(k.rsplit("_", 1)[0], float(k.rsplit("_", 1)[1])): z[k]
                 for k in z.files}
        print(f"empty prints loaded ({len(EMPTY)})", flush=True)
    else:
        print("=== pass 1: empty prints", flush=True)
        rng = np.random.default_rng(45)
        store = {}
        with Pool(NPROC) as pool:
            for env in an.environment.unique():
                for band in sorted(an.wifi_band.unique()):
                    g = an[(an.environment == env) & (an.wifi_band == band)
                           & (an.number_of_users == 0)]
                    labs = list(g.label.values)
                    rng.shuffle(labs)
                    res = [r for r in pool.map(read_products, labs[:NEMPTY])
                           if r is not None]
                    key = f"{env}_{band}"
                    store[key] = np.mean([r.mean(0) for r in res], 0
                                         ).astype(np.complex64)
                    print(f"  {env}/{band}: {len(res)} empty samples",
                          flush=True)
        np.savez(epf, **store)
        EMPTY = {(k.rsplit("_", 1)[0], float(k.rsplit("_", 1)[1])): v
                 for k, v in store.items()}
    jobs = [(r.label, r.environment, r.wifi_band) for r in an.itertuples()]
    rng = np.random.default_rng(45)
    rng.shuffle(jobs)
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"=== pass 2: tokenize {len(jobs)} samples", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            done += r
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(jobs)} ({done} ok)", flush=True)
    print(f"wimans tokenize done: {done}/{len(jobs)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
