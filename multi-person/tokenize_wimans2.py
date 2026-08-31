#!/usr/bin/env python3
"""WiMANS tokenizer v2 — empty-room-anchored ADDITIVE clean, per-RX tokens.

Replaces the product-domain pipeline (which erased absolute phase/delay and
averaged the array).  Per packet, the measured EMPTY ROOM is the reference:

  x_t = g_t * e^{j th_t} * e^{j a_t k} * (room + persons + noise)

  1. STO ramp a_t: phase slope across subcarriers of <x_t, room>
  2. complex c_t (gain*CFO): least-squares scalar fit after de-ramping
  3. sanitize  s_t = deramp(x_t)/c_t     (absolute phase+delay PRESERVED)
  4. de-room   D_t = s_t - room          (additive person channels)

Template bootstrap: empty packets aligned within-recording to a reference
packet, averaged; recording templates aligned to a master and averaged.
(Template is defined up to ONE fixed calibration per (env,band) — a common
basis, harmless.)

Tokens: per RX antenna (NEVER averaged across RX).  Slepian STFT of D over
that RX's 3tx x 30subc block; a (rx, window, doppler) bin becomes a token
iff its energy beats the EMPTY-ROOM noise floor (PCTL percentile of the
same bin statistic over 0-user recordings run through the identical
pipeline) — so token COUNT is an absolute occupancy measure.
Features: [w, f, aod, psi, logE, rx]
  aod  phase progression across the 3 TX antennas (angle of departure)
  psi  phase slope across subcarriers (delay — preserved by this clean)

  NPROC=8 python3 multi-person/tokenize_wimans2.py
"""
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal.windows import dpss

W = os.path.expanduser(os.environ.get("WIMANS", "~/zerdani/buffer/wimans"))
OUT = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/octonet/wimans_tokens2"))
NPROC = int(os.environ.get("NPROC", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NEMPTY = int(os.environ.get("NEMPTY", "40"))
PCTL = float(os.environ.get("PCTL", "99"))
FS, WINF, HOPF = 400.0, 256, 32
KTAP = 4
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND].astype(np.float32)
TAPERS = dpss(WINF, 2.5, KTAP).astype(np.float32)
K30 = np.arange(30)

def read_packets(label):
    """raw packets + times: (T, 3rx, 3tx, 30), AGC removed per packet."""
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
    x = np.stack(cs)
    t = np.array(ts)
    t = np.unwrap(t, period=2 ** 32) * 1e-6
    t = t - t[0]
    if not (0.5 < t[-1] < 30):
        t = np.arange(len(x)) / (len(x) / 3.0)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-12
    return (x / g), t

def fit_pack(x, R):
    """x, R (3,3,30) -> sanitized s (absolute frame of R), or None."""
    P = (x * np.conj(R)).sum(axis=(0, 1))                # (30,)
    d = (P[1:] * np.conj(P[:-1])).sum()
    if np.abs(d) < 1e-12: return None
    a = np.angle(d)
    xr = x * np.exp(-1j * a * K30)[None, None, :]
    c = (xr * np.conj(R)).sum() / ((np.abs(R) ** 2).sum() + 1e-12)
    if np.abs(c) < 0.05: return None
    return (xr / c).astype(np.complex64)

def rec_template(label):
    """within-recording coherent mean, aligned to its median-power packet."""
    r = read_packets(label)
    if r is None: return None
    x, _ = r
    pw = (np.abs(x) ** 2).sum(axis=(1, 2, 3))
    ref = x[np.argsort(pw)[len(pw) // 2]]
    acc, n = np.zeros_like(ref), 0
    for p in x[:: max(1, len(x) // 400)]:                # ~400 packets enough
        s = fit_pack(p, ref)
        if s is not None: acc, n = acc + s, n + 1
    return acc / n if n >= 50 else None

def sanitize_stream(label, R):
    """-> D (nb, 3, 3, 30) person channels on the 400 Hz grid, or None."""
    r = read_packets(label)
    if r is None: return None
    x, t = r
    nb = int(t[-1] * FS)
    if nb < WINF + 2 * HOPF: return None
    D = np.zeros((len(x), 3, 3, 30), np.complex64)
    ok = np.zeros(len(x), bool)
    for i in range(len(x)):
        s = fit_pack(x[i], R)
        if s is None: continue
        D[i] = s - R
        ok[i] = True
    if ok.mean() < 0.5: return None
    idx = np.minimum((t * FS).astype(int), nb - 1)[ok]
    Df = D[ok].reshape(ok.sum(), -1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, Df.shape[1]), np.complex64)
    np.add.at(s.real, idx, Df.real)
    np.add.at(s.imag, idx, Df.imag)
    mg = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(
            0, len(good) - 1)]
        mg[bad] = mg[near]
    return mg.reshape(nb, 3, 3, 30)

def stft_rx(D):
    """-> S (rx, KTAP, nw, nf, 3tx, 30), eng (rx, nw, nf)."""
    T = len(D)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None, None, 0
    nf = int(PBAND.sum())
    S = np.empty((3, KTAP, nw, nf, 3, 30), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None, None]
        for w in range(nw):
            seg = np.fft.fft(D[w * HOPF:w * HOPF + WINF] * tap,
                             axis=0)[PBAND]                 # (nf,3,3,30)
            S[:, k, w] = np.transpose(seg, (1, 0, 2, 3))
    eng = (np.abs(S) ** 2).mean(axis=(1, 4, 5))             # (rx, nw, nf)
    return S, eng, nw

def tokens_from(S, eng, nw, floor):
    toks = []
    for rx in range(3):
        ws, fs = np.where(eng[rx] >= floor[rx][None, :])
        for b in range(len(ws)):
            M = S[rx, :, ws[b], fs[b]]                      # (KTAP, 3tx, 30)
            p = (M[:, 1] * np.conj(M[:, 0])).sum() \
                + (M[:, 2] * np.conj(M[:, 1])).sum()
            q = (M[:, :, 1:] * np.conj(M[:, :, :-1])).sum()
            toks.append((float(ws[b]), float(FPOS[fs[b]]),
                         float(np.angle(p)), float(np.angle(q)),
                         float(np.log10(eng[rx, ws[b], fs[b]] + 1e-12)),
                         float(rx)))
    return np.array(toks, np.float32) if toks else None

TMPL, FLOOR = {}, {}

def one(job):
    label, env, band = job
    of = f"{OUT}/tokens/{label}.npz"
    if os.path.exists(of): return 1
    try:
        R = TMPL[(env, float(band))]
        D = sanitize_stream(label, R)
        if D is None: return 0
        S, eng, nw = stft_rx(D)
        if S is None: return 0
        toks = tokens_from(S, eng, nw, FLOOR[(env, float(band))])
        if toks is None:
            toks = np.zeros((0, 6), np.float32)
        np.savez(of, toks=toks, nw=np.int64(nw))
        return 1
    except Exception:
        return 0

def main():
    global TMPL, FLOOR
    os.makedirs(f"{OUT}/tokens", exist_ok=True)
    an = pd.read_csv(f"{W}/annotation.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    an.to_csv(f"{OUT}/manifest.csv", index=False)
    tf = f"{OUT}/templates.npz"
    rng = np.random.default_rng(52)

    if os.path.exists(tf):
        z = np.load(tf)
        for key in z.files:
            if key.startswith("T_"):
                e_, b_ = key[2:].rsplit("_", 1)
                TMPL[(e_, float(b_))] = z[key]
            elif key.startswith("F_"):
                e_, b_ = key[2:].rsplit("_", 1)
                FLOOR[(e_, float(b_))] = z[key]
        print(f"templates+floors loaded ({len(TMPL)})", flush=True)
    else:
        print("=== pass 1: room templates from empty recordings", flush=True)
        store = {}
        with Pool(NPROC) as pool:
            for env in an.environment.unique():
                for band in sorted(an.wifi_band.unique()):
                    g = an[(an.environment == env) & (an.wifi_band == band)
                           & (an.number_of_users == 0)]
                    labs = list(g.label.values)
                    rng.shuffle(labs)
                    tl = [t for t in pool.map(rec_template, labs[:NEMPTY])
                          if t is not None]
                    master = tl[0]
                    acc, n = master.copy(), 1
                    for t in tl[1:]:
                        s = fit_pack(t, master)
                        if s is not None: acc, n = acc + s, n + 1
                    TMPL[(env, float(band))] = (acc / n).astype(np.complex64)
                    print(f"  {env}/{band}: template from {n}/{len(tl)} "
                          f"empties", flush=True)
        print("=== pass 1b: noise floors (empties through the pipeline)",
              flush=True)
        for (env, band), R in TMPL.items():
            g = an[(an.environment == env) & (an.wifi_band == band)
                   & (an.number_of_users == 0)]
            labs = list(g.label.values)
            rng.shuffle(labs)
            es = []
            for lb in labs[:NEMPTY]:
                D = sanitize_stream(lb, R)
                if D is None: continue
                _, eng, nw = stft_rx(D)
                if eng is not None: es.append(eng)
            allE = np.concatenate([e.reshape(3, -1, e.shape[2])
                                   for e in es], 1)
            FLOOR[(env, band)] = np.percentile(
                allE, PCTL, axis=1).astype(np.float32)      # (3, nf)
            print(f"  {env}/{band}: floor from {len(es)} empties", flush=True)
            store = None
        np.savez(tf, **{f"T_{e}_{b}": v for (e, b), v in TMPL.items()},
                 **{f"F_{e}_{b}": v for (e, b), v in FLOOR.items()})

    jobs = [(r.label, r.environment, r.wifi_band) for r in an.itertuples()]
    rng.shuffle(jobs)
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"=== pass 2: tokenize {len(jobs)} samples", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            done += r
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(jobs)} ({done} ok)", flush=True)
    print(f"wimans tokenize2 done: {done}/{len(jobs)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
