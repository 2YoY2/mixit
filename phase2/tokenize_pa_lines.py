#!/usr/bin/env python3
"""Line-spectral tokenizer — super-resolution moved to the DOPPLER axis
(user's diagnosis: SR was wasted on a 2-element spatial aperture while the
signal axis got Fourier-limited STFT; that's why fine windows failed).

Per short window, per recording: model the CMN'd product streams as K<=4
complex exponential LINES (limb micro-Doppler components):
  zero-padded cross-channel spectrum -> off-grid peak (parabolic) ->
  per-channel complex amplitudes by projection -> deflate -> repeat.
Line frequency precision beats 1/T for high-SNR lines -> fine time AND
fine Doppler together.  Spatial is read plainly per line (the only thing
a 2-element aperture supports): phi = weighted antenna-pair phase,
psi = weighted subcarrier phase-ramp slope of the line's spatial vector.

Token: [w, f_line(Hz, continuous), phi, psi, log10(lineE), phase]
~K lines/window -> ~10x fewer, richer tokens than per-bin MUSIC.

Gate target: WINF=128 failed FFT gate at +0.134/57%; this must recover
to >= +0.21/73% (the WINF=256 overlap benchmark) at 2x finer time.

  WINF=128 HOPF=32 NSEL=100 NPROC=8 python3 phase2/tokenize_pa_lines.py
"""
import os, importlib.util
from multiprocessing import Pool
import numpy as np
import pandas as pd

_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tkz", os.path.join(_dir, "tokenize_pa.py"))
tkz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tkz)

TOK = os.path.expanduser(os.environ.get("TOKREF", "~/zerdani/buffer/octonet/pa_tokens"))
PREP1 = os.path.expanduser(os.environ.get(
    "PREP1", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
OUT = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/octonet/pa_tokens_lines"))
NSEL = int(os.environ.get("NSEL", "100"))
NPROC = int(os.environ.get("NPROC", "8"))
KLINES = int(os.environ.get("KLINES", "4"))
RESID = float(os.environ.get("RESID", "0.10"))
NFFT = int(os.environ.get("NFFT", "1024"))
FS = 400.0
WINF, HOPF = tkz.WINF, tkz.HOPF

FGRID = np.fft.fftfreq(NFFT, 1 / FS)
FMASK = (np.abs(FGRID) >= 2) & (np.abs(FGRID) <= 150)
FI = np.where(FMASK)[0]

def lines_window(x):
    """x (T, C) complex -> [(f, a_c (C,), energy)] greedy off-grid lines"""
    T = len(x)
    t = np.arange(T) / FS
    win = np.hanning(T).astype(np.float32)
    xw = x * win[:, None]
    out, e0 = [], None
    for _ in range(KLINES):
        X = np.fft.fft(xw, n=NFFT, axis=0)
        pw = (np.abs(X[FI]) ** 2).sum(1)
        j = int(np.argmax(pw))
        jm, jp = (j - 1) % len(FI), (j + 1) % len(FI)
        d = pw[jm] - 2 * pw[j] + pw[jp]
        off = float(np.clip(0.5 * (pw[jm] - pw[jp]) / d, -0.5, 0.5)) \
            if abs(d) > 1e-20 else 0.0
        f = float(FGRID[FI[j]] + off * (FS / NFFT))
        s = np.exp(2j * np.pi * f * t).astype(np.complex64)
        sw = s * win
        den = float((np.abs(sw) ** 2).sum())
        a = (xw * np.conj(sw[:, None])).sum(0) / den          # (C,)
        e = float((np.abs(a) ** 2).sum() * den)
        if e0 is None: e0 = e
        if e < RESID * e0 or e <= 0: break
        out.append((f, a, e))
        xw = xw - np.outer(sw, a)
    return out

def spatial_of(a):
    """phi (antenna-pair phase) and psi (subcarrier ramp) of a (2,57) line
    vector, energy-weighted plain readout — all a 2-element array supports"""
    w2 = (np.abs(a[0]) * np.abs(a[1]))
    phi = float(np.angle((a[1] * np.conj(a[0]) * w2).sum() /
                         (w2.sum() + 1e-12)))
    r = a[:, 1:] * np.conj(a[:, :-1])
    wr = np.abs(r)
    psi = float(np.angle((r * wr).sum() / (wr.sum() + 1e-12)))
    return phi, psi

def tokenize_lines(y):
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None, 0
    C = dyn.reshape(T, -1)                                    # (T, 114)
    toks = []
    for w in range(nw):
        for f, a, e in lines_window(C[w * HOPF:w * HOPF + WINF]):
            av = a.reshape(2, 57)
            phi, psi = spatial_of(av)
            ph0 = float(np.angle(a[np.argmax(np.abs(a))]))
            toks.append((float(w), abs(f), phi, psi,
                         float(np.log10(e + 1e-12)), ph0))
    if len(toks) < 2 * nw: return None, nw
    return np.array(toks, np.float32), nw

def one(job):
    rid, f = job
    of = f"{OUT}/tokens/{rid:06d}.npz"
    if os.path.exists(of): return 1
    try:
        y = tkz.read_products(os.path.join(tkz.ROOT, f))
        if y is None: return 0
        toks, nw = tokenize_lines(y)
        if toks is None: return 0
        np.savez(of, toks=toks, nw=np.int64(nw))
        return 1
    except Exception:
        return 0

def main():
    os.makedirs(f"{OUT}/tokens", exist_ok=True)
    if not os.path.lexists(f"{OUT}/manifest.csv"):
        os.symlink(f"{TOK}/manifest.csv", f"{OUT}/manifest.csv")
    man = pd.read_csv(f"{TOK}/manifest.csv")
    meta1 = pd.read_csv(f"{PREP1}/meta.csv")
    ok = set(meta1[meta1.imu_ok == 1].file)
    g = man[man.scene == 1].sample(frac=1, random_state=42)
    g = g[g.file.isin(ok)].head(NSEL)
    jobs = [(int(r.rid), r.file) for r in g.itertuples()]
    print(f"[lines] tokenize {len(jobs)} recs (WINF={WINF} HOPF={HOPF} "
          f"K={KLINES}) -> {OUT}", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            done += r
    print(f"[lines] done: {done}/{len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
