#!/usr/bin/env python3
"""Phase-2 token prep: PerceptAlign raw .mats -> MUSIC tokens for the limb
clustering model. The tokenizer (user's spec, validated by probes 25-28):

  raw CSI -> clean (hardware-AGC removal; CFO/SFO/CTO cancelled exactly by
             cross-antenna conjugate products, one LO) -> 400 Hz
  -> normalize (spectral-domain rule: SUBTRACT the complex static -- multipath
     is additive in H; divide by |static| only -- gain is multiplicative;
     phases carrying delay/angle never touched)
  -> Slepian multitaper STFT (K=4, NW=2.5)
  -> per-TF-bin MUSIC over the antenna x subcarrier aperture (batched eigh)
  -> tokens [window, Doppler f, angle phi, delay psi, logE] per kept bin.

One .npz per .mat in $OUT/tokens/{rid:06d}.npz + meta.csv.
Splits: scenes 1-3 train, 4-5 test. Resume-safe (skips existing files).

  OMP_NUM_THREADS=1 NPROC=6 python3 prep/tokenize_pa.py
"""
import os, glob, re
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py
from scipy.signal.windows import dpss

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
OUT = os.path.expanduser(os.environ.get("OUT", "~/zerdani/buffer/octonet/pa_tokens"))
NPROC = int(os.environ.get("NPROC", "6"))
LIMIT = int(os.environ.get("LIMIT", "0"))
FS, WINF, HOPF = 400.0, 256, 128
KTAP, L = 4, 20
NPH, NPS = 37, 37
CHUNK = 128                                   # bins per steering-proj chunk
PAT = re.compile(r"Scene(\d+)/user(\d+)/action(\d+)/([^/]+)/csi_mat/(\d+)-r(\d)\.mat$")
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
    y = x[:, 1:, :] * np.conj(x[:, :1, :])               # CFO/SFO/CTO cancel
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
    yb = y.mean(0)                                       # SUBTRACT static,
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)           # divide |static| only
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None, 0
    nf = int(PBAND.sum())
    S = np.empty((KTAP, nw, nf, 2, 57), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))          # (nw, nf)
    floor = np.median(eng)
    ws, fs = np.where(eng >= floor)
    if len(ws) < 4 * nw: return None, nw
    nbin = len(ws)
    A = np.empty((nbin, KTAP * (57 - L + 1), 2 * L), np.complex64)
    for b in range(nbin):
        M = S[:, ws[b], fs[b]]                           # (K, 2, 57)
        A[b] = np.concatenate([M[:, :, k:k + L].reshape(KTAP, -1)
                               for k in range(57 - L + 1)], 0)
    R = A.conj().transpose(0, 2, 1) @ A                  # (nbin, 40, 40)
    ew, ev = np.linalg.eigh(R)
    # D=1 noise projection via completeness: ||En^H a||^2 = 1 - |v_top^H a|^2
    # (STEER rows are unit-norm conj steering) -- exact, 40x cheaper than
    # projecting on the 39-dim noise subspace explicitly.
    vtop = ev[:, :, -1]                                  # (nbin, 2L)
    sv = STEER @ vtop.T                                  # (grid, nbin)
    P = 1.0 / np.maximum(1.0 - np.abs(sv) ** 2, 1e-6)
    peak = P.argmax(0)
    toks = np.c_[ws.astype(np.float32), FPOS[fs],
                 PH[IPH[peak]].astype(np.float32),
                 PS[IPS[peak]].astype(np.float32),
                 np.log10(eng[ws, fs] + 1e-12).astype(np.float32)]
    return toks.astype(np.float32), nw

def one(job):
    rid, f = job
    of = f"{OUT}/tokens/{rid:06d}.npz"
    mm = PAT.search(f)
    if mm is None: return None
    scene, user, act = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
    clip, take, rx = mm.group(4), int(mm.group(5)), int(mm.group(6))
    row = (rid, f, f"s{scene}_u{user}_a{act}_{clip}_t{take}_r{rx}", f"r{rx}",
           scene, user, act, take, "train" if scene <= 3 else "test")
    if os.path.exists(of):
        try:
            z = np.load(of)
            return row + (int(z["nw"]), len(z["toks"]))
        except Exception:
            pass
    try:
        y = read_products(os.path.join(ROOT, f))
        if y is None: return None
        toks, nw = tokenize(y)
        if toks is None: return None
        np.savez(of, toks=toks, nw=np.int64(nw))
        return row + (nw, len(toks))
    except Exception:
        return None

def main():
    os.makedirs(f"{OUT}/tokens", exist_ok=True)
    files = sorted(os.path.relpath(f, ROOT) for f in
                   glob.glob(f"{ROOT}/Scene*/user*/action*/*/csi_mat/*.mat"))
    jobs = list(enumerate(files))
    # scenes 1,4,5 first: train-core + both test rooms land early so the
    # trainer can be brought up while scenes 2-3 (train bulk) fill in behind
    prio = {"Scene1": 0, "Scene4": 1, "Scene5": 2, "Scene2": 3, "Scene3": 4}
    jobs.sort(key=lambda j: prio.get(j[1].split("/")[0], 9))
    if LIMIT: jobs = jobs[:LIMIT]
    mrows = []
    for rid, f in jobs:                       # manifest upfront: the trainer
        mm = PAT.search(f)                    # starts before this pass ends
        if mm is None: continue
        sc, us, ac = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        cl, tk, rx = mm.group(4), int(mm.group(5)), int(mm.group(6))
        mrows.append((rid, f, f"s{sc}_u{us}_a{ac}_{cl}_t{tk}_r{rx}", f"r{rx}",
                      sc, us, ac, tk, "train" if sc <= 3 else "test"))
    pd.DataFrame(mrows, columns=["rid", "file", "name", "node", "scene",
                                 "subject", "act", "trial", "split"]
                 ).to_csv(f"{OUT}/manifest.csv", index=False)
    print(f"{len(jobs)} mats -> {OUT}", flush=True)
    rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if r is not None: rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(jobs)} ({len(rows)} ok)", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "file", "name", "node", "scene",
                                     "subject", "act", "trial", "split",
                                     "nw", "ntok"]).sort_values("rid")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print(f"\n{len(df)} kept / {len(jobs)}")
    print(df.groupby(["scene", "split"]).agg(recs=("rid", "size"),
                                             ntok_med=("ntok", "median")))

if __name__ == "__main__":
    main()
