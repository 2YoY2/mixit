#!/usr/bin/env python3
"""Matching-pursuit tokenizer — the MUSIC replacement (user: MUSIC is the
worst SR algorithm in the wild; here its sins are specific: coherent
multipath breaks the subspace, D=1 drops multi-limb bins, the 37-bin grid
quantizes phi/psi, and amplitudes are discarded).

Per TF bin: greedy deflation on the subband-snapshot matrix —
  Bartlett peak on the grid -> parabolic OFF-GRID refinement (phi,psi) ->
  per-snapshot complex amplitudes by projection -> de-ramped mean phase ->
  deflate -> repeat (K<=3, stop when atom energy < RESID x first atom).
Coherent sources handled natively (no covariance subspace), K>1 keeps
overlapping limbs, off-grid kills the 0.17-rad quantization, amplitudes
make the token domain invertible.

Token: [w, f, phi, psi, log10(binE*share), phase]  (6 cols; col 0-4
compatible with every existing consumer).

Small-set driver: tokenizes the first NSEL scene-1 recordings of the
probe-42 gate's exact sampling order (random_state=42, imu_ok) so the
clusterer can verify on them directly.

  WINF=256 HOPF=32 NSEL=100 NPROC=8 OUT=.../pa_tokens_mp_test \\
  python3 phase2/tokenize_pa_mp.py
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
    "OUT", "~/zerdani/buffer/octonet/pa_tokens_mp_test"))
NSEL = int(os.environ.get("NSEL", "100"))
NPROC = int(os.environ.get("NPROC", "8"))
KMAX = int(os.environ.get("KMAX", "3"))
RESID = float(os.environ.get("RESID", "0.15"))
L, NPH, NPS = tkz.L, tkz.NPH, tkz.NPS
PH, PS = tkz.PH, tkz.PS
IPH, IPS = tkz.IPH, tkz.IPS
GRID = tkz.STEER.conj()                       # (1369, 2L) unit steering
GB_PH, GB_PS = 2 * np.pi / NPH, 2 * np.pi / NPS

def steer(phi, psi):
    a = np.concatenate([np.exp(1j * psi * np.arange(L)),
                        np.exp(1j * phi) * np.exp(1j * psi * np.arange(L))])
    return (a / np.sqrt(2 * L)).astype(np.complex64)

def para(pm, p0, pp):
    d = pm - 2 * p0 + pp
    if abs(d) < 1e-12: return 0.0
    return float(np.clip(0.5 * (pm - pp) / d, -0.5, 0.5))

def mp_bin(A):
    """greedy atoms of one bin's snapshot matrix A (nsnap, 2L)"""
    atoms, e0 = [], None
    A = A.copy()
    for _ in range(KMAX):
        R = (A.conj().T @ A) / len(A)
        P = np.real(((GRID @ R) * GRID.conj()).sum(1))
        j = int(np.argmax(P))
        iph, ips = int(IPH[j]), int(IPS[j])
        jm = lambda di, dj: ((iph + di) % NPH) * NPS + (ips + dj) % NPS
        dphi = para(P[jm(-1, 0)], P[j], P[jm(1, 0)]) * GB_PH
        dpsi = para(P[jm(0, -1)], P[j], P[jm(0, 1)]) * GB_PS
        phi = float((PH[iph] + dphi + np.pi) % (2 * np.pi) - np.pi)
        psi = float((PS[ips] + dpsi + np.pi) % (2 * np.pi) - np.pi)
        v = steer(phi, psi)
        e = float(np.real(v.conj() @ R @ v))
        if e0 is None: e0 = e
        if e < RESID * e0 or e <= 0: break
        a = A @ v.conj()                                  # (nsnap,)
        q = np.arange(len(A)) // tkz.KTAP                 # subband offset
        abar = complex((a * np.exp(-1j * psi * q)).mean())
        atoms.append((phi, psi, e, float(np.angle(abar))))
        A = A - np.outer(a, v)
    tot = sum(a[2] for a in atoms) + 1e-12
    return [(p, s, e / tot, ph) for p, s, e, ph in atoms]

def tokenize_mp(y):
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - tkz.WINF) // tkz.HOPF + 1
    if nw < 8: return None, 0
    nf = int(tkz.PBAND.sum())
    S = np.empty((tkz.KTAP, nw, nf, 2, 57), np.complex64)
    for k in range(tkz.KTAP):
        tap = tkz.TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * tkz.HOPF:w * tkz.HOPF + tkz.WINF]
                                 * tap, axis=0)[tkz.PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))
    floor = np.median(eng)
    ws, fs = np.where(eng >= floor)
    if len(ws) < 4 * nw: return None, nw
    toks = []
    for b in range(len(ws)):
        M = S[:, ws[b], fs[b]]                            # (K, 2, 57)
        A = np.concatenate([M[:, :, k:k + L].reshape(tkz.KTAP, -1)
                            for k in range(57 - L + 1)], 0)
        for phi, psi, share, ph in mp_bin(A):
            toks.append((float(ws[b]), float(tkz.FPOS[fs[b]]), phi, psi,
                         float(np.log10(eng[ws[b], fs[b]] * share + 1e-12)),
                         ph))
    if not toks: return None, nw
    return np.array(toks, np.float32), nw

def one(job):
    rid, f = job
    of = f"{OUT}/tokens/{rid:06d}.npz"
    if os.path.exists(of): return 1
    try:
        y = tkz.read_products(os.path.join(tkz.ROOT, f))
        if y is None: return 0
        toks, nw = tokenize_mp(y)
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
    g = g[g.file.isin(ok)].head(NSEL)          # the gate's exact ordering
    jobs = [(int(r.rid), r.file) for r in g.itertuples()]
    print(f"MP tokenize {len(jobs)} recs (WINF={tkz.WINF} HOPF={tkz.HOPF} "
          f"KMAX={KMAX} RESID={RESID}) -> {OUT}", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            done += r
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(jobs)} ({done} ok)", flush=True)
    print(f"mp tokenize done: {done}/{len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
