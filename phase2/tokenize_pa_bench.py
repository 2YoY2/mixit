#!/usr/bin/env python3
"""Per-bin estimator BAKE-OFF (user: try other methods before choosing).

Same front end, same overlap grid, same 100 gate recordings, same D=1
token-per-bin format; only the (phi,psi) estimator changes.  All methods
get the same parabolic off-grid refinement, so the estimator is the only
variable.  METHOD env:
  music     top-eigenvector completeness identity (current tokenizer) +
            off-grid refinement
  bartlett  plain beamformer peak of v^H R v (control: is MUSIC helping?)
  capon     MVDR 1/(v^H R^-1 v), diagonal loading 1e-3 tr/40
  fbmusic   forward-backward-averaged covariance -> MUSIC (the textbook
            coherent-multipath fix; J = anti-identity works for our
            Kronecker ULAxULA layout)

  METHOD=capon WINF=256 HOPF=32 NSEL=100 NPROC=8 \\
  OUT=.../pa_tokens_bench_capon python3 phase2/tokenize_pa_bench.py
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
METHOD = os.environ.get("METHOD", "bartlett")
OUT = os.path.expanduser(os.environ.get(
    "OUT", f"~/zerdani/buffer/octonet/pa_tokens_bench_{METHOD}"))
NSEL = int(os.environ.get("NSEL", "100"))
NPROC = int(os.environ.get("NPROC", "8"))
L, NPH, NPS = tkz.L, tkz.NPH, tkz.NPS
PH, PS, IPH, IPS = tkz.PH, tkz.PS, tkz.IPH, tkz.IPS
GRID = tkz.STEER.conj()                       # (1369, 2L) unit steering
GB_PH, GB_PS = 2 * np.pi / NPH, 2 * np.pi / NPS
J40 = np.eye(2 * L)[::-1].astype(np.complex64)

def para(pm, p0, pp):
    d = pm - 2 * p0 + pp
    if abs(d) < 1e-12: return 0.0
    return float(np.clip(0.5 * (pm - pp) / d, -0.5, 0.5))

def refine(P, j):
    iph, ips = int(IPH[j]), int(IPS[j])
    jm = lambda di, dj: ((iph + di) % NPH) * NPS + (ips + dj) % NPS
    dphi = para(P[jm(-1, 0)], P[j], P[jm(1, 0)]) * GB_PH
    dpsi = para(P[jm(0, -1)], P[j], P[jm(0, 1)]) * GB_PS
    phi = float((PH[iph] + dphi + np.pi) % (2 * np.pi) - np.pi)
    psi = float((PS[ips] + dpsi + np.pi) % (2 * np.pi) - np.pi)
    return phi, psi

def est_bin(A):
    R = (A.conj().T @ A) / len(A)
    if METHOD == "bartlett":
        P = np.real(((GRID @ R) * GRID.conj()).sum(1))
    elif METHOD == "capon":
        Rl = R + (1e-3 * np.real(np.trace(R)) / (2 * L)) * np.eye(2 * L)
        Ri = np.linalg.inv(Rl)
        q = np.real(((GRID @ Ri) * GRID.conj()).sum(1))
        P = 1.0 / np.maximum(q, 1e-12)
    elif METHOD in ("music", "fbmusic"):
        Rm = 0.5 * (R + J40 @ R.conj() @ J40) if METHOD == "fbmusic" else R
        ew, ev = np.linalg.eigh(Rm)
        vtop = ev[:, -1]
        sv = GRID.conj() @ vtop                # completeness identity
        P = 1.0 / np.maximum(1.0 - np.abs(sv) ** 2, 1e-6)
    else:
        raise ValueError(METHOD)
    return refine(P, int(np.argmax(P)))

def tokenize_m(y):
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
        M = S[:, ws[b], fs[b]]
        A = np.concatenate([M[:, :, k:k + L].reshape(tkz.KTAP, -1)
                            for k in range(57 - L + 1)], 0)
        phi, psi = est_bin(A)
        toks.append((float(ws[b]), float(tkz.FPOS[fs[b]]), phi, psi,
                     float(np.log10(eng[ws[b], fs[b]] + 1e-12))))
    return np.array(toks, np.float32), nw

def one(job):
    rid, f = job
    of = f"{OUT}/tokens/{rid:06d}.npz"
    if os.path.exists(of): return 1
    try:
        y = tkz.read_products(os.path.join(tkz.ROOT, f))
        if y is None: return 0
        toks, nw = tokenize_m(y)
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
    print(f"[{METHOD}] tokenize {len(jobs)} recs -> {OUT}", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            done += r
    print(f"[{METHOD}] done: {done}/{len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
