#!/usr/bin/env python3
"""HYPOTHESIS TEST (user's design): room-aware canonicalization -- express
token coordinates RELATIVE to the static's MUSIC peak (phi0, psi0), the
room's dominant standing path. Room-colored (phi, psi) made body-referenced
by fixed algebra; no learnable pathway touches the static.

Probe 27/28 rerun, 4 arms, same recordings, same k-means, same scoring:
  canon   [cos/sin dphi, cos/sin dpsi, f]   phi-phi0, psi-psi0
  orig    [cos/sin phi,  cos/sin psi,  f]   probe-27 replication anchor
  canap   dominant aperture vector phase-referenced to the static + f
  dopp    [f]                               the control that matched all

Probes 27/28 verdict was 'aperture mute' ON ROOM-COLORED coords. If canon
beats dopp where orig does not -> canonicalization unlocks the spatial axes.

  NRID=150 NPROC=6 python3 diagnostics/29_canonical_clustering.py
"""
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py
from scipy.signal.windows import dpss

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
NRID = int(os.environ.get("NRID", "150"))
NPROC = int(os.environ.get("NPROC", "6"))
MAXCORR = float(os.environ.get("MAXCORR", "0.7"))
FS, WINF, HOPF = 400.0, 256, 128
KTAP, L = 4, 20
NPH, NPS = 37, 37
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND].astype(np.float32)
DEV = ["LW", "RW", "LP", "RP", "HD"]
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

def smooth_subs(V):
    """V (..., 2, 57) -> (..., 38, 40) subcarrier-smoothed sub-vectors."""
    return np.stack([V[..., :, k:k + L].reshape(*V.shape[:-2], 2 * L)
                     for k in range(57 - L + 1)], -2)

def tokenize(y):
    yb = y.mean(0)
    # static reference peak (phi0, psi0) + static direction s0 (fixed algebra)
    sb = smooth_subs(yb[None])[0]                        # (38, 40)
    Rs = sb.conj().T @ sb
    ews, evs = np.linalg.eigh(Rs)
    s0 = evs[:, -1]
    P0 = 1.0 / np.maximum(1.0 - np.abs(STEER @ s0) ** 2, 1e-6)
    j0 = int(P0.argmax())
    phi0, psi0 = PH[IPH[j0]], PS[IPS[j0]]
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    nf = int(PBAND.sum())
    S = np.empty((KTAP, nw, nf, 2, 57), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))
    floor = np.median(eng)
    ws, fs = np.where(eng >= floor)
    if len(ws) < 4 * nw: return None
    nbin = len(ws)
    A = np.empty((nbin, KTAP * 38, 2 * L), np.complex64)
    for b in range(nbin):
        A[b] = smooth_subs(S[:, ws[b], fs[b]]).reshape(-1, 2 * L)
    R = A.conj().transpose(0, 2, 1) @ A
    ew, ev = np.linalg.eigh(R)
    vtop = ev[:, :, -1]                                  # (nbin, 40)
    sv = STEER @ vtop.T
    P = 1.0 / np.maximum(1.0 - np.abs(sv) ** 2, 1e-6)
    peak = P.argmax(0)
    phi, psi = PH[IPH[peak]], PS[IPS[peak]]
    sref = s0 / np.maximum(np.abs(s0), 1e-9)
    vcan = vtop * np.conj(sref)[None, :]                 # canonical aperture
    return dict(w=ws, f=FPOS[fs], e=eng[ws, fs], nw=nw,
                phi=phi.astype(np.float32), psi=psi.astype(np.float32),
                dphi=np.arctan2(np.sin(phi - phi0), np.cos(phi - phi0)).astype(np.float32),
                dpsi=np.arctan2(np.sin(psi - psi0), np.cos(psi - psi0)).astype(np.float32),
                vcan=vcan.astype(np.complex64))

def kmeans2(F, wgt, iters=30, seeds=8):
    best, bl = None, np.inf
    rng = np.random.default_rng(0)
    for s_ in range(seeds):
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
    rid, matf = job
    try:
        gf = f"{TOK}/imu/{rid:06d}.npy"
        if not os.path.exists(gf): return None
        y = read_products(f"{ROOT}/{matf}")
        if y is None: return None
        tk = tokenize(y)
        if tk is None: return None
        nw = tk["nw"]
        gi = np.asarray(np.load(gf), np.float32)
        T = min(len(y), len(gi)); gi = gi[:T]
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(T, np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0) for w in range(nw)])
        order = np.argsort(-G.mean(0))
        li, lj = int(order[0]), int(order[1])
        c12 = corr(G[:, li], G[:, lj])
        if not np.isfinite(c12) or abs(c12) > MAXCORR: return None
        fdim = (tk["f"] / 150.0)[:, None]
        arms = {
            "canon": np.c_[np.cos(tk["dphi"]), np.sin(tk["dphi"]),
                           np.cos(tk["dpsi"]), np.sin(tk["dpsi"]), fdim],
            "orig": np.c_[np.cos(tk["phi"]), np.sin(tk["phi"]),
                          np.cos(tk["psi"]), np.sin(tk["psi"]), fdim],
            "canap": np.c_[tk["vcan"].real, tk["vcan"].imag, fdim],
            "dopp": fdim,
        }
        wgt = np.sqrt(tk["e"])
        out = []
        for nm in ("canon", "orig", "canap", "dopp"):
            a = kmeans2(arms[nm], wgt)
            E = np.zeros((2, nw))
            for k in (0, 1):
                np.add.at(E[k], tk["w"][a == k].astype(int), tk["e"][a == k])
            if E[0].std() < 1e-12 or E[1].std() < 1e-12: return None
            def score(Gm):
                p1 = np.nanmean([corr(E[0], Gm[:, li]), corr(E[1], Gm[:, lj])])
                p2 = np.nanmean([corr(E[0], Gm[:, lj]), corr(E[1], Gm[:, li])])
                return (max(p1, p2), min(p1, p2))
            m, x = score(G)
            n_, _ = score(np.roll(G, nw // 2, 0))
            out += [m, x, n_]
        return (DEV[li], DEV[lj]) + tuple(out)
    except Exception:
        return None

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
for grp, scenes in (("scenes123", [1, 2, 3]), ("rooms45", [4, 5])):
    sub = man[man.scene.isin(scenes)]
    cand = sub.iloc[rng.permutation(len(sub))]
    jobs = [(int(r.rid), r.file) for r in cand.itertuples()][:NRID * 5]
    res, tried = [], 0
    with Pool(NPROC) as pool:
        for r in pool.imap_unordered(one_rid, jobs, chunksize=1):
            tried += 1
            if r is not None:
                res.append(r)
                if len(res) >= NRID: break
    print(f"\n== {grp}: {len(res)} scored / {tried} tried")
    if not res: continue
    pair = np.array(["+".join(sorted((r[0], r[1]))) for r in res])
    A = np.array([r[2:] for r in res], float)
    for i, nm in enumerate(("canon", "orig", "canap", "dopp")):
        M, Xc, N = A[:, 3*i], A[:, 3*i+1], A[:, 3*i+2]
        print(f"  [{nm:6s}] matched {np.nanmedian(M):+.3f}  "
              f"gap {np.nanmedian(M-Xc):+.3f}  null {np.nanmedian(N):+.3f}  "
              f"win {np.mean(M > N)*100:.0f}%")
    print("  per-pair (n>=15): canon win | orig win | dopp win")
    for p in pd.Series(pair).value_counts().index:
        m = pair == p
        if m.sum() < 15: continue
        print(f"    {p:6s} n={m.sum():3d}: {np.mean(A[m,0]>A[m,2])*100:3.0f}%"
              f" | {np.mean(A[m,3]>A[m,5])*100:3.0f}%"
              f" | {np.mean(A[m,9]>A[m,11])*100:3.0f}%")
print("""
READ: canon >> dopp where orig ~ dopp -> canonicalization unlocks the
spatial axes; the room-aware separator (v4) is justified. canon ~ orig ~
dopp -> spatial axes stay mute even body-referenced.""")
