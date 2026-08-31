#!/usr/bin/env python3
"""Probe 41: the ROOM-MAP hypothesis (user's): the vector w that maps room
A's ensemble static to room B's should also map A's DYNAMICS to B's.

Physics: in the cross-antenna product domain (probe-27 front end,
y (T,2,57)), the dynamic part is the s.p* cross-term -> the room enters the
dynamics exactly through its static s.  If s_B = w.s_A then the s-mediated
dynamics transform by ~w, and w is SMOOTH in f (walls unresolvable at
20 MHz, probes 38/39) so it acts cleanly on products.  After phase-safe CMN
|w| cancels -> the observable effect is a phase rotation of the phi/psi
tokens (exactly the room-coded part the pose/act models consume).

Rooms 1 -> 2; scene 3 gives the wrong-map control (w13 applied instead).

GUARD (scene-1 clips, limb GT from prep_pa_xrf400): the probe-27 clusterer
score on native / mapped / wrong-mapped clips.  Doppler-only must be
IDENTICAL (map invariance); full-token score preserved => w didn't break
dynamic structure.

ALIGNMENT (no GT): per-clip energy-weighted phi/psi histograms -> ridge
room classifier (5-fold CV, per node):
    native1  vs native2   baseline separability (expect high)
    mapped1  vs native2   hypothesis works -> falls toward 0.5
    wrong1   vs native2   control (stays high)

  NST=40 NRID=60 NAL2=60 NPROC=8 python3 diagnostics/41_room_map.py
"""
import os
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from scipy.signal.windows import dpss

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
PREP1 = os.path.expanduser(os.environ.get(
    "PREP1", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
NST = int(os.environ.get("NST", "40"))       # clips per (scene,node) ensemble
NRID = int(os.environ.get("NRID", "60"))     # guard clips (scene 1)
NAL2 = int(os.environ.get("NAL2", "60"))     # alignment clips (scene 2)
NPROC = int(os.environ.get("NPROC", "8"))
FS, WINF, HOPF = 400.0, 256, 128
KTAP = 4
L, NPH, NPS = 20, 37, 37
MAXCORR = 0.7
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

PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

def tokenize(y):
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 10: return None, 0
    S = np.empty((KTAP, nw, PBAND.sum(), 2, 57), np.complex64)
    for k in range(KTAP):
        tap = TAPERS[k][:, None, None]
        for w in range(nw):
            S[k, w] = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                 axis=0)[PBAND]
    eng = (np.abs(S) ** 2).mean(axis=(0, 3, 4))
    floor = np.median(eng)
    toks = []
    for w in range(nw):
        for i in np.where(eng[w] >= floor)[0]:
            subs = np.concatenate(
                [S[:, w, i, :, k:k + L].reshape(KTAP, -1)
                 for k in range(57 - L + 1)], 0)
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

def histfeat(toks):
    """energy-weighted phi/psi histograms (the room-coded token axes)"""
    wgt = toks[:, 4] / (toks[:, 4].sum() + 1e-12)
    hphi = np.histogram(toks[:, 2], bins=NPH, range=(-np.pi, np.pi),
                        weights=wgt)[0]
    hpsi = np.histogram(toks[:, 3], bins=NPS, range=(-np.pi, np.pi),
                        weights=wgt)[0]
    return np.r_[hphi, hpsi].astype(np.float64)

def static_one(file):
    try:
        y = read_products(f"{ROOT}/{file}")
        return None if y is None else y.mean(0)
    except Exception:
        return None

def clip_one(job):
    """returns (node, histfeat) and, when GT given, the probe-27 scores"""
    file, node, W, gtf = job
    try:
        y = read_products(f"{ROOT}/{file}")
        if y is None: return None
        if W is not None: y = y * W[None]
        toks, nw = tokenize(y)
        if toks is None or len(toks) < 8 * nw: return None
        hf = histfeat(toks)
        if gtf is None: return (node, hf, None)
        gi = np.asarray(np.load(gtf), np.float32)
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
        if not np.isfinite(c12) or abs(c12) > MAXCORR: return (node, hf, None)
        F_full = np.c_[np.cos(toks[:, 2]), np.sin(toks[:, 2]),
                       np.cos(toks[:, 3]), np.sin(toks[:, 3]),
                       toks[:, 1] / 150.0]
        F_dopp = toks[:, 1:2] / 150.0
        wgt = np.sqrt(toks[:, 4])
        out = []
        for F in (F_full, F_dopp):
            a = kmeans2(F, wgt)
            E = np.zeros((2, nw))
            for k_ in (0, 1):
                sel = a == k_
                for w, e in zip(toks[sel, 0].astype(int), toks[sel, 4]):
                    E[k_, int(w)] += e
            if E[0].std() < 1e-12 or E[1].std() < 1e-12: return (node, hf, None)
            def score(Gm):
                p1 = np.nanmean([corr(E[0], Gm[:, li]), corr(E[1], Gm[:, lj])])
                p2 = np.nanmean([corr(E[0], Gm[:, lj]), corr(E[1], Gm[:, li])])
                return (max(p1, p2), min(p1, p2))
            m_, x_ = score(G)
            n_, _ = score(np.roll(G, nw // 2, 0))
            out += [m_, x_, n_]
        return (node, hf, tuple(out))
    except Exception:
        return None

def ridge_cv(Xa, Xb, folds=5, lam=1.0):
    """5-fold CV accuracy of ridge-to-{-1,+1} classifier a-vs-b"""
    X = np.r_[Xa, Xb]
    yv = np.r_[np.full(len(Xa), -1.0), np.full(len(Xb), 1.0)]
    rng = np.random.default_rng(0)
    ix = rng.permutation(len(X))
    X, yv = X[ix], yv[ix]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    accs = []
    for f in range(folds):
        te = np.arange(len(X)) % folds == f
        tr = ~te
        A = X[tr]
        wv = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]),
                             A.T @ yv[tr])
        accs.append(float(np.mean(np.sign(X[te] @ wv) == yv[te])))
    return float(np.mean(accs))

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    meta1 = pd.read_csv(f"{PREP1}/meta.csv")
    gtmap = {r.file: f"{PREP1}/imu/{int(r.rid):06d}.npy"
             for r in meta1[meta1.imu_ok == 1].itertuples()}
    rng = np.random.default_rng(41)
    NODES = ["r1", "r2", "r3"]

    print("=== pass 1: ensemble product statics per (scene, node)", flush=True)
    S = {}
    with Pool(NPROC) as pool:
        for sc in (1, 2, 3):
            for nd in NODES:
                g = man[(man.scene == sc) & (man.node == nd)]
                files = list(g.file.values)
                rng.shuffle(files)
                res = [r for r in pool.map(static_one, files[:NST])
                       if r is not None]
                S[(sc, nd)] = np.mean(res, 0)
                print(f"  ({sc},{nd}): {len(res)} clips", flush=True)
    W12, W13 = {}, {}
    for nd in NODES:
        for Wd, sc in ((W12, 2), (W13, 3)):
            num = S[(sc, nd)] * np.conj(S[(1, nd)])
            den = np.abs(S[(1, nd)]) ** 2
            Wd[nd] = (num / (den + 0.05 * np.median(den))).astype(np.complex64)
        print(f"  {nd}: |w12| med {np.median(np.abs(W12[nd])):.2f}  "
              f"|w13| med {np.median(np.abs(W13[nd])):.2f}", flush=True)

    g1 = man[man.scene == 1].sample(frac=1, random_state=41)
    g1 = g1[g1.file.isin(gtmap)].head(NRID * 2)
    g2 = man[man.scene == 2].sample(frac=1, random_state=41).head(NAL2 * 2)
    arms = {
        "native1": [(r.file, r.node, None, gtmap[r.file])
                    for r in g1.itertuples()],
        "mapped1": [(r.file, r.node, W12[r.node], gtmap[r.file])
                    for r in g1.itertuples()],
        "wrong1": [(r.file, r.node, W13[r.node], gtmap[r.file])
                   for r in g1.itertuples()],
        "native2": [(r.file, r.node, None, None) for r in g2.itertuples()],
    }
    feats, scores = {}, {}
    with Pool(NPROC) as pool:
        for arm, jobs in arms.items():
            cap = NAL2 if arm == "native2" else NRID
            got_f, got_s = [], []
            for r in pool.imap_unordered(clip_one, jobs, chunksize=1):
                if r is None: continue
                nd, hf, sc_ = r
                got_f.append((nd, hf))
                if sc_ is not None: got_s.append(sc_)
                if len(got_f) >= cap: break
            feats[arm], scores[arm] = got_f, got_s
            print(f"  arm {arm}: {len(got_f)} clips ({len(got_s)} scored)",
                  flush=True)

    print("\n=== GUARD: probe-27 clusterer scores (scene-1 clips)", flush=True)
    for arm in ("native1", "mapped1", "wrong1"):
        A = np.array(scores[arm], float)
        if not len(A): print(f"  [{arm}] no scored clips"); continue
        for nm, o in (("full", 0), ("dopp", 3)):
            M, N = A[:, o], A[:, o + 2]
            print(f"  [{arm:8s}|{nm}] matched {np.nanmedian(M):+.3f}  null "
                  f"{np.nanmedian(N):+.3f}  win {np.mean(M > N)*100:.0f}%  "
                  f"(n={len(A)})", flush=True)

    print("\n=== ALIGNMENT: room classifier (phi/psi hists, 5-fold ridge)",
          flush=True)
    for cmp_, arm in (("native1 vs native2", "native1"),
                      ("mapped1 vs native2", "mapped1"),
                      ("wrong1  vs native2", "wrong1")):
        accs = []
        for nd in NODES:
            Xa = np.array([h for n_, h in feats[arm] if n_ == nd])
            Xb = np.array([h for n_, h in feats["native2"] if n_ == nd])
            n = min(len(Xa), len(Xb))
            if n < 10: continue
            accs.append(ridge_cv(Xa[:n], Xb[:n]))
        print(f"  {cmp_}: acc {np.mean(accs)*100:.1f}%  "
              f"(per-node {['%.0f' % (a*100) for a in accs]}, chance 50)",
              flush=True)
    print("READ: mapped1-vs-native2 near 50 while native1/wrong1 stay high "
          "= the static-derived map aligns the DYNAMICS' room coding -> "
          "the room-map hypothesis holds.", flush=True)

if __name__ == "__main__":
    main()
