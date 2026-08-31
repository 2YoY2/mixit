#!/usr/bin/env python3
"""Probe 42: clusterer gate on CACHED tokens — the instrument that verifies
a tokenization (user ruling: the clusterer judges, not us).

Probe-27 protocol, but reading token npz files instead of recomputing from
raw: per recording, k-means K=2 on token features -> the 2 clusters'
energy envelopes vs the 2 most-active limbs' GT envelopes (prep imu @400Hz,
window-averaged on THIS tokenization's grid), best permutation, rolled
null.  Reference numbers (coarse grid, scene 1): matched +0.21, win 72%.

  TOKDIR=~/zerdani/buffer/octonet/pa_tokens       WINF=256 HOPF=128  (coarse)
  TOKDIR=~/zerdani/buffer/octonet/pa_tokens_fine  WINF=128 HOPF=32   (fine)
  NRID=60 python3 diagnostics/42_fine_gate.py
"""
import os
import numpy as np
import pandas as pd

TOKDIR = os.path.expanduser(os.environ.get(
    "TOKDIR", "~/zerdani/buffer/octonet/pa_tokens"))
PREP1 = os.path.expanduser(os.environ.get(
    "PREP1", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
NRID = int(os.environ.get("NRID", "60"))
WINF = int(os.environ.get("WINF", "256"))
HOPF = int(os.environ.get("HOPF", "128"))
MINTOK = int(os.environ.get("MINTOK", "4"))  # min tokens per window (line
                                             # tokenizers are sparse: use 1)
MAXCORR = 0.7
DEV = ["LW", "RW", "LP", "RP", "HD"]

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

def one(tokf, gtf):
    z = np.load(tokf)
    toks, nw = z["toks"].astype(np.float64), int(z["nw"])
    if len(toks) < MINTOK * nw or nw < 8: return None
    eng = 10.0 ** toks[:, 4]
    gi = np.asarray(np.load(gtf), np.float32)
    g2 = gi.copy()
    T = len(gi)
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(T, np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                  for w in range(nw) if w * HOPF + WINF <= T])
    nw = len(G)
    if nw < 8: return None
    order = np.argsort(-G.mean(0))
    li, lj = int(order[0]), int(order[1])
    c12 = corr(G[:, li], G[:, lj])
    if not np.isfinite(c12) or abs(c12) > MAXCORR: return None
    F_full = np.c_[np.cos(toks[:, 2]), np.sin(toks[:, 2]),
                   np.cos(toks[:, 3]), np.sin(toks[:, 3]),
                   toks[:, 1] / 150.0]
    F_dopp = toks[:, 1:2] / 150.0
    wgt = np.sqrt(eng)
    out = []
    for F in (F_full, F_dopp):
        a = kmeans2(F, wgt)
        E = np.zeros((2, nw))
        for k_ in (0, 1):
            sel = (a == k_) & (toks[:, 0] < nw)
            for w, e in zip(toks[sel, 0].astype(int), eng[sel]):
                E[k_, w] += e
        if E[0].std() < 1e-12 or E[1].std() < 1e-12: return None
        def score(Gm):
            p1 = np.nanmean([corr(E[0], Gm[:, li]), corr(E[1], Gm[:, lj])])
            p2 = np.nanmean([corr(E[0], Gm[:, lj]), corr(E[1], Gm[:, li])])
            return max(p1, p2)
        m_ = score(G)
        n_ = score(np.roll(G, nw // 2, 0))
        out += [m_, n_]
    return (DEV[li], DEV[lj]) + tuple(out)

def main():
    man = pd.read_csv(f"{TOKDIR}/manifest.csv")
    meta1 = pd.read_csv(f"{PREP1}/meta.csv")
    gtmap = {r.file: f"{PREP1}/imu/{int(r.rid):06d}.npy"
             for r in meta1[meta1.imu_ok == 1].itertuples()}
    g = man[man.scene == 1].sample(frac=1, random_state=42)
    g = g[g.file.isin(gtmap)]
    res, tried = [], 0
    for r in g.itertuples():
        tokf = f"{TOKDIR}/tokens/{int(r.rid):06d}.npz"
        if not os.path.exists(tokf): continue
        tried += 1
        try:
            out = one(tokf, gtmap[r.file])
        except Exception:
            out = None
        if out is not None:
            res.append(out)
            if len(res) >= NRID: break
    print(f"{TOKDIR} (WINF={WINF} HOPF={HOPF}): {len(res)} scored / "
          f"{tried} tried", flush=True)
    if not res: return
    A = np.array([r[2:] for r in res], float)   # mF nF mD nD
    for nm, o in (("full tokens", 0), ("Doppler-only", 2)):
        M, N = A[:, o], A[:, o + 1]
        print(f"  [{nm}] matched {np.nanmedian(M):+.3f}  null "
              f"{np.nanmedian(N):+.3f}  win {np.mean(M > N)*100:.0f}%",
              flush=True)

if __name__ == "__main__":
    main()
