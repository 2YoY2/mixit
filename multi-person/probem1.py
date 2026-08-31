#!/usr/bin/env python3
"""probem1 — does the token CLOUD get coarser as humans are added (1..5)?

Identity-free geometry of real recordings' cached token clouds, per n users
and per motion stratum (labels used ONLY to stratify, never to score):
  STILL   all users' activity == 'nothing' (bodies present, not moving)
  MOVING  no user is 'nothing'
  MIXED   some of each
Per-sample, label-free measures:
  ntok/nw        cloud size per window
  logE           total dynamic energy
  ent            normalized energy entropy over tokens (spread vs conc.)
  top10          energy share of the top-10% tokens
  phid           energy-weighted circular dispersion of phi (spatial spread)
  modes          phi histogram peak count (energy-weighted, smoothed)
  sil(k)         silhouette of k-means on spatial feats, k=2..6
  bestk          argmax_k sil — does the cloud announce its own n?
  ARI@n          split-half k-means stability at k=n (partition robustness)
All from cached tokens — no raw reads, no training, no GPU.

  python3 multi-person/probem1.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
SAMPN = int(os.environ.get("SAMPN", "150"))
MAXN = int(os.environ.get("MAXN", "400"))
rng = np.random.default_rng(51)

def kmeans(X, k, iters=40, restarts=3):
    best, blab = None, None
    for r in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        lab = np.zeros(len(X), int)
        for _ in range(iters):
            d = ((X[:, None] - C[None]) ** 2).sum(-1)
            nl = d.argmin(1)
            if (nl == lab).all(): break
            lab = nl
            for j in range(k):
                if (lab == j).any(): C[j] = X[lab == j].mean(0)
        cost = ((X - C[lab]) ** 2).sum()
        if best is None or cost < best: best, blab = cost, lab.copy()
    return blab

def silhouette(X, lab):
    k = lab.max() + 1
    if k < 2: return -1.0
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() < 2: continue
        a = D[i][same].sum() / (same.sum() - 1)
        b = min(D[i][lab == j].mean() for j in range(k)
                if j != lab[i] and (lab == j).any())
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s)) if s else -1.0

def ari(a, b):
    n = len(a)
    ct = {}
    for x, y in zip(a, b): ct[(x, y)] = ct.get((x, y), 0) + 1
    A = {}; Bc = {}
    for (x, y), v in ct.items():
        A[x] = A.get(x, 0) + v
        Bc[y] = Bc.get(y, 0) + v
    c2 = lambda m: m * (m - 1) / 2
    sidx = sum(c2(v) for v in ct.values())
    sa = sum(c2(v) for v in A.values())
    sb = sum(c2(v) for v in Bc.values())
    exp = sa * sb / max(c2(n), 1e-9)
    mx = 0.5 * (sa + sb)
    return float((sidx - exp) / max(mx - exp, 1e-9))

def cloud_stats(t, nw, n_users):
    e = 10.0 ** t[:, 4].astype(np.float64)
    w = e / e.sum()
    ent = float(-(w * np.log(w + 1e-12)).sum() / np.log(len(w)))
    top10 = float(np.sort(w)[-max(1, len(w) // 10):].sum())
    z = (w * np.exp(1j * t[:, 2])).sum()
    phid = float(1.0 - np.abs(z))
    h, _ = np.histogram(t[:, 2], bins=37, range=(-np.pi, np.pi), weights=w)
    hs = h + np.roll(h, 1) * 0.5 + np.roll(h, -1) * 0.5
    pk = int(((hs > np.roll(hs, 1)) & (hs >= np.roll(hs, -1))
              & (hs > 0.25 * hs.max())).sum())
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]),
              np.sin(t[:, 3]), np.cos(t[:, 3])]
    if len(X) > MAXN:
        ix = rng.choice(len(X), MAXN, replace=False)
        X = X[ix]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    sil = {}
    for k in range(2, 7):
        if len(X) <= k * 3: sil[k] = -1.0; continue
        sil[k] = silhouette(X, kmeans(X, k))
    bestk = max(sil, key=sil.get)
    stab = np.nan
    kn = min(max(n_users, 2), 6)
    if len(X) > 2 * kn * 4:
        h1 = rng.permutation(len(X)) < len(X) // 2
        l1 = kmeans(X[h1], kn); l2 = kmeans(X[~h1], kn)
        C1 = np.stack([X[h1][l1 == j].mean(0) for j in range(kn)
                       if (l1 == j).any()])
        C2 = np.stack([X[~h1][l2 == j].mean(0) for j in range(kn)
                       if (l2 == j).any()])
        a1 = ((X[:, None] - C1[None]) ** 2).sum(-1).argmin(1)
        a2 = ((X[:, None] - C2[None]) ** 2).sum(-1).argmin(1)
        stab = ari(a1, a2)
    return dict(ntokw=len(t) / nw, logE=float(np.log10(e.sum())), ent=ent,
                top10=top10, phid=phid, modes=pk, bestk=bestk, stab=stab,
                **{f"sil{k}": v for k, v in sil.items()})

def main():
    an = pd.read_csv(f"{OUT}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    rows = []
    for r in an.itertuples():
        n = int(r.number_of_users)
        acts = [str(getattr(r, f"user_{k}_activity")).strip()
                for k in range(1, n + 1)] if n else []
        if n and any(a in ("", "nan") for a in acts): continue
        still = sum(a == "nothing" for a in acts)
        strat = ("EMPTY" if n == 0 else
                 "STILL" if still == n else
                 "MOVING" if still == 0 else "MIXED")
        rows.append((r.label, n, strat, r.environment))
    df = pd.DataFrame(rows, columns=["label", "n", "strat", "env"])
    print(df.groupby(["strat", "n"]).size(), flush=True)

    recs = []
    for (strat, n), g in df.groupby(["strat", "n"]):
        labs = list(g.label.values)
        rng.shuffle(labs)
        got = 0
        for lb in labs:
            if got >= SAMPN: break
            f = f"{OUT}/tokens/{lb}.npz"
            if not os.path.exists(f): continue
            z = np.load(f)
            t = z["toks"]
            if len(t) < 24: continue
            st = cloud_stats(t, int(z["nw"]), n)
            st.update(n=n, strat=strat)
            recs.append(st)
            got += 1
        print(f"  {strat} n={n}: {got}", flush=True)
    d = pd.DataFrame(recs)

    cols = ["ntokw", "logE", "ent", "top10", "phid", "modes", "bestk", "stab"]
    for strat in ["EMPTY", "STILL", "MIXED", "MOVING"]:
        s = d[d.strat == strat]
        if not len(s): continue
        print(f"\n=== {strat}", flush=True)
        print("  n    " + "  ".join(f"{c:>6s}" for c in cols) +
              "   sil2  sil3  sil4  sil5  sil6", flush=True)
        for n in sorted(s.n.unique()):
            g = s[s.n == n]
            line = "  ".join(f"{g[c].mean():6.3f}" for c in cols)
            sils = "  ".join(f"{g[f'sil{k}'].mean():.3f}" for k in range(2, 7))
            print(f"  {n}    {line}   {sils}   (N={len(g)})", flush=True)
        if strat in ("MOVING", "MIXED"):
            m = s[s.n >= 1]
            if len(m) > 20 and m.n.nunique() > 1:
                rho = spearmanr(m.bestk, m.n).statistic
                acc = float((m.bestk == m.n).mean())
                print(f"  bestk vs n: spearman {rho:+.3f}  exact "
                      f"{acc:.3f}", flush=True)
                for n in sorted(m.n.unique()):
                    h = np.bincount(m[m.n == n].bestk, minlength=7)[2:7]
                    print(f"    n={n} bestk hist 2..6: {list(h)}", flush=True)
    print("probem1 done", flush=True)

if __name__ == "__main__":
    main()
