#!/usr/bin/env python3
"""Probe 45: do WiMANS tokens correlate with the NUMBER OF PEOPLE?

Per-sample features from the empty-normalized tokens:
  staticdev   ||recording mean - empty print|| / ||print||  (occupancy in
              the static — standing bodies, kept by the empty-room CMN)
  logntok     token count (dynamic activity extent)
  logE        total dynamic energy
  fmean/fstd  energy-weighted Doppler stats
Tests: Spearman vs n_users per (env, band) and pooled; ridge 6-way count
readout, within-env 5-fold AND cross-env (train 2 rooms -> test the third
— the transfer question in miniature).  Runs on whatever tokens exist.

  python3 diagnostics/45_wimans_count.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))

def feats(label):
    f = f"{OUT}/tokens/{label}.npz"
    if not os.path.exists(f): return None
    z = np.load(f)
    t = z["toks"]
    if len(t) < 4: return None
    e = 10.0 ** t[:, 4].astype(np.float64)
    wsum = e.sum() + 1e-12
    return [float(z["staticdev"]), np.log10(1 + len(t)),
            float(np.log10(wsum)), float((e * t[:, 1]).sum() / wsum),
            float(np.sqrt((e * (t[:, 1] - (e * t[:, 1]).sum() / wsum) ** 2
                           ).sum() / wsum))]

FN = ["staticdev", "logntok", "logE", "fmean", "fstd"]

def ridge_acc(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(6)[ytr]
    Wm = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                         Xtr.T @ Y)
    pred = (Xte @ Wm).argmax(1)
    return float((pred == yte).mean()), pred

def main():
    an = pd.read_csv(f"{OUT}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    rows = []
    for r in an.itertuples():
        fv = feats(r.label)
        if fv is None: continue
        rows.append((r.environment, float(r.wifi_band),
                     int(r.number_of_users), fv))
    print(f"{len(rows)} samples with tokens", flush=True)
    if len(rows) < 100: return
    env = np.array([r[0] for r in rows])
    band = np.array([r[1] for r in rows])
    yu = np.array([r[2] for r in rows])
    X = np.array([r[3] for r in rows])

    print("\n=== Spearman(feature, n_users)", flush=True)
    for e_ in list(np.unique(env)) + ["POOLED"]:
        m = np.ones(len(env), bool) if e_ == "POOLED" else env == e_
        line = "  ".join(f"{n}:{spearmanr(X[m, i], yu[m]).statistic:+.2f}"
                         for i, n in enumerate(FN))
        print(f"  {e_:12s} (n={m.sum()}): {line}", flush=True)

    print("\n=== ridge 6-way count readout", flush=True)
    rng = np.random.default_rng(45)
    for e_ in np.unique(env):
        m = env == e_
        ix = rng.permutation(np.where(m)[0])
        h = int(len(ix) * 0.8)
        acc, _ = ridge_acc(X[ix[:h]], yu[ix[:h]], X[ix[h:]], yu[ix[h:]])
        base = max(np.bincount(yu[m]) / m.sum())
        print(f"  within {e_:12s}: acc {acc*100:.0f}%  (majority "
              f"{base*100:.0f}%)", flush=True)
    for e_ in np.unique(env):
        tr, te = env != e_, env == e_
        acc, pred = ridge_acc(X[tr], yu[tr], X[te], yu[te])
        off1 = float((np.abs(pred - yu[te]) <= 1).mean())
        print(f"  cross ->{e_:12s}: acc {acc*100:.0f}%  |err|<=1 "
              f"{off1*100:.0f}%", flush=True)

if __name__ == "__main__":
    main()
