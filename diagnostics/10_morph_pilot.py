#!/usr/bin/env python3
"""Closed-form morph pilot -- the Phase-0 gate for the cross-prediction objective.

For two recordings A,B of one node, the old objectives' optimum leaves A's
per-recording room deviation in the BODY slot (that is the observed leakage).
The new objective claims a tiny multiplicative family

    M = { gain  x  delay-ramp(tau)  x  smooth spectral filter (Chebyshev, deg D) }

maps A's static to B's, so the deviation can live in the ROOM slot instead.
This script measures that claim with no training and no GPU:

    m_A, m_B = time-means of the prepped complex streams      (the statics)
    r_L      = min over m in M_L of || m(m_A) - m_B ||^2      (closed form per tau)

Levels: L1 gain only | L2 gain+delay | L3 gain+delay+smooth.

READ
  expl_diff(L) = 1 - r_L / ||m_B - m_A||^2
      of the same-node static DIFFERENCE, the fraction a level-L morph explains.
      This is the ceiling on what the new objective can move from the private
      slot into the room slot.
  expl_diff(L3) high (>0.6) on cross-subject pairs -> M captures room deviations; go.
  low (<0.2) everywhere -> statics differ by more than M spans; enrich M or rethink.
  same-subject >> cross-subject -> the unexplained part is person-static
      (the b-bar floor); no within-node objective may claim it. Expected.
  NOTE if multiple scenes hide under one node id, cross-scene pairs deflate
      everything; check the census before over-reading a low number.
"""
import os
import numpy as np
import pandas as pd

D     = os.path.expanduser(os.environ.get("MIXIT_DATA", "~/zerdani/buffer/octonet/mixit_data_v2"))
NPAIR = int(os.environ.get("NPAIR", "600"))    # total pairs across nodes
DEG   = int(os.environ.get("DEG", "2"))        # L3 smooth-filter degree
TMAX  = float(os.environ.get("TMAX", "6.0"))   # delay grid half-width, samples
TSTEP = float(os.environ.get("TSTEP", "0.1"))
SEED  = int(os.environ.get("SEED", "0"))

def static_of(rid):
    a = np.load(f"{D}/streams/{rid:06d}.npy", mmap_mode="r")
    m = np.asarray(a).mean(0)
    return m if np.isfinite(m).all() and np.abs(m).sum() > 0 else None

def fit(u, v, taus, deg):
    """min over tau and complex coeffs of ||cheb(coeffs) * ramp(tau) * u - v||^2."""
    K = len(u)
    V = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, K), deg)
    k = np.arange(K)
    best = np.inf
    for tau in taus:
        b = (np.exp(-2j * np.pi * k * tau / K) * u)[:, None] * V
        c, *_ = np.linalg.lstsq(b, v, rcond=None)
        r = float((np.abs(v - b @ c) ** 2).sum())
        if r < best: best = r
    return best

meta = pd.read_csv(f"{D}/meta.csv").merge(
    pd.read_csv(f"{D}/imu_meta.csv")[["rid", "node", "subject", "act"]], on="rid")
rng = np.random.default_rng(SEED)
by = {n: g.reset_index(drop=True) for n, g in meta.groupby("node") if len(g) > 1}
quota = max(1, NPAIR // max(len(by), 1))
taus = np.arange(-TMAX, TMAX + 1e-9, TSTEP)
cache, rows = {}, []
for n, g in by.items():
    for _ in range(quota):
        i, j = rng.choice(len(g), 2, replace=False)
        ra, rb = int(g.rid[i]), int(g.rid[j])
        for r in (ra, rb):
            if r not in cache: cache[r] = static_of(r)
        u, v = cache[ra], cache[rb]
        if u is None or v is None or len(u) != len(v): continue
        base = float((np.abs(v - u) ** 2).sum())
        vv = float((np.abs(v) ** 2).sum())
        if base < 1e-20 or vv < 1e-20: continue
        r1 = fit(u, v, [0.0], 0)
        r2 = fit(u, v, taus, 0)
        r3 = fit(u, v, taus, DEG)
        rows.append((n, bool(g.subject[i] == g.subject[j]), base / vv,
                     1 - r1 / base, 1 - r2 / base, 1 - r3 / base))
df = pd.DataFrame(rows, columns=["node", "same_subj", "gap", "L1", "L2", "L3"])
print(f"{len(df)} pairs over {df.node.nunique()} nodes "
      f"({int(df.same_subj.sum())} same-subject, {int((~df.same_subj).sum())} cross)")
print(f"static gap |m_B - m_A|^2 / |m_B|^2: median {df.gap.median():.3f}\n")
print("explained fraction of the static DIFFERENCE (median [mean]):")
for L, nm in (("L1", "gain only        "), ("L2", "gain + delay     "),
              ("L3", f"+ smooth (deg {DEG}) ")):
    print(f"  {nm}: {df[L].median():+.3f}  [{df[L].mean():+.3f}]")
print("\nby pair type (L3):")
for s, lbl in ((True, "same-subject "), (False, "cross-subject")):
    v = df[df.same_subj == s].L3
    if len(v): print(f"  {lbl}: median {v.median():+.3f}   n={len(v)}")
print("\nper-node L3 median:")
for n, g in df.groupby("node"):
    print(f"  node {n}: {g.L3.median():+.3f}  (n={len(g)})")
print("""
READ: L3 cross-subject is the number. >0.6 -> the morph family captures room
deviations, the cross-prediction objective has a ceiling worth training toward.
<0.2 -> same-node statics differ by more than M spans; enrich M before training.
""")
