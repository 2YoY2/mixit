#!/usr/bin/env python3
"""Morph ceiling on prep_v3 AMPLITUDE statics -- the honest re-run of pilot 10
(whose 94% was measured through the broken conj rep and means nothing).

For two recordings of one (node, date) group -- same receiver, same room, same
day -- fit a small deformation mapping A's static to B's:

    M_L1: per-antenna gain
    M_L2: + fractional subcarrier shift (delay change moves the ripple)
    M_L3: + smooth spectral reshape (Chebyshev deg DEG, per antenna block)

and report the explained fraction of the static DIFFERENCE. This is the
ceiling on what the cross-prediction objective can move from the private slot
to the room slot.

CONTROL: pairs from the same node but DIFFERENT dates (different session,
possibly different room state). If morphs explain cross-date statics as well
as within-date, statics are generic hardware shape, not room identity --
and the whole static story needs a rethink. Expect within >> cross.

  PREP_OUT=~/zerdani/buffer/octonet/prep_v3 python3 diagnostics/12_morph_pilot_amp.py
"""
import os
import numpy as np
import pandas as pd

OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v3"))
NPAIR = int(os.environ.get("NPAIR", "600"))
NCTRL = int(os.environ.get("NCTRL", "200"))
DEG   = int(os.environ.get("DEG", "2"))
SMAX  = float(os.environ.get("SMAX", "6.0"))
SSTEP = float(os.environ.get("SSTEP", "0.25"))
K = 114

def static_of(rid):
    y = np.load(f"{OUT}/streams/{rid:06d}.npy", mmap_mode="r")
    return np.asarray(y, np.float32).mean(0)

def fit_block(u, v, shifts, deg):
    """min over shift, real cheb coeffs of ||cheb * shift(u) - v||^2 on one block."""
    V = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, K), deg)
    F = np.fft.rfft(u); fq = np.arange(len(F))
    best = np.inf
    for s in shifts:
        us = np.fft.irfft(F * np.exp(-2j * np.pi * fq * s / K), K)
        B = V * us[:, None]
        c, *_ = np.linalg.lstsq(B, v, rcond=None)
        best = min(best, float(((v - B @ c) ** 2).sum()))
    return best

def levels(u, v):
    shifts = np.arange(-SMAX, SMAX + 1e-9, SSTEP)
    out = []
    for sh, dg in (([0.0], 0), (shifts, 0), (shifts, DEG)):
        r = sum(fit_block(u[a * K:(a + 1) * K], v[a * K:(a + 1) * K], sh, dg)
                for a in (0, 1))
        out.append(r)
    return out

meta = pd.read_csv(f"{OUT}/meta.csv")
meta = meta[meta.split == "train"].reset_index(drop=True)
rng = np.random.default_rng(0)
groups = {k: g.reset_index(drop=True) for k, g in meta.groupby(["node", "date"]) if len(g) > 1}
keys = list(groups)
cache, rows = {}, []

def get(rid):
    if rid not in cache: cache[rid] = static_of(rid)
    return cache[rid]

def add_pair(ga, ia, gb, ib, kind):
    ra, rb = int(ga.rid[ia]), int(gb.rid[ib])
    u, v = get(ra), get(rb)
    base = float(((v - u) ** 2).sum())
    if base < 1e-12: return
    r1, r2, r3 = levels(u, v)
    rows.append((kind, ga.scene[ia], bool(ga.subject[ia] == gb.subject[ib]),
                 base / float((v ** 2).sum()),
                 1 - r1 / base, 1 - r2 / base, 1 - r3 / base))

quota = max(1, NPAIR // len(keys))
for k in keys:
    g = groups[k]
    for _ in range(quota):
        i, j = rng.choice(len(g), 2, replace=False)
        add_pair(g, i, g, j, "within")
for _ in range(NCTRL):
    k1, k2 = rng.choice(len(keys), 2, replace=False)
    if keys[k1][0] != keys[k2][0]: continue          # same node, different date only
    g1, g2 = groups[keys[k1]], groups[keys[k2]]
    add_pair(g1, rng.integers(len(g1)), g2, rng.integers(len(g2)), "cross-date")

df = pd.DataFrame(rows, columns=["kind", "scene", "same_subj", "gap", "L1", "L2", "L3"])
w = df[df.kind == "within"]
print(f"{len(w)} within-(node,date) pairs, {len(df) - len(w)} cross-date controls")
print(f"static gap |v-u|^2/|v|^2: within median {w.gap.median():.3f}\n")
print("explained fraction of the static DIFFERENCE (median [mean]):")
for L, nm in (("L1", "gain            "), ("L2", "+ shift         "), ("L3", f"+ smooth deg {DEG}  ")):
    print(f"  {nm}: within {w[L].median():+.3f} [{w[L].mean():+.3f}]")
c = df[df.kind == "cross-date"]
if len(c):
    print(f"\ncross-date control (same node, different session):")
    print(f"  L3 median {c.L3.median():+.3f}   vs within {w.L3.median():+.3f}"
          f"   gap {c.gap.median():.3f} vs {w.gap.median():.3f}")
for s in sorted(w.scene.unique()):
    v = w[w.scene == s]
    print(f"  scene {s}: L3 {v.L3.median():+.3f}  (n={len(v)})")
ss = w[w.same_subj]; cs = w[~w.same_subj]
print(f"  same-subject {ss.L3.median():+.3f} (n={len(ss)})   "
      f"cross-subject {cs.L3.median():+.3f} (n={len(cs)})" if len(cs) else
      f"  same-subject {ss.L3.median():+.3f} (n={len(ss)})   cross-subject n=0")
print("""
READ: within-L3 is the trainable ceiling (>0.5 -> objective worth training).
cross-date L3 well BELOW within-L3 -> statics carry session/room identity, good.
cross-date ~ within -> statics are generic hardware shape; rethink before training.
""")
