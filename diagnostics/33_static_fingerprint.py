#!/usr/bin/env python3
"""Is the super-resolved static a ROOM fingerprint? (user's verification)
Per recording: product static -> smoothed covariance -> Bartlett spectrum
on the (phi, psi) grid -> unit-norm vector. Compare cosines:
  A  same room, same receiver, SAME user      (repeatability)
  B  same room, same receiver, DIFFERENT user (actor-invariance)
  C  same receiver label, DIFFERENT room      (room specificity)
A~B >> C = room fingerprint, robust to actors; A-B gap = stance footprint.
Reference column: raw 399-static cosine (probe-23 style).

  python3 diagnostics/33_static_fingerprint.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
NPER = int(os.environ.get("NPER", "8"))      # recordings per (scene,node,user)
NUSR = int(os.environ.get("NUSR", "5"))
L, NPH, NPS = 20, 37, 37
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64)

def spectrum(rid):
    f = f"{TOK}/statics/{rid:06d}.npy"
    if not os.path.exists(f): return None, None
    v = np.load(f)
    y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
    sb = np.stack([y[:, k:k + L].reshape(-1) for k in range(57 - L + 1)], 0)
    P = (np.abs(sb @ STEER.conj().T) ** 2).mean(0)
    P = np.log10(P + 1e-12)
    P = P - P.mean()
    return P / (np.linalg.norm(P) + 1e-12), v / (np.linalg.norm(v) + 1e-12)

def cos(a, b): return float(a @ b)

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
S = {}
for (sc, nd), g in man.groupby(["scene", "node"]):
    users = rng.permutation(g.subject.unique())[:NUSR]
    for u in users:
        rids = rng.permutation(g[g.subject == u].rid.values)[:NPER]
        for r in rids:
            sp, rv = spectrum(int(r))
            if sp is not None:
                S.setdefault((sc, nd, u), []).append((sp, rv))

def pairs_same_key():
    cs, cr = [], []
    for k, lst in S.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                cs.append(cos(lst[i][0], lst[j][0]))
                cr.append(cos(lst[i][1], lst[j][1]))
    return np.mean(cs), np.mean(cr), len(cs)

def pairs_cross_user():
    cs, cr = [], []
    keys = list(S)
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            if ki[0] == kj[0] and ki[1] == kj[1] and ki[2] != kj[2]:
                for a in S[ki][:3]:
                    for b in S[kj][:3]:
                        cs.append(cos(a[0], b[0]))
                        cr.append(cos(a[1], b[1]))
    return np.mean(cs), np.mean(cr), len(cs)

def pairs_cross_room():
    cs, cr = [], []
    keys = list(S)
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            if ki[0] != kj[0] and ki[1] == kj[1]:
                for a in S[ki][:2]:
                    for b in S[kj][:2]:
                        cs.append(cos(a[0], b[0]))
                        cr.append(cos(a[1], b[1]))
    return np.mean(cs), np.mean(cr), len(cs)

a = pairs_same_key(); b = pairs_cross_user(); c = pairs_cross_room()
print(f"{'comparison':34s}{'SR-spectrum':>12s}{'raw-static':>12s}{'pairs':>8s}")
print(f"{'A same room+rx+user':34s}{a[0]:12.3f}{a[1]:12.3f}{a[2]:8d}")
print(f"{'B same room+rx, diff user':34s}{b[0]:12.3f}{b[1]:12.3f}{b[2]:8d}")
print(f"{'C diff room, same rx label':34s}{c[0]:12.3f}{c[1]:12.3f}{c[2]:8d}")
print(f"\nstance footprint (A-B): SR {a[0]-b[0]:+.3f}  raw {a[1]-b[1]:+.3f}")
print(f"room specificity (B-C): SR {b[0]-c[0]:+.3f}  raw {b[1]-c[1]:+.3f}")
print("""
READ: B >> C -> the super-resolved static IS a room fingerprint, stable
across actors. A > B by a real margin -> the per-user/stance component is
visible in the SR domain (the thing worth tokenizing as 'stance atoms').""")
