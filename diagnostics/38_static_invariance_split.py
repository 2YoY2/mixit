#!/usr/bin/env python3
"""Invariance-signature decomposition of the static (user's design):
per-ACTOR ensemble static -> joint (phi,psi) FB-MUSIC components -> classify:
  shared across actors (same scene+rx)          -> walls + DC
  of those, shared across rx / across rooms     -> DC
  remainder of actor-shared                     -> WALLS (rx- & room-specific)
  not shared across actors                      -> person(-average)

  python3 diagnostics/38_static_invariance_split.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
LA, D, K = 44, 8, 8
NPH, NPS = 73, 181
MINR = 40
TOLP, TOLS = 0.35, 0.10          # match tolerance (phi, psi) rad
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(LA)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * LA)
STEER = (STEER / np.sqrt(2 * LA)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

def components(y):
    nsh = 57 - LA + 1
    f = np.stack([y[:, k:k + LA].reshape(-1) for k in range(nsh)], 0)
    sb = np.concatenate([f, np.conj(f[:, ::-1])], 0)
    R = sb.conj().T @ sb / len(sb)
    ew, ev = np.linalg.eigh(R)
    En = ev[:, :2 * LA - D]
    P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
    order = np.argsort(-P)
    picks = []
    for j in order:
        if all(abs(int(IPH[j]) - int(IPH[q])) > 6 or
               abs(int(IPS[j]) - int(IPS[q])) > 8 for q in picks):
            picks.append(int(j))
        if len(picks) >= K: break
    return [(float(PH[IPH[j]]), float(PS[IPS[j]]), float(P[j]))
            for j in picks]

def match(c, cs, tolp=TOLP, tols=TOLS):
    for (p2, s2, _) in cs:
        dp = abs(c[0] - p2); dp = min(dp, 2 * np.pi - dp)
        ds = abs(c[1] - s2); ds = min(ds, 2 * np.pi - ds)
        if dp < tolp and ds < tols: return True
    return False

man = pd.read_csv(f"{TOK}/manifest.csv")
COMP = {}
for (sc, nd), g in man.groupby(["scene", "node"]):
    for u in sorted(g.subject.unique()):
        rids = g[g.subject == u].rid.values
        acc, n = np.zeros((2, 57), np.complex128), 0
        for r in rids:
            f = f"{TOK}/statics/{int(r):06d}.npy"
            if not os.path.exists(f): continue
            v = np.load(f)
            acc += (v[171:285] + 1j * v[285:]).reshape(2, 57)
            n += 1
        if n < MINR: continue
        COMP[(sc, nd, u)] = components((acc / n).astype(np.complex64))
print(f"{len(COMP)} per-actor ensembles", flush=True)

# 1) actor-shared components per (scene, node)
SHARED = {}
for (sc, nd) in sorted(set((k[0], k[1]) for k in COMP)):
    actors = [k for k in COMP if k[0] == sc and k[1] == nd]
    if len(actors) < 3: continue
    ref = COMP[actors[0]]
    shared, personal = [], []
    for c in ref:
        frac = np.mean([match(c, COMP[a]) for a in actors[1:]])
        (shared if frac >= 0.7 else personal).append(c + (frac,))
    SHARED[(sc, nd)] = shared
    tot = sum(c[2] for c in ref)
    print(f"scene{sc} {nd}: {len(shared)}/{len(ref)} actor-invariant "
          f"({sum(c[2] for c in shared)/tot*100:.0f}% of energy); "
          f"{len(personal)} personal", flush=True)

# 2) of actor-shared: shared across rx? across rooms? (delay axis only for
#    rx/room -- angle is frame-dependent per rx)
print("\nactor-invariant components classified:", flush=True)
for (sc, nd), shared in sorted(SHARED.items()):
    rows = []
    for c in shared:
        other_rx = [SHARED.get((sc, o)) for o in ("r1", "r2", "r3")
                    if o != nd and (sc, o) in SHARED]
        rx_frac = np.mean([match(c, cs, tolp=np.pi, tols=TOLS)
                           for cs in other_rx]) if other_rx else np.nan
        other_rm = [SHARED.get((s2, nd)) for s2 in (1, 2, 3, 4, 5)
                    if s2 != sc and (s2, nd) in SHARED]
        rm_frac = np.mean([match(c, cs, tolp=np.pi, tols=TOLS)
                           for cs in other_rm]) if other_rm else np.nan
        kind = ("DC" if (rx_frac > 0.6 and rm_frac > 0.6) else
                "WALL" if rx_frac < 0.6 or rm_frac < 0.6 else "?")
        rows.append((c[0], c[1], c[2], rx_frac, rm_frac, kind))
    rows.sort(key=lambda r: -r[2])
    for (p, s_, pw, rxf, rmf, kind) in rows[:5]:
        print(f"  scene{sc} {nd}: psi{s_:+.3f} phi{p:+.2f} pw{10*np.log10(pw):5.1f}dB"
              f"  rx-shared {rxf:.2f} room-shared {rmf:.2f} -> {kind}",
              flush=True)
