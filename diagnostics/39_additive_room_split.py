#!/usr/bin/env python3
"""The full static decomposition on ADDITIVE statics (real delay axis,
3-element angle axis). Per (scene,node):
  1. per-ACTOR ensemble additive static -> joint (phi,psi) FB-MUSIC
  2. actor-shared components -> walls+DC; personal remainder
  3. among actor-shared: shared across rx / rooms (delay axis, honest
     tolerances vs occupied range) -> DC vs WALLS
  + saves ensemble spectra for the room-grid plot.

  python3 diagnostics/39_additive_room_split.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
OUTF = os.path.expanduser("~/zerdani/buffer/octonet/archive2/additive_room_dist.npz")
LA, D, K = 40, 8, 8
MINR = 40
NPH, NPS = 73, 181
TOLP, TOLS = 0.35, 0.06
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH), np.exp(2j * PH)], 1)  # 3 el
A_SUB = np.exp(1j * np.outer(PS, np.arange(LA)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 3 * LA)
STEER = (STEER / np.sqrt(3 * LA)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

def load_add(rid):
    f = f"{TOK}/statics_add/{rid:06d}.npy"
    if not os.path.exists(f): return None
    v = np.load(f)
    return (v[:171] + 1j * v[171:]).reshape(3, 57).astype(np.complex64)

def spec_and_comps(y):
    nsh = 57 - LA + 1
    f = np.stack([y[:, k:k + LA].reshape(-1) for k in range(nsh)], 0)
    sb = np.concatenate([f, np.conj(f[:, ::-1])], 0)
    R = sb.conj().T @ sb / len(sb)
    ew, ev = np.linalg.eigh(R)
    En = ev[:, :3 * LA - D]
    P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
    order = np.argsort(-P)
    picks = []
    for j in order:
        if all(abs(int(IPH[j]) - int(IPH[q])) > 6 or
               abs(int(IPS[j]) - int(IPS[q])) > 6 for q in picks):
            picks.append(int(j))
        if len(picks) >= K: break
    comps = [(float(PH[IPH[j]]), float(PS[IPS[j]]), float(P[j]))
             for j in picks]
    return 10 * np.log10(P.reshape(NPH, NPS)).astype(np.float32), comps

def match(c, cs, tolp=TOLP, tols=TOLS, use_phi=True):
    for tup in cs:
        p2, s2 = tup[0], tup[1]
        dp = abs(c[0] - p2); dp = min(dp, 2 * np.pi - dp)
        ds = abs(c[1] - s2); ds = min(ds, 2 * np.pi - ds)
        if (dp < tolp or not use_phi) and ds < tols: return True
    return False

man = pd.read_csv(f"{TOK}/manifest.csv")
COMP, SPECS, SLAB = {}, [], []
for (sc, nd), g in man.groupby(["scene", "node"]):
    room_acc, room_n = np.zeros((3, 57), np.complex128), 0
    for u in sorted(g.subject.unique()):
        acc, n = np.zeros((3, 57), np.complex128), 0
        for r in g[g.subject == u].rid.values:
            y = load_add(int(r))
            if y is None: continue
            acc += y; n += 1
        if n < MINR: continue
        COMP[(sc, nd, u)] = spec_and_comps((acc / n).astype(np.complex64))[1]
        room_acc += acc; room_n += n
    if room_n > 100 and nd == "r1":
        sp, _ = spec_and_comps((room_acc / room_n).astype(np.complex64))
        SPECS.append(sp); SLAB.append(f"scene{sc}")
np.savez(OUTF, spec=np.stack(SPECS), labels=np.array(SLAB),
         ph=PH.astype(np.float32), ps=PS.astype(np.float32))
print(f"{len(COMP)} actor ensembles; {len(SPECS)} room spectra saved", flush=True)

occ = [abs(c[1]) for v in COMP.values() for c in v]
print(f"delay-axis occupancy: median |psi| {np.median(occ):.3f}  "
      f"p90 {np.percentile(occ, 90):.3f}  (product domain was <=0.05)",
      flush=True)

SHARED = {}
for (sc, nd) in sorted(set((k[0], k[1]) for k in COMP)):
    actors = [k for k in COMP if k[0] == sc and k[1] == nd]
    if len(actors) < 3: continue
    ref = COMP[actors[0]]
    shared = [c for c in ref
              if np.mean([match(c, COMP[a]) for a in actors[1:]]) >= 0.7]
    SHARED[(sc, nd)] = shared
    tot = sum(c[2] for c in ref) + 1e-12
    print(f"scene{sc} {nd}: {len(shared)}/{len(ref)} actor-invariant "
          f"({sum(c[2] for c in shared)/tot*100:.0f}% energy)", flush=True)

print("\nDC vs WALL (delay-axis matching, honest tolerance):", flush=True)
for (sc, nd), shared in sorted(SHARED.items()):
    for c in sorted(shared, key=lambda x: -x[2])[:4]:
        orx = [SHARED.get((sc, o)) for o in ("r1", "r2", "r3")
               if o != nd and (sc, o) in SHARED]
        rxf = np.mean([match(c, cs, use_phi=False) for cs in orx]) \
            if orx else np.nan
        orm = [SHARED.get((s2, nd)) for s2 in (1, 2, 3, 4, 5)
               if s2 != sc and (s2, nd) in SHARED]
        rmf = np.mean([match(c, cs, use_phi=False) for cs in orm]) \
            if orm else np.nan
        kind = "DC" if (rxf >= 0.5 and rmf >= 0.5) else "WALL"
        print(f"  scene{sc} {nd}: psi{c[1]:+.3f} phi{c[0]:+.2f} "
              f"pw{10*np.log10(c[2]):5.1f}dB rx{rxf:.2f} room{rmf:.2f} "
              f"-> {kind}", flush=True)
