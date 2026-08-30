#!/usr/bin/env python3
"""THE ROOM static, human removed (user's spec): ensemble complex mean over
ALL recordings at a (scene, node) -- persons vary across recordings and
average out; the room persists. Split-half validation: template from user
half A vs half B must MATCH within a room (human-free, replicable) and
DIFFER across rooms. Joint angle x delay MUSIC on each half-template.
Saves spectra -> archive2/room_ensemble_dist.npz

  python3 diagnostics/37_room_ensemble_static.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
OUTF = os.path.expanduser("~/zerdani/buffer/octonet/archive2/room_ensemble_dist.npz")
NODE = os.environ.get("NODE", "r1")
LA, D = 44, 6
NPH, NPS = 73, 181
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(LA)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * LA)
STEER = (STEER / np.sqrt(2 * LA)).astype(np.complex64).conj()

def spectrum(y):
    nsh = 57 - LA + 1
    f = np.stack([y[:, k:k + LA].reshape(-1) for k in range(nsh)], 0)
    sb = np.concatenate([f, np.conj(f[:, ::-1])], 0)
    R = sb.conj().T @ sb / len(sb)
    ew, ev = np.linalg.eigh(R)
    En = ev[:, :2 * LA - D]
    P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
    return 10 * np.log10(P.reshape(NPH, NPS)).astype(np.float32)

man = pd.read_csv(f"{TOK}/manifest.csv")
out, labels = [], []
for sc in [1, 2, 3, 4, 5]:
    g = man[(man.scene == sc) & (man.node == NODE)]
    users = sorted(g.subject.unique())
    halves = {"A": users[0::2], "B": users[1::2]}
    for hname, us in halves.items():
        acc, n = np.zeros((2, 57), np.complex128), 0
        for r in g[g.subject.isin(us)].rid.values:
            f = f"{TOK}/statics/{int(r):06d}.npy"
            if not os.path.exists(f): continue
            v = np.load(f)
            acc += (v[171:285] + 1j * v[285:]).reshape(2, 57)
            n += 1
        if n < 50: continue
        tpl = (acc / n).astype(np.complex64)
        out.append(spectrum(tpl))
        labels.append(f"scene{sc}_{hname}")
        print(f"{labels[-1]}: {n} recordings averaged", flush=True)
np.savez(OUTF, spec=np.stack(out), labels=np.array(labels),
         ph=PH.astype(np.float32), ps=PS.astype(np.float32))
print(f"saved {len(out)} -> {OUTF}", flush=True)
