#!/usr/bin/env python3
"""Joint (angle x delay) MUSIC distribution of the STATIC, per actor per
room (user's request: run on actor 1's static, actor 2's, ... compare).
FB subcarrier smoothing L=44, aperture 2x44=88, pooled over NREC recordings
per (scene, node, user). Saves spectra to archive2/static_music_dist.npz.

  python3 diagnostics/36_static_music_dist.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
OUTF = os.path.expanduser("~/zerdani/buffer/octonet/archive2/static_music_dist.npz")
SCENES = [int(s) for s in os.environ.get("SCENES", "1,4,5").split(",")]
NODE = os.environ.get("NODE", "r1")
NUSR = int(os.environ.get("NUSR", "4"))
NREC = int(os.environ.get("NREC", "6"))
LA, D = 44, 6
NPH, NPS = 73, 181
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)          # (NPH, 2)
A_SUB = np.exp(1j * np.outer(PS, np.arange(LA)))              # (NPS, LA)
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * LA)
STEER = (STEER / np.sqrt(2 * LA)).astype(np.complex64).conj()

def snaps(y):
    """(2,57) -> FB smoothed sub-vectors (2*(57-LA+1), 2*LA)"""
    nsh = 57 - LA + 1
    f = np.stack([y[:, k:k + LA].reshape(-1) for k in range(nsh)], 0)
    b = np.conj(f[:, ::-1])
    return np.concatenate([f, b], 0)

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
out, labels = [], []
for sc in SCENES:
    g = man[(man.scene == sc) & (man.node == NODE)]
    users = sorted(g.subject.unique())[:NUSR]
    for u in users:
        rids = rng.permutation(g[g.subject == u].rid.values)[:NREC]
        S = []
        for r in rids:
            f = f"{TOK}/statics/{int(r):06d}.npy"
            if not os.path.exists(f): continue
            v = np.load(f)
            y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
            S.append(snaps(y))
        if not S: continue
        S = np.concatenate(S, 0)
        R = S.conj().T @ S / len(S)
        ew, ev = np.linalg.eigh(R)
        En = ev[:, :2 * LA - D]
        P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
        out.append(10 * np.log10(P.reshape(NPH, NPS)).astype(np.float32))
        labels.append(f"scene{sc}_u{u}")
        print(labels[-1], "done", flush=True)
np.savez(OUTF, spec=np.stack(out), labels=np.array(labels),
         ph=PH.astype(np.float32), ps=PS.astype(np.float32))
print(f"saved {len(out)} spectra -> {OUTF}", flush=True)
