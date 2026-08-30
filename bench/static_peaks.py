#!/usr/bin/env python3
"""Static MUSIC peak (phi0, psi0) per recording -- the canonical reference
for room-aware coordinates. Fixed algebra on the stored statics.
-> pa_tokens/static_peaks.npz {rids, phi0, psi0}

  python3 bench/static_peaks.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
L, NPH, NPS = 20, 37, 37
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

man = pd.read_csv(f"{TOK}/manifest.csv")
rids, p0s, s0s = [], [], []
for i, r in enumerate(man.itertuples()):
    f = f"{TOK}/statics/{int(r.rid):06d}.npy"
    if not os.path.exists(f): continue
    v = np.load(f)
    y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
    sb = np.stack([y[:, k:k + L].reshape(-1) for k in range(57 - L + 1)], 0)
    R = sb.conj().T @ sb
    ew, ev = np.linalg.eigh(R)
    s0 = ev[:, -1]
    P = 1.0 / np.maximum(1.0 - np.abs(STEER @ s0) ** 2, 1e-6)
    j = int(P.argmax())
    rids.append(int(r.rid)); p0s.append(PH[IPH[j]]); s0s.append(PS[IPS[j]])
    if (i + 1) % 10000 == 0: print(f"  {i+1}/{len(man)}", flush=True)
np.savez(f"{TOK}/static_peaks.npz", rids=np.array(rids, np.int64),
         phi0=np.array(p0s, np.float32), psi0=np.array(s0s, np.float32))
print(f"{len(rids)} peaks saved", flush=True)
