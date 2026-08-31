#!/usr/bin/env python3
"""Probe 41c: the room map applied at TOKEN (cluster) level — user's
correction of 41b.  A room change moves the (phi, psi) constellation; so
estimate the shift between the rooms' STATIC signatures on the tokenizer's
own grid and translate room-4's DYNAMIC tokens by it.  No CSI touched.

  per (scene,node): coherent ensemble mean of cached product statics
  (pa_tokens/statics/{rid}.npy, complex (2,57)) -> Bartlett spectrum on the
  37x37 (phi,psi) tokenizer grid (L=20 smoothing) -> dominant peak.
  map room4->room1: (dphi, dpsi)[node] = peak1 - peak4 (circular).
  scene-4 tokens: phi += dphi, psi += dpsi (wrapped), written to TOK3 with
  the same symlink overlay as 41b.  Downstream evals run with TOK=TOK3.

  NST=200 python3 phase3/tokmap41.py
"""
import os, glob
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
TOK3 = os.path.expanduser(os.environ.get(
    "TOK3", "~/zerdani/buffer/octonet/pa_tokens_tokmap41"))
NST = int(os.environ.get("NST", "200"))
# DCRM=1: subtract each antenna row's subcarrier mean from the ensemble
# static before peak estimation — removes the psi=0 zero-delay (DC)
# component exactly (room-shared per probes 38/39), so the peak is the
# dominant ROOM-SPECIFIC path and the map gains a real psi axis.
DCRM = int(os.environ.get("DCRM", "0"))
NODES = ["r1", "r2", "r3"]
L, NPH, NPS = 20, 37, 37
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
ST = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
ST = (ST / np.sqrt(2 * L)).astype(np.complex64)
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))

def peak_of(c):
    """dominant Bartlett peak (phi, psi) of a complex (2,57) static"""
    sb = np.stack([c[:, k:k + L].reshape(-1) for k in range(57 - L + 1)], 0)
    P = (np.abs(sb @ ST.conj().T) ** 2).mean(0)
    j = int(np.argmax(P))
    return PH[IPH[j]], PS[IPS[j]]

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    os.makedirs(f"{TOK3}/tokens", exist_ok=True)
    rng = np.random.default_rng(41)
    DP = {}
    for nd in NODES:
        pk = {}
        for sc in (1, 4):
            rids = list(man[(man.scene == sc) & (man.node == nd)].rid)
            rng.shuffle(rids)
            cs = []
            for r in rids[:NST]:
                f = f"{TOK}/statics/{int(r):06d}.npy"
                if not os.path.exists(f): continue
                v = np.load(f)
                cs.append((v[171:285] + 1j * v[285:]).reshape(2, 57))
            cm = np.mean(cs, 0).astype(np.complex64)
            if DCRM: cm = cm - cm.mean(axis=1, keepdims=True)
            pk[sc] = peak_of(cm)
            print(f"  ({sc},{nd}): n={len(cs)} peak phi {pk[sc][0]:+.2f} "
                  f"psi {pk[sc][1]:+.2f}", flush=True)
        DP[nd] = (wrap(pk[1][0] - pk[4][0]), wrap(pk[1][1] - pk[4][1]))
        print(f"  map4->1 {nd}: dphi {DP[nd][0]:+.2f}  dpsi {DP[nd][1]:+.2f}",
              flush=True)

    for name in ("manifest.csv", "pose", "imu", "statics", "statics_add",
                 "tenv", "static_peaks.npz"):
        src, dst = f"{TOK}/{name}", f"{TOK3}/{name}"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for cf in glob.glob(f"{TOK}/*_1-2-3.pkl"):
        dst = f"{TOK3}/{os.path.basename(cf)}"
        if not os.path.lexists(dst): os.symlink(cf, dst)
    nlink = ndone = 0
    for r in man.itertuples():
        rid, sc = int(r.rid), int(r.scene)
        src = f"{TOK}/tokens/{rid:06d}.npz"
        dst = f"{TOK3}/tokens/{rid:06d}.npz"
        if not os.path.exists(src) or os.path.lexists(dst): continue
        if sc != 4:
            os.symlink(src, dst); nlink += 1
            continue
        z = np.load(src)
        toks = z["toks"].copy()
        dphi, dpsi = DP[r.node]
        toks[:, 2] = wrap(toks[:, 2] + dphi)
        toks[:, 3] = wrap(toks[:, 3] + dpsi)
        np.savez(dst, toks=toks.astype(np.float32), nw=z["nw"])
        ndone += 1
        if ndone % 1000 == 0: print(f"  {ndone} mapped", flush=True)
    print(f"tokmap done: {ndone} scene-4 recs mapped, {nlink} symlinked -> "
          f"{TOK3}", flush=True)

if __name__ == "__main__":
    main()
