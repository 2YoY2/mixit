#!/usr/bin/env python3
"""Probe 41e: the room map as a MATRIX change of basis (user's correction
of the scalar/translation arms).

Per node, T = C1^{1/2} C4^{-1/2} (complex CORAL) aligns the covariance of
room-4's static subband snapshots (40-dim MUSIC space, ensembles of
per-recording cached statics) with room-1's.  The induced TOKEN map is
computed through the steering grid: a(phi,psi) -> T a -> re-fit on the
grid -> (phi',psi') lookup table (1369 entries/node); scene-4 tokens'
angles are exact grid values, so the remap is a table lookup.  The rigid
(dphi,dpsi) translation of 41c is the special case T = phase diagonal.

  NST=200 python3 phase3/tokmap41_mat.py
"""
import os, glob
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
TOK3 = os.path.expanduser(os.environ.get(
    "TOK3", "~/zerdani/buffer/octonet/pa_tokens_tokmap41mat"))
NST = int(os.environ.get("NST", "200"))
DCRM = int(os.environ.get("DCRM", "0"))
RIDGE = float(os.environ.get("RIDGE", "0.05"))
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

def snapshots(c):
    """(38, 40) subband snapshots of a complex (2,57) static"""
    return np.stack([c[:, k:k + L].reshape(-1)
                     for k in range(57 - L + 1)], 0)

def msqrt(C, inv=False):
    ew, ev = np.linalg.eigh(C)
    ew = np.maximum(ew, RIDGE * ew.max())
    d = ew ** (-0.5 if inv else 0.5)
    return (ev * d[None]) @ ev.conj().T

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    os.makedirs(f"{TOK3}/tokens", exist_ok=True)
    rng = np.random.default_rng(41)
    LUT = {}
    for nd in NODES:
        C = {}
        for sc in (1, 4):
            rids = list(man[(man.scene == sc) & (man.node == nd)].rid)
            rng.shuffle(rids)
            rows = []
            for r in rids[:NST]:
                f = f"{TOK}/statics/{int(r):06d}.npy"
                if not os.path.exists(f): continue
                c = (np.load(f)[171:285]
                     + 1j * np.load(f)[285:]).reshape(2, 57)
                if DCRM: c = c - c.mean(axis=1, keepdims=True)
                rows.append(snapshots(c.astype(np.complex64)))
            V = np.concatenate(rows, 0)                  # (N*38, 40)
            C[sc] = (V.conj().T @ V) / len(V)
            print(f"  ({sc},{nd}): {len(rows)} statics -> cov", flush=True)
        T = msqrt(C[1]) @ msqrt(C[4], inv=True)          # room4 -> room1
        B = (T @ ST.T).T                                 # (1369, 40) mapped
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        M = np.abs(Bn @ ST.conj().T) ** 2                # match vs grid
        LUT[nd] = M.argmax(1)                            # grid -> grid
        moved = float(np.mean(LUT[nd] != np.arange(NPH * NPS)))
        dphi = np.median(np.abs((PH[IPH[LUT[nd]]] - PH[IPH]
                                 + np.pi) % (2 * np.pi) - np.pi))
        dpsi = np.median(np.abs((PS[IPS[LUT[nd]]] - PS[IPS]
                                 + np.pi) % (2 * np.pi) - np.pi))
        print(f"  T {nd}: {moved*100:.0f}% grid cells move, median "
              f"|dphi| {dphi:.2f} |dpsi| {dpsi:.2f}", flush=True)

    for name in ("manifest.csv", "pose", "imu", "statics", "statics_add",
                 "tenv", "static_peaks.npz"):
        src, dst = f"{TOK}/{name}", f"{TOK3}/{name}"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for cf in glob.glob(f"{TOK}/*_1-2-3.pkl"):
        dst = f"{TOK3}/{os.path.basename(cf)}"
        if not os.path.lexists(dst): os.symlink(cf, dst)
    nlink = ndone = 0
    iph_of = {(round(float(PH[i]), 5)): None for i in range(NPH)}
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
        gi = (np.searchsorted(PH, toks[:, 2] - 1e-4).clip(0, NPH - 1)
              * NPS
              + np.searchsorted(PS, toks[:, 3] - 1e-4).clip(0, NPS - 1))
        j = LUT[r.node][gi.astype(int)]
        toks[:, 2] = PH[IPH[j]]
        toks[:, 3] = PS[IPS[j]]
        np.savez(dst, toks=toks.astype(np.float32), nw=z["nw"])
        ndone += 1
        if ndone % 1000 == 0: print(f"  {ndone} mapped", flush=True)
    print(f"tokmap-mat done: {ndone} scene-4 recs mapped, {nlink} "
          f"symlinked -> {TOK3}", flush=True)

if __name__ == "__main__":
    main()
