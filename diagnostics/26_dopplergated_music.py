#!/usr/bin/env python3
"""Doppler-gated super-resolution (probe 25 corrected per the literature:
mD-Track / Widar2.0 joint multi-dimensional estimation). Probe 25 pooled all
motion-band bins into ONE covariance -- Doppler collapsed, torso masked the
hand, Rayleigh wall. Fix: estimate angle/delay PER DOPPLER BIN: the joint
(Doppler x angle x delay) space resolves scatterers that share an angle lobe
but differ in micro-Doppler (hand 15-80 Hz vs torso 2-8 Hz).

Chain: raw .mat -> hardware-AGC removal -> cross-antenna products (CFO/SFO
cancel, angle survives) -> 400 Hz -> phase-safe CMN -> STFT -> per positive
Doppler bin: subcarrier-smoothed covariance -> MUSIC(D=1) -> joint map
P_f(phi,psi).  Outputs per clip:
  map  log P_f(phi) stacked over f (Doppler-angle image), 3 rx concat
  dphi = circ(hand-band angle) - circ(torso-band angle): array-orientation-
         free laterality scalar (could transfer across rooms)
Battery: dphi table per (room,act,rx) + prototype/within/LORO on the map.

  CLIPKEY=1-1-1 python3 diagnostics/26_dopplergated_music.py
"""
import os, glob, re
import numpy as np
import h5py

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
SCENES = [int(s) for s in os.environ.get("SCENES", "1,4,5").split(",")]
ACTS = [int(a) for a in os.environ.get("ACTS", "1,2").split(",")]
CLIPKEY = os.environ.get("CLIPKEY", "1-1-1")
FS, WINF, HOPF = 400.0, 256, 128
L, NPH, NPS = 20, 49, 49
TORSO = tuple(float(v) for v in os.environ.get("TORSO", "2,8").split(","))
HAND = tuple(float(v) for v in os.environ.get("HAND", "15,80").split(","))
LAM = float(os.environ.get("LAM", "100"))
NSH = int(os.environ.get("NSH", "20"))
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)
FPOS = freqs[PBAND]

def read_products(path):
    with h5py.File(path, "r") as h:
        c = h["csi/csi"][...]
        ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    x = (c["real"] + 1j * c["imag"]).astype(np.complex64)
    dt = float(np.median(np.diff(ts)))
    rate, t = None, None
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
    if rate is None:
        rate = 810.0; t = np.arange(x.shape[-1]) / rate
    keep = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., keep], t[keep]
    if float(t[-1]) < 2.0: return None
    x = np.moveaxis(x, -1, 0)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    x = x / g                                   # hardware AGC removal
    y = x[:, 1:, :] * np.conj(x[:, :1, :])
    nb = int(float(t[-1]) * FS)
    if nb < WINF + HOPF: return None
    yf = y.reshape(len(y), -1)
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, yf.shape[1]), np.complex64)
    np.add.at(s.real, idx, yf.real.astype(np.float32))
    np.add.at(s.imag, idx, yf.imag.astype(np.float32))
    m = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(0, len(good) - 1)]
        m[bad] = m[near]
    return m.reshape(nb, 2, 57)

PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()

def doppler_music(y):
    """per-bin MUSIC: returns doppler-angle log map (nf,NPH), band energies,
    and per-bin angle resultants for the dphi summary."""
    yb = y.mean(0)
    ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
    dyn = ((y - yb) / ga).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 4: return None
    han = np.hanning(WINF)[:, None, None].astype(np.float32)
    S = np.stack([np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * han, axis=0)[PBAND]
                  for w in range(nw)], 1)              # (nf, nw, 2, 57)
    nf = S.shape[0]
    amap = np.empty((nf, NPH), np.float32)
    res = np.empty(nf, np.complex128)
    eng = np.empty(nf, np.float64)
    for i in range(nf):
        subs = np.concatenate([S[i, :, :, k:k + L].reshape(nw, -1)
                               for k in range(57 - L + 1)], 0)
        eng[i] = float((np.abs(subs) ** 2).mean())
        R = (subs.conj().T @ subs) / len(subs)
        ew, ev = np.linalg.eigh(R)
        En = ev[:, :2 * L - 1]                          # D=1 per Doppler bin
        P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
        Pg = P.reshape(NPH, NPS)
        marg = Pg.sum(1)
        amap[i] = np.log10(marg)
        w = marg / marg.sum()
        res[i] = (w * np.exp(1j * PH)).sum()
    amap -= amap.mean()
    def band_phi(lo, hi):
        m = (FPOS >= lo) & (FPOS <= hi)
        if not m.any(): return np.nan
        wts = eng[m] / (eng[m].sum() + 1e-30)
        return float(np.angle((wts * res[m]).sum()))
    return amap.ravel(), band_phi(*TORSO), band_phi(*HAND)

def ridge_acc(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    G = (Xtr - mu) / sd
    a = np.linalg.solve(G @ G.T + LAM * np.eye(len(G)), ytr)
    return float((np.sign(((Xte - mu) / sd) @ (G.T @ a)) == yte).mean())

rng = np.random.default_rng(0)
X, y, DPHI = {}, {}, {}
for sc in SCENES:
    rows, labs = [], []
    for act in ACTS:
        bytake = {}
        for f in sorted(glob.glob(f"{ROOT}/Scene{sc}/user*/action{act}/"
                                  f"{CLIPKEY}/csi_mat/*.mat")):
            mm = re.search(r"user(\d+)/action\d+/[^/]+/csi_mat/(\d+)-r(\d)\.mat", f)
            if mm:
                bytake.setdefault((mm.group(1), mm.group(2)), {})[
                    int(mm.group(3))] = f
        for key in sorted(bytake):
            g = bytake[key]
            if set(g) != {1, 2, 3}: continue
            maps, dp = [], []
            for rx in (1, 2, 3):
                yb = read_products(g[rx])
                r = doppler_music(yb) if yb is not None else None
                if r is None: break
                maps.append(r[0])
                d = r[2] - r[1]                        # hand minus torso angle
                dp.append(np.arctan2(np.sin(d), np.cos(d)))
            if len(maps) == 3:
                rows.append(np.concatenate(maps)); labs.append(act)
                DPHI.setdefault((sc, act), []).append(dp)
    if not rows: continue
    X[sc] = np.array(rows)
    y[sc] = np.where(np.array(labs) == ACTS[0], 1.0, -1.0)
    n0 = int((y[sc] > 0).sum())
    print(f"room {sc}: n={n0}/{len(labs) - n0}", flush=True)
rooms = sorted(X)

print(f"\nE) dphi = hand-band({HAND}) minus torso-band({TORSO}) angle "
      f"(circ-mean deg, per rx):")
for (sc, act), v in sorted(DPHI.items()):
    V = np.exp(1j * np.array(v))
    mdeg = np.degrees(np.angle(V.mean(0)))
    spr = 1 - np.abs(V.mean(0))
    print(f"  room {sc} act {act}: " + "  ".join(
        f"rx{i+1} {mdeg[i]:+7.1f} (spread {spr[i]:.2f})" for i in range(3)))

print("\nA) prototype cosines on the Doppler-angle maps:")
labs2, protos = [], []
for sc in rooms:
    for act, an in ((ACTS[0], 1), (ACTS[1], -1)):
        p = X[sc][y[sc] == an].mean(0)
        protos.append(p / (np.linalg.norm(p) + 1e-12))
        labs2.append(f"a{act}@r{sc}")
P = np.array(protos)
cm = P @ P.T
print("      " + "".join(f"{l:>8s}" for l in labs2))
for i, l in enumerate(labs2):
    print(f"{l:>6s}" + "".join(f"{cm[i, j]:8.3f}" for j in range(len(labs2))))

print("B) within-room 70/30:")
for sc in rooms:
    ix = rng.permutation(len(y[sc])); k = int(len(ix) * 0.7)
    if k < 2 or len(ix) - k < 2: print(f"  room {sc}: n too small"); continue
    print(f"  room {sc}: acc {ridge_acc(X[sc][ix[:k]], y[sc][ix[:k]], X[sc][ix[k:]], y[sc][ix[k:]]):.3f}")

print("C) leave-one-room-out vs shuffle null:")
for sc in rooms:
    tr = [r for r in rooms if r != sc]
    Xt = np.concatenate([X[r] for r in tr]); yt = np.concatenate([y[r] for r in tr])
    acc = ridge_acc(Xt, yt, X[sc], y[sc])
    nul = np.mean([ridge_acc(Xt, rng.permutation(yt), X[sc], y[sc])
                   for _ in range(NSH)])
    print(f"  test room {sc}: acc {acc:.3f}   shuffle-null {nul:.3f}")

print("""
READ: E is the money row -- if act1 vs act2 dphi separates (gap > spread,
consistent sign per rx within a room), the hand's angle is readable once
Doppler-gated: laterality EXISTS coherently. dphi consistency ACROSS rooms
would be the transferable version. All ~ null -> joint-space closure too.""")
