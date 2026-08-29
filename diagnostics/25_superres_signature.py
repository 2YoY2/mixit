#!/usr/bin/env python3
"""Super-resolution limb-signature probe (user's design): STFT isolates the
motion component, then MUSIC super-resolves it over the antenna x subcarrier
aperture -- the coherent test the island prep structurally cannot do (islands
are within-antenna products; cross-antenna phase never survives the prep).

Per raw .mat (3 ant, 57 subc, ~900 Hz):
  conj-mult antennas 2,3 against antenna 1 per packet  -> CFO/SFO cancel
    exactly (one LO), relative antenna phase of the DYNAMIC path survives
  bin to 400 Hz, subtract time-mean = phase-preserving static removal
    (dividing would rotate each element arbitrarily and break steering)
  STFT 0.64 s; snapshots = POSITIVE 2-150 Hz bins only (one-sided selection
    isolates dyn*conj(static) from its mirror static*conj(dyn))
  subcarrier smoothing (L=20) -> covariance -> MUSIC over
    (antenna phase step phi) x (subcarrier phase step psi)
  signature = log MUSIC pseudo-spectrum, per rx, 3 rx concatenated;
  plus phi_hat = circular mean direction of the phi marginal.

Battery as probe 23 (prototypes / within-room / LORO / shuffle null) + a
per-(room,act,rx) phi_hat table. CAVEATS: psi is delay relative to the static
response; absolute phi does NOT transfer across rooms (arrays sit elsewhere)
-- the decisive read is WITHIN-room separation.

  CLIPKEY=1-1-1 python3 diagnostics/25_superres_signature.py
"""
import os, glob, re
import numpy as np
import h5py

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
SCENES = [int(s) for s in os.environ.get("SCENES", "1,4,5").split(",")]
ACTS = [int(a) for a in os.environ.get("ACTS", "1,2").split(",")]
CLIPKEY = os.environ.get("CLIPKEY", "1-1-1")
FS, WINF, HOPF = 400.0, 256, 128
L, D, NPH, NPS = 20, 2, 49, 97
LAM = float(os.environ.get("LAM", "100"))
NSH = int(os.environ.get("NSH", "20"))
freqs = np.fft.fftfreq(WINF, 1 / FS)
PBAND = (freqs >= 2) & (freqs <= 150)          # positive side only

def read_products(path):
    """(T,2,57) conj-mult products on a uniform 400 Hz grid, or None."""
    with h5py.File(path, "r") as h:
        c = h["csi/csi"][...]
        ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    x = (c["real"] + 1j * c["imag"]).astype(np.complex64)      # (3,57,T)
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
    x = np.moveaxis(x, -1, 0)                                  # (T,3,57)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    x = x / g
    y = x[:, 1:, :] * np.conj(x[:, :1, :])                     # (T,2,57)
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
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)           # (NPH,2)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))                # (NPS,L)
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()

def music_sig(y):
    """log MUSIC spectrum (NPH*NPS,) + circular-mean antenna phase, or None."""
    dyn = (y - y.mean(0)).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    han = np.hanning(WINF)[:, None, None].astype(np.float32)
    snaps = []
    for w in range(nw):
        F = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * han, axis=0)
        snaps.append(F[PBAND])
    S = np.concatenate(snaps, 0)                               # (N,2,57)
    subs = np.concatenate([S[:, :, k:k + L].reshape(len(S), -1)
                           for k in range(57 - L + 1)], 0)
    R = (subs.conj().T @ subs) / len(subs)
    ew, ev = np.linalg.eigh(R)
    En = ev[:, :2 * L - D]
    P = 1.0 / np.maximum((np.abs(STEER @ En) ** 2).sum(1), 1e-12)
    lp = np.log10(P).astype(np.float32)
    marg = P.reshape(NPH, NPS).sum(1)
    phi = float(np.angle((marg * np.exp(1j * PH)).sum()))
    return lp - lp.mean(), phi

def ridge_acc(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    G = (Xtr - mu) / sd
    B = np.linalg.solve(G.T @ G + LAM * np.eye(G.shape[1]), G.T @ ytr)
    return float((np.sign(((Xte - mu) / sd) @ B) == yte).mean())

rng = np.random.default_rng(0)
X, y, PHI = {}, {}, {}
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
            sigs, phis = [], []
            for rx in (1, 2, 3):
                yb = read_products(g[rx])
                r = music_sig(yb) if yb is not None else None
                if r is None: break
                sigs.append(r[0]); phis.append(r[1])
            if len(sigs) == 3:
                rows.append(np.concatenate(sigs)); labs.append(act)
                PHI.setdefault((sc, act), []).append(phis)
    if not rows: continue
    X[sc] = np.array(rows)
    y[sc] = np.where(np.array(labs) == ACTS[0], 1.0, -1.0)
    n0 = int((y[sc] > 0).sum())
    print(f"room {sc}: n={n0}/{len(labs) - n0}", flush=True)
rooms = sorted(X)

print("\nE) super-resolved mover direction phi_hat (circ-mean deg, per rx):")
for (sc, act), v in sorted(PHI.items()):
    V = np.exp(1j * np.array(v))                               # (n,3)
    mdeg = np.degrees(np.angle(V.mean(0)))
    spr = 1 - np.abs(V.mean(0))
    print(f"  room {sc} act {act}: " + "  ".join(
        f"rx{i+1} {mdeg[i]:+7.1f} (spread {spr[i]:.2f})" for i in range(3)))

print("\nA) prototype cosines (rows/cols = act@room):")
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
READ: decisive row is WITHIN-room (B, and E's act-vs-act phi gap vs spread):
coherent laterality present -> per-limb ID possible with per-site calibration;
also C >> null would mean it even transfers. All ~ null -> the closure now
covers every channel the hardware recorded.""")
