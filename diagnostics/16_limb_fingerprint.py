#!/usr/bin/env python3
"""Stage-0 gate for limb extraction on XRF V2: do different moving limbs have
distinguishable spatial fingerprints in the CSI?

Representation: per-antenna coherent islands (inter-antenna phase is broken in
this release, within-antenna spectral phase is clean, sig 0.088): adjacent-
subcarrier conj per antenna -> (T, 3rx x 3ant x 29) = 261 complex features,
per-feature normalised. Rayleigh does not bound this: the limb's component is
TAGGED IN TIME by its own IMU, and a tagged component's spatial signature is
estimable below the resolution cell (rep.txt Thm 3-4).

Per recording:
  e_i     = motion envelope of each of the 5 IMU devices (LW RW LP RP GL)
  w_i     = e_i residualised against the other four, clamped >= 0
            (isolates limb-SPECIFIC motion timing; correlated gait drops out)
  v_i     = top eigenvector of the w_i-weighted dynamic covariance
            = limb i's spatial fingerprint
Metrics (all |cos|, medians over recordings where both limbs are active):
  pair    cross-limb fingerprint alignment  -> LOW = distinguishable
  stab    split-half stability of one limb  -> HIGH = fingerprint is real
  null    v_i vs fingerprint from a time-shifted (unrelated) weight
READ: stab >> pair ~ null  -> fingerprints exist, limb slots identifiable;
      pair ~ stab          -> limbs indistinguishable, keep body-level scope.

  NREC=120 python3 diagnostics/16_limb_fingerprint.py
"""
import os
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

DIR  = os.path.expanduser(os.environ.get("XRF", "~/zerdani/buffer/xrfv2"))
NREC = int(os.environ.get("NREC", "120"))
MINW = float(os.environ.get("MINW", "0.05"))   # min residual-energy share to count a limb as active
DEV = ["LW", "RW", "LP", "RP", "GL"]

def envelopes(imu):                    # (T,5,6) -> (T,5)
    x = imu.astype(np.float32)
    hp = x - uniform_filter1d(x, 25, axis=0)
    e = np.sqrt((hp ** 2).sum(-1))
    return uniform_filter1d(e, 15, axis=0)

def top_vec(D, w):
    if w.sum() <= 0: return None
    C = (D * (w / w.sum())[:, None]).conj().T @ D
    _, vec = np.linalg.eigh(C)
    return vec[:, -1]

def cos(a, b):
    if a is None or b is None: return np.nan
    return float(np.abs(np.vdot(a, b)) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

wifi = h5py.File(f"{DIR}/wifi_50hz_853_video_aligned.h5")
imuf = h5py.File(f"{DIR}/imu_50hz_853_video_aligned.h5")
names = [x.decode() if isinstance(x, bytes) else x
         for x in wifi["sample_names"][...]]
rng = np.random.default_rng(0)
sel = list(rng.permutation(names)[:NREC])
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
P = {p: [] for p in pairs}; S = {i: [] for i in range(5)}; NUL = []
used = 0
for n in sel:
    try:
        g = wifi["samples"][n]
        c = g["amp"][...] * np.exp(1j * g["pha"][...])
        imu = imuf["samples"][n][...]
    except Exception:
        continue
    T = min(len(c), len(imu))
    if T < 1000: continue
    c, imu = c[:T], imu[:T]
    z = (c[..., :-1] * np.conj(c[..., 1:])).reshape(T, -1)     # (T, 261)
    m = z.mean(0)
    sc = np.sqrt((np.abs(z - m) ** 2).mean(0)) + 1e-9
    D = (z - m) / sc
    e = envelopes(imu)                                          # (T, 5)
    v, act, W = {}, [], {}
    for i in range(5):
        others = [j for j in range(5) if j != i]
        A = np.c_[e[:, others], np.ones(T)]
        beta, *_ = np.linalg.lstsq(A, e[:, i], rcond=None)
        w = np.clip(e[:, i] - A @ beta, 0, None)
        if e[:, i].std() < 1e-6 or (w ** 2).sum() < MINW * (e[:, i] ** 2).sum():
            continue
        W[i] = w
        v[i] = top_vec(D, w)
        act.append(i)
    if len(act) < 2: continue
    used += 1
    h = T // 2
    for i in act:
        S[i].append(cos(top_vec(D[:h], W[i][:h]), top_vec(D[h:], W[i][h:])))
    for (i, j) in pairs:
        if i in act and j in act:
            P[(i, j)].append(cos(v[i], v[j]))
    i, j = act[0], act[-1]
    NUL.append(cos(v[i], top_vec(D, np.roll(W[j], T // 3))))

print(f"{used}/{len(sel)} recordings with >=2 active limbs\n")
print("cross-limb fingerprint |cos| (median, n):")
for (i, j) in pairs:
    if P[(i, j)]:
        print(f"  {DEV[i]}-{DEV[j]}: {np.nanmedian(P[(i,j)]):.3f}  (n={len(P[(i,j)])})")
print("\nsplit-half stability per limb (median, n):")
for i in range(5):
    if S[i]:
        print(f"  {DEV[i]}: {np.nanmedian(S[i]):.3f}  (n={len(S[i])})")
print(f"\nnull (shifted-weight) |cos| median: {np.nanmedian(NUL):.3f}  (n={len(NUL)})")
print("""
READ: stability >> cross-limb ~ null -> limbs carry distinct, repeatable
spatial fingerprints; limb slots are identifiable and the closed-form
projection control is buildable. cross-limb ~ stability -> the aperture
cannot tell limbs apart; keep body-level scope.
""")
