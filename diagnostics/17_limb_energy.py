#!/usr/bin/env python3
"""Limb ENERGY-footprint probe (user's model, replacing 16's complex-direction
estimator which phase-rotation invalidates):

    dyn_energy(t, f)  ~=  b0(f) + sum_i e_i(t) * p_i(f)

Each limb i has a fixed nonneg energy footprint p_i across the 270 amplitude
features (3 rx x 3 ant x 30 subc), scaled by its IMU envelope e_i. Solved
jointly by least squares over time (handles correlated envelopes without
pre-residualizing). Also directly tests the proportionality claim via R^2.

Metrics (medians over recordings):
  R2        variance of per-feature dynamic energy explained by the 5 envelopes
  pair      cross-limb cosine of centered footprints  -> LOW = distinguishable
  stab      interleaved-half stability of a footprint -> HIGH = real
  null      footprint refit with that limb's envelope time-rolled
READ: stab >> pair ~ null and R2 substantial -> limb energy footprints exist.

  NREC=120 python3 diagnostics/17_limb_energy.py
"""
import os
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

DIR  = os.path.expanduser(os.environ.get("XRF", "~/zerdani/buffer/xrfv2"))
NREC = int(os.environ.get("NREC", "120"))
DEV = ["LW", "RW", "LP", "RP", "GL"]
SM = 25                                        # 0.5 s smoothing @50 Hz

def envelopes(imu):
    x = imu.astype(np.float32)
    hp = x - uniform_filter1d(x, SM, axis=0)
    return uniform_filter1d(np.sqrt((hp ** 2).sum(-1)), SM, axis=0)

def feat_energy(amp):                          # (T,3,3,30) -> (T,270) smoothed dyn energy
    a = amp.reshape(len(amp), -1)
    hp = a - uniform_filter1d(a, SM, axis=0)
    return uniform_filter1d(hp ** 2, SM, axis=0)

def fit(E, Y):
    """E (T,6 with intercept), Y (T,F) -> P (6,F), R2"""
    P, *_ = np.linalg.lstsq(E, Y, rcond=None)
    resid = Y - E @ P
    r2 = 1 - (resid ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12)
    return P, float(r2)

def ccos(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

wifi = h5py.File(f"{DIR}/wifi_50hz_853_video_aligned.h5")
imuf = h5py.File(f"{DIR}/imu_50hz_853_video_aligned.h5")
names = [x.decode() if isinstance(x, bytes) else x for x in wifi["sample_names"][...]]
rng = np.random.default_rng(0)
sel = list(rng.permutation(names)[:NREC])
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
P_ = {p: [] for p in pairs}; S_ = {i: [] for i in range(5)}; NUL, R2S = [], []
used = 0
for n in sel:
    try:
        g = wifi["samples"][n]
        amp = g["amp"][...]
        pha = g["pha"][...] if os.environ.get("PH", "1") == "1" else None
        imu = imuf["samples"][n]["imu"][...]
    except Exception as ex:
        print(f"  skip {n}: {type(ex).__name__}"); continue
    T = min(len(amp), len(imu))
    if T < 1000: continue
    amp = amp[:T]
    if os.environ.get("AGC", "1") == "1":
        # per-packet per-receiver gain normalisation: AGC multiplies every
        # antenna/subcarrier of a receiver at once and TRACKS body motion --
        # a motion-correlated global common mode in every footprint. Kill it
        # exactly; the spatial pattern survives.
        rms = np.sqrt((amp ** 2).mean(axis=(2, 3), keepdims=True)) + 1e-9
        amp = amp / rms
    Y = feat_energy(amp)
    if pha is not None:
        # phase-structure energy: islands (adjacent-subcarrier conj, CFO-free)
        # on the AGC-cleaned complex; dynamic energy of re+im per feature.
        c = amp * np.exp(1j * pha[:T])
        z = (c[..., :-1] * np.conj(c[..., 1:])).reshape(T, -1)
        zr = np.c_[z.real, z.imag]
        hp = zr - uniform_filter1d(zr, SM, axis=0)
        ph_e = uniform_filter1d(hp[:, :z.shape[1]] ** 2
                                + hp[:, z.shape[1]:] ** 2, SM, axis=0)
        Y = np.c_[Y, ph_e]
    if os.environ.get("NORM", "static") == "static":
        # remove the ROOM's statistical property: dyn energy at feature f is
        # (limb pattern) x |static field at f|^2 to first order -- the room's
        # ripple is a COMMON multiplicative factor in every limb's footprint,
        # inflating cross-limb similarity. Divide it out.
        st = amp[:T].reshape(T, -1).mean(0) ** 2
        Y = Y / (st + st.mean() * 1e-3)
        Y = Y / (Y.mean() + 1e-12)
    else:
        Y = Y / (Y.mean(0) + 1e-12)            # old per-feature scale norm
    e = envelopes(imu[:T])
    e = e / (e.std(0) + 1e-9)
    act = [i for i in range(5) if e[:, i].std() > 1e-6]
    if len(act) < 2: continue
    used += 1
    E = np.c_[e, np.ones(T)]
    P, r2 = fit(E, Y); R2S.append(r2)
    blk = (np.arange(T) // 100) % 2
    Pa, _ = fit(E[blk == 0], Y[blk == 0]); Pb, _ = fit(E[blk == 1], Y[blk == 1])
    if os.environ.get("DIFF", "1") == "1":
        # kill the common mode in FOOTPRINT space: every normalisation leaves
        # some shared factor (room ripple, inverse noise floor); comparing
        # limb footprints MINUS their cross-limb mean isolates what is
        # actually limb-specific.
        for M in (P, Pa, Pb):
            M[act] = M[act] - M[act].mean(0, keepdims=True)
    fp = {i: P[i] for i in act}
    for i in act:
        S_[i].append(ccos(Pa[i], Pb[i]))
    for (i, j) in pairs:
        if i in act and j in act:
            P_[(i, j)].append(ccos(fp[i], fp[j]))
    i = act[0]
    En = E.copy(); En[:, i] = np.roll(En[:, i], T // 3)
    Pn, _ = fit(En, Y)
    NUL.append(ccos(fp[i], Pn[i]))

print(f"{used}/{len(sel)} recordings used")
print(f"\nR2 of energy ~ envelopes (proportionality claim): median {np.median(R2S):.3f}")
print("\ncross-limb footprint cosine (median, n):")
for (i, j) in pairs:
    if P_[(i, j)]:
        print(f"  {DEV[i]}-{DEV[j]}: {np.nanmedian(P_[(i,j)]):.3f}  (n={len(P_[(i,j)])})")
print("\ninterleaved stability per limb:")
for i in range(5):
    if S_[i]:
        print(f"  {DEV[i]}: {np.nanmedian(S_[i]):.3f}  (n={len(S_[i])})")
print(f"\nnull (rolled envelope) cosine: median {np.nanmedian(NUL):.3f}  (n={len(NUL)})")
print("""
READ: stab >> pair ~ null with decent R2 -> limb energy footprints are real
and separable; the limbs x slots program revives on the ENERGY model, and a
closed-form control exists (project energies onto footprints). stab ~ pair
-> even energy footprints don't separate limbs here.
""")
