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
    CLEAN = os.environ.get("CLEAN", "1") == "1"
    if CLEAN:
        # Hampel: impulse outliers (interference bursts, firmware glitches)
        from scipy.ndimage import median_filter
        a2 = amp.reshape(T, -1)
        med = median_filter(a2, size=(31, 1))
        dev = np.abs(a2 - med)
        mad = median_filter(dev, size=(31, 1)) * 1.4826 + 1e-9
        a2 = np.where(dev > 3 * mad, med, a2)
        amp = a2.reshape(amp.shape)
    st_amp = amp.reshape(T, -1).mean(0)
    snr_w = st_amp / (st_amp + np.percentile(st_amp, 25))   # damp weak features
    def dyn_pca(F, k=20):
        hp = F - uniform_filter1d(F, SM, axis=0)
        if not CLEAN: return hp
        m = hp.mean(0); hpc = hp - m
        U, S, Vt = np.linalg.svd(hpc, full_matrices=False)
        return (U[:, :k] * S[:k]) @ Vt[:k] + m                # signal subspace only
    hp_a = dyn_pca(amp.reshape(T, -1)) * snr_w[None, :]
    Y = uniform_filter1d(hp_a ** 2, SM, axis=0)
    if pha is not None:
        c = amp * np.exp(1j * pha[:T])
        z = c[..., :-1] * np.conj(c[..., 1:])                # (T,3,3,29) CFO-free
        if CLEAN:
            # SFO/PDD residual: per-packet per-antenna common rotation of the
            # island -- remove it or it injects common-mode "energy" everywhere
            u = z / (np.abs(z) + 1e-12)
            phi = np.angle(u.mean(-1, keepdims=True))
            z = z * np.exp(-1j * phi)
        z = z.reshape(T, -1)
        zw = np.sqrt(snr_w.reshape(3, 3, 30)[..., :-1].reshape(-1)
                     * snr_w.reshape(3, 3, 30)[..., 1:].reshape(-1))
        zr = np.c_[z.real, z.imag]
        hp_z = dyn_pca(zr)
        ph_e = uniform_filter1d((hp_z[:, :z.shape[1]] ** 2
                                 + hp_z[:, z.shape[1]:] ** 2) * zw[None, :] ** 2,
                                SM, axis=0)
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
    if os.environ.get("LAG", "1") == "1":
        # the release is only coarsely aligned (measured: median +0.4 s offset,
        # IQR -0.2..+0.8 s, 5% within 0.1 s). One clock offset per recording:
        # estimate from TOTAL motion (unbiased wrt limbs), shift, then attribute.
        ec = Y.sum(1); ec = (ec - ec.mean()) / (ec.std() + 1e-9)
        ei = e.sum(1); ei = (ei - ei.mean()) / (ei.std() + 1e-9)
        L = 75
        cc = [float((ec[max(0, -l):T - max(0, l)]
                     * ei[max(0, l):T - max(0, -l)]).mean())
              for l in range(-L, L + 1)]
        l = int(np.argmax(cc)) - L
        if l > 0:   e, Y = e[l:], Y[:T - l]
        elif l < 0: e, Y = e[:T + l], Y[-l:]
        T = len(e)
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
