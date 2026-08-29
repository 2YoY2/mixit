#!/usr/bin/env python3
"""PA limb-GT x Doppler validation: clean the complex CSI (PA's inter-antenna
phase is coherent -- real Doppler exists here, unlike XRF), build per-limb
speeds from BODY25 keypoints3d, and correlate in the Doppler-shift domain.

Per clip (one receiver):
  CSI  : timestamps -> ~900 Hz complex; AGC per-packet norm; cross-antenna
         conj pairs (CFO/SFO/PDD cancel); bin-average to 200 Hz; STFT with
         0.5 s window / 0.1 s hop, window-mean removed (static stays out of
         the spectrum, NOT out of the data) -> Doppler energy S(t, f), +-100 Hz
  GT   : keypoints3d/*.json, joints [LWrist7 RWrist4 LHip12 RHip9 Head0],
         confidence-masked 3D speeds -> v_i(t) on the STFT frame grid
Metrics per limb (medians over clips, null = circularly shifted v):
  r_tot   corr(v_i, total Doppler energy)
  r_band  corr(v_i, energy in 2-10 / 10-30 / 30-100 Hz bands)
  r_cent  corr(v_i, Doppler centroid)
Bonus physics check: slope of centroid vs TOTAL speed -> implied 2/lambda
(expect ~35 Hz per m/s at 5.3 GHz) -- validates keypoint units AND carrier.

  NREC=200 python3 diagnostics/18_pa_limb_doppler.py
"""
import os, glob, json, re
import numpy as np
import h5py
from scipy.ndimage import uniform_filter1d

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
NREC = int(os.environ.get("NREC", "200"))
FS, WINS, HOPS = 200.0, 100, 20            # 200 Hz grid, 0.5 s window, 0.1 s hop
J = [7, 4, 12, 9, 0]                       # LW RW LP RP Head (BODY25)
DEV = ["LW", "RW", "LP", "RP", "HD"]
BANDS = [(2, 10), (10, 30), (30, 100)]

def load_csi(f):
    with h5py.File(f, "r") as h:
        c = h["csi/csi"][...]
        ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    x = (c["real"] + 1j * c["imag"]).astype(np.complex64)     # (3,57,T)
    dt = float(np.median(np.diff(ts)))
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            t = (ts - ts[0]) * unit
            break
    else:
        t = np.arange(x.shape[-1]) / 810.0
    k = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., k], t[k]
    x = np.moveaxis(x, -1, 0)                                 # (T,3,57)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    return x / g, t

def doppler(x, t):
    y = np.c_[x[:, 0] * np.conj(x[:, 1]), x[:, 2] * np.conj(x[:, 1])]  # (T,114)
    dur = float(t[-1])
    nb = int(dur * FS)
    if nb < 3 * WINS: return None, None
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, y.shape[1]), np.complex64)
    np.add.at(s.real, idx, y.real); np.add.at(s.imag, idx, y.imag)
    z = s / np.maximum(cnt, 1)[:, None]
    frames = range(0, nb - WINS, HOPS)
    S = np.empty((len(list(frames)), WINS), np.float32)
    for i, s0 in enumerate(range(0, nb - WINS, HOPS)):
        w = z[s0:s0 + WINS]
        w = w - w.mean(0, keepdims=True)
        F = np.fft.fft(w, axis=0)
        S[i] = (np.abs(F) ** 2).sum(1)
    tf = (np.arange(len(S)) * HOPS + WINS / 2) / FS
    return S, tf                                              # S: (F, WINS bins)

def gt_speeds(d, dur):
    fs = sorted(glob.glob(f"{d}/*.json"))
    if len(fs) < 20: return None, None
    P = np.full((len(fs), 25, 3), np.nan)
    for i, f in enumerate(fs):
        try:
            j = json.load(open(f))
            if j:
                kp = np.asarray(j[0]["keypoints3d"], np.float32)
                m = kp[:, 3] > 0.3
                P[i, m] = kp[m, :3]
        except Exception:
            pass
    fps = len(fs) / dur
    scale = 1.0
    med = np.nanmedian(np.abs(P))
    if med > 50: scale = 1e-3                                 # mm -> m
    P = P * scale
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1) * fps     # (F-1, 25)
    tv = (np.arange(len(v)) + 0.5) / fps
    return v[:, J], tv

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() < 1e-9 or b[ok].std() < 1e-9: return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

mats = sorted(glob.glob(f"{ROOT}/Scene*/user*/action*/*/csi_mat/*-r1.mat"))
rng = np.random.default_rng(0)
rng.shuffle(mats)
freqs = np.fft.fftfreq(WINS, 1 / FS)
res = {d: {"tot": [], "cent": [], "bands": [[] for _ in BANDS], "null": []}
       for d in DEV}
cent_pts = []
used = 0
for f in mats:
    if used >= NREC: break
    kd = os.path.join(os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d")
    if not os.path.isdir(kd): continue
    try:
        x, t = load_csi(f)
        S, tf = doppler(x, t)
        if S is None: continue
        v, tv = gt_speeds(kd, float(t[-1]))
        if v is None: continue
    except Exception:
        continue
    used += 1
    vi = np.stack([np.interp(tf, tv, np.nan_to_num(v[:, i]), left=np.nan,
                             right=np.nan) for i in range(5)], 1)
    vi = uniform_filter1d(np.nan_to_num(vi), 3, axis=0)
    pos = (freqs > 2) & (freqs < 100)
    tot = S[:, pos].sum(1)
    cent = (S[:, pos] * freqs[pos]).sum(1) / (tot + 1e-12)
    for i, dv in enumerate(DEV):
        res[dv]["tot"].append(corr(vi[:, i], tot))
        res[dv]["cent"].append(corr(vi[:, i], cent))
        for bi, (lo, hi) in enumerate(BANDS):
            bm = (np.abs(freqs) >= lo) & (np.abs(freqs) < hi)
            res[dv]["bands"][bi].append(corr(vi[:, i], S[:, bm].sum(1)))
        res[dv]["null"].append(corr(np.roll(vi[:, i], len(vi) // 3), tot))
    vt = vi.sum(1)
    ok = np.isfinite(vt) & (vt > 0.05)
    if ok.sum() > 10:
        sl = np.polyfit(vt[ok], cent[ok], 1)[0]
        cent_pts.append(sl)

print(f"{used} clips analysed\n")
print(f"{'limb':5s}{'r_tot':>8s}{'r_2-10':>8s}{'r_10-30':>9s}{'r_30-100':>10s}"
      f"{'r_cent':>8s}{'null':>7s}")
for dv in DEV:
    r = res[dv]
    print(f"{dv:5s}{np.nanmedian(r['tot']):8.3f}"
          f"{np.nanmedian(r['bands'][0]):8.3f}{np.nanmedian(r['bands'][1]):9.3f}"
          f"{np.nanmedian(r['bands'][2]):10.3f}{np.nanmedian(r['cent']):8.3f}"
          f"{np.nanmedian(r['null']):7.3f}")
print(f"\nimplied 2/lambda from centroid~speed slope: median "
      f"{np.nanmedian(cent_pts):.1f} Hz/(m/s)  (5.3 GHz predicts ~35)")
print("""
READ: r_tot >> null validates the limb GT against cleaned CSI. Band ordering
(hips/torso in low bands, wrists reaching higher bands) + a sane implied
2/lambda validates the DOPPLER bridge and keypoint units. This gates the full
prep_pa_limbgt build and the limb-aware fine-tune/gate.
""")
