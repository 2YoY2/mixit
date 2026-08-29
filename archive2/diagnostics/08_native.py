#!/usr/bin/env python3
"""Bypass the streams. Build the motion envelope at NATIVE packet timing --
no interpolation, no resample -- and redo the activity ranking vs IMU.

Binning by timestamp is averaging, not interpolation, so it adds no content.

  rho jumps  -> preprocess_v2's 75->100 Hz complex interpolation is the culprit
  rho stays  -> the CSI genuinely does not track the wearer at this packet rate
"""
import os, glob, pickle, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.expanduser("~/zerdani/buffer/octonet/OctoNet-upload")
D = os.path.expanduser(os.environ.get("MIXIT_DATA", "~/zerdani/buffer/octonet/mixit_data_v2"))
N = int(os.environ.get("NREC", "400"))
BIN = 0.25                                   # s, matches the IMU envelope smoothing
# bands where 75 Hz packets are UNAMBIGUOUS: |v| < 1.09 m/s at 5.18 GHz
SLOW = (0.2, 2.0)      # breathing + torso; 93 s record = 20-45 breathing cycles
MID  = (2.0, 10.0)     # gestures
FAST = (10.0, 30.0)    # limbs (aliasing starts above ~37.5 Hz)

def native_env(f):
    d = pickle.load(open(f, "rb"))
    ts = d.get("timestamp", d.get("timestamps"))
    x = np.asarray(d["data"])
    if x.ndim != 3 or x.shape[1] != 2: return None, None, None
    t = np.array([(v - ts[0]).total_seconds() for v in ts])
    k = np.concatenate([[True], np.diff(t) > 0]); t, x = t[k], x[k]
    y = x[:, 0, :] * np.conj(x[:, 1, :])                 # kill common hardware phase
    mag = np.abs(x).mean((0, 1)); valid = mag > 0.1 * np.median(mag)
    y = y[:, valid]
    y /= np.maximum(np.abs(y).mean(0), 1e-12)
    nb = int(t[-1] / BIN)
    if nb < 20: return None, None, None
    idx = np.clip((t / BIN).astype(int), 0, nb - 1)
    # per-bin dynamic energy: variance WITHIN the bin (no cross-bin interpolation)
    e = np.zeros(nb)
    for b in range(nb):
        m = idx == b
        if m.sum() < 3: continue
        seg = y[m]; e[b] = float((np.abs(seg - seg.mean(0)) ** 2).mean())
    # band-resolved envelopes over the FULL record, native timing:
    # Lomb-Scargle-free approach -- bin to a uniform 1/BIN grid (averaging, not
    # interpolation) then band-filter the ENVELOPE, which is a 4 Hz quantity.
    bands = {}
    fe = np.fft.rfft(e - e.mean()); fq = np.fft.rfftfreq(len(e), BIN)
    for nm, (lo, hi) in (("slow", SLOW), ("mid", MID), ("fast", FAST)):
        m = (fq >= lo) & (fq < hi)
        bands[nm] = float((np.abs(fe[m]) ** 2).sum() / max((np.abs(fe) ** 2).sum(), 1e-30))
    return e, t[-1], float(len(t) / t[-1]), bands

meta = pd.read_csv(f"{D}/meta.csv").merge(
    pd.read_csv(f"{D}/imu_meta.csv")[["rid", "imu_ok", "act"]], on="rid")
meta = meta[meta.imu_ok == 1].reset_index(drop=True)
rng = np.random.default_rng(0)
sel = meta.iloc[rng.choice(len(meta), min(N, len(meta)), replace=False)]
rows, rates = [], []
for r in sel.itertuples():
    try:
        e, dur, rate, bands = native_env(os.path.join(ROOT, r.file))
        if e is None: continue
        g = np.load(f"{D}/imu_env/{r.rid:06d}.npy", mmap_mode="r")
        gi = np.array(g)[::int(round(0.25 * 100))][:len(e)]
        m = min(len(e), len(gi))
        if m < 20: continue
        a, b = e[:m], gi[:m]
        ok = a > 0
        if ok.sum() < 20: continue
        def bandfilt(x, lo, hi):
            f = np.fft.rfft(x - x.mean()); q = np.fft.rfftfreq(len(x), BIN)
            f[(q < lo) | (q >= hi)] = 0
            return np.fft.irfft(f, len(x))
        rr_ = {}
        for nm, (lo, hi) in (("slow", SLOW), ("mid", MID), ("fast", FAST)):
            aa, bb = bandfilt(a[ok], lo, hi), bandfilt(b[ok], lo, hi)
            d_ = np.sqrt((aa*aa).sum() * (bb*bb).sum())
            rr_[nm] = float((aa*bb).sum()/d_) if d_ > 1e-20 else 0.0
        rows.append((r.act, float(b[ok].mean()), float(a[ok].mean()),
                     float(np.corrcoef(a[ok], b[ok])[0, 1]),
                     rr_["slow"], rr_["mid"], rr_["fast"], bands["slow"]))
        rates.append(rate)
    except Exception: continue
df = pd.DataFrame(rows, columns=["act", "imu", "csi", "r_time",
                                 "r_slow", "r_mid", "r_fast", "e_slow"])
print(f"{len(df)} recordings, native packet rate {np.mean(rates):.2f} Hz "
      f"(Nyquist {np.mean(rates)/2:.1f} Hz -> |v| < {np.mean(rates)/2/34.5:.2f} m/s)\n")
print(f"WITHIN-recording temporal r (native, no resample, FULL record):")
print(f"  broadband : mean {df.r_time.mean():+.3f}  median {df.r_time.median():+.3f}")
for nm in ("slow", "mid", "fast"):
    v = df[f"r_{nm}"]
    print(f"  {nm:9s}: mean {v.mean():+.3f}  median {v.median():+.3f}  "
          f"|r|>0.2 in {(v.abs()>0.2).mean()*100:.0f}% of recordings")
print(f"  envelope energy in slow band: {df.e_slow.mean()*100:.1f}%")
g = df.groupby("act").agg(imu=("imu", "mean"), csi=("csi", "mean"), n=("act", "size"))
g = g[g.n >= 3].sort_values("csi", ascending=False)
print("\n  TOP :", ", ".join(f"{a}({v:.3g})" for a, v in g.csi.head(8).items()))
print("  BOT :", ", ".join(f"{a}({v:.3g})" for a, v in g.csi.tail(8).items()))
rho, p = spearmanr(g.imu, g.csi)
print(f"\n  BETWEEN-activity Spearman vs IMU: rho = {rho:+.3f} (p={p:.2g})   [resampled was +0.054]")
rr, pp = spearmanr(df.imu, df.csi)
print(f"  per-recording: rho = {rr:+.3f} (p={pp:.2g})   [resampled was +0.063]")
print(f"  csi cv = {df.csi.std()/df.csi.mean():.2f}   [resampled was 0.17]")
