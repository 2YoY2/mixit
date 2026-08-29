#!/usr/bin/env python3
"""Validate prep_v3 for pre-training, and (re)write the split column:
scenes 0,1 -> train, scene 2 -> test. Original meta kept as meta.orig.csv.

Checks -- fail loudly, nothing passes silently:
  A  meta<->files: every row's stream exists, stored length == nsamp,
     imu file exists iff imu_ok, same length as the stream
  B  numeric health (SAMP sampled streams): finite, unit RMS within 2%,
     static-frac distribution (expect ~0.9-0.99: room-dominated amplitude)
  C  signal survival (SAMP sampled imu_ok streams): dynamic-energy envelope
     of the PREPPED stream vs the aligned IMU envelope, 0.25 s bins --
     the probe measured this on raw packets (med r 0.071, 35% |r|>0.2,
     rec-spearman +0.33); binning + f16 must not have eaten it.

  SAMP=200 python3 prep/01_validate.py
"""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT  = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v3"))
SAMP = int(os.environ.get("SAMP", "200"))
FS, BIN = 40, 10                       # grid Hz, envelope bin = 10 samples = 0.25 s

meta = pd.read_csv(f"{OUT}/meta.csv")
if not os.path.exists(f"{OUT}/meta.orig.csv"):
    meta.to_csv(f"{OUT}/meta.orig.csv", index=False)
meta["split"] = np.where(meta.scene <= 1, "train", "test")
meta.to_csv(f"{OUT}/meta.csv", index=False)
print("split rewritten: scenes 0,1 -> train, scene 2 -> test")
print(meta.groupby(["split", "scene"]).agg(recs=("rid", "size"),
      subj=("subject", "nunique"), imu=("imu_ok", "mean")).round(3), "\n")

# A -- full consistency sweep (mmap headers only, cheap)
bad = []
for r in meta.itertuples():
    sf = f"{OUT}/streams/{r.rid:06d}.npy"
    gf = f"{OUT}/imu/{r.rid:06d}.npy"
    if not os.path.exists(sf): bad.append((r.rid, "stream missing")); continue
    n = len(np.load(sf, mmap_mode="r"))
    if n != r.nsamp: bad.append((r.rid, f"len {n} != nsamp {r.nsamp}"))
    if bool(r.imu_ok) != os.path.exists(gf): bad.append((r.rid, "imu_ok/file mismatch"))
    elif r.imu_ok and len(np.load(gf, mmap_mode="r")) != n:
        bad.append((r.rid, "imu/stream length mismatch"))
orphans = set(int(os.path.basename(p)[:6]) for p in glob.glob(f"{OUT}/streams/*.npy")) \
          - set(meta.rid)
print(f"A  consistency: {len(bad)} bad rows, {len(orphans)} orphan streams "
      f"{'-- PASS' if not bad and not orphans else '-- FAIL: ' + str((bad + [(o, 'orphan') for o in sorted(orphans)])[:8])}")

# B + C -- sampled deep checks
rng = np.random.default_rng(0)
sel = meta[meta.imu_ok == 1].sample(min(SAMP, int(meta.imu_ok.sum())), random_state=0)
rms_off, sfrac, rs, im_mean, cs_mean, nonfinite = [], [], [], [], [], 0
for r in sel.itertuples():
    y = np.load(f"{OUT}/streams/{r.rid:06d}.npy").astype(np.float32)
    if not np.isfinite(y).all(): nonfinite += 1; continue
    rms_off.append(abs(float(np.sqrt((y ** 2).mean())) - 1.0))
    st = y.mean(0)
    sfrac.append(float((st ** 2).sum() / np.maximum((y ** 2).mean(0), 1e-12).sum()))
    nb = len(y) // BIN
    seg = y[:nb * BIN].reshape(nb, BIN, -1)
    e = ((seg - seg.mean(1, keepdims=True)) ** 2).mean((1, 2))
    g = np.load(f"{OUT}/imu/{r.rid:06d}.npy").astype(np.float32)
    gi = g[:nb * BIN].reshape(nb, BIN).mean(1)
    m = min(len(e), len(gi)); a, b = e[:m], gi[:m]
    if a.std() > 0 and b.std() > 0:
        rs.append(float(np.corrcoef(a, b)[0, 1]))
        im_mean.append(float(b.mean())); cs_mean.append(float(a.mean()))
print(f"B  health: nonfinite {nonfinite}/{len(sel)}, |rms-1| max {max(rms_off):.4f}, "
      f"static-frac median {np.median(sfrac):.3f} (p10 {np.percentile(sfrac, 10):.3f})")
r = np.array(rs); rho, p = spearmanr(im_mean, cs_mean)
print(f"C  imu tracking on prepped streams ({len(r)} recs): "
      f"median r {np.median(r):+.3f}  mean {r.mean():+.3f}  "
      f"|r|>0.2 in {(np.abs(r) > 0.2).mean()*100:.0f}%  rec-spearman {rho:+.3f} (p={p:.1g})")
print("\nREAD: A must PASS. B: static-frac ~0.9-0.99. C: numbers at or above the "
      "raw-packet probe (med 0.071 / 35% / +0.33) -> dataset READY for pre-training.")
