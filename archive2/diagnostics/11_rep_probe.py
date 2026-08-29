#!/usr/bin/env python3
"""Which representation survives OctoNet's per-chain phase noise?

09_phase verdict: the two stored antenna slices carry INDEPENDENT per-packet
random phase (sig_conj ~ 1.9 rad ~ the uniform limit pi/sqrt(3) = 1.81), while
amplitude is clean (dyn|c| ~ 0.9%). Cross-antenna conj -- the PerceptAlign
sanitization -- is therefore destroyed on this dataset, and everything computed
through it (08's near-zero IMU correlation, 10's morph ceiling on v2 streams)
measured phase noise, not channel. This probe picks the replacement.

Candidates, all computable per packet with no cross-chain phase reference:
  amp    |h0(k,t)|                    phase-free; body appears via the
                                      2*Re(room* body) interference term
  aconj  h0(k,t) * conj(h0(k+1,t))    adjacent-subcarrier product WITHIN one
                                      chain: per-packet common phase cancels
                                      exactly; residual is the per-packet STO
                                      ramp step e^{j d(t)} -- measured by sig
  xconj  h0 * conj(h1)                the broken one, kept as control

Per rep x NREC recordings (native timing, binned 0.25 s, no interpolation):
  sig    per-packet phase-increment std, mid subcarrier   (0 for amp)
  dyn    dynamic energy fraction after per-column normalisation
  r_imu  dynamic-energy envelope vs worn-IMU envelope, full record
READ: the winner has sig << 1.8, dyn in a sane 0.01-0.3 band, and the highest
r_imu / %|r|>0.2. If aconj wins, prep_v3 keeps complex structure; if only amp
survives, prep_v3 is amplitude-based and Doppler sign is gone for good.
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/octonet/OctoNet-upload"))
D    = os.path.expanduser(os.environ.get("MIXIT_DATA", "~/zerdani/buffer/octonet/mixit_data_v2"))
N    = int(os.environ.get("NREC", "200"))
BIN  = 0.25
REPS = ("amp", "aconj", "xconj")

def load(f):
    d = pickle.load(open(f, "rb"))
    ts = d.get("timestamp", d.get("timestamps"))
    x = np.asarray(d["data"])
    if x.ndim != 3 or x.shape[1] != 2: return None, None
    t = np.array([(v - ts[0]).total_seconds() for v in ts])
    k = np.concatenate([[True], np.diff(t) > 0])
    return t[k], x[k]

def make_rep(x, name):
    h0, h1 = x[:, 0, :], x[:, 1, :]
    if name == "amp":   y = np.abs(h0).astype(np.complex64)
    if name == "aconj": y = h0[:, :-1] * np.conj(h0[:, 1:])
    if name == "xconj": y = h0 * np.conj(h1)
    mag = np.abs(y).mean(0); v = mag > 0.1 * np.median(mag)
    y = y[:, v]
    return y / np.maximum(np.abs(y).mean(0), 1e-12)

def stats(y, t):
    k = y.shape[1] // 2
    ph = np.angle(y[:, k])
    sig = float(np.std(np.diff(np.unwrap(ph)))) if np.abs(y[:, k]).std() > 0 else 0.0
    d = y - y.mean(0, keepdims=True)
    dyn = float((np.abs(d) ** 2).mean() / (np.abs(y) ** 2).mean())
    nb = int(t[-1] / BIN)
    if nb < 20: return sig, dyn, None
    idx = np.clip((t / BIN).astype(int), 0, nb - 1)
    e = np.zeros(nb)
    for b in range(nb):
        m = idx == b
        if m.sum() >= 3:
            seg = y[m]; e[b] = float((np.abs(seg - seg.mean(0)) ** 2).mean())
    return sig, dyn, e

meta = pd.read_csv(f"{D}/meta.csv").merge(
    pd.read_csv(f"{D}/imu_meta.csv")[["rid", "imu_ok", "act"]], on="rid")
meta = meta[meta.imu_ok == 1].reset_index(drop=True)
rng = np.random.default_rng(0)
sel = meta.iloc[rng.choice(len(meta), min(N, len(meta)), replace=False)]
acc = {r: {"sig": [], "dyn": [], "r": [], "imu": [], "csi": []} for r in REPS}
for row in sel.itertuples():
    try:
        t, x = load(os.path.join(ROOT, row.file))
        if t is None: continue
        g = np.load(f"{D}/imu_env/{row.rid:06d}.npy", mmap_mode="r")
        gi = np.array(g)[::int(round(BIN * 100))]
        for rep in REPS:
            sig, dyn, e = stats(make_rep(x, rep), t)
            if e is None: continue
            m = min(len(e), len(gi))
            if m < 20: continue
            a, b = e[:m], gi[:m]
            ok = a > 0
            if ok.sum() < 20: continue
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            acc[rep]["sig"].append(sig); acc[rep]["dyn"].append(dyn)
            acc[rep]["r"].append(r)
            acc[rep]["imu"].append(float(b[ok].mean()))
            acc[rep]["csi"].append(float(a[ok].mean()))
    except Exception:
        continue

print(f"{len(acc[REPS[0]]['r'])} recordings scored per rep (native, BIN={BIN}s)\n")
print(f"{'rep':7s}{'sig':>8s}{'dyn':>9s}{'r_imu med':>11s}{'r_imu mean':>12s}"
      f"{'%|r|>0.2':>10s}{'rec-spearman':>14s}")
print("-" * 61)
for rep in REPS:
    a = acc[rep]
    if not a["r"]: print(f"{rep:7s}  (no data)"); continue
    r = np.array(a["r"])
    rho, p = spearmanr(a["imu"], a["csi"])
    print(f"{rep:7s}{np.mean(a['sig']):8.3f}{np.mean(a['dyn']):9.4f}"
          f"{np.median(r):11.3f}{np.mean(r):12.3f}{(np.abs(r) > 0.2).mean()*100:9.0f}%"
          f"{rho:+11.3f} (p={p:.1g})")
print("""
READ: winner = sig << 1.8, dyn in 0.01-0.3, highest r_imu / spearman.
  amp wins, aconj sig ~1.8  -> STO also jitters; amplitude-only prep_v3.
  aconj wins with small sig -> complex prep_v3 on adjacent-subcarrier products.
  everything flat           -> the wifi genuinely can't see the wearer at this
                               packet rate; escalate before writing any prep.
""")
