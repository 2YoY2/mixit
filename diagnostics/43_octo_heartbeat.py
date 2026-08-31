#!/usr/bin/env python3
"""Probe 43: does OctoNet WiFi CSI carry the HEARTBEAT?  (user: OctoNet has
Polar H10 HR GT — check CSI x heartbeat correlation first.)

Per recording (prep_v75 amplitude streams, ~95 s @ ~73 Hz — 0.01 Hz
spectral resolution; OctoNet phase is dead, amplitude-only regime):
  z-score per column -> Hann FFT -> PSD summed over the 228 columns ->
  peak in the cardiac band (0.8-2.3 Hz) -> f_csi vs Polar mean HR/60.
Also the 2nd harmonic (cardiac motion is impulsive) and the breath band
(0.15-0.5 Hz) peak for reference.

Arms: STILL activities (sit, sleep, ...) = the claim; MOVING activities
(walk, dance, jog) = destruction control (body motion must mask cardiac);
shuffled HR pairing = chance null for the hit-rate.

  ACTS=sit,sleep CTRL=walk,dance NRID=40 python3 diagnostics/43_octo_heartbeat.py
"""
import os, glob, pickle
import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get(
    "OROOT", "~/zerdani/buffer/octonet/OctoNet-upload"))
PREP = os.path.expanduser(os.environ.get(
    "OPREP", "~/zerdani/buffer/octonet/prep_v75"))
ACTS = os.environ.get("ACTS", "sit,sleep").split(",")
CTRL = os.environ.get("CTRL", "walk,dance,jog").split(",")
NRID = int(os.environ.get("NRID", "40"))
CB = (0.8, 2.3)                              # cardiac band (48-138 bpm)
BB = (0.15, 0.5)                             # breath band
TOL = float(os.environ.get("TOL", "0.15"))   # Hz (~9 bpm)

def polar_hr(row):
    pat = (f"{ROOT}/node_{row.node}/polar/data/{row.date}*_node_{row.node}"
           f"_modality_heartrate_subject_{row.subject}_activity_{row.act}"
           f"_trial_{row.trial}.pickle")
    fs = glob.glob(pat)
    if not fs: return None
    recs = []
    with open(fs[0], "rb") as h:
        while True:
            try: recs.append(pickle.load(h))
            except Exception: break
    hr = np.array([r["data"] for r in recs
                   if isinstance(r, dict) and "data" in r], float)
    hr = hr[(hr > 35) & (hr < 220)]
    if len(hr) < 20: return None
    return float(np.median(hr))

def csi_peaks(rid, rate):
    f = f"{PREP}/streams/{rid:06d}.npy"
    if not os.path.exists(f):
        f = f"{PREP}/streams/{rid}.npy"
        if not os.path.exists(f): return None
    x = np.load(f).astype(np.float32)
    T = len(x)
    if T < 30 * rate: return None
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    w = np.hanning(T)[:, None].astype(np.float32)
    X = np.fft.rfft(x * w, axis=0)
    psd = (np.abs(X) ** 2).sum(1)
    fr = np.fft.rfftfreq(T, 1.0 / rate)
    out = {}
    for name, (lo, hi) in (("card", CB), ("breath", BB)):
        m = (fr >= lo) & (fr <= hi)
        pk = np.argmax(psd[m])
        out[name] = float(fr[m][pk])
        out[name + "_prom"] = float(psd[m][pk] / (np.median(psd[m]) + 1e-12))
    return out

def run_arm(man, acts, tag):
    rows, rng = [], np.random.default_rng(43)
    sub = man[man.act.isin(acts)].sample(frac=1, random_state=43)
    for r in sub.itertuples():
        hr = polar_hr(r)
        if hr is None: continue
        pk = csi_peaks(int(r.rid), float(r.rate))
        if pk is None: continue
        rows.append((hr / 60.0, pk["card"], pk["card_prom"], pk["breath"]))
        if len(rows) >= NRID: break
    if len(rows) < 8:
        print(f"[{tag}] only {len(rows)} matched — skipping", flush=True)
        return
    A = np.array(rows)
    fhr, fcsi = A[:, 0], A[:, 1]
    e1 = np.abs(fcsi - fhr)
    e2 = np.abs(fcsi - 2 * fhr)
    eb = np.minimum(e1, e2)                   # fundamental or harmonic
    hit = float(np.mean(eb <= TOL))
    nulls = []
    for _ in range(200):
        p = rng.permutation(fhr)
        nulls.append(np.mean(np.minimum(np.abs(fcsi - p),
                                        np.abs(fcsi - 2 * p)) <= TOL))
    c = float(np.corrcoef(fhr, fcsi)[0, 1]) if len(A) > 3 else np.nan
    print(f"[{tag}] n={len(A)}  hit@{TOL}Hz {hit*100:.0f}%  "
          f"null {np.mean(nulls)*100:.0f}%  corr(f_csi,f_hr) {c:+.2f}  "
          f"med|err| {np.median(eb)*60:.0f} bpm  "
          f"card-prominence med {np.median(A[:,2]):.1f}", flush=True)

def detect_arm(man, acts, tag):
    """detection test: excess PSD at the TRUE HR (best columns), vs the
    same statistic at other recordings' HRs (null)"""
    pairs = []
    sub = man[man.act.isin(acts)].sample(frac=1, random_state=43)
    for r in sub.itertuples():
        hr = polar_hr(r)
        if hr is None: continue
        f = f"{PREP}/streams/{int(r.rid):06d}.npy"
        if not os.path.exists(f): continue
        x = np.load(f).astype(np.float32)
        rate = float(r.rate)
        if len(x) < 30 * rate: continue
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        w = np.hanning(len(x))[:, None].astype(np.float32)
        X = np.abs(np.fft.rfft(x * w, axis=0)) ** 2       # (F, 228)
        fr = np.fft.rfftfreq(len(x), 1.0 / rate)
        cb = (fr >= CB[0]) & (fr <= CB[1])
        Xc = X[cb] / (np.median(X[cb], 0, keepdims=True) + 1e-12)
        pairs.append((fr[cb], Xc, hr / 60.0))
        if len(pairs) >= NRID: break
    if len(pairs) < 8:
        print(f"[det {tag}] only {len(pairs)} matched", flush=True)
        return
    def score(frv, Xc, f0):
        m = np.abs(frv - f0) <= 0.06
        if not m.any(): return np.nan
        return float(np.sort(Xc[m].max(0))[-5:].mean())   # top-5 columns
    true_s = [score(f_, X_, h_) for f_, X_, h_ in pairs]
    null_s = []
    hrs = [h for _, _, h in pairs]
    for i, (f_, X_, h_) in enumerate(pairs):
        for j, h2 in enumerate(hrs):
            if j != i and abs(h2 - h_) > 0.12:
                null_s.append(score(f_, X_, h2))
    true_s = np.array([s for s in true_s if np.isfinite(s)])
    null_s = np.array([s for s in null_s if np.isfinite(s)])
    thr = np.percentile(null_s, 95)
    print(f"[det {tag}] n={len(true_s)}  med excess@HR {np.median(true_s):.2f}"
          f" vs null {np.median(null_s):.2f}  frac>null95 "
          f"{np.mean(true_s > thr)*100:.0f}% (chance 5%)", flush=True)

def main():
    man = pd.read_csv(f"{PREP}/meta.csv")
    print("acts available:", {a: int((man.act == a).sum())
                              for a in ACTS + CTRL}, flush=True)
    run_arm(man, ACTS, "STILL " + "+".join(ACTS))
    run_arm(man, CTRL, "MOVING " + "+".join(CTRL))
    for a in ACTS:
        detect_arm(man, [a], a)
    detect_arm(man, CTRL, "moving-ctrl")

if __name__ == "__main__":
    main()
