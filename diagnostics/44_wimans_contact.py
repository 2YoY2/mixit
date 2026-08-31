#!/usr/bin/env python3
"""Probe 44: WiMANS first contact — go/no-go physics before any modeling.

A) PHASE ALIVENESS (the OctoNet landmine test, probe-09 protocol): the
   circular spread of per-packet cross-antenna product phases.  Alive
   (PA-like, one LO): product phase stable within a recording (spread
   << uniform limit ~1.9).  Dead (OctoNet-like): spread ~= uniform.
B) QUANTIZATION FLOOR: CMN'd dynamic band energy vs the +-0.5 LSB
   quantization noise (LSB estimated from the amplitude value lattice).
C) EMPTY-ROOM PRINT: 0-user samples per (environment, band) — split-half
   correlation of the mean product static (the room print), and its
   correlation to the OCCUPIED-ensemble mean (validates the label-free
   proxy the site-calibration law relies on).

  N0=20 NOCC=20 python3 diagnostics/44_wimans_contact.py
"""
import os, glob
import numpy as np
import pandas as pd
import scipy.io as sio

W = os.path.expanduser(os.environ.get("WIMANS", "~/zerdani/buffer/wimans"))
N0 = int(os.environ.get("N0", "20"))
NOCC = int(os.environ.get("NOCC", "20"))

def load_csi(label):
    f = f"{W}/wifi_csi/mat/{label}.mat"
    if not os.path.exists(f): return None
    try:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        tr = m["trace"]
        cs = np.stack([np.asarray(p.csi, np.complex64) for p in tr
                       if hasattr(p, "csi") and
                       np.asarray(p.csi).shape == (3, 3, 30)])
        return cs                                       # (T, 3rx, 3tx, 30)
    except Exception:
        return None

def phase_spread(cs):
    """circular std of per-packet cross-RX product phase (rx2*conj(rx1)),
    per (tx, subcarrier), median over channels — probe-09 statistic"""
    y = cs[:, 1] * np.conj(cs[:, 0])                    # (T, 3tx, 30)
    ph = np.angle(y)
    R = np.abs(np.exp(1j * ph).mean(0))                 # (3, 30)
    sig = np.sqrt(-2 * np.log(np.maximum(R, 1e-6)))
    return float(np.median(sig))

def qfloor(cs):
    """dynamic-band RMS vs quantization LSB (value-lattice estimate)"""
    a = np.abs(cs).ravel()
    a = a[a > 0]
    vals = np.unique(np.round(a[:200000], 6))
    lsb = np.median(np.diff(vals[:2000])) if len(vals) > 10 else np.nan
    dyn = cs - cs.mean(0, keepdims=True)
    return float(np.sqrt((np.abs(dyn) ** 2).mean())), float(lsb)

def prod_static(cs):
    g = np.sqrt((np.abs(cs) ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-9
    cs = cs / g
    y = cs[:, 1:] * np.conj(cs[:, :1])                  # (T, 2, 3, 30)
    return y.mean(0).ravel()

def cc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.abs((a * np.conj(b)).sum())
                 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    an = pd.read_csv(f"{W}/annotation.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    print("envs:", an.environment.value_counts().to_dict(), flush=True)
    print("users:", an.number_of_users.value_counts().sort_index().to_dict(),
          flush=True)
    rng = np.random.default_rng(44)

    print("\n=== A/B: phase + quantization (10 mixed samples)", flush=True)
    sub = an.sample(frac=1, random_state=44).head(40)
    sigs, dyns, lsbs = [], [], []
    for r in sub.itertuples():
        cs = load_csi(r.label)
        if cs is None or len(cs) < 300: continue
        sigs.append(phase_spread(cs))
        d, l = qfloor(cs)
        dyns.append(d); lsbs.append(l)
        if len(sigs) >= 10: break
    print(f"  phase spread med {np.median(sigs):.2f} "
          f"(alive << 1.9 uniform; PA-like ~0.1-0.3)", flush=True)
    print(f"  dyn RMS med {np.median(dyns):.4f} vs LSB/sqrt(12) "
          f"{np.median(lsbs)/np.sqrt(12):.4f} -> ratio "
          f"{np.median(dyns)/(np.median(lsbs)/np.sqrt(12)+1e-12):.1f}x",
          flush=True)

    print("\n=== C: empty-room print per (env, band)", flush=True)
    for env in an.environment.unique():
        for band in sorted(an.wifi_band.unique()):
            g0 = an[(an.environment == env) & (an.wifi_band == band)
                    & (an.number_of_users == 0)]
            gocc = an[(an.environment == env) & (an.wifi_band == band)
                      & (an.number_of_users >= 1)]
            s0 = [prod_static(c) for r in g0.sample(
                frac=1, random_state=44).head(N0).itertuples()
                if (c := load_csi(r.label)) is not None and len(c) > 300]
            so = [prod_static(c) for r in gocc.sample(
                frac=1, random_state=44).head(NOCC).itertuples()
                if (c := load_csi(r.label)) is not None and len(c) > 300]
            if len(s0) < 6 or len(so) < 6:
                print(f"  {env}/{band}: too few (n0={len(s0)})", flush=True)
                continue
            h = len(s0) // 2
            m1, m2 = np.mean(s0[:h], 0), np.mean(s0[h:], 0)
            e_split = cc(m1, m2)
            proxy = cc(np.mean(s0, 0), np.mean(so, 0))
            print(f"  {env}/{band}: n0={len(s0)} empty split-half {e_split:.3f}"
                  f"  |  occupied-proxy vs truth {proxy:.3f}", flush=True)

if __name__ == "__main__":
    main()
