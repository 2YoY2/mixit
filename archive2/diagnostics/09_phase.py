#!/usr/bin/env python3
"""Do the two 'antennas' share a phase reference? Direct measurement, no theory.

If they share an LO/ADC (same NIC, same packet), then
    angle( x[:,0,k] * conj(x[:,1,k]) )
is a FIXED geometric phase difference plus slow body Doppler -> nearly flat,
tiny per-packet increments.
If they do not, it random-walks with per-packet std sigma, and the predicted
dynamic fraction is  1 - exp(-sigma^2).  The workflow's fit needs sigma=1.52.

Reference from stage_two (2 antennas, one NIC, known-good): conj dyn frac 0.130.
"""
import os, glob, pickle
import numpy as np
ROOT = os.path.expanduser("~/zerdani/buffer/octonet/OctoNet-upload")
NF = int(os.environ.get("NFILES", "12"))

def dyn(y):
    d = y - y.mean(0, keepdims=True)
    return float((np.abs(d) ** 2).mean() / (np.abs(y) ** 2).mean())

files = sorted(glob.glob(f"{ROOT}/node_*/wifi/*/*.pickle"))[:NF]
print(f"{len(files)} files\n")
print(f"{'file':38s}{'sig_a0':>9s}{'sig_conj':>10s}{'pred_dyn':>10s}"
      f"{'dyn|c|':>9s}{'dyn_a0':>9s}{'dyn_conj':>10s}")
print("-" * 95)
agg = []
for f in files:
    d = pickle.load(open(f, "rb"))
    x = np.asarray(d["data"])
    if x.ndim != 3 or x.shape[1] != 2: continue
    k = x.shape[2] // 2                                  # mid subcarrier
    a0 = x[:, 0, k]
    cj = x[:, 0, :] * np.conj(x[:, 1, :])
    # per-packet phase increment std (the sigma the fit needs)
    s_a0 = float(np.std(np.diff(np.unwrap(np.angle(a0)))))
    s_cj = float(np.std(np.diff(np.unwrap(np.angle(cj[:, k])))))
    pred = 1 - np.exp(-s_cj ** 2)
    row = (os.path.basename(os.path.dirname(f))[4:40], s_a0, s_cj, pred,
           dyn(np.abs(x[:, 0, :])), dyn(x[:, 0, :]), dyn(cj))
    agg.append(row[1:])
    print(f"{row[0]:38s}{row[1]:9.3f}{row[2]:10.3f}{row[3]:10.3f}"
          f"{row[4]:9.3f}{row[5]:9.3f}{row[6]:10.3f}")
a = np.array(agg)
print("-" * 95)
print(f"{'MEAN':38s}{a[:,0].mean():9.3f}{a[:,1].mean():10.3f}{a[:,2].mean():10.3f}"
      f"{a[:,3].mean():9.3f}{a[:,4].mean():9.3f}{a[:,5].mean():10.3f}")
print(f"""
READ:
  sig_conj near 0 and dyn_conj near 0.13  -> antennas DO share a reference.
      The 0.90 came from somewhere else; audit that measurement first.
  sig_conj ~1.5 rad and dyn_conj ~0.90    -> the common phase is NOT cancelling.
      Then check whether the two slices are the same packet before blaming clocks.
  dyn|c| (amplitude only, phase-immune)   -> if this is already high, it is AGC,
      not phase, and the whole phase story is wrong.
  stage_two reference: dyn_conj = 0.130 on 2 antennas of one NIC.
""")
