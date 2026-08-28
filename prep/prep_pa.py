#!/usr/bin/env python3
"""prep_pa: PerceptAlign raw .mat -> the SAME model format as prep_v3.

Raw: Scene{1..5}/user*/action*/<clip>/csi_mat/<take>-r<rx>.mat, v7.3 HDF5,
csi/csi (3 ant, 57 subc, T) complex compound, csi/timestamp (1, T) uint64.

Mapping to the OctoNet-trained model's input, (T, 228) amplitude @ 40 Hz:
  1. |CSI| of the two OUTER antennas (0, 2) -- most separated pair
  2. per-antenna global gain norm (spectral shape kept, as in prep_v3)
  3. subcarrier axis 57 -> 114 by linear interpolation (magnitude-only,
     frequency axis -- no temporal content invented)
  4. sample rate from the .mat TIMESTAMPS (unit auto-detected) -- no fps
     assumptions, which retires the old Scene4/5 fps landmine; the per-scene
     rate census printed at the end is the verification
  5. bin-average to 40 Hz, ffill gaps, unit RMS, float16
     skip: dur < 2 s, fill > 0.35

meta.csv: same schema as prep_v3. node = "r{rx}", date = "s{scene}_u{user}"
so the (node, date) group key means same receiver + same layout, imu_ok = 0.
split: Scenes 1-3 -> train, Scenes 4-5 -> test (the transfer gate).

  LIMIT=12 NPROC=4 python3 prep/prep_pa.py     # smoke
  nohup python3 prep/prep_pa.py > ../log_prep_pa.txt 2>&1 &
"""
import os, glob, re
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py

ROOT  = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_pa"))
FS    = float(os.environ.get("FS", "40"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NPROC = int(os.environ.get("NPROC", "6"))
PAT = re.compile(r"Scene(\d+)/user(\d+)/action(\d+)/([^/]+)/csi_mat/(\d+)-r(\d)\.mat$")
K57, K = 57, 114
src, dst = np.linspace(0, 1, K57), np.linspace(0, 1, K)
Wm = np.zeros((K, K57), np.float32)
for i, d in enumerate(dst):
    j = min(max(np.searchsorted(src, d), 1), K57 - 1)
    w = (d - src[j - 1]) / (src[j] - src[j - 1])
    Wm[i, j - 1], Wm[i, j] = 1 - w, w

def bin_mean(t, y, nb):
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, y.shape[1]), np.float32)
    np.add.at(s, idx, y.astype(np.float32))
    m = s / np.maximum(cnt, 1)[:, None]
    m[cnt == 0] = np.nan
    m = pd.DataFrame(m).ffill().bfill().to_numpy(np.float32)
    return m, float((cnt == 0).mean())

def one(job):
    rid, f = job
    try:
        mm = PAT.search(f)
        with h5py.File(os.path.join(ROOT, f), "r") as h:
            c = h["csi/csi"][...]
            ts = h["csi/timestamp"][...].ravel().astype(np.float64)
        amp = np.hypot(c["real"], c["imag"])              # (3, 57, T)
        dt = float(np.median(np.diff(ts)))
        rate, t = None, None
        for unit in (1.0, 1e-3, 1e-6, 1e-9):
            if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
                rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
        if rate is None:
            rate = 810.0; t = np.arange(amp.shape[-1]) / rate
        keep = np.concatenate([[True], np.diff(t) > 0])
        amp, t = amp[..., keep], t[keep]
        dur = float(t[-1])
        if dur < 2.0: return None
        a = np.stack([amp[0], amp[2]], 0)                 # outer antennas
        a = a / np.maximum(a.mean(axis=(1, 2), keepdims=True), 1e-12)
        v = np.concatenate([(Wm @ a[0]).T, (Wm @ a[1]).T], 1)   # (T, 228)
        nb = int(dur * FS)
        if nb < 60: return None
        y, fill = bin_mean(t, v, nb)
        if fill > 0.35: return None
        y = y / max(float(np.sqrt((y ** 2).mean())), 1e-12)
        np.save(f"{OUT}/streams/{rid:06d}.npy", y.astype(np.float16))
        scene, user = int(mm.group(1)), int(mm.group(2))
        act, clip, take, rx = int(mm.group(3)), mm.group(4), int(mm.group(5)), int(mm.group(6))
        return (rid, f, f"r{rx}", f"s{scene}_u{user}", scene, user, user, act,
                take, nb, round(rate, 2), round(fill, 4), 0,
                "train" if scene <= 3 else "test")
    except Exception:
        if os.environ.get("DEBUG"):
            import traceback; traceback.print_exc()
        return None

def main():
    os.makedirs(f"{OUT}/streams", exist_ok=True)
    files = sorted(os.path.relpath(f, ROOT) for f in
                   glob.glob(f"{ROOT}/Scene*/user*/action*/*/csi_mat/*.mat"))
    jobs = list(enumerate(files))
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"{len(jobs)} mats -> {OUT} (FS={FS}, {NPROC} procs)", flush=True)
    rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            if r is not None: rows.append(r)
            if (i + 1) % 500 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "file", "node", "date", "scene",
                                     "subject", "base_subject", "act", "trial",
                                     "nsamp", "rate", "fill", "imu_ok", "split"]
                      ).sort_values("rid")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print(f"\n{len(df)} kept / {len(jobs)}")
    print(df.groupby("scene").agg(recs=("rid", "size"), users=("subject", "nunique"),
                                  groups=("date", "nunique"),
                                  nsamp_med=("nsamp", "median"),
                                  rate_med=("rate", "median"),
                                  rate_p10=("rate", lambda v: np.percentile(v, 10)),
                                  rate_p90=("rate", lambda v: np.percentile(v, 90))).round(1))
    print("\nREAD: rate_med per scene is the fps-landmine verdict from raw "
          "timestamps. nsamp_med bounds the fine-tune WIN (use <= p10 of test).",
          flush=True)

if __name__ == "__main__":
    main()
