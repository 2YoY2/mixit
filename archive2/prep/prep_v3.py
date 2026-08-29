#!/usr/bin/env python3
"""prep_v3: OctoNet raw -> model-ready AMPLITUDE streams. Design fixed by the
diagnostics, do not "improve" it without re-running them:

  09_phase  : the two stored antenna slices carry INDEPENDENT per-packet random
              phase -> cross-antenna conj is dead on this dataset.
  11_rep    : amplitude tracks the worn IMU (rec-level spearman +0.33, p=2e-6);
              aconj is phase-coherent (sig 0.15) but tracks worse.
  manifest  : 4 simultaneous nodes; packet rate 45-99 Hz, monotonic timestamps;
              3 date-disjoint campaigns = the 3 scenes; subject prefix encodes
              the scene, base id the person (10 people appear in all 3 scenes).

Per recording (one wifi pickle = one recording dir):
  1. |CSI| of both antennas: (N, 2, 114) -> (N, 228)
  2. per-antenna GLOBAL gain normalisation only (AGC / tx power). The
     cross-subcarrier amplitude profile IS the room information -- per-column
     normalisation would erase it; never add it.
  3. bin-average onto a uniform FS grid (averaging adds no content -- 08's
     lesson; no complex interpolation anywhere). Empty bins: forward-fill,
     fraction recorded in meta as `fill`.
  4. unit RMS over the whole stream; float16.
     Skip: rate < 30 Hz, duration < 20 s, fill > 0.35.
  5. worn-IMU envelope on the SAME grid (free-accel norm, 0.25 s smoothing),
     naive-datetime clock alignment (the v2 timezone trap), if a paired imu
     pickle exists within 10 s.

Output under $PREP_OUT (default ~/zerdani/buffer/octonet/prep_v3):
  streams/{rid:06d}.npy  (T, 228) float16
  imu/{rid:06d}.npy      (T,)     float16   (only when imu_ok)
  meta.csv   rid,file,node,date,scene,subject,base_subject,act,trial,
             nsamp,rate,fill,imu_ok,split
  templates.npz          per (node, date) mean static, key "{node}_{date}"

scene: dates <= 20240801 -> 0, 202409xx -> 1, >= 20241029 -> 2.
split: scenes 0,1 -> train, scene 2 -> test (zero-shot room; the identity
cohort's ~10 shared people are seen in the training rooms and reappear in
the held-out one -- the honest cross-environment identity setup).

  LIMIT=16 NPROC=4 python3 prep/prep_v3.py       # smoke run
  nohup python3 prep/prep_v3.py > ../log_prep3.txt 2>&1 &
"""
import os, glob, pickle, re
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT  = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/octonet/OctoNet-upload"))
OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v3"))
FS    = float(os.environ.get("FS", "40"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NPROC = int(os.environ.get("NPROC", "8"))
SMOOTH_SEC, MAX_GAP = 0.25, 10
WPAT = re.compile(r"exp-(\d{14})_node_(\d)_modality_wifi_subject_(\d+)_activity_(.+?)_trial_(\d+)")
IPAT = re.compile(r"(\d{14})_node_\d_modality_imu")
IMU_PATHS, IMU_KEYS = {}, np.array([])

def naive(v):
    return pd.Timestamp(v).to_pydatetime().replace(tzinfo=None)

def rel_times(ts):
    t0 = naive(ts[0])
    t = np.array([(naive(v) - t0).total_seconds() for v in ts])
    k = np.concatenate([[True], np.diff(t) > 0])
    return t, k, t0

def bin_mean(t, y, nb):
    """(N,) times + (N, C) values -> (nb, C) bin means, ffilled; returns fill frac."""
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, y.shape[1]), np.float32)
    np.add.at(s, idx, y.astype(np.float32))
    m = s / np.maximum(cnt, 1)[:, None]
    m[cnt == 0] = np.nan
    m = pd.DataFrame(m).ffill().bfill().to_numpy(np.float32)
    return m, float((cnt == 0).mean())

def imu_env(path):
    d = pickle.load(open(path, "rb"))
    X = np.asarray(d["data"])                          # (Ni, 13, 17)
    a = np.linalg.norm(X[:, 0:3, :], axis=1).mean(1)   # free-accel magnitude
    t, k, t0 = rel_times(list(d["timestamps"]))
    return a[k], t[k], t0

def one(job):
    rid, f = job
    try:
        d = pickle.load(open(os.path.join(ROOT, f), "rb"))
        m = WPAT.match(os.path.basename(os.path.dirname(f)))
        x = np.asarray(d["data"])
        if x.ndim != 3 or x.shape[1] != 2: return None
        t, k, w0 = rel_times(d.get("timestamp", d.get("timestamps")))
        t, x = t[k], x[k]
        dur = float(t[-1])
        rate = len(t) / dur if dur > 0 else 0.0
        if dur < 20 or rate < 30: return None
        amp = np.abs(x).astype(np.float32)
        amp /= np.maximum(amp.mean(axis=(0, 2), keepdims=True), 1e-12)
        nb = int(dur * FS)
        y, fill = bin_mean(t, amp.reshape(len(amp), -1), nb)
        if fill > 0.35: return None
        y = y / max(float(np.sqrt((y ** 2).mean())), 1e-12)
        np.save(f"{OUT}/streams/{rid:06d}.npy", y.astype(np.float16))
        stamp, node = int(m.group(1)), int(m.group(2))
        subj, act, trial = int(m.group(3)), m.group(4), int(m.group(5))
        imu_ok = 0
        if len(IMU_KEYS):
            kk = int(IMU_KEYS[np.argmin(np.abs(IMU_KEYS - stamp))])
            if abs(kk - stamp) <= MAX_GAP:
                try:
                    a, ti, i0 = imu_env(IMU_PATHS[kk])
                    src = ti + (i0 - w0).total_seconds()
                    keep = (src >= 0) & (src <= dur)
                    if keep.sum() > 20 and src[keep][-1] > 0.5 * dur:
                        g, _ = bin_mean(src[keep], a[keep, None], nb)
                        g = uniform_filter1d(g[:, 0], max(3, int(SMOOTH_SEC * FS) | 1))
                        if g.std() > 0:
                            np.save(f"{OUT}/imu/{rid:06d}.npy", g.astype(np.float16))
                            imu_ok = 1
                except Exception:
                    pass
        date = str(stamp)[:8]
        scene = 0 if date <= "20240801" else (1 if date < "20241000" else 2)
        row = (rid, f, node, date, scene, subj, subj % 100, act, trial,
               nb, round(rate, 2), round(fill, 4), imu_ok,
               "train")   # all 3 OctoNet rooms pre-train; PA scenes 4/5 are the gate
        return row, y.mean(0)
    except Exception:
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        return None

def main():
    global IMU_PATHS, IMU_KEYS
    os.makedirs(f"{OUT}/streams", exist_ok=True)
    os.makedirs(f"{OUT}/imu", exist_ok=True)
    IMU_PATHS = {int(m.group(1)): p for p in glob.glob(f"{ROOT}/imu/*.pickle")
                 if (m := IPAT.match(os.path.basename(p)))}
    IMU_KEYS = np.array(sorted(IMU_PATHS))
    files = sorted(os.path.relpath(f, ROOT)
                   for f in glob.glob(f"{ROOT}/node_*/wifi/*/*.pickle"))
    jobs = list(enumerate(files))
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"{len(jobs)} recordings -> {OUT}  (FS={FS} Hz, {NPROC} procs)", flush=True)
    rows, statics = [], {}
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            if r is not None:
                rows.append(r[0])
                statics.setdefault((r[0][2], r[0][3]), []).append(r[1])
            if (i + 1) % 200 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "file", "node", "date", "scene",
                                     "subject", "base_subject", "act", "trial",
                                     "nsamp", "rate", "fill", "imu_ok", "split"]
                      ).sort_values("rid")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    np.savez(f"{OUT}/templates.npz",
             **{f"{n}_{d}": np.mean(v, axis=0) for (n, d), v in statics.items()})
    print(f"\n{len(df)} kept / {len(jobs)}   imu_ok {df.imu_ok.mean()*100:.1f}%   "
          f"templates {len(statics)} (node,date) groups")
    print(df.groupby("scene").agg(recs=("rid", "size"), subj=("subject", "nunique"),
                                  imu=("imu_ok", "mean")).round(3))
    print(f"disk: ~{sum(r[9] for r in rows) * 228 * 2 / 1e9:.1f} GB streams", flush=True)

if __name__ == "__main__":
    main()
