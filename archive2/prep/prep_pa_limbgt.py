#!/usr/bin/env python3
"""prep_pa_limbgt: per-limb GT envelopes for prep_pa_xrf streams, from BODY25
keypoints3d. Validated at routing grade by diagnostics/18 (r~0.3 vs null 0).

Per clip (shared by its 3 receivers): keypoints3d/*.json -> person 0,
joints [LWrist7 RWrist4 LHip12 RHip9 Head0], confidence>0.3, 3D speeds ->
(T,5) envelopes on the clip's 50 Hz grid (start-synced; 0.5 s smoothing gives
the alignment tolerance diagnostics/18 operated under). Written as
imu/{rid:06d}.npy for every receiver rid of the clip; meta.csv imu_ok set to 1.

  CLIPS=3000 python3 prep/prep_pa_limbgt.py     # subset, probe-sized
  CLIPS=0    -> all clips
"""
import os, glob, json
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT  = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_pa_xrf"))
CLIPS = int(os.environ.get("CLIPS", "3000"))
NPROC = int(os.environ.get("NPROC", "6"))
FS    = float(os.environ.get("FS", "50"))  # must match the target prep's grid
SMK   = max(3, int(FS / 2)) | 1            # 0.5 s smoothing at any rate
JB = [7, 4, 12, 9, 0]                     # BODY25 LW RW LHip RHip Nose

def one(job):
    kd, rows = job                        # rows: [(rid, nsamp), ...] per receiver
    try:
        fs = sorted(glob.glob(f"{kd}/*.json"))
        if len(fs) < 20: return None
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
        if np.nanmedian(np.abs(P)) > 50: P *= 1e-3          # mm -> m
        out = []
        for rid, nsamp in rows:
            dur = nsamp / FS
            fps = len(fs) / dur
            v = np.linalg.norm(np.diff(P[:, JB], axis=0), axis=-1) * fps
            v = np.where(np.isfinite(v), v, 0)
            tv = (np.arange(len(v)) + 0.5) / fps
            ti = np.arange(nsamp) / FS
            e = np.stack([np.interp(ti, tv, v[:, i]) for i in range(5)], 1)
            e = uniform_filter1d(e.astype(np.float32), SMK, axis=0)
            if e.std() < 1e-9: continue
            e = e / (e.std(0) + 1e-9)
            np.save(f"{OUT}/imu/{rid:06d}.npy", e.astype(np.float16))
            out.append(rid)
        return out
    except Exception:
        return None

def main():
    os.makedirs(f"{OUT}/imu", exist_ok=True)
    meta = pd.read_csv(f"{OUT}/meta.csv")
    spl = os.environ.get("SPLIT", "train")
    tr = meta[meta.split == spl].copy()
    tr["kd"] = tr.file.map(lambda f: os.path.join(
        ROOT, os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d"))
    groups = [(kd, [(int(r.rid), int(r.nsamp)) for r in g.itertuples()])
              for kd, g in tr.groupby("kd")]
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    if CLIPS: groups = groups[:CLIPS]
    print(f"{len(groups)} clips -> limb GT", flush=True)
    done = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, groups, chunksize=8)):
            if r: done.extend(r)
            if (i + 1) % 500 == 0: print(f"  {i+1}/{len(groups)}", flush=True)
    meta.loc[meta.rid.isin(done), "imu_ok"] = 1
    meta.to_csv(f"{OUT}/meta.csv", index=False)
    print(f"{len(done)} rids got limb GT | imu_ok now "
          f"{meta[meta.split=='train'].imu_ok.mean()*100:.1f}% of train")

if __name__ == "__main__":
    main()
