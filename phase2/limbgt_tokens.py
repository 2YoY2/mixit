#!/usr/bin/env python3
"""Uniform limb GT for the token pipeline: the EXACT phase-1 recipe
(prep_pa_limbgt: start-synced proportional mapping + interp to the CSI grid
+ 0.5 s smoothing -- same as the dataset authors' own preprocessing, which
also start-syncs; no lag correction), applied identically to ALL five scenes
so rooms 1-5 share one GT convention, on the 400 Hz token grid.

BODY25 keypoints3d (fresh3d/) -> person 0, joints [LW7 RW4 LHip12 RHip9
Head0], conf>0.3, 3D speeds -> (nsamp,5) envelopes -> pa_tokens/imu/{rid}.npy
for every receiver rid of the clip. Resume-safe (skips existing files and
clips whose token files are not tokenized yet -- rerun after tokenize_pa).

  CLIPS=20 python3 prep/limbgt_tokens.py    # smoke
  CLIPS=0  NPROC=8 python3 prep/limbgt_tokens.py
"""
import os, glob, json
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT  = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK   = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
CLIPS = int(os.environ.get("CLIPS", "0"))
NPROC = int(os.environ.get("NPROC", "8"))
FS    = 400.0
WINF, HOPF = 256, 128
SMK = max(3, int(FS / 2)) | 1                # 0.5 s smoothing (phase-1)
JB = [7, 4, 12, 9, 0]                        # BODY25 LW RW LHip RHip Nose

def one(job):
    kd, rows = job
    try:
        fs = sorted(glob.glob(f"{kd}/*.json"))
        if len(fs) < 20: return 0
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
        n = 0
        for rid in rows:
            of = f"{TOK}/imu/{rid:06d}.npy"
            if os.path.exists(of): n += 1; continue
            tf = f"{TOK}/tokens/{rid:06d}.npz"
            if not os.path.exists(tf): continue
            nw = int(np.load(tf)["nw"])
            nsamp = (nw - 1) * HOPF + WINF
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
            np.save(of, e.astype(np.float16))
            n += 1
        return n
    except Exception:
        return 0

def main():
    os.makedirs(f"{TOK}/imu", exist_ok=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man["kd"] = man.file.map(lambda f: os.path.join(
        ROOT, os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d"))
    groups = [(kd, [int(r) for r in g.rid.values])
              for kd, g in man.groupby("kd")]
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    if CLIPS: groups = groups[:CLIPS]
    print(f"{len(groups)} clips -> limb GT on token grid", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, groups, chunksize=8)):
            done += r
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(groups)} clips ({done} rids)", flush=True)
    print(f"{done} rids have GT", flush=True)

if __name__ == "__main__":
    main()
