#!/usr/bin/env python3
"""Pose GT on the token window grid, for the pose-transfer probe: BODY25
keypoints3d -> ROOT-RELATIVE (MidHip) 3D positions of 15 joints, interpolated
to STFT window centers (same start-sync convention as everywhere). Root-
relative because world coordinates differ per room -- absolute cross-room
pose is ill-posed; pose SHAPE is the transferable target.

Writes pa_tokens/pose/{rid:06d}.npy float16 (nw, 15, 3), NaN where missing.

  SCENES=1,4 NPROC=8 python3 prep/posegt_tokens.py
"""
import os, glob, json
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
SCENES = {int(s) for s in os.environ.get("SCENES", "1,4").split(",")}
NPROC = int(os.environ.get("NPROC", "8"))
FS, WINF, HOPF = 400.0, 256, 128
NJ = 15                                       # BODY25 joints 0..14, root=8

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
        if np.nanmedian(np.abs(P)) > 50: P *= 1e-3
        rel = P[:, :NJ] - P[:, 8:9]                       # root-relative
        # fill joint gaps by linear interp over frames where >=50% finite
        for jx in range(NJ):
            for c in range(3):
                v = rel[:, jx, c]
                ok = np.isfinite(v)
                if ok.mean() >= 0.5 and ok.sum() >= 2:
                    rel[:, jx, c] = np.interp(np.arange(len(v)),
                                              np.where(ok)[0], v[ok])
                elif ok.sum() < 2:
                    rel[:, jx, c] = np.nan
        n = 0
        for rid in rows:
            of = f"{TOK}/pose/{rid:06d}.npy"
            if os.path.exists(of): n += 1; continue
            tf = f"{TOK}/tokens/{rid:06d}.npz"
            if not os.path.exists(tf): continue
            nw = int(np.load(tf)["nw"])
            nsamp = (nw - 1) * HOPF + WINF
            fps = len(fs) / (nsamp / FS)
            tv = (np.arange(len(fs)) + 0.5) / fps
            tc = (np.arange(nw) * HOPF + WINF / 2) / FS
            out = np.empty((nw, NJ, 3), np.float32)
            for jx in range(NJ):
                for c in range(3):
                    v = rel[:, jx, c]
                    out[:, jx, c] = np.interp(tc, tv, v) \
                        if np.isfinite(v).all() else np.nan
            np.save(of, out.astype(np.float16))
            n += 1
        return n
    except Exception:
        return 0

def main():
    os.makedirs(f"{TOK}/pose", exist_ok=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin(SCENES)]
    man["kd"] = man.file.map(lambda f: os.path.join(
        ROOT, os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d"))
    groups = [(kd, [int(r) for r in g.rid.values])
              for kd, g in man.groupby("kd")]
    print(f"{len(groups)} clips -> pose GT (scenes {sorted(SCENES)})", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, groups, chunksize=8)):
            done += r
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(groups)} ({done} rids)", flush=True)
    print(f"{done} rids have pose GT", flush=True)

if __name__ == "__main__":
    main()
