#!/usr/bin/env python3
"""Pose-anchor variant for the XRF anchor-modality A/B: writes pose3d-derived
limb envelopes as imu_pose/{rid}.npy + meta_pose.csv (imu_ok = pose available),
leaving streams and the IMU anchors untouched. Lag-aligned to CSI via each
stream's own amp-block dynamic energy.

  python3 prep/prep_xrf_poseanchor.py
"""
import os
from multiprocessing import Pool
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

SRC = os.path.expanduser(os.environ.get("XRF", "~/zerdani/buffer/xrfv2"))
OUT = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_xrf"))
NPROC = int(os.environ.get("NPROC", "6"))
JP = [9, 10, 11, 12, 0]
_H = {}

def pf():
    if "p" not in _H:
        _H["p"] = h5py.File(f"{SRC}/pose3d_coco17_depth_853_video_aligned.h5", "r")
    return _H["p"]

def pose_env(po, T):
    kp = po["pose3d_camera"][...]
    ts = po["timestamps_sec"][...].astype(np.float64)
    xyz = kp[:, :, :3].astype(np.float64)
    xyz[kp[:, :, 3] < 0.3] = np.nan
    k = np.concatenate([[True], np.diff(ts) > 0])
    xyz, ts = xyz[k], ts[k]
    if len(ts) < 30: return None
    v = np.linalg.norm(np.diff(xyz[:, JP], axis=0), axis=-1) / np.diff(ts)[:, None]
    v = np.where(np.isfinite(v), v, 0)
    tv = (ts[1:] + ts[:-1]) / 2 - ts[0]
    ti = np.arange(T) / 50.0
    e = np.stack([np.interp(ti, tv, v[:, i]) for i in range(5)], 1)
    return uniform_filter1d(e.astype(np.float32), 15, axis=0)

def one(row):
    rid, name, nsamp = row
    try:
        y = np.asarray(np.load(f"{OUT}/streams/{rid:06d}.npy",
                               mmap_mode="r")[:, :90], np.float32)
        T = len(y)
        hp = y - uniform_filter1d(y, 25, axis=0)
        ec = uniform_filter1d((hp ** 2).sum(1), 10)
        ecz = (ec - ec.mean()) / (ec.std() + 1e-9)
        ep = pose_env(pf()["samples"][name], T)
        if ep is None: return None
        L = 75
        ei = ep.sum(1); eiz = (ei - ei.mean()) / (ei.std() + 1e-9)
        cc = np.array([float((ecz[max(0, -l):T - max(0, l)]
                              * eiz[max(0, l):T - max(0, -l)]).mean())
                       for l in range(-L, L + 1)])
        lag = int(np.argmax(cc)) - L
        peak = float(cc.max())
        if peak <= 0.15: return None
        ep = ep[lag:] if lag > 0 else (ep[:T + lag] if lag < 0 else ep)
        e = np.zeros((T, 5), np.float32)
        e[:len(ep)] = ep[:T]
        e = e / (e.std(0) + 1e-9)
        np.save(f"{OUT}/imu_pose/{rid:06d}.npy", e.astype(np.float16))
        return rid
    except Exception:
        return None

def main():
    os.makedirs(f"{OUT}/imu_pose", exist_ok=True)
    meta = pd.read_csv(f"{OUT}/meta.csv")
    rows = [(int(r.rid), r.name, int(r.nsamp)) for r in meta.itertuples()]
    done = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, rows, chunksize=8)):
            if r is not None: done.append(r)
            if (i + 1) % 500 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
    mp = meta.copy()
    mp["imu_ok"] = mp.rid.isin(done).astype(int)
    mp.to_csv(f"{OUT}/meta_pose.csv", index=False)
    print(f"{len(done)}/{len(rows)} rids got pose anchors "
          f"({len(done)/len(rows)*100:.1f}%) -> imu_pose/ + meta_pose.csv")

if __name__ == "__main__":
    main()
