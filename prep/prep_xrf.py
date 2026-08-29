#!/usr/bin/env python3
"""prep_xrf: XRF V2 release -> model-ready per-receiver streams with the full
cleaning stack (every cleaner was individually validated in diagnostics 16/17):

  1. AGC       per-packet per-receiver gain normalisation (motion-correlated
               common mode -- measured, it faked R2 and swamped footprints)
  2. Hampel    impulse outliers (31-sample window, 3 MAD)
  3. SFO/PDD   per-packet per-antenna common rotation removed from the
               adjacent-subcarrier islands (CFO already cancels in the product)
  4. SNR       weak features damped by st/(st+p25) (quantisation-dominated
               subcarriers otherwise amplified by any later normalisation)
  5. IMU lag   ONE clock offset per sequence (release is only coarsely
               aligned: median +0.4 s, IQR -0.2..+0.8 s), estimated from
               TOTAL motion cross-correlation, applied to all receivers.

Output per (sequence, rx): streams/{rid:06d}.npy float16 (T, 264)
  [90 amp | 87 island re | 87 island im], amp and island blocks separately
  unit-RMS then jointly scaled; imu/{rid:06d}.npy float16 (T, 5) lag-aligned
  per-limb envelopes [LW RW LP RP GL]; meta.csv with rid,name,scene,subject,
  take,rx,nsamp,lag,peak,imu_ok,split (all train).

  LIMIT=6 python3 prep/prep_xrf.py     # smoke
"""
import os
from multiprocessing import Pool
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, median_filter

SRC  = os.path.expanduser(os.environ.get("XRF", "~/zerdani/buffer/xrfv2"))
OUT  = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_xrf"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NPROC = int(os.environ.get("NPROC", "6"))
SM = 25
SCENE = {"Diningroom": 0, "Studyroom": 1, "bedroom": 2}
_H = {}

def files():
    if "w" not in _H:
        _H["w"] = h5py.File(f"{SRC}/wifi_50hz_853_video_aligned.h5", "r")
        _H["i"] = h5py.File(f"{SRC}/imu_50hz_853_video_aligned.h5", "r")
        _H["p"] = h5py.File(f"{SRC}/pose3d_coco17_depth_853_video_aligned.h5", "r")
    return _H["w"], _H["i"], _H["p"]

def envelopes(imu):
    x = imu.astype(np.float32)
    hp = x - uniform_filter1d(x, SM, axis=0)
    return uniform_filter1d(np.sqrt((hp ** 2).sum(-1)), SM, axis=0)

JP = [9, 10, 11, 12, 0]        # COCO17 LWrist RWrist LHip RHip Nose = LW RW LP RP GL

def pose_env(po, Timu):
    """pose3d -> (T,5) limb-speed envelopes on the 50 Hz grid, conf-masked."""
    kp = po["pose3d_camera"][...]                      # (F,17,4) xyz+conf
    ts = po["timestamps_sec"][...].astype(np.float64)
    xyz = kp[:, :, :3].astype(np.float64)
    xyz[kp[:, :, 3] < 0.3] = np.nan
    k = np.concatenate([[True], np.diff(ts) > 0])
    xyz, ts = xyz[k], ts[k]
    if len(ts) < 30: return None
    v = np.linalg.norm(np.diff(xyz[:, JP], axis=0), axis=-1) / \
        np.diff(ts)[:, None]
    v = np.where(np.isfinite(v), v, 0)
    tv = (ts[1:] + ts[:-1]) / 2 - ts[0]
    ti = np.arange(Timu) / 50.0
    e = np.stack([np.interp(ti, tv, v[:, i]) for i in range(5)], 1)
    return uniform_filter1d(e.astype(np.float32), 15, axis=0)

def lag_align(ecz, e, T):
    """one clock offset for anchor e (T,5) vs CSI dyn-energy z-score ecz."""
    ei = e.sum(1)
    eiz = (ei - ei.mean()) / (ei.std() + 1e-9)
    L = 75
    cc = np.array([float((ecz[max(0, -l):T - max(0, l)]
                          * eiz[max(0, l):T - max(0, -l)]).mean())
                   for l in range(-L, L + 1)])
    return int(np.argmax(cc)) - L, float(cc.max())

def one(job):
    idx, name = job
    try:
        w, im, pf = files()
        g = w["samples"][name]
        amp = g["amp"][...].astype(np.float32)
        pha = g["pha"][...].astype(np.float32)
        imu = im["samples"][name]["imu"][...]
        T = min(len(amp), len(imu))
        if T < 600: return None
        amp, pha, imu = amp[:T], pha[:T], imu[:T]
        rms = np.sqrt((amp ** 2).mean(axis=(2, 3), keepdims=True)) + 1e-9   # AGC
        amp = amp / rms
        a2 = amp.reshape(T, -1)                                            # Hampel
        med = median_filter(a2, size=(31, 1))
        dev = np.abs(a2 - med)
        mad = median_filter(dev, size=(31, 1)) * 1.4826 + 1e-9
        amp = np.where(dev > 3 * mad, med, a2).reshape(amp.shape)
        # per-anchor clock offsets vs CSI (all three clocks drift mutually)
        hp = a2 - uniform_filter1d(a2, SM, axis=0)
        ec = uniform_filter1d((hp ** 2).sum(1), 10)
        ecz = (ec - ec.mean()) / (ec.std() + 1e-9)
        e = envelopes(imu)                                                 # (T,5)
        lag, peak = lag_align(ecz, e, T)
        ep, lag_p, peak_p = None, 0, 0.0
        if os.environ.get("POSE", "0") == "1":     # user: no pose GT in pretrain
            try:                                    # (validated; enable later)
                ep = pose_env(pf["samples"][name], T)
                if ep is not None:
                    lag_p, peak_p = lag_align(ecz, ep, T)
            except Exception:
                ep = None
        def shift(v, l):
            return v[l:] if l > 0 else (v[:len(v) + l] if l < 0 else v)
        e = shift(e, lag)
        if ep is not None: ep = shift(ep, lag_p)
        T2 = min(len(e), len(ep) if ep is not None else len(e), T)
        amp, pha, e = amp[:T2], pha[:T2], e[:T2]
        # fuse anchors per limb: z-scored mean of whichever passed its lag scan
        ez = e / (e.std(0) + 1e-9)
        use_imu, use_pose = peak > 0.2, ep is not None and peak_p > 0.2
        if use_pose:
            epz = ep[:T2] / (ep[:T2].std(0) + 1e-9)
            fused = (ez + epz) / 2 if use_imu else epz
        else:
            fused = ez
        e = fused
        imu_ok_flag = int(use_imu or use_pose)
        subj, scene_s, take = name.split("_")
        rows = []
        for rx in range(3):
            a = amp[:, rx]                                    # (T2,3,30)
            st = a.reshape(T2, -1).mean(0)
            wgt = st / (st + np.percentile(st, 25))           # SNR damping
            aw = a.reshape(T2, -1) * wgt[None, :]
            c = a * np.exp(1j * pha[:, rx])
            z = c[..., :-1] * np.conj(c[..., 1:])             # (T2,3,29)
            u = z / (np.abs(z) + 1e-12)                       # SFO/PDD detrend
            z = z * np.exp(-1j * np.angle(u.mean(-1, keepdims=True)))
            zw = np.sqrt(wgt.reshape(3, 30)[:, :-1] * wgt.reshape(3, 30)[:, 1:]
                         ).reshape(-1)
            zf = z.reshape(T2, -1) * zw[None, :]
            aw = aw / (np.sqrt((aw ** 2).mean()) + 1e-12)
            zs = np.sqrt((np.abs(zf) ** 2).mean()) + 1e-12
            y = np.c_[aw, zf.real / zs, zf.imag / zs].astype(np.float16)
            rid = idx * 3 + rx
            np.save(f"{OUT}/streams/{rid:06d}.npy", y)
            np.save(f"{OUT}/imu/{rid:06d}.npy",
                    (e / (e.std(0) + 1e-9)).astype(np.float16))
            rows.append((rid, name, SCENE.get(scene_s, -1), int(subj), int(take),
                         rx, T2, lag, round(peak, 3), imu_ok_flag, "train"))
        return rows
    except Exception:
        if os.environ.get("DEBUG"):
            import traceback; traceback.print_exc()
        return None

def main():
    os.makedirs(f"{OUT}/streams", exist_ok=True)
    os.makedirs(f"{OUT}/imu", exist_ok=True)
    w = files()[0]
    names = [x.decode() if isinstance(x, bytes) else x
             for x in w["sample_names"][...]]
    jobs = list(enumerate(sorted(names)))
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"{len(jobs)} sequences x 3 rx -> {OUT}", flush=True)
    rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if r: rows.extend(r)
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "name", "scene", "subject", "take",
                                     "rx", "nsamp", "lag", "peak", "imu_ok",
                                     "split"]).sort_values("rid")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print(f"\n{len(df)} streams  imu_ok {df.imu_ok.mean()*100:.1f}%  "
          f"lag median {df.lag.median():+.0f} IQR "
          f"[{df.lag.quantile(.25):+.0f},{df.lag.quantile(.75):+.0f}]")
    print(df.groupby("scene").agg(streams=("rid", "size"),
                                  subj=("subject", "nunique"),
                                  nsamp_med=("nsamp", "median")))

if __name__ == "__main__":
    main()
