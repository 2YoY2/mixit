#!/usr/bin/env python3
"""Per-recording IMU motion envelopes aligned to mixit_data_v2/streams/<rid>.npy.

Mirrors preprocess_v2.py's grid exactly (same dedup, same np.arange(0, t[-1],
1/100)), so env[rid][s:s+WIN] lines up with the window grab() slices at offset s.

Envelope = ||free acceleration|| (imu ch0:3). Gravity already removed by Xsens,
so no integration and no drift.

TIMEZONE TRAP: imu stamps are pandas.Timestamp, wifi are datetime.datetime,
both NAIVE. .timestamp() reads pandas-naive as UTC and stdlib-naive as LOCAL,
faking a 3600/7200 s offset that looks like broken sync. Subtract the naive
datetimes directly instead. (Note preprocess_v2.py is unaffected: it does
t -= t[0] within one file, so the bias cancels.)

  python3 imu_stream_v2.py                  # -> imu_env/<rid>.npy + imu_meta.csv
  python3 imu_stream_v2.py --per-segment    # (T, 17) instead of (T,)
"""
import os, glob, pickle, re, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT = os.path.expanduser("~/zerdani/buffer/octonet/OctoNet-upload")
OUT  = os.path.expanduser(os.environ.get("MIXIT_DATA", "~/zerdani/buffer/octonet/mixit_data_v2"))
FS = 100.0
SMOOTH_SEC = 0.25
MAX_STAMP_GAP = 10
IPAT = re.compile(r"(\d{14})_node_\d_modality_imu")
WPAT = re.compile(r"exp-(\d{14})_node_(\d)_modality_wifi_subject_(\d+)_activity_(.+?)_trial_(\d+)")

def imu_index():
    idx = {}
    for p in sorted(glob.glob(f"{ROOT}/imu/*.pickle")):
        m = IPAT.match(os.path.basename(p))
        if m: idx[int(m.group(1))] = p
    return idx, np.array(sorted(idx))

def imu_envelope(path, per_segment):
    d = pickle.load(open(path, "rb"))
    X = np.asarray(d["data"])                        # (N, 13, 17)
    ts = list(d["timestamps"])
    a = np.linalg.norm(X[:, 0:3, :], axis=1)         # (N, 17) free-accel magnitude
    rel = np.array([(v - ts[0]).total_seconds() for v in ts])
    ok = np.concatenate([[True], np.diff(rel) > 0])
    a, rel = a[ok], rel[ok]
    if not per_segment: a = a.mean(1, keepdims=True)
    return a, rel, ts[0]

def main(args):
    meta = pd.read_csv(f"{OUT}/meta.csv")
    imus, keys = imu_index()
    os.makedirs(f"{OUT}/imu_env", exist_ok=True)
    print(f"{len(meta)} streams, {len(imus)} imu files -> {OUT}/imu_env/")
    cache, rows = {}, []
    for r in meta.itertuples():
        st, why = 0, "ok"
        try:
            full = os.path.join(ROOT, r.file)
            m = WPAT.match(os.path.basename(os.path.dirname(full)))
            if not m: raise ValueError("name")
            s = int(m.group(1)); k = keys[np.argmin(np.abs(keys - s))]
            if abs(k - s) > MAX_STAMP_GAP: raise ValueError("no-imu")
            d = pickle.load(open(full, "rb"))
            ts = d.get("timestamp", d.get("timestamps"))
            t = np.array([pd.Timestamp(v).timestamp() for v in ts]); t -= t[0]
            keep = np.concatenate([[True], np.diff(t) > 0]); t = t[keep]
            grid = np.arange(0, t[-1], 1 / FS)
            if len(grid) != r.nsamp:
                grid = grid[:r.nsamp] if len(grid) > r.nsamp else np.pad(grid, (0, r.nsamp - len(grid)), mode="edge")
            if k not in cache: cache[k] = imu_envelope(imus[k], args.per_segment)
            a, rel, it0 = cache[k]
            w0 = pd.Timestamp(ts[0]).to_pydatetime().replace(tzinfo=None)
            i0 = pd.Timestamp(it0).to_pydatetime().replace(tzinfo=None)
            src = rel + (i0 - w0).total_seconds()               # imu time on the wifi clock
            if src[-1] < 0.9 * t[-1] or src[0] > 0.1 * t[-1]: raise ValueError("bad-clock")
            g = np.stack([np.interp(grid, src, a[:, j], left=np.nan, right=np.nan)
                          for j in range(a.shape[1])])
            kk = max(3, int(SMOOTH_SEC * FS) | 1)
            g = uniform_filter1d(np.nan_to_num(g, nan=0.0), kk, axis=-1).astype(np.float32)
            if g.std() <= 0: raise ValueError("flat")
            out = g.T if args.per_segment else g[0]              # (T,17) or (T,)
            np.save(f"{OUT}/imu_env/{r.rid:06d}.npy", out); st = 1
        except Exception as e:
            why = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}"
        mm = WPAT.match(os.path.basename(os.path.dirname(os.path.join(ROOT, r.file))))
        rows.append((r.rid, st, why, mm.group(2) if mm else "", mm.group(3) if mm else "",
                     mm.group(4) if mm else ""))
        if (r.Index + 1) % 300 == 0: print(f"  {r.Index+1}/{len(meta)}", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "imu_ok", "why", "node", "subject", "act"])
    df.to_csv(f"{OUT}/imu_meta.csv", index=False)
    print(f"\nimu_ok = {df.imu_ok.mean()*100:.1f}%   reasons: {df[df.imu_ok==0].why.value_counts().to_dict()}")
    good = df[df.imu_ok == 1].rid.values
    if len(good):
        v = np.concatenate([np.load(f"{OUT}/imu_env/{g:06d}.npy").ravel() for g in good[:200]])
        print(f"envelope over 200 recordings: mean {v.mean():.3f}  p50 {np.median(v):.3f}  p99 {np.percentile(v,99):.3f} m/s2")
    print(f"nodes {sorted(df.node.unique())}  subjects {len(df.subject.unique())}  activities {len(df.act.unique())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-segment", action="store_true")
    main(ap.parse_args())
