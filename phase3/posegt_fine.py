#!/usr/bin/env python3
"""Pose GT on an arbitrary token grid (rebuild of the lost posegt script).

Recipe (phase-1 convention, mirrors phase2/limbgt_tokens.py): BODY25
keypoints3d (fresh3d/, person 0, conf>0.3, mm->m heuristic), proportional
start-sync (fps = nframes / clip-duration-from-tokens), joints = BODY25
[:15], ROOT-RELATIVE (minus MidHip j8).  Window w value = keypoint track
linearly interpolated at the window-center time (NaN joints stay NaN).

VALIDATE=1: rebuild the COARSE grid (WINF=256 HOPF=128 vs TOKREF) for
NVAL clips and report agreement with the cached pose/ files — the recipe
must reproduce them before any fine-grid generation is trusted.

Build:  TOKDIR=~/.../pa_tokens_fine2 WINF=256 HOPF=32 NPROC=8 \\
        python3 phase3/posegt_fine.py
"""
import os, glob, json
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOKDIR = os.path.expanduser(os.environ.get(
    "TOKDIR", "~/zerdani/buffer/octonet/pa_tokens_fine2"))
TOKREF = os.path.expanduser(os.environ.get(
    "TOKREF", "~/zerdani/buffer/octonet/pa_tokens"))
WINF = int(os.environ.get("WINF", "256"))
HOPF = int(os.environ.get("HOPF", "32"))
NPROC = int(os.environ.get("NPROC", "8"))
VALIDATE = int(os.environ.get("VALIDATE", "0"))
ABSROOT = int(os.environ.get("ABSROOT", "0"))
SCENES = [int(v) for v in os.environ.get("SCENES", "").split(",") if v]
NVAL = int(os.environ.get("NVAL", "12"))
FS, NJ, ROOTJ = 400.0, 15, 8

def load_kp(kd):
    fs = sorted(glob.glob(f"{kd}/*.json"))
    if len(fs) < 20: return None
    P = np.full((len(fs), 25, 3), np.nan, np.float32)
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
    return P[:, :NJ]

def grid_pose(P, nw, winf, hopf):
    """window-center linear interp of the root-relative 15-joint track"""
    nsamp = (nw - 1) * hopf + winf
    fps = len(P) / (nsamp / FS)
    R = P if ABSROOT else P - P[:, ROOTJ:ROOTJ + 1]
    tc = (np.arange(nw) * hopf + winf / 2) / FS          # window centers (s)
    fi = tc * fps - 0.5                                  # frame coordinate
    f0 = np.clip(np.floor(fi).astype(int), 0, len(P) - 1)
    f1 = np.clip(f0 + 1, 0, len(P) - 1)
    a = np.clip(fi - f0, 0, 1)[:, None, None]
    return (1 - a) * R[f0] + a * R[f1]                   # (nw, 15, 3)

def one(job):
    kd, rows = job
    try:
        P = load_kp(kd)
        if P is None: return 0
        n = 0
        for rid in rows:
            of = f"{TOKDIR}/pose/{rid:06d}.npy"
            if os.path.exists(of): n += 1; continue
            tf = f"{TOKDIR}/tokens/{rid:06d}.npz"
            if not os.path.exists(tf): continue
            nw = int(np.load(tf)["nw"])
            G = grid_pose(P, nw, WINF, HOPF)
            if not np.isfinite(G).any(): continue
            np.save(of, G.astype(np.float16))
            n += 1
        return n
    except Exception:
        return 0

def validate():
    man = pd.read_csv(f"{TOKREF}/manifest.csv")
    man["kd"] = man.file.map(lambda f: os.path.join(
        ROOT, os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d"))
    rng = np.random.default_rng(0)
    rows = man.sample(frac=1, random_state=7).itertuples()
    done, stats = 0, []
    for r in rows:
        pf = f"{TOKREF}/pose/{int(r.rid):06d}.npy"
        tf = f"{TOKREF}/tokens/{int(r.rid):06d}.npz"
        if not (os.path.exists(pf) and os.path.exists(tf)): continue
        ref = np.asarray(np.load(pf), np.float32)
        P = load_kp(r.kd)
        if P is None: continue
        G = grid_pose(P, len(ref), 256, 128)
        both = np.isfinite(ref) & np.isfinite(G)
        if both.sum() < 100: continue
        d = np.abs(ref[both] - G[both])
        stats.append((float(np.median(d)), float(d.mean()),
                      float(np.corrcoef(ref[both], G[both])[0, 1]),
                      float((np.isfinite(ref) == np.isfinite(G)).mean())))
        done += 1
        if done >= NVAL: break
    A = np.array(stats)
    print(f"VALIDATE vs cached coarse pose ({done} clips): median|d| "
          f"{np.median(A[:,0])*1000:.1f} mm  mean|d| {np.median(A[:,1])*1000:.1f} mm  "
          f"corr {np.median(A[:,2]):.4f}  nan-agree {np.median(A[:,3])*100:.1f}%",
          flush=True)

def main():
    if VALIDATE:
        validate(); return
    os.makedirs(f"{TOKDIR}/pose", exist_ok=True)
    man = pd.read_csv(f"{TOKDIR}/manifest.csv")
    if SCENES: man = man[man.scene.isin(SCENES)]
    man["kd"] = man.file.map(lambda f: os.path.join(
        ROOT, os.path.dirname(os.path.dirname(f)), "fresh3d/keypoints3d"))
    groups = [(kd, [int(r) for r in g.rid.values])
              for kd, g in man.groupby("kd")]
    tot = 0
    with Pool(NPROC) as pool:
        for i, n in enumerate(pool.imap_unordered(one, groups, chunksize=2)):
            tot += n
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(groups)} clip-dirs, {tot} pose files",
                      flush=True)
    print(f"posegt done: {tot} files -> {TOKDIR}/pose", flush=True)

if __name__ == "__main__":
    main()
