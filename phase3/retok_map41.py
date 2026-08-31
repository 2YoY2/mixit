#!/usr/bin/env python3
"""Probe 41b (functional): remap ROOM 4 -> ROOM 1 via the static-derived
per-channel map w, retokenize scene 4 with the exact phase-2 tokenizer, and
build a parallel token dir for end-to-end tests of the trained models:

    TOK2/tokens/{rid}.npz   scene-4 rids: retokenized from w-mapped streams
                            all other rids: symlinks to the originals
    everything else (manifest, pose/, imu/, statics*, tenv/) symlinked;
    scenes-1-3 build caches (*_1-2-3.pkl) symlinked (identical content),
    scene-4 caches left absent so they rebuild from the mapped tokens.

Then (chained by the launcher): act_from_tokens TOK=TOK2 (movement
classification on mapped room 4 vs the native 22.8% cap), eval_v2big
TOK=TOK2 (pose), rerank_mh TOK=TOK2 (MH selected/oracle PCK).

  NST=60 NPROC=8 python3 phase3/retok_map41.py
"""
import os, glob, importlib.util
from multiprocessing import Pool
import numpy as np
import pandas as pd

_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tkz", os.path.join(_dir, "..", "phase2", "tokenize_pa.py"))
tkz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tkz)

ROOT = tkz.ROOT
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
TOK2 = os.path.expanduser(os.environ.get(
    "TOK2", "~/zerdani/buffer/octonet/pa_tokens_map41"))
NST = int(os.environ.get("NST", "60"))
NPROC = int(os.environ.get("NPROC", "8"))
NODES = ["r1", "r2", "r3"]

W = {}

def static_one(file):
    try:
        y = tkz.read_products(os.path.join(ROOT, file))
        return None if y is None else y.mean(0)
    except Exception:
        return None

def remap_one(job):
    rid, file, node = job
    of = f"{TOK2}/tokens/{rid:06d}.npz"
    if os.path.exists(of): return 1
    try:
        y = tkz.read_products(os.path.join(ROOT, file))
        if y is None: return 0
        y = y * W[node][None]
        toks, nw = tkz.tokenize(y)
        if toks is None: return 0
        np.savez(of, toks=toks, nw=np.int64(nw))
        return 1
    except Exception:
        return 0

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    os.makedirs(f"{TOK2}/tokens", exist_ok=True)
    rng = np.random.default_rng(41)

    print("=== ensemble product statics (scenes 1, 4)", flush=True)
    S = {}
    with Pool(NPROC) as pool:
        for sc in (1, 4):
            for nd in NODES:
                files = list(man[(man.scene == sc) & (man.node == nd)].file)
                rng.shuffle(files)
                res = [r for r in pool.map(static_one, files[:NST])
                       if r is not None]
                S[(sc, nd)] = np.mean(res, 0)
                print(f"  ({sc},{nd}): {len(res)}", flush=True)
    for nd in NODES:
        num = S[(1, nd)] * np.conj(S[(4, nd)])
        den = np.abs(S[(4, nd)]) ** 2
        W[nd] = (num / (den + 0.05 * np.median(den))).astype(np.complex64)
        print(f"  w4->1 {nd}: |w| med {np.median(np.abs(W[nd])):.2f}",
              flush=True)

    print("=== symlink overlay", flush=True)
    for name in ("manifest.csv", "pose", "imu", "statics", "statics_add",
                 "tenv", "static_peaks.npz"):
        src, dst = f"{TOK}/{name}", f"{TOK2}/{name}"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for cf in glob.glob(f"{TOK}/*_1-2-3.pkl"):
        dst = f"{TOK2}/{os.path.basename(cf)}"
        if not os.path.lexists(dst): os.symlink(cf, dst)
    nlink = 0
    for r in man[man.scene != 4].itertuples():
        src = f"{TOK}/tokens/{int(r.rid):06d}.npz"
        dst = f"{TOK2}/tokens/{int(r.rid):06d}.npz"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst); nlink += 1
    print(f"  {nlink} token symlinks (non-scene-4)", flush=True)

    jobs = [(int(r.rid), r.file, r.node)
            for r in man[man.scene == 4].itertuples()]
    print(f"=== retokenize {len(jobs)} scene-4 recordings (mapped)",
          flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(remap_one, jobs,
                                                  chunksize=4)):
            done += r
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(jobs)} ({done} ok)", flush=True)
    print(f"retok done: {done}/{len(jobs)} scene-4 recs -> {TOK2}",
          flush=True)

if __name__ == "__main__":
    main()
