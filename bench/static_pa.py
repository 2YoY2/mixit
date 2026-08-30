#!/usr/bin/env python3
"""Per-recording statics for the pose model: [log-amp static |H| (3x57=171),
conj-product static re/im (2x57x2=228)] = 399 floats -> pa_tokens/statics/.
Amp static is CFO-immune; product static is CFO-cancelled (phase valid).

  SCENES=1,2,3,4 NPROC=10 python3 bench/static_pa.py
"""
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
SCENES = {int(s) for s in os.environ.get("SCENES", "1,2,3,4").split(",")}
NPROC = int(os.environ.get("NPROC", "10"))

def one(job):
    rid, f = job
    of = f"{TOK}/statics/{rid:06d}.npy"
    if os.path.exists(of): return 1
    try:
        with h5py.File(os.path.join(ROOT, f), "r") as h:
            c = h["csi/csi"][...]
        x = (c["real"] + 1j * c["imag"]).astype(np.complex64)   # (3,57,T)
        x = np.moveaxis(x, -1, 0)
        g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
        x = x / g
        amp = np.log10(np.abs(x).mean(0).ravel() + 1e-9)        # (171,)
        y = (x[:, 1:, :] * np.conj(x[:, :1, :])).mean(0)        # (2,57)
        y = y / (np.sqrt((np.abs(y) ** 2).mean()) + 1e-12)
        st = np.r_[amp, y.real.ravel(), y.imag.ravel()].astype(np.float32)
        np.save(of, st)
        return 1
    except Exception:
        return 0

def main():
    os.makedirs(f"{TOK}/statics", exist_ok=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin(SCENES)]
    jobs = [(int(r.rid), r.file) for r in man.itertuples()]
    print(f"{len(jobs)} statics", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=16)):
            done += r
            if (i + 1) % 5000 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    print(f"{done} done", flush=True)

if __name__ == "__main__":
    main()
