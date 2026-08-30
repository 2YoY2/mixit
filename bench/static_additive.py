#!/usr/bin/env python3
"""Additive statics: sanitize (STO ramp fit + CFO de-rotation, additivity
preserved) -> time-mean -> (3,57) complex per recording. Real delay axis,
true 3-element angle axis. -> pa_tokens/statics_add/{rid}.npy (342,) f32

  NPROC=10 python3 bench/static_additive.py
"""
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
NPROC = int(os.environ.get("NPROC", "10"))

def sanitize(c):
    A, S, T = c.shape
    k = np.arange(S) - (S - 1) / 2.0
    ph = np.unwrap(np.angle(c.mean(0)), axis=0)
    slope = (k[:, None] * ph).sum(0) / (k ** 2).sum()
    c = c * np.exp(-1j * slope[None, None, :] * k[None, :, None])
    ref = c.mean(-1, keepdims=True)
    proj = (c * ref.conj()).sum((0, 1))
    return c * np.exp(-1j * np.angle(proj))[None, None, :]

def one(job):
    rid, f = job
    of = f"{TOK}/statics_add/{rid:06d}.npy"
    if os.path.exists(of): return 1
    try:
        with h5py.File(os.path.join(ROOT, f), "r") as h:
            c = h["csi/csi"][...]
        x = (c["real"] + 1j * c["imag"]).astype(np.complex64)
        g = np.sqrt((np.abs(x) ** 2).mean((0, 1), keepdims=True)) + 1e-12
        x = sanitize(x / g)
        st = x.mean(-1)                                   # (3,57) complex
        st = st / (np.sqrt((np.abs(st) ** 2).mean()) + 1e-12)
        np.save(of, np.r_[st.real.ravel(), st.imag.ravel()].astype(np.float32))
        return 1
    except Exception:
        return 0

def main():
    os.makedirs(f"{TOK}/statics_add", exist_ok=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    jobs = [(int(r.rid), r.file) for r in man.itertuples()]
    print(f"{len(jobs)} additive statics", flush=True)
    done = 0
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=16)):
            done += r
            if (i + 1) % 5000 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    print(f"{done} done", flush=True)

if __name__ == "__main__":
    main()
