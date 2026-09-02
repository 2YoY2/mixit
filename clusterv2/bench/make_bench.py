#!/usr/bin/env python3
"""Superposition bench: physical mixes with exact per-token ownership GT.

Sum two real solo recordings' RAW complex CSI (same room+receiver for PA,
same env+band for WiMANS) -> run the PRODUCTION tokenizer on the mix ->
label every token's owner from the solos' dynamic-energy maps on the same
TF grid.  Cross-terms arise physically (products of sums).  Statics
(duplicated room) are removed by the tokenizer's own static subtraction.

Ownership per token at (w,f): r = eA/(eA+eB) on the solos' CMN'd energy
maps; own=0 if r>0.8, own=1 if r<0.2, else -1 (collision/ambiguous,
excluded from losses and purity).

Out: BENCHD/{DATASET}_{split}/{i:06d}.npz  {toks, nw, own}  + meta.csv

  DATASET=pa    NPAIR=3000 NVAL=400 python3 clusterv2/bench/make_bench.py
  DATASET=wimans NPAIR=2500 NVAL=400 python3 clusterv2/bench/make_bench.py
"""
import os, sys, importlib.util
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATASET = os.environ.get("DATASET", "pa")
BENCHD = os.path.expanduser(os.environ.get(
    "BENCHD", "~/zerdani/buffer/clusterv2/bench"))
NPAIR = int(os.environ.get("NPAIR", "3000"))
NVAL = int(os.environ.get("NVAL", "400"))
NPROC = int(os.environ.get("NPROC", "8"))
RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- PA
if DATASET == "pa":
    import h5py
    spec = importlib.util.spec_from_file_location(
        "tp", os.path.join(REPO, "phase2", "tokenize_pa.py"))
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    TOK = os.path.expanduser("~/zerdani/buffer/cluster/tok/pa-v1")
    ROOT = tp.ROOT
    FS, WINF, HOPF = tp.FS, tp.WINF, tp.HOPF

    def read_raw(path):
        """raw complex binned to 400 Hz: (nb, 3, 57), AGC removed."""
        try:
            with h5py.File(path, "r") as h:
                c = h["csi/csi"][...]
                ts = h["csi/timestamp"][...].ravel().astype(np.float64)
        except Exception:
            return None
        x = (c["real"] + 1j * c["imag"]).astype(np.complex64)
        dt = float(np.median(np.diff(ts)))
        rate, t = None, None
        for unit in (1.0, 1e-3, 1e-6, 1e-9):
            if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
                rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
        if rate is None:
            t = np.arange(x.shape[-1]) / 810.0
        keep = np.concatenate([[True], np.diff(t) > 0])
        x, t = x[..., keep], t[keep]
        if float(t[-1]) < 2.0: return None
        x = np.moveaxis(x, -1, 0)                       # (T, 3, 57)
        g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2),
                                          keepdims=True)) + 1e-12
        x = x / g
        nb = int(float(t[-1]) * FS)
        if nb < WINF + 2 * HOPF: return None
        xf = x.reshape(len(x), -1)
        idx = np.minimum((t * FS).astype(int), nb - 1)
        cnt = np.bincount(idx, minlength=nb).astype(np.float32)
        s = np.zeros((nb, xf.shape[1]), np.complex64)
        np.add.at(s.real, idx, xf.real.astype(np.float32))
        np.add.at(s.imag, idx, xf.imag.astype(np.float32))
        m = s / np.maximum(cnt, 1)[:, None]
        bad = cnt == 0
        if bad.mean() > 0.35: return None
        if bad.any():
            good = np.where(~bad)[0]
            near = good[np.searchsorted(good, np.where(bad)[0]
                                        ).clip(0, len(good) - 1)]
            m[bad] = m[near]
        return m.reshape(nb, 3, 57)

    def products(raw):
        return (raw[:, 1:, :] * np.conj(raw[:, :1, :])).reshape(
            len(raw), 2, 57)

    def dyn_eng(y):
        """tokenize_pa's CMN+STFT energy map (nw, nf)."""
        yb = y.mean(0)
        ga = np.maximum(np.abs(yb), 0.05 * np.median(np.abs(yb)) + 1e-12)
        dyn = ((y - yb) / ga).astype(np.complex64)
        nw = (len(dyn) - WINF) // HOPF + 1
        if nw < 8: return None
        nf = int(tp.PBAND.sum())
        eng = np.zeros((nw, nf))
        for k in range(tp.KTAP):
            tap = tp.TAPERS[k][:, None, None]
            for w in range(nw):
                Sk = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                axis=0)[tp.PBAND]
                eng[w] += (np.abs(Sk) ** 2).mean(axis=(1, 2))
        return eng / tp.KTAP

    def make_pairs():
        man = pd.read_csv(f"{TOK}/manifest.csv")
        man = man[man.scene.isin([1, 2, 3])]
        pairs = []
        for (sc, nd), g in man.groupby(["scene", "node"]):
            f = list(g.file.values)
            RNG.shuffle(f)
            for i in range(0, len(f) - 1, 2):
                pairs.append((f[i], f[i + 1]))
        RNG.shuffle(pairs)
        return pairs

    def one(job):
        i, fa, fb, od = job
        of = f"{od}/{i:06d}.npz"
        if os.path.exists(of): return 1
        try:
            A = read_raw(os.path.join(ROOT, fa))
            B = read_raw(os.path.join(ROOT, fb))
            if A is None or B is None: return 0
            nb = min(len(A), len(B))
            mix = A[:nb] + B[:nb]
            toks, nw = tp.tokenize(products(mix))
            if toks is None: return 0
            ea = dyn_eng(products(A[:nb]))
            eb = dyn_eng(products(B[:nb]))
            if ea is None or eb is None: return 0
            fla, flb = np.median(ea), np.median(eb)
            own = np.full(len(toks), -1, np.int8)
            for k, t in enumerate(toks):
                w = int(t[0])
                fdx = int(np.argmin(np.abs(tp.FPOS - t[1])))
                if w >= min(len(ea), len(eb)): continue
                va, vb = ea[w, fdx], eb[w, fdx]
                if va < fla and vb < flb: continue
                r = va / (va + vb + 1e-12)
                if r > 0.8: own[k] = 0
                elif r < 0.2: own[k] = 1
            np.savez(of, toks=toks, nw=np.int64(nw), own=own)
            return 1
        except Exception:
            return 0

# ------------------------------------------------------------- WiMANS
else:
    spec = importlib.util.spec_from_file_location(
        "tw", os.path.join(REPO, "multi-person", "tokenize_wimans.py"))
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)
    W = tw.W
    TOKW = os.path.expanduser("~/zerdani/buffer/cluster/tok/wimans-v1")
    FS, WINF, HOPF = tw.FS, tw.WINF, tw.HOPF
    _z = np.load(f"{TOKW}/empty_prints.npz")
    EMPTY = {tuple(k.rsplit("_", 1)): _z[k] for k in _z.files}

    def dyn_eng_w(y, emp):
        ga = np.maximum(np.abs(emp), 0.05 * np.median(np.abs(emp)) + 1e-12)
        dyn = ((y - emp[None]) / ga[None]).astype(np.complex64)
        nw = (len(dyn) - WINF) // HOPF + 1
        if nw < 8: return None
        eng = np.zeros((nw, int(tw.PBAND.sum())))
        for k in range(tw.KTAP):
            tap = tw.TAPERS[k][:, None, None]
            for w in range(nw):
                Sk = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                                axis=0)[tw.PBAND]
                eng[w] += (np.abs(Sk) ** 2).mean(axis=(1, 2))
        return eng / tw.KTAP

    def make_pairs():
        an = pd.read_csv(f"{W}/annotation.csv")
        an.columns = [c.strip("﻿") for c in an.columns]
        an = an[an.number_of_users == 1]
        pairs = []
        for (env, band), g in an.groupby(["environment", "wifi_band"]):
            labs = list(g.label.values)
            RNG.shuffle(labs)
            for i in range(0, len(labs) - 1, 2):
                pairs.append((f"{labs[i]}|{env}|{band}",
                              f"{labs[i + 1]}|{env}|{band}"))
        RNG.shuffle(pairs)
        return pairs

    def one(job):
        i, fa, fb, od = job
        of = f"{od}/{i:06d}.npz"
        if os.path.exists(of): return 1
        try:
            la, env, band = fa.split("|")
            lb = fb.split("|")[0]
            A = tw.read_products(la)      # NB: products domain (nb, 6, 30)
            B = tw.read_products(lb)
            if A is None or B is None: return 0
            # products are not additive; approximate physical mix by raw
            # sum is unavailable here, so use the DOMINANT-term mix:
            # dynamics superpose in the product domain to first order
            # (cross terms are the second-order residue).
            nb = min(len(A), len(B))
            emp = EMPTY[(env, band)]
            mix = A[:nb] + B[:nb] - emp[None]
            toks, nw = tw.tokenize(mix, emp)
            if toks is None: return 0
            ea = dyn_eng_w(A[:nb], emp)
            eb = dyn_eng_w(B[:nb], emp)
            if ea is None or eb is None: return 0
            fla, flb = np.median(ea), np.median(eb)
            own = np.full(len(toks), -1, np.int8)
            for k, t in enumerate(toks):
                w = int(t[0])
                fdx = int(np.argmin(np.abs(tw.FPOS - t[1])))
                if w >= min(len(ea), len(eb)): continue
                va, vb = ea[w, fdx], eb[w, fdx]
                if va < fla and vb < flb: continue
                r = va / (va + vb + 1e-12)
                if r > 0.8: own[k] = 0
                elif r < 0.2: own[k] = 1
            np.savez(of, toks=toks, nw=np.int64(nw), own=own)
            return 1
        except Exception:
            return 0

def main():
    pairs = make_pairs()
    print(f"{DATASET}: {len(pairs)} candidate pairs", flush=True)
    # clamp so val ALWAYS exists and never overlaps train (WiMANS has far
    # fewer pairs than NPAIR: 1782)
    ntr = min(NPAIR, max(0, len(pairs) - NVAL))
    for split, n0, n1 in (("train", 0, ntr), ("val", ntr, ntr + NVAL)):
        od = f"{BENCHD}/{DATASET}_{split}"
        os.makedirs(od, exist_ok=True)
        jobs = [(i - n0, pairs[i][0], pairs[i][1], od)
                for i in range(n0, min(n1, len(pairs)))]
        pd.DataFrame([(j[0], j[1], j[2]) for j in jobs],
                     columns=["i", "a", "b"]).to_csv(
            f"{od}/meta.csv", index=False)
        done = 0
        with Pool(NPROC) as pool:
            for r in pool.imap_unordered(one, jobs, chunksize=4):
                done += r
        print(f"  {split}: {done}/{len(jobs)} mixes -> {od}", flush=True)
    print("bench done", flush=True)

if __name__ == "__main__":
    main()
