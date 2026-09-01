#!/usr/bin/env python3
"""Probe 50: TOKENIZER ORACLE — with token-level GT, are tokens per-person?

Construction (GT by superposition): take n solo (1-user) recordings of the
same (env, band), bin their RAW CSI to the 400 Hz grid, sum them and remove
the duplicated room (n-1 copies of the raw empty print), then run the
PRODUCTION tokenizer on the mix.  Cross-terms between people arise exactly
as they would physically (products of sums).  GT = each solo's own dynamic
energy map on the same TF grid.

Per mix token at bin (w,f):  owner share r_k = e_k / sum(e).  Metrics:
  pure      max share > 0.8  (atom belongs to one person)
  collision >=2 solos above their own floor in that bin, no 0.8 owner
  emergent  no solo above floor there (cross-term / noise artifact)
  phi-fid   |mix phi - owner solo-token phi| (circular) on pure bins where
            the owner emitted a token — does the spatial readout survive?
Purity curve over n = 2, 3, 4.  Caveat: superposition ignores body-body
shadowing (the exchange-game epsilon) — this is the OPTIMISTIC ceiling.

  NPAIR=30 python3 multi-person/50_token_oracle.py
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tw", os.path.join(HERE, "tokenize_wimans.py"))
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)          # defs + constants only; main() guarded

W = tw.W
NPAIR = int(os.environ.get("NPAIR", "30"))
NEMPTY = int(os.environ.get("NEMPTY", "30"))
FS, WINF, HOPF = tw.FS, tw.WINF, tw.HOPF

def read_raw_binned(label):
    """like tw.read_products but stops BEFORE products: (nb, 3, 3, 30)."""
    import scipy.io as sio
    f = f"{W}/wifi_csi/mat/{label}.mat"
    if not os.path.exists(f): return None
    try:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        tr = m["trace"]
    except Exception:
        return None
    cs, ts = [], []
    for p in np.atleast_1d(tr):
        c = np.asarray(getattr(p, "csi", None))
        if c is None or c.shape != (3, 3, 30): continue
        cs.append(c.astype(np.complex64))
        ts.append(float(getattr(p, "timestamp_low", 0)))
    if len(cs) < 300: return None
    x = np.stack(cs)
    t = np.array(ts)
    t = np.unwrap(t, period=2 ** 32) * 1e-6
    t = t - t[0]
    if not (0.5 < t[-1] < 30):
        t = np.arange(len(x)) / (len(x) / 3.0)
    g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2, 3), keepdims=True)) + 1e-12
    x = x / g
    nb = int(t[-1] * FS)
    if nb < WINF + 2 * HOPF: return None
    xf = x.reshape(len(x), -1)
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, xf.shape[1]), np.complex64)
    np.add.at(s.real, idx, xf.real)
    np.add.at(s.imag, idx, xf.imag)
    mg = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(
            0, len(good) - 1)]
        mg[bad] = mg[near]
    return mg.reshape(nb, 3, 3, 30)

def products(raw):
    y = raw[:, 1:] * np.conj(raw[:, :1])
    return y.reshape(len(y), 6, 30)

def dyn_eng(y, emp):
    """CMN + Slepian STFT -> (nw, nf) mean dynamic energy (tokenize's eng)."""
    ga = np.maximum(np.abs(emp), 0.05 * np.median(np.abs(emp)) + 1e-12)
    dyn = ((y - emp[None]) / ga[None]).astype(np.complex64)
    T = len(dyn)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None
    nf = int(tw.PBAND.sum())
    eng = np.zeros((nw, nf))
    for k in range(tw.KTAP):
        tap = tw.TAPERS[k][:, None, None]
        for w in range(nw):
            Sk = np.fft.fft(dyn[w * HOPF:w * HOPF + WINF] * tap,
                            axis=0)[tw.PBAND]
            eng[w] += (np.abs(Sk) ** 2).mean(axis=(1, 2))
    return eng / tw.KTAP

def circd(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

def main():
    an = pd.read_csv(f"{W}/annotation.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    rng = np.random.default_rng(50)
    print("building raw + product empty prints per (env, band)", flush=True)
    RAWE, PRODE = {}, {}
    for env in sorted(an.environment.unique()):
        for band in sorted(an.wifi_band.unique()):
            g = an[(an.environment == env) & (an.wifi_band == band)
                   & (an.number_of_users == 0)]
            labs = list(g.label.values)
            rng.shuffle(labs)
            raws, prods = [], []
            for lb in labs[:NEMPTY]:
                r = read_raw_binned(lb)
                if r is None: continue
                raws.append(r.mean(0))
                prods.append(products(r).mean(0))
            RAWE[(env, band)] = np.mean(raws, 0)
            PRODE[(env, band)] = np.mean(prods, 0)
            print(f"  {env}/{band}: {len(raws)} empties", flush=True)

    res = {n: [] for n in (2, 3, 4)}
    for env in sorted(an.environment.unique()):
        for band in sorted(an.wifi_band.unique()):
            g = an[(an.environment == env) & (an.wifi_band == band)
                   & (an.number_of_users == 1)]
            labs = list(g.label.values)
            emp_r, emp_p = RAWE[(env, band)], PRODE[(env, band)]
            for n in (2, 3, 4):
                done = 0
                tries = 0
                while done < NPAIR and tries < NPAIR * 4:
                    tries += 1
                    pick = rng.choice(labs, n, replace=False)
                    raws = [read_raw_binned(lb) for lb in pick]
                    if any(r is None for r in raws): continue
                    nb = min(len(r) for r in raws)
                    raws = [r[:nb] for r in raws]
                    mix_raw = np.sum(raws, 0) - (n - 1) * emp_r[None]
                    ym = products(mix_raw)
                    toks, _ = tw.tokenize(ym, emp_p)
                    if toks is None: continue
                    engs = [dyn_eng(products(r), emp_p) for r in raws]
                    if any(e is None for e in engs): continue
                    floors = [np.median(e) for e in engs]
                    solo_tok = []
                    for r in raws:
                        tt, _ = tw.tokenize(products(r), emp_p)
                        d = {} if tt is None else \
                            {(int(t[0]), float(t[1])): t[2] for t in tt}
                        solo_tok.append(d)
                    pure = coll = emer = 0
                    fid, fid_c = [], []
                    for t in toks:
                        wdx, fval = int(t[0]), float(t[1])
                        fdx = int(np.argmin(np.abs(tw.FPOS - fval)))
                        if wdx >= len(engs[0]): continue
                        e = np.array([eng[wdx, fdx] for eng in engs])
                        alive = e > np.array(floors)
                        share = e / (e.sum() + 1e-12)
                        k = int(np.argmax(share))
                        if not alive.any():
                            emer += 1
                        elif share[k] > 0.8:
                            pure += 1
                            ph = solo_tok[k].get((wdx, fval))
                            if ph is not None:
                                fid.append(circd(t[2], ph))
                        else:
                            coll += 1
                            ph = solo_tok[k].get((wdx, fval))
                            if ph is not None:
                                fid_c.append(circd(t[2], ph))
                    tot = pure + coll + emer
                    if tot < 20: continue
                    res[n].append((pure / tot, coll / tot, emer / tot,
                                   np.median(fid) if fid else np.nan,
                                   np.median(fid_c) if fid_c else np.nan))
                    done += 1
                print(f"  {env}/{band} n={n}: {done} mixes", flush=True)

    print("\n=== ORACLE PURITY (production tokenizer, superposed solos)",
          flush=True)
    print("  n   pure   collision  emergent   phi-err(pure)  phi-err(coll)",
          flush=True)
    for n in (2, 3, 4):
        if not res[n]: continue
        a = np.array(res[n], float)
        print(f"  {n}  {np.nanmean(a[:,0]):.3f}   {np.nanmean(a[:,1]):.3f}"
              f"      {np.nanmean(a[:,2]):.3f}      "
              f"{np.nanmedian(a[:,3]):.3f} rad      "
              f"{np.nanmedian(a[:,4]):.3f} rad   (N={len(a)})", flush=True)
    print("\npure = one person owns >80% of the bin -> a clusterer CAN "
          "assign it.\ncollision/emergent = information the tokenizer has "
          "already destroyed.", flush=True)
    print("probe 50 done", flush=True)

if __name__ == "__main__":
    main()
