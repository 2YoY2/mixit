#!/usr/bin/env python3
"""Probe 51: who correlates best with TOTAL body movement —
raw CSI, cleaned CSI, or the (named) slots?

Per recording (scenes 1 and 4), Pearson corr vs the GT total-motion
envelope (sum of the 5 limb keypoint-speed envelopes, window means):
  RAW    amplitude frame-diff envelope of the raw .mat (hardware AGC and
         CFO still in — CSI as measured)
  CLEAN  total token energy per window (sanitized + CMN + STFT dynamics;
         equals the sum over ALL slots by construction)
  SLOTS  energy of the predominance-NAMED limb slots only (limbsel map)
         — the slots acting as a learned denoiser of the clean envelope
NB per-LIMB movement is out of scope here: raw/clean produce one global
envelope and cannot attribute; only slots can (probe 49).

  N=300 python3 diagnostics/51_motion_corr.py
"""
import os, re, time
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/octonet/pa_tokens"))
ROOT = os.path.expanduser(os.environ.get(
    "ROOT", "~/zerdani/buffer/PerceptAlign"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
SELF = os.path.expanduser(os.environ.get(
    "SELF", "~/zerdani/buffer/octonet/limbsel_slots.npz"))
N = int(os.environ.get("N", "300"))
FS, WINF, HOPF = 400.0, 256, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/best.pt", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x):
        return torch.softmax(self.head(self.enc(self.inp(x))), -1)

sep = SetSep(); sep.load_state_dict(ck["model"])
sep = sep.to(dev).eval()
z = np.load(SELF)
SMAP = {int(r): int(m) for r, m in zip(z["rids"], z["mask"])}

def raw_env(path, nw):
    """amplitude frame-diff of the raw mat, on the token window grid."""
    try:
        with h5py.File(path, "r") as h:
            c = h["csi/csi"][...]
            ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    except Exception:
        return None
    x = np.abs(c["real"] + 1j * c["imag"]).astype(np.float32)
    dt = float(np.median(np.diff(ts)))
    t = None
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            t = (ts - ts[0]) * unit; break
    if t is None:
        t = np.arange(x.shape[-1]) / 810.0
    keep = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., keep], t[keep]
    x = np.moveaxis(x, -1, 0)                    # (T, ant, subc)
    d = np.linalg.norm((x[1:] - x[:-1]).reshape(len(x) - 1, -1), axis=1)
    tb = t[1:]
    nb = int(float(t[-1]) * FS)
    if nb < WINF + 2 * HOPF: return None
    idx = np.minimum((tb * FS).astype(int), nb - 1)
    acc = np.zeros(nb); cnt = np.zeros(nb)
    np.add.at(acc, idx, d); np.add.at(cnt, idx, 1)
    e = acc / np.maximum(cnt, 1)
    env = np.array([e[w * HOPF:w * HOPF + WINF].mean()
                    for w in range(min(nw, (nb - WINF) // HOPF + 1))])
    return env

def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 8 or a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(51)
    for scene in (1, 4):
        ms = man[man.scene == scene]
        rids = rng.permutation(np.array(ms.rid.values))[:N * 2]
        r2f = {int(r.rid): r.file for r in ms.itertuples()}
        rows = []
        t0 = time.time()
        for rid in rids:
            if len(rows) >= N: break
            rid = int(rid)
            tf = f"{TOK}/tokens/{rid:06d}.npz"
            gf = f"{TOK}/imu/{rid:06d}.npy"
            if not (os.path.exists(tf) and os.path.exists(gf)): continue
            zt = np.load(tf); t = zt["toks"]; nw = int(zt["nw"])
            if len(t) < 16: continue
            gi = np.asarray(np.load(gf), np.float32)
            gtot = gi.sum(1)
            G = np.array([gtot[w * HOPF:w * HOPF + WINF].mean()
                          for w in range(nw)])
            if len(G) < 8 or G.std() < 1e-9: continue
            # CLEAN + SLOTS from cached tokens
            le = t[:, 4]
            zle = (le - le.mean()) / (le.std() + 1e-6)
            X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                      np.cos(t[:, 3]), t[:, 1] / 150.0,
                      t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
            e = (10.0 ** le).astype(np.float64)
            widx = t[:, 0].astype(int)
            with torch.no_grad():
                a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
            hard = a.argmax(1)
            msk = SMAP.get(rid, 0)
            named = np.array([(msk >> s) & 1 > 0 for s in hard])
            ec = np.zeros(nw); np.add.at(ec, widx, e)
            es = np.zeros(nw); np.add.at(es, widx[named], e[named])
            esum = np.zeros(nw)
            best = -2.0
            for m in range(M):
                em = np.zeros(nw)
                np.add.at(em, widx[hard == m], e[hard == m])
                esum += em
                c_ = corr(em, G)
                if np.isfinite(c_) and c_ > best: best = c_
            renv = raw_env(os.path.join(ROOT, r2f[rid]), nw)
            if renv is None: continue
            rows.append((corr(renv, G), corr(ec, G), corr(es, G),
                         corr(esum, G), best))
        A = np.array(rows, float)
        A = A[np.isfinite(A).all(1)]
        print(f"\n=== SCENE {scene} (N={len(A)}, "
              f"{(time.time()-t0)/60:.1f}min)", flush=True)
        for i, nm in enumerate(("RAW     ", "CLEAN   ", "NAMED   ",
                                "SLOT-SUM", "BEST-1  ")):
            print(f"  {nm}: median r {np.median(A[:, i]):+.3f}  "
                  f"mean {A[:, i].mean():+.3f}", flush=True)
        print(f"  paired: CLEAN>RAW {np.mean(A[:,1]>A[:,0])*100:.0f}%  "
              f"SLOTSUM>CLEAN {np.mean(A[:,3]>A[:,1])*100:.0f}%  "
              f"BEST1>CLEAN {np.mean(A[:,4]>A[:,1])*100:.0f}%  "
              f"NAMED>CLEAN {np.mean(A[:,2]>A[:,1])*100:.0f}%", flush=True)
    print("probe 51 done", flush=True)

if __name__ == "__main__":
    main()
