#!/usr/bin/env python3
"""Probe 52: who EXPLAINS the movement best — raw, clean, or slots?

Same yardstick for all three: 8 basis envelopes per representation,
least-squares fit of the 5 GT limb envelopes, EVEN/ODD-window
cross-validated R^2 (fit even, score odd; in-sample R^2 also shown).
  RAW-8    amplitude frame-diff energy in 8 antenna x subcarrier blocks
           (AGC/CFO in — CSI as measured)
  CLEAN-8  token energy in 8 Doppler bands (sanitize+CMN+STFT, no slots)
  SLOT-8   the 8 limbtok12 slot envelopes
Fair by construction: same count of curves, same fit, same target.

  N=200 python3 diagnostics/52_explain_compare.py
"""
import os, time
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
N = int(os.environ.get("N", "200"))
FS, WINF, HOPF = 400.0, 256, 128
NB8 = 8
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

def raw_groups(path, nw):
    """8 amplitude frame-diff group envelopes on the window grid."""
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
    T = len(x)
    d = np.abs(x[1:] - x[:-1]).reshape(T - 1, -1)   # (T-1, ant*subc)
    nch = d.shape[1]
    bounds = np.linspace(0, nch, NB8 + 1).astype(int)
    dg = np.stack([d[:, bounds[i]:bounds[i + 1]].sum(1)
                   for i in range(NB8)], 1)          # (T-1, 8)
    tb = t[1:]
    nb = int(float(t[-1]) * FS)
    if nb < WINF + 2 * HOPF: return None
    idx = np.minimum((tb * FS).astype(int), nb - 1)
    acc = np.zeros((nb, NB8)); cnt = np.zeros(nb)
    np.add.at(acc, idx, dg); np.add.at(cnt, idx, 1)
    e = acc / np.maximum(cnt, 1)[:, None]
    nww = min(nw, (nb - WINF) // HOPF + 1)
    return np.stack([e[w * HOPF:w * HOPF + WINF].mean(0)
                     for w in range(nww)])            # (nww, 8)

def cv_r2(E, Y):
    """even/odd window CV: fit even, score odd (and in-sample)."""
    n = len(E)
    if n < 12: return np.nan, np.nan
    ev = np.arange(n) % 2 == 0
    A = np.c_[E, np.ones(n)]
    beta, *_ = np.linalg.lstsq(A[ev], Y[ev], rcond=None)
    prd = A[~ev] @ beta
    sst = ((Y[~ev] - Y[ev].mean(0)) ** 2).sum()
    r2cv = 1 - ((Y[~ev] - prd) ** 2).sum() / max(sst, 1e-12)
    beta2, *_ = np.linalg.lstsq(A, Y, rcond=None)
    sst2 = ((Y - Y.mean(0)) ** 2).sum()
    r2in = 1 - ((Y - A @ beta2) ** 2).sum() / max(sst2, 1e-12)
    return float(r2cv), float(r2in)

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(52)
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
            z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
            if len(t) < 16: continue
            gi = np.asarray(np.load(gf), np.float32)
            R = raw_groups(os.path.join(ROOT, r2f[rid]), nw)
            if R is None or len(R) < 12: continue
            nww = len(R)
            Y = np.stack([[gi[:, j][w * HOPF:w * HOPF + WINF].mean()
                           for w in range(nww)] for j in range(5)], 1)
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
            S8 = np.zeros((nww, M))
            for m in range(M):
                em = np.zeros(nw)
                np.add.at(em, widx[hard == m], e[hard == m])
                S8[:, m] = em[:nww]
            fb = np.linspace(2, 150, NB8 + 1)
            C8 = np.zeros((nww, NB8))
            bi = np.clip(np.searchsorted(fb, t[:, 1]) - 1, 0, NB8 - 1)
            for b in range(NB8):
                em = np.zeros(nw)
                mb = bi == b
                np.add.at(em, widx[mb], e[mb])
                C8[:, b] = em[:nww]
            rows.append((cv_r2(R, Y), cv_r2(C8, Y), cv_r2(S8, Y)))
        A = np.array(rows, float)    # (n, 3, 2)
        ok = np.isfinite(A).all((1, 2))
        A = A[ok]
        print(f"\n=== SCENE {scene} (N={len(A)}, "
              f"{(time.time()-t0)/60:.1f}min)  [CV-R2 | in-sample R2]",
              flush=True)
        for i, nm in enumerate(("RAW-8  ", "CLEAN-8", "SLOT-8 ")):
            print(f"  {nm}: CV {np.median(A[:, i, 0]):+.3f}  "
                  f"in-sample {np.median(A[:, i, 1]):+.3f}", flush=True)
        print(f"  paired CV: SLOT>CLEAN {np.mean(A[:,2,0]>A[:,1,0])*100:.0f}%"
              f"  SLOT>RAW {np.mean(A[:,2,0]>A[:,0,0])*100:.0f}%  "
              f"CLEAN>RAW {np.mean(A[:,1,0]>A[:,0,0])*100:.0f}%", flush=True)
    print("probe 52 done", flush=True)

if __name__ == "__main__":
    main()
