#!/usr/bin/env python3
"""Cross-DATASET stress test (user's design): tokenize XRF 50 Hz island
streams -- degenerate tokens (no aperture: phi=psi=0; band 2-25 Hz) -- feed
the FROZEN PA-trained 12h limb-cluster separator, score slot envelopes vs
XRF's IMU limb envelopes (probe-27 battery). Window length kept physically
identical (0.64 s / 0.32 s -> WINF=32, HOPF=16 @ 50 Hz).
Arms: rawf (f as-is), stretch (f*6: 2-25 -> 12-150, the band the model
knows), kmdopp (doppler k-means control on same tokens).

  python3 diagnostics/32_xrf_transfer.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

XRF = os.path.expanduser(os.environ.get("PREP", "~/zerdani/buffer/octonet/prep_xrf"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
NREC = int(os.environ.get("NREC", "250"))
MAXCORR = 0.7
FS, WINF, HOPF = 50.0, 32, 16
freqs = np.fft.fftfreq(WINF, 1 / FS)
FSEL = (freqs >= 2) & (freqs <= 25)
FPOS = freqs[FSEL]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
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

import time
sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        time.sleep(60)

def tokenize50(x):
    """(T,264) islands @50Hz -> tokens [w, f, E]; pooled island energy."""
    z = x[:, 90:177] + 1j * x[:, 177:264]
    zb = z.mean(0)
    gz = np.abs(zb); thr = 0.05 * np.median(gz) + 1e-9
    zb = np.where(gz < thr, thr + 0j, zb)
    z = z / zb - 1.0
    T = len(z)
    nw = (T - WINF) // HOPF + 1
    if nw < 8: return None
    han = np.hanning(WINF)[:, None]
    ws, fs_, es = [], [], []
    E = np.empty((nw, int(FSEL.sum())))
    for w in range(nw):
        F = np.fft.fft(z[w * HOPF:w * HOPF + WINF] * han, axis=0)
        E[w] = (np.abs(F[FSEL]) ** 2).mean(1)
    floor = np.median(E)
    ws, fs_ = np.where(E >= floor)
    if len(ws) < 4 * nw: return None
    return ws, FPOS[fs_], E[ws, fs_], nw

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def kmeans1d(f, wgt):
    c = np.percentile(f, [25, 75]).astype(np.float64)
    for _ in range(25):
        a = (np.abs(f[:, None] - c[None]) ** 2).argmin(1)
        for k in (0, 1):
            m = a == k
            if m.any(): c[k] = (f[m] * wgt[m]).sum() / wgt[m].sum()
    return a

meta = pd.read_csv(f"{XRF}/meta.csv")
if "imu_ok" in meta.columns: meta = meta[meta.imu_ok == 1]
if "split" in meta.columns and (meta.split == "test").any():
    meta = meta[meta.split == "test"]
rng = np.random.default_rng(0)
rows = []
for rid in rng.permutation(meta.rid.values):
    rid = int(rid)
    sf, gf = f"{XRF}/streams/{rid:06d}.npy", f"{XRF}/imu/{rid:06d}.npy"
    if not (os.path.exists(sf) and os.path.exists(gf)): continue
    x = np.asarray(np.load(sf), np.float32)
    tk = tokenize50(x)
    if tk is None: continue
    ws, fv, ev, nw = tk
    gi = np.asarray(np.load(gf), np.float32)[:len(x)]
    if gi.ndim != 2 or gi.shape[1] < 5: continue
    gi = gi[:, :5]
    g2 = gi.copy()
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0) for w in range(nw)])
    order = np.argsort(-G.mean(0))
    li, lj = int(order[0]), int(order[1])
    c12 = corr(G[:, li], G[:, lj])
    if not np.isfinite(c12) or abs(c12) > MAXCORR: continue
    le = np.log10(ev + 1e-12)
    zle = (le - le.mean()) / (le.std() + 1e-6)
    res = []
    for arm, fmul in (("rawf", 1.0), ("stretch", 6.0)):
        X7 = np.c_[np.zeros(len(ws)), np.ones(len(ws)),
                   np.zeros(len(ws)), np.ones(len(ws)),
                   fv * fmul / 150.0, ws / max(nw - 1, 1),
                   zle].astype(np.float32)
        with torch.no_grad():
            a = sep(torch.from_numpy(X7)[None].to(dev))[0].cpu().numpy()
        Em = np.zeros((M, nw))
        for m in range(M):
            np.add.at(Em[m], ws, a[:, m] * ev)
        def score(Gm):
            C = np.zeros((M, 2))
            for m in range(M):
                C[m, 0] = corr(Em[m], Gm[:, li]) or 0
                C[m, 1] = corr(Em[m], Gm[:, lj]) or 0
            C = np.nan_to_num(C)
            return max((C[m1, 0] + C[m2, 1]) / 2 for m1 in range(M)
                       for m2 in range(M) if m1 != m2)
        res += [score(G), score(np.roll(G, nw // 2, 0))]
    ak = kmeans1d(fv, np.sqrt(ev))
    Ek = np.zeros((2, nw))
    for k in (0, 1):
        np.add.at(Ek[k], ws[ak == k], ev[ak == k])
    def score2(Gm):
        p1 = np.nanmean([corr(Ek[0], Gm[:, li]), corr(Ek[1], Gm[:, lj])])
        p2 = np.nanmean([corr(Ek[0], Gm[:, lj]), corr(Ek[1], Gm[:, li])])
        return max(p1, p2)
    res += [score2(G), score2(np.roll(G, nw // 2, 0))]
    rows.append(res)
    if len(rows) >= NREC: break

A = np.array(rows, float)
print(f"{len(rows)} XRF recordings scored (frozen PA separator, 50 Hz tokens)")
for i, nm in enumerate(("sep rawf", "sep stretch", "km dopp")):
    Mv, N = A[:, 2 * i], A[:, 2 * i + 1]
    print(f"  [{nm:11s}] matched {np.nanmedian(Mv):+.3f}  "
          f"null {np.nanmedian(N):+.3f}  win {np.mean(Mv > N)*100:.0f}%")
print("""
READ: sep >> null on ANOTHER DATASET at a third of the bandwidth with no
aperture = the clustering generalizes shockingly. sep ~ km ~ null = expected;
the model needs its native token distribution.""")
