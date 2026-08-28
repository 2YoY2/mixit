#!/usr/bin/env python3
"""GT audit v2: per-recording clock-offset search, then channel correlations.

13_gt_corr came back ~0 for EVERY channel including raw CSI -> the alignment
(mocap timestamps vs the 1 s wifi dir-stamp) is the bottleneck, not the model.
Fix: slide the GT speed envelope +-SRCH s against the RAW dynamic-energy
envelope (search on raw = unbiased toward either output channel), lock the
best offset per recording, then score body/room/raw there.

Honesty guard: searching 201 offsets over ~240-frame envelopes inflates noise
correlations to ~0.15-0.2. The check is the OFFSET STRUCTURE: a real clock
offset is near-constant within a date (one session) -- so the per-date offset
IQR is printed, and tight IQR + high r is signal; scattered offsets + modest r
is the search fitting noise. Judge from that, not the medians alone.

  NREC=200 CUDA_VISIBLE_DEVICES= python3 diagnostics/14_gt_sync.py
"""
import os, re, glob
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/octonet/OctoNet-upload"))
OUT  = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v75"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/xpred75_runs"))
NREC = int(os.environ.get("NREC", "200"))
CAP_S = float(os.environ.get("CAP_S", "60"))
SRCH = float(os.environ.get("SRCH", "10"))
FS, C = 75.0, 228
MPAT = re.compile(r"Take (\d{4}-\d{2}-\d{2}) (\d{2}\.\d{2}\.\d{2}) (AM|PM)")

ck = torch.load(f"{RUNS}/best.pt", map_location="cpu", weights_only=False)
from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    def __init__(self, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(C, nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=2, out_chan=nf, n_blocks=8,
                                n_repeats=4, bn_chan=192, hid_chan=512,
                                skip_chan=192, mask_act="linear",
                                causal=ck["cfg"]["causal"], norm_type="cLN")
        self.dec = nn.ConvTranspose1d(nf, C, L, stride=S)
    def forward(self, x):
        T = x.shape[-1]
        e = self.enc(x)
        y = self.dec((self.masker(e) * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, 2, C, T)
        return y[:, 0], y[:, 1] + (x - y.sum(1))
net = Sep().eval(); net.load_state_dict(ck["model"])

takes = {}
for p in glob.glob(f"{ROOT}/mocap_pose/*.npy"):
    m = MPAT.search(os.path.basename(p))
    if m:
        t = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}",
                              "%Y-%m-%d %I.%M.%S %p")
        takes[int(t.strftime("%Y%m%d%H%M%S"))] = p
tkeys = np.array(sorted(takes))

def naive(v): return pd.Timestamp(v).to_pydatetime().replace(tzinfo=None)

def gt_speed(path, w0):
    d = np.load(path, allow_pickle=True).item()
    pos = np.asarray(d["positions"], np.float64)
    t = np.array([(naive(v) - w0).total_seconds() for v in d["timestamps"]])
    k = np.concatenate([[True], np.diff(t) > 0])
    pos, t = pos[k], t[k]
    if len(t) < 240 or t[0] > 20 or t[-1] < 10: return None, None
    v = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / np.diff(t)[:, None]
    return t[1:], np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=1)

def env(e1d):
    n = (len(e1d) - 16) // 8 + 1
    idx = np.arange(n)[:, None] * 8 + np.arange(16)[None, :]
    return e1d[idx].mean(1)

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 24 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12: return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

def resid(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 8: return y
    b = np.polyfit(x[ok], y[ok], 1)
    return y - (b[0] * x + b[1])

meta = pd.read_csv(f"{OUT}/meta.csv")
meta["stamp"] = meta.file.str.extract(r"exp-(\d{14})_").astype(np.int64)
rng = np.random.default_rng(0)
meta = meta.iloc[rng.permutation(len(meta))]
offs = np.arange(-SRCH, SRCH + 1e-9, 0.1)
rows = []
for r in meta.itertuples():
    if len(rows) >= NREC: break
    k = int(tkeys[np.argmin(np.abs(tkeys - r.stamp))])
    if abs(k - r.stamp) > 10: continue
    w0 = datetime.strptime(str(int(r.stamp)), "%Y%m%d%H%M%S")
    tg, gv = gt_speed(takes[k], w0)
    if tg is None: continue
    x = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy"), np.float32)
    T = min(len(x), int(CAP_S * FS)); T -= (T - 16) % 8
    if T < 300: continue
    x = x[:T]
    with torch.no_grad():
        s, p = net(torch.from_numpy(x.T[None]))
    s, p = s[0].numpy().T, p[0].numpy().T
    ex = env(((x - x.mean(0)) ** 2).sum(1))
    eb = env(((p - p.mean(0)) ** 2).sum(1))
    er = env(((s - s.mean(0)) ** 2).sum(1))
    tc = (np.arange(len(ex)) * 8 + 8) / FS
    best, r0 = (np.nan, -2), np.nan
    for d_ in offs:
        gs = np.interp(tc + d_, tg, gv, left=np.nan, right=np.nan)
        rr = corr(ex, gs)
        if d_ == 0.0 or abs(d_) < 0.05: r0 = rr
        if rr == rr and rr > best[1]: best = (d_, rr)
    if best[1] <= -2: continue
    d_ = best[0]
    gs = np.interp(tc + d_, tg, gv, left=np.nan, right=np.nan)
    gsf = np.nan_to_num(gs, nan=np.nanmean(gs))
    rows.append((r.scene, str(r.date), d_, r0, best[1], corr(eb, gs), corr(er, gs),
                 corr(resid(er, eb), resid(gsf, eb))))
    if len(rows) % 50 == 0: print(f"  {len(rows)}/{NREC}", flush=True)

df = pd.DataFrame(rows, columns=["scene", "date", "off", "r_raw0", "r_raw",
                                 "r_body", "r_room", "r_room_g_body"])
print(f"\n{len(df)} recordings\n")
print(f"raw at zero offset : med {np.nanmedian(df.r_raw0):+.3f}")
print(f"raw at best offset : med {np.nanmedian(df.r_raw):+.3f}   "
      f"|r|>0.3 {np.nanmean(np.abs(df.r_raw) > 0.3)*100:.0f}%")
print(f"body at best offset: med {np.nanmedian(df.r_body):+.3f}   "
      f"|r|>0.3 {np.nanmean(np.abs(df.r_body) > 0.3)*100:.0f}%")
print(f"room at best offset: med {np.nanmedian(df.r_room):+.3f}")
print(f"room|body (partial): med {np.nanmedian(df.r_room_g_body):+.3f}")
print("\noffset structure per date (median [IQR] s, n):")
for d, g in df.groupby("date"):
    q1, q3 = np.percentile(g.off, [25, 75])
    print(f"  {d}: {np.median(g.off):+6.2f}  [{q3-q1:5.2f}]  n={len(g)}")
print("""
READ: tight per-date offset IQR (<1 s) with raw jumping well above its
zero-offset value -> real clock offset found; body/room numbers at the best
offset are then meaningful. Scattered offsets (IQR ~ several s) with raw
~0.15-0.2 -> the search fit noise; treat as null.
""")
