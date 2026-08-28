#!/usr/bin/env python3
"""GT audit: correlate the pretrained separator's room/body channels against
OctoNet MoCap ground truth (skeleton joint speed). Read-only; CPU-only so a
concurrently training arm keeps the GPU.

GT envelope: mean joint speed from mocap_pose/*.npy skeletons (120 Hz, take
start = wall-clock in the filename, matched to the wifi stamp within 10 s).
Model: xpred75_runs/best.pt on prep_v75 streams (first CAP_S seconds).

Per recording: dynamic-energy envelopes of body, room, raw (0.21 s frames)
vs GT speed envelope ->
  r_body / r_room / r_raw   Pearson per recording
  r_room|body               PARTIAL correlation of room vs GT given body --
                            the scale-blind leakage test (old project 8.3:
                            plain correlation cannot see leakage; this can).
PASS pattern: r_body ~ r_raw and high, r_room|body ~ 0.

  NREC=200 CUDA_VISIBLE_DEVICES= python3 diagnostics/13_gt_corr.py
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
FS, K, C = 75.0, 114, 228
MPAT = re.compile(r"Take (\d{4}-\d{2}-\d{2}) (\d{2}\.\d{2}\.\d{2}) (AM|PM)")
WPAT = re.compile(r"exp-(\d{14})_")

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
net = Sep().eval()
net.load_state_dict(ck["model"])
print(f"ckpt step {ck['step']} cfg {ck['cfg']}")

takes = {}
for p in glob.glob(f"{ROOT}/mocap_pose/*.npy"):
    m = MPAT.search(os.path.basename(p))
    if not m: continue
    t = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}",
                          "%Y-%m-%d %I.%M.%S %p")
    takes[int(t.strftime("%Y%m%d%H%M%S"))] = p
tkeys = np.array(sorted(takes))
print(f"{len(tkeys)} mocap takes")

def gt_speed(path):
    x = np.load(path)
    ax = int(np.argmax(x.shape))
    x = np.moveaxis(x, ax, 0).astype(np.float64)      # time first
    x = x.reshape(len(x), -1, 3) if x.ndim == 2 and x.shape[1] % 3 == 0 else x
    if x.ndim != 3: return None
    v = np.linalg.norm(np.diff(x, axis=0), axis=-1) * 120.0
    return np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=1)

def env(e1d):
    n = (len(e1d) - 16) // 8 + 1
    if n < 8: return None
    idx = np.arange(n)[:, None] * 8 + np.arange(16)[None, :]
    return e1d[idx].mean(1)

def corr(a, b):
    if a is None or b is None: return np.nan
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12: return np.nan
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
rows = []
for r in meta.itertuples():
    if len(rows) >= NREC: break
    k = int(tkeys[np.argmin(np.abs(tkeys - r.stamp))])
    if abs(k - r.stamp) > 10: continue
    g = gt_speed(takes[k])
    if g is None or len(g) < 240: continue
    x = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy"), np.float32)
    T = min(len(x), int(CAP_S * FS)); T -= (T - 16) % 8
    if T < 300: continue
    x = x[:T]
    with torch.no_grad():
        s, p = net(torch.from_numpy(x.T[None]))
    s, p = s[0].numpy().T, p[0].numpy().T
    eb = env(((p - p.mean(0)) ** 2).sum(1))
    er = env(((s - s.mean(0)) ** 2).sum(1))
    ex = env(((x - x.mean(0)) ** 2).sum(1))
    if eb is None: continue
    tc = (np.arange(len(eb)) * 8 + 8) / FS
    gs = np.interp(tc, np.arange(len(g)) / 120.0, g, left=np.nan, right=np.nan)
    rb, rr, rx = corr(eb, gs), corr(er, gs), corr(ex, gs)
    rpart = corr(resid(er, eb), resid(np.nan_to_num(gs, nan=np.nanmean(gs)), eb))
    rows.append((r.scene, rb, rr, rx, rpart))
    if len(rows) % 50 == 0: print(f"  {len(rows)}/{NREC}", flush=True)

df = pd.DataFrame(rows, columns=["scene", "r_body", "r_room", "r_raw", "r_room_g_body"])
print(f"\n{len(df)} recordings matched to mocap takes\n")
def line(v): return f"med {np.nanmedian(v):+.3f}  mean {np.nanmean(v):+.3f}  |r|>0.2 {np.nanmean(np.abs(v) > 0.2)*100:.0f}%"
for c in ("r_body", "r_room", "r_raw", "r_room_g_body"):
    print(f"  {c:14s}: {line(df[c])}")
print("\nper scene (median r_body / r_room / r_room|body):")
for s, g in df.groupby("scene"):
    print(f"  scene {s}: {np.nanmedian(g.r_body):+.3f} / {np.nanmedian(g.r_room):+.3f} / "
          f"{np.nanmedian(g.r_room_g_body):+.3f}  (n={len(g)})")
print("""
READ: r_body ~ r_raw and clearly > 0 -> body channel tracks the GT skeleton.
r_room may look nonzero (scale-blind Pearson); r_room|body is the honest
leakage number -- near 0 means the room channel carries no GT information
beyond what body explains. Sync caveat: take start is filename wall-clock,
so +-1-2 s offsets depress all correlations equally; compare, don't absolutize.
""")
