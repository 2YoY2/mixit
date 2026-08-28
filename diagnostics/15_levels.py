#!/usr/bin/env python3
"""Signal-level accounting of the separation: what does removing the room
actually remove, in dB, on the held-out scenes?

Per test recording (model = an arm's best.pt), decompose x, s (room), p (body)
into static (time-mean) and dynamic (residual) power:

  lvl_s, lvl_p       channel level relative to x           (dB)
  dyn_x              raw dynamic level (the motion budget) (dB rel. x)
  dyncap = dyn_p/dyn_x   fraction of raw dynamics kept in the BODY channel
  dynleak = dyn_s/dyn_x  fraction of raw dynamics lost to the ROOM channel
  statp = stat_p/stat_x  fraction of raw statics kept in the body channel
                         (the level view of room-in-body leakage)

PASS shape: dyncap ~ 1 (removing the room removed no motion), dynleak ~ 0,
lvl_p ~ dyn_x + a small static allowance, statp small but NOT ~0 (a still
person must survive; ~0 here would be the static/dynamic collapse).

  MIXIT_RUNS=~/zerdani/buffer/octonet/pa400_scratch_runs \
  PREP_OUT=~/zerdani/buffer/octonet/prep_pa400 \
  CUDA_VISIBLE_DEVICES= python3 diagnostics/15_levels.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT  = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_pa400"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/pa400_scratch_runs"))
NREC = int(os.environ.get("NREC", "120"))
C = 228

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
print(f"ckpt step {ck['step']} val {ck.get('val'):.3f} cfg {ck['cfg']}")

def split(z):
    st = float((z.mean(0) ** 2).sum())
    dy = float(((z - z.mean(0)) ** 2).mean(0).sum())
    return st, dy

meta = pd.read_csv(f"{OUT}/meta.csv")
meta = meta[meta.split == "test"].sample(NREC, random_state=0)
rows = []
for r in meta.itertuples():
    x = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy"), np.float32)
    T = len(x) - ((len(x) - 16) % 8)
    x = x[:T]
    with torch.no_grad():
        s, p = net(torch.from_numpy(x.T[None]))
    s, p = s[0].numpy().T, p[0].numpy().T
    px = float((x ** 2).mean(0).sum())
    stx, dyx = split(x); sts, dys = split(s); stp, dyp = split(p)
    rows.append((float((s**2).mean(0).sum())/px, float((p**2).mean(0).sum())/px,
                 dyx/px, dyp/max(dyx,1e-12), dys/max(dyx,1e-12),
                 stp/max(stx,1e-12), sts/max(stx,1e-12)))
d = np.median(np.array(rows), axis=0)
dB = lambda v: 10*np.log10(max(v, 1e-12))
print(f"\n{len(rows)} test recordings (medians)")
print(f"  room  level  : {d[0]*100:6.2f}%  ({dB(d[0]):+6.2f} dB)")
print(f"  body  level  : {d[1]*100:6.2f}%  ({dB(d[1]):+6.2f} dB)")
print(f"  raw dynamics : {d[2]*100:6.2f}%  ({dB(d[2]):+6.2f} dB)   <- the motion budget")
print(f"  dyncap  (dyn kept in body) : {d[3]*100:6.1f}%")
print(f"  dynleak (dyn lost to room) : {d[4]*100:6.1f}%")
print(f"  statics kept in body       : {d[5]*100:6.2f}%   (watchdog: ~0 = collapse)")
print(f"  statics kept in room       : {d[6]*100:6.2f}%")
print("""
READ: dyncap near 100 with dynleak near 0 -> removing the room removed no
motion. body level ~ raw-dynamics level + small static part is the healthy
shape. statics-in-body ~0 would mean the still person is being deleted.
""")
