#!/usr/bin/env python3
"""eval_xrf -- the gate, island representation. Audits a train_xrf checkpoint
on the untouched test split vs the two closed-form controls, on single
recordings. Same battery as eval_xpred, ported to (T,264) streams @50 Hz:
pairSNR / leak / psf / stress hp-room. PA has no IMU -> no route rows.

  MIXIT_RUNS=~/zerdani/buffer/octonet/pa_xrf_ft_runs \
  PREP_OUT=~/zerdani/buffer/octonet/prep_pa_xrf python3 eval/eval_xrf.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_pa_xrf"))
RUNS  = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/pa_xrf_ft_runs"))
CKPT  = os.environ.get("CKPT", "best.pt")
NPAIR = int(os.environ.get("NPAIR", "300"))
MINN  = int(os.environ.get("MINN", "100"))
FSR   = float(os.environ.get("FS", "50"))
C = 264
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/{CKPT}", map_location=dev, weights_only=False)
assert ck.get("xrf"), "not a train_xrf checkpoint"
DEG, SELF = ck["cfg"]["DEG"], ck["cfg"].get("SELF", 1)
CH30 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 30), DEG)
CH29 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 29), DEG)
print(f"ckpt {CKPT} step {ck['step']} val {ck.get('val')} cfg {ck['cfg']} dev={dev}")

from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    def __init__(self, cin=C, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(cin * (1 + SELF), nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=2, out_chan=nf, n_blocks=8,
                                n_repeats=4, bn_chan=192, hid_chan=512,
                                skip_chan=192, mask_act="linear",
                                causal=True, norm_type="cLN")
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x):
        T = x.shape[-1]
        if SELF:
            rm = torch.cumsum(x, -1) / torch.arange(1, T + 1, device=x.device)
            xin = torch.cat([x, rm], 1)
        else:
            xin = x
        e = self.enc(xin)
        y = self.dec((self.masker(e) * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, 2, C, T)
        return y[:, 0], y[:, 1] + (x - y.sum(1))
net = Sep().to(dev).eval()
net.load_state_dict(ck["model"])

def morph(sb, sa):
    out = np.empty(C, np.float32)
    for a in range(3):
        sl = slice(a * 30, (a + 1) * 30)
        Bm = CH30 * sb[sl][:, None]
        cf, *_ = np.linalg.lstsq(Bm, sa[sl], rcond=None)
        out[sl] = Bm @ cf
    zb = sb[90:177] + 1j * sb[177:264]
    za = sa[90:177] + 1j * sa[177:264]
    for a in range(3):
        sl = slice(a * 29, (a + 1) * 29)
        Bm = CH29 * zb[sl][:, None]
        cf, *_ = np.linalg.lstsq(Bm, za[sl], rcond=None)
        fit = Bm @ cf
        out[90 + sl.start:90 + sl.stop] = fit.real
        out[177 + sl.start:177 + sl.stop] = fit.imag
    return out

def sfrac(p):
    sp = p.mean(0)
    return float((sp ** 2).sum() / max((p ** 2).mean(0).sum(), 1e-12))

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

meta = pd.read_csv(f"{OUT}/meta.csv")
meta = meta[(meta.split == "test") & (meta.nsamp >= MINN)].reset_index(drop=True)
print(f"{len(meta)} test recordings, scenes {sorted(meta.scene.unique())}")
statics = {int(r.rid): np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy",
                                          mmap_mode="r"), np.float32).mean(0)
           for r in meta.itertuples()}
tmpl = {k: np.mean([statics[int(r)] for r in g.rid], 0)
        for k, g in meta.groupby(["node", "date"])}

def arms(x, key):
    with torch.no_grad():
        s, p = net(torch.from_numpy(x.T[None]).to(dev))
    s, p = s[0].cpu().numpy().T, p[0].cpu().numpy().T
    t = tmpl[key]
    fit = morph(t, x.mean(0))
    return {"model": (s, p), "trivial": (np.broadcast_to(t, x.shape), x - t),
            "deflate": (np.broadcast_to(fit, x.shape), x - fit)}

ARMS = ("model", "trivial", "deflate")
per = {a: {"psf": [], "pstat": {}} for a in ARMS}
hp_room = []
for r in meta.itertuples():
    x = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy"), np.float32)
    T = len(x) - ((len(x) - 16) % 8)
    x = x[:T]
    key = (r.node, r.date)
    if r.Index % 5 == 0:
        w = max(3, int(FSR)) | 1
        ma = pd.DataFrame(x).rolling(w, center=True, min_periods=1).mean() \
               .to_numpy(np.float32)
        xh = x - ma
        with torch.no_grad():
            sh, _ = net(torch.from_numpy(xh.T[None]).to(dev))
        hp_room.append(float((sh[0].cpu().numpy() ** 2).mean()
                             / max((xh ** 2).mean(), 1e-12)))
    for a, (s, p) in arms(x, key).items():
        per[a]["psf"].append(sfrac(p))
        per[a]["pstat"][int(r.rid)] = p.mean(0)

rng = np.random.default_rng(0)
groups = {k: list(g.rid.astype(int)) for k, g in meta.groupby(["node", "date"])
          if len(g) > 1}
keys = list(groups)
snr = {a: [] for a in ARMS}; leak = {a: [] for a in ARMS}
for _ in range(NPAIR):
    k = keys[rng.integers(len(keys))]
    ra, rb = rng.choice(groups[k], 2, replace=False)
    xa = np.asarray(np.load(f"{OUT}/streams/{ra:06d}.npy"), np.float32)
    T = len(xa) - ((len(xa) - 16) % 8)
    xa = xa[:T]
    tgt = xa - morph(statics[rb], statics[ra])[None, :]
    for a, (s, p) in arms(xa, k).items():
        snr[a].append(10 * np.log10(max((tgt ** 2).sum(), 1e-12)
                                    / max(((tgt - p) ** 2).sum(), 1e-12)))
        leak[a].append(corr(per[a]["pstat"][ra], per[a]["pstat"][rb]))

print(f"\nstress hp-room (model): median {np.median(hp_room):.3f}  "
      f"-- low = adaptive, high = memorised room\n")
print(f"{'arm':9s}{'pairSNR':>9s}{'leak':>8s}{'psf':>7s}")
print("-" * 34)
for a in ARMS:
    print(f"{a:9s}{np.median(snr[a]):9.2f}{np.nanmedian(leak[a]):8.3f}"
          f"{np.median(per[a]['psf']):7.3f}")
print("\nREAD: PASS = model pairSNR >= controls, leak < controls, psf sane, "
      "stress low.\nArm-1 reference (amplitude rep): gaps -13.5/-36 dB, "
      "leak 0.986, psf 0.316, stress 7.9.")
