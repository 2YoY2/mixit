#!/usr/bin/env python3
"""eval_xpred -- the room-2 gate. Audits best.pt on the UNTOUCHED test scene
against the two controls it must beat, on single recordings.

Arms (all produce a division s + p = x per recording):
  model    s, p = net(x)                       -- single recording, no reference,
                                                  no template, no IMU
  trivial  s = site template, p = x - template -- template = mean static of the
                                                  recording's (node,date) group,
                                                  the deployment site-calibration
  deflate  s = m*(template -> own static)      -- closed-form morph (deg from
           p = x - s                              ckpt cfg): the training
                                                  objective's no-network optimum

Metrics (ground-truth-free + IMU-referenced; each with its blind spot noted):
  pairSNR  SNR(p_A, x_A - m*(static_B)) over same-group pairs -- the training
           objective measured on the unseen room. Cannot see person statics.
  leak     corr of private STATICS across same-group pairs. Bodies are
           independent across recordings, so any correlation is room content
           sitting in the body channel. THE headline: lower = cleaner body.
  psf      private static-frac. Near 0 = static/dynamic collapse (founding
           rule broken); the watchdog, not a score.
  route    of IMU-attributable motion, fraction landing in the body slot
           (imu_ok recordings only). Blind to statics.
  r_body   corr(body dynamic envelope, IMU envelope); r_raw printed as the
           reference the body channel should not fall far below.

  MIXIT_RUNS=~/zerdani/buffer/octonet/xpred_runs python3 eval/eval_xpred.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v3"))
RUNS  = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/xpred_runs"))
CKPT  = os.environ.get("CKPT", "best.pt")
NPAIR = int(os.environ.get("NPAIR", "300"))
K, C = 114, 228
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/{CKPT}", map_location=dev, weights_only=False)
assert ck.get("xpred"), "not an xpred checkpoint"
DEG = ck["cfg"]["DEG"]
CHEB = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, K), DEG)
print(f"ckpt {CKPT} step {ck['step']} cfg {ck['cfg']} dev={dev}")

HINT = int(ck["cfg"].get("HINT", 0))
from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    def __init__(self, cin=C, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(cin * (1 + HINT), nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=2, out_chan=nf, n_blocks=8,
                                n_repeats=4, bn_chan=192, hid_chan=512,
                                skip_chan=192, mask_act="linear",
                                causal=ck["cfg"]["causal"], norm_type="cLN")
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x, h=None):
        T = x.shape[-1]
        xin = torch.cat([x, h.unsqueeze(-1).expand(-1, -1, T)], 1) if HINT else x
        e = self.enc(xin)
        y = self.dec((self.masker(e) * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, 2, C, T)
        return y[:, 0], y[:, 1] + (x - y.sum(1))

net = Sep().to(dev).eval()
net.load_state_dict(ck["model"])

def morph(u, v):
    out = np.empty(C, np.float32)
    for a in (0, 1):
        Bm = CHEB * u[a * K:(a + 1) * K][:, None]
        c, *_ = np.linalg.lstsq(Bm, v[a * K:(a + 1) * K], rcond=None)
        out[a * K:(a + 1) * K] = Bm @ c
    return out

def env(e1d):                      # (T,) energy -> framed envelope, k=16 s=8
    n = (len(e1d) - 16) // 8 + 1
    if n < 4: return None
    idx = np.arange(n)[:, None] * 8 + np.arange(16)[None, :]
    return e1d[idx].mean(1)

def corr(a, b):
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def sfrac(p):
    sp = p.mean(0)
    return float((sp ** 2).sum() / max((p ** 2).mean(0).sum(), 1e-12))

meta = pd.read_csv(f"{OUT}/meta.csv")
meta = meta[(meta.split == "test") & (meta.nsamp >= 256)].reset_index(drop=True)
print(f"{len(meta)} test recordings, scene {sorted(meta.scene.unique())}")

statics = {int(r.rid): np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy",
                                          mmap_mode="r"), np.float32).mean(0)
           for r in meta.itertuples()}
tmpl = {k: np.mean([statics[int(r)] for r in g.rid], 0)
        for k, g in meta.groupby(["node", "date"])}

def arms(x, key):                  # x (T,228) trimmed -> {arm: (s, p)}
    with torch.no_grad():
        xt = torch.from_numpy(x.T[None]).to(dev)
        ht = torch.from_numpy(tmpl[key].astype(np.float32)[None]).to(dev)
        s, p = net(xt, ht)
    s, p = s[0].cpu().numpy().T, p[0].cpu().numpy().T
    t = tmpl[key]
    fit = morph(t, x.mean(0))
    return {"model": (s, p), "trivial": (np.broadcast_to(t, x.shape), x - t),
            "deflate": (np.broadcast_to(fit, x.shape), x - fit)}

ARMS = ("model", "trivial", "deflate")
per = {a: {"psf": [], "route": [], "r_body": [], "pstat": {}} for a in ARMS}
r_raw = []
for r in meta.itertuples():
    x = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy"), np.float32)
    T = len(x) - ((len(x) - 16) % 8)
    x = x[:T]
    key = (r.node, r.date)
    gi = None
    if r.imu_ok:
        gi = env(np.asarray(np.load(f"{OUT}/imu/{r.rid:06d}.npy"), np.float32)[:T] ** 1)
        ex = env(((x - x.mean(0)) ** 2).sum(1))
        rr = corr(ex, gi)
        if rr == rr: r_raw.append(rr)
    for a, (s, p) in arms(x, key).items():
        per[a]["psf"].append(sfrac(p))
        per[a]["pstat"][int(r.rid)] = p.mean(0)
        if gi is not None:
            eb = env(((p - p.mean(0)) ** 2).sum(1))
            es = env(((s - s.mean(0)) ** 2).sum(1))
            rb, rs = corr(eb, gi), corr(es, gi)
            if rb == rb and rs == rs:
                per[a]["r_body"].append(rb)
                per[a]["route"].append(max(rb, 0) / max(max(rb, 0) + max(rs, 0), 1e-9))

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

print(f"\nr_raw (raw CSI env vs IMU) median: {np.median(r_raw):+.3f}\n")
print(f"{'arm':9s}{'pairSNR':>9s}{'leak':>8s}{'psf':>7s}{'route':>8s}{'r_body':>9s}")
print("-" * 50)
for a in ARMS:
    print(f"{a:9s}{np.median(snr[a]):9.2f}{np.nanmedian(leak[a]):8.3f}"
          f"{np.median(per[a]['psf']):7.3f}{np.median(per[a]['route']):8.3f}"
          f"{np.median(per[a]['r_body']):9.3f}")
print("""
READ (gate to PerceptAlign):
  PASS needs ALL of: model pairSNR >= both controls; model leak < both
  controls; model psf not << trivial's (collapse watch); route > 0.5 with
  r_body not far below r_raw.
  pairSNR cannot see person statics; leak is the honest room-in-body number;
  psf is a watchdog, not a score. trivial/deflate need a site template --
  the model is the only single-recording arm. Judge accordingly.
""")
