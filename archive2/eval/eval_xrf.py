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
M = ck["cfg"].get("M", 2)
LIMB = int(ck["cfg"].get("LIMB", 0)) and M > 2
CMN = int(ck["cfg"].get("CMN", 0))
CH30 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 30), DEG)
CH29 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 29), DEG)
print(f"ckpt {CKPT} step {ck['step']} val {ck.get('val')} cfg {ck['cfg']} dev={dev}")

from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    def __init__(self, cin=C, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(cin * (1 + SELF), nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=M, out_chan=nf, n_blocks=8,
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
        y = y.view(-1, M, C, T)
        res = x - y.sum(1)
        return torch.cat([y[:, :1], (y[:, 1] + res).unsqueeze(1), y[:, 2:]], 1)
class PISep(nn.Module):
    def __init__(self, H=64, E=8):
        super().__init__()
        self.emb = nn.Embedding(3, E)
        tid = torch.zeros(C, dtype=torch.long)
        tid[90:177] = 1; tid[177:] = 2
        self.register_buffer("tid", tid)
        self.inp = nn.Conv1d(1 + E, H, 1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.ConstantPad1d((4 * d, 0), 0.0),
            nn.Conv1d(H, H, 5, dilation=d), nn.PReLU(),
            nn.Conv1d(H, H, 1)) for d in (1, 2, 4, 8, 16, 32, 64, 128)])
        self.head = nn.Sequential(nn.Conv1d(2 * H, H, 1), nn.PReLU(),
                                  nn.Conv1d(H, M, 1))
    def forward(self, x):
        B, Cc, T = x.shape
        e = self.emb(self.tid)
        u = torch.cat([x.reshape(B * Cc, 1, T),
                       e.repeat(B, 1)[:, :, None].expand(B * Cc, e.shape[1], T)], 1)
        z = self.inp(u)
        for b in self.blocks:
            z = z + b(z)
        H_ = z.shape[1]
        g = z.reshape(B, Cc, H_, T).mean(1)
        gz = g[:, None].expand(B, Cc, H_, T).reshape(B * Cc, H_, T)
        m = self.head(torch.cat([z, gz], 1))
        m = torch.softmax(m, 1).reshape(B, Cc, M, T).permute(0, 2, 1, 3)
        return m * x[:, None]

net = (PISep() if ck["cfg"].get("ARCH", "conv") == "pi" else Sep()).to(dev).eval()
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

SPLITEVAL = os.environ.get("SPLITEVAL", "test")
MAXR = int(os.environ.get("MAXR", "0"))
meta = pd.read_csv(f"{OUT}/meta.csv")
meta = meta[(meta.split == SPLITEVAL) & (meta.nsamp >= MINN)].reset_index(drop=True)
if MAXR and len(meta) > MAXR:
    meta = meta.sample(MAXR, random_state=0).reset_index(drop=True)
print(f"{len(meta)} test recordings, scenes {sorted(meta.scene.unique())}")
statics = {int(r.rid): np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy",
                                          mmap_mode="r"), np.float32).mean(0)
           for r in meta.itertuples()}
tmpl = {k: np.mean([statics[int(r)] for r in g.rid], 0)
        for k, g in meta.groupby(["node", "date"])}

def cmn_maps(sa):
    aa = np.abs(sa[:90])
    ga_ = np.maximum(aa, 0.05 * np.median(aa) + 1e-9)
    zb_ = sa[90:177] + 1j * sa[177:264]
    gz = np.abs(zb_)
    thr = 0.05 * np.median(gz) + 1e-9
    zb_ = np.where(gz < thr, thr + 0j, zb_)
    def nm(v, shift):
        va = v[..., :90] / ga_
        vz = (v[..., 90:177] + 1j * v[..., 177:264]) / zb_
        if shift: va = va - 1.0; vz = vz - 1.0
        return np.concatenate([va, vz.real, vz.imag], -1).astype(np.float32)
    def dn(v):
        va = v[..., :90] * ga_
        vz = (v[..., 90:177] + 1j * v[..., 177:264]) * zb_
        return np.concatenate([va, vz.real, vz.imag], -1).astype(np.float32)
    return nm, dn

def arms(x, key, sa):
    xin = x
    nm = dn = None
    if CMN:
        nm, dn = cmn_maps(sa)
        xin = nm(x, True)
    with torch.no_grad():
        y = net(torch.from_numpy(xin.T[None]).to(dev))[0].cpu().numpy()
    p = y[1:].sum(0).T
    if CMN: p = dn(p)
    s = x - p
    t = tmpl[key]
    fit = morph(t, x.mean(0))
    return {"model": (s, p), "trivial": (np.broadcast_to(t, x.shape), x - t),
            "deflate": (np.broadcast_to(fit, x.shape), x - fit)}, y

ARMS = ("model", "trivial", "deflate")
per = {a: {"psf": [], "pstat": {}} for a in ARMS}
hp_room = []
LIMBROWS = {i: {"own": [], "cross": [], "room": []} for i in range(5)}
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
        xh_in = cmn_maps(statics[int(r.rid)])[0](xh, False) if CMN else xh
        with torch.no_grad():
            yh = net(torch.from_numpy(xh_in.T[None]).to(dev))
        hp_room.append(float((yh[0, 0].cpu().numpy() ** 2).mean()
                             / max((xh_in ** 2).mean(), 1e-12)))
    armd, yfull = arms(x, key, statics[int(r.rid)])
    for a, (s, p) in armd.items():
        per[a]["psf"].append(sfrac(p))
        per[a]["pstat"][int(r.rid)] = p.mean(0)
    if LIMB and os.path.exists(f"{OUT}/imu/{r.rid:06d}.npy"):
        gi = np.asarray(np.load(f"{OUT}/imu/{r.rid:06d}.npy"),
                        np.float32)[:T]
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(T, np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        def envnp(z):
            d = z - z.mean(0); e = (d ** 2).sum(1)
            n = (len(e) - 16) // 8 + 1
            ix = np.arange(n)[:, None] * 8 + np.arange(16)
            return e[ix].mean(1)
        def poolg(v):
            n = (len(v) - 16) // 8 + 1
            ix = np.arange(n)[:, None] * 8 + np.arange(16)
            return v[ix].mean(1)
        er = envnp(yfull[0].T)
        evs = [envnp(yfull[2 + i].T) for i in range(5)]
        ga = [poolg(g2[:, i]) for i in range(5)]
        for i in range(5):
            if ga[i].std() < 1e-9: continue
            own = corr(evs[i], ga[i])
            crs = np.nanmean([corr(evs[j], ga[i]) for j in range(5) if j != i])
            LIMBROWS[i]["own"].append(own)
            LIMBROWS[i]["cross"].append(crs)
            LIMBROWS[i]["room"].append(corr(er, ga[i]))

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
    for a, (s, p) in arms(xa, k, statics[ra])[0].items():
        snr[a].append(10 * np.log10(max((tgt ** 2).sum(), 1e-12)
                                    / max(((tgt - p) ** 2).sum(), 1e-12)))
        leak[a].append(corr(per[a]["pstat"][ra], per[a]["pstat"][rb]))

if any(LIMBROWS[i]["own"] for i in range(5)):
    print("\nlimb routing on UNSEEN rooms (median r: own slot / other slots / room slot):")
    for i, d in enumerate(["LW", "RW", "LP", "RP", "HD"]):
        R = LIMBROWS[i]
        if R["own"]:
            print(f"  {d}: {np.nanmedian(R['own']):+.3f} / "
                  f"{np.nanmedian(R['cross']):+.3f} / {np.nanmedian(R['room']):+.3f}"
                  f"  (n={len(R['own'])})")
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
