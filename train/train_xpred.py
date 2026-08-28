#!/usr/bin/env python3
"""train_xpred -- cross-prediction separator on prep_v3 amplitude streams.

FRESH code: nothing imported from the legacy MixIT/exchange stack (user
policy). One model, CSI input ONLY -- the IMU appears in the loss, never the
forward pass. Train split = scenes 0+1 (5% of (node,date) groups held out as
val); scene 2 is never touched by this script.

Objective, per item (recordings A,B drawn from one (node,date) group = same
receiver, same room, same day):

    m*      = closed-form morph fitting static_B -> static_A
              (per-antenna Chebyshev reshape, deg DEG; pilot 12: gain/shift
              explain nothing, smooth deg 6 ~ 70% of the static difference)
    tgt_p   = window_A - m*(static_B)          # body + morph-unexplained static
    s, p    = model(window_A)                  # mixture consistency: s + p = x
    L       = -SNRclip(p, tgt_p)  +  warmup * IMUW * L_route

-SNRclip scores the PRIVATE side, so errors are priced at body scale, not
20 dB below room scale. L_route (IMU naming, scale-blind + static-blind):
fraction of IMU-attributable motion energy that lands in the SHARED slot -> 0.
A window's own static stays in tgt_p by construction, so a still person is
never pushed to the room slot by any term here (founding rule).

The model amortises the pair statistic into a single-recording function: at
inference it takes one recording -- no reference, no template, no IMU.

  MIXIT_RUNS=~/zerdani/buffer/octonet/xpred_runs \
  nohup python3 train/train_xpred.py > ../log_xpred.txt 2>&1 &
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUT    = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_v3"))
RUNS   = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/xpred_runs"))
B      = int(os.environ.get("B", "24"))
WIN    = int(os.environ.get("WIN", "512"))        # 12.8 s @ 40 Hz; (WIN-16)%8==0
DEG    = int(os.environ.get("DEG", "6"))
LR     = float(os.environ.get("LR", "2e-4"))
STEPS  = int(os.environ.get("STEPS", "60000"))
HOURS  = float(os.environ.get("HOURS", "8"))
IMUW   = float(os.environ.get("IMUW", "5.0"))
SNRMAX = float(os.environ.get("SNRMAX", "30.0"))
WARM   = int(os.environ.get("WARM", "2000"))
SEED   = int(os.environ.get("SEED", "0"))
NW     = int(os.environ.get("NW", "6"))
HINT   = int(os.environ.get("HINT", "1"))         # site template as extra input:
# room-2 gate v1 (HINT=0) failed -- leak 0.909, pairSNR 0.1 vs deflate 13.8.
# Only 2 physical training rooms exist; a template-free net cannot learn to
# infer an unseen room's static and memorises instead (Run 1-2 redux). The
# hint is deployment-honest: a base station accumulates it from idle traffic.
K, C = 114, 228
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
CHEB = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, K), DEG)  # (K, DEG+1)

# ---------- statics ----------------------------------------------------------
def build_statics(meta):
    cachef = f"{OUT}/statics_cache.npz"
    if os.path.exists(cachef):
        z = np.load(cachef)
        if all(str(r) in z for r in meta.rid):
            return {int(r): z[str(r)] for r in meta.rid}
    print("building statics cache ...", flush=True)
    st = {}
    for i, r in enumerate(meta.itertuples()):
        st[int(r.rid)] = np.asarray(np.load(f"{OUT}/streams/{r.rid:06d}.npy",
                                            mmap_mode="r"), np.float32).mean(0)
        if (i + 1) % 500 == 0: print(f"  {i+1}/{len(meta)}", flush=True)
    np.savez(cachef, **{str(k): v for k, v in st.items()})
    return st

def morph(static_b, static_a):
    """Closed-form m*(static_B): per-antenna smooth reshape toward static_A."""
    out = np.empty(C, np.float32)
    for a in (0, 1):
        u, v = static_b[a * K:(a + 1) * K], static_a[a * K:(a + 1) * K]
        Bm = CHEB * u[:, None]
        c, *_ = np.linalg.lstsq(Bm, v, rcond=None)
        out[a * K:(a + 1) * K] = Bm @ c
    return out

# ---------- data -------------------------------------------------------------
class Pairs(Dataset):
    def __init__(self, groups, statics, seed):
        self.groups, self.statics = groups, statics
        self.tmpl = [np.mean([statics[r[0]] for r in g], 0).astype(np.float32)
                     for g in groups]
        w = np.array([len(g) for g in groups], float)
        self.w = w / w.sum()
        self.rng = np.random.default_rng(seed)
    def __len__(self): return 10 ** 9
    def __getitem__(self, _):
        gi_ = self.rng.choice(len(self.groups), p=self.w)
        g = self.groups[gi_]
        ia, ib = self.rng.choice(len(g), 2, replace=False)
        (ra, na, ioka), (rb, _, _) = g[ia], g[ib]
        s0 = int(self.rng.integers(0, na - WIN + 1))
        x = np.asarray(np.load(f"{OUT}/streams/{ra:06d}.npy",
                               mmap_mode="r")[s0:s0 + WIN], np.float32)
        tgt = x - morph(self.statics[rb], self.statics[ra])[None, :]
        if ioka:
            gi = np.asarray(np.load(f"{OUT}/imu/{ra:06d}.npy",
                                    mmap_mode="r")[s0:s0 + WIN], np.float32)
        else:
            gi = np.zeros(WIN, np.float32)
        return (torch.from_numpy(x.T), torch.from_numpy(tgt.T),
                torch.from_numpy(gi), float(ioka),
                torch.from_numpy(self.tmpl[gi_]))

def make_groups(meta):
    out = []
    for _, g in meta.groupby(["node", "date"]):
        rows = [(int(r.rid), int(r.nsamp), int(r.imu_ok))
                for r in g.itertuples() if r.nsamp >= WIN]
        if len(rows) > 1: out.append(rows)
    return out

# ---------- model ------------------------------------------------------------
from asteroid.masknn import TDConvNet

class Sep(nn.Module):
    def __init__(self, cin=C, nf=512, L=16, S=8, hint=HINT):
        super().__init__()
        self.hint = hint
        self.enc = nn.Conv1d(cin * (1 + hint), nf, L, stride=S)
        try:
            self.masker = TDConvNet(in_chan=nf, n_src=2, out_chan=nf, n_blocks=8,
                                    n_repeats=4, bn_chan=192, hid_chan=512,
                                    skip_chan=192, mask_act="linear",
                                    causal=True, norm_type="cLN")
            self.causal = True
        except TypeError:
            self.masker = TDConvNet(in_chan=nf, n_src=2, out_chan=nf, n_blocks=8,
                                    n_repeats=4, bn_chan=192, hid_chan=512,
                                    skip_chan=192, mask_act="linear")
            self.causal = False
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x, h=None):               # x (B, 228, T), h (B, 228)
        T = x.shape[-1]
        xin = torch.cat([x, h.unsqueeze(-1).expand(-1, -1, T)], 1) \
              if self.hint else x
        e = self.enc(xin)
        m = self.masker(e)                      # (B, 2, nf, F)
        y = self.dec((m * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, 2, x.shape[1], T)
        res = x - y.sum(1)
        y = torch.stack([y[:, 0], y[:, 1] + res], 1)   # residual -> private
        return y[:, 0], y[:, 1]                        # s, p

# ---------- losses -----------------------------------------------------------
def neg_snr(est, ref):
    num = ref.pow(2).sum((-2, -1)) + 1e-10
    den = (ref - est).pow(2).sum((-2, -1)) + 1e-10
    return -torch.clamp(10 * torch.log10(num / den), max=SNRMAX)

def route_loss(s, p, imu, ok):
    """Fraction of IMU-attributable motion energy in the SHARED slot. [0,1].
    Static-blind: envelopes are variance around the window time-mean, so a
    still person's reflection contributes nothing to either side."""
    def env(z):
        d = z - z.mean(-1, keepdim=True)
        return F.avg_pool1d(d.pow(2).sum(1, keepdim=True), 16, 8)[:, 0]
    es, ep = env(s), env(p)
    gi = F.avg_pool1d(imu.unsqueeze(1), 16, 8)[:, 0]
    def zs(x):
        x = x - x.mean(-1, keepdim=True)
        return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)
    live = ((es.var(-1) > 1e-12) & (ep.var(-1) > 1e-12)
            & (gi.var(-1) > 1e-12) & (ok > 0.5))
    if live.sum() == 0: return s.new_zeros(())
    rs = (zs(es) * zs(gi)).mean(-1).clamp_min(0)
    rp = (zs(ep) * zs(gi)).mean(-1).clamp_min(0)
    return (rs / (rs + rp + 1e-8))[live].mean()

# ---------- main -------------------------------------------------------------
def main():
    os.makedirs(RUNS, exist_ok=True)
    meta = pd.read_csv(f"{OUT}/meta.csv")
    meta = meta[(meta.split == "train") & (meta.nsamp >= WIN)].reset_index(drop=True)
    statics = build_statics(meta)
    rng = np.random.default_rng(SEED)
    gkeys = sorted(meta.groupby(["node", "date"]).groups)
    val_keys = set(map(tuple, rng.choice(np.array(gkeys, dtype=object),
                                         max(2, len(gkeys) // 20), replace=False)))
    isval = meta.apply(lambda r: (r.node, r.date) in val_keys, axis=1)
    gtr, gva = make_groups(meta[~isval]), make_groups(meta[isval])
    print(f"{len(gtr)} train groups, {len(gva)} val groups | dev={dev} | "
          f"DEG={DEG} IMUW={IMUW} WIN={WIN} B={B}", flush=True)
    dl = DataLoader(Pairs(gtr, statics, SEED), batch_size=B, num_workers=NW,
                    pin_memory=(dev == "cuda"), persistent_workers=NW > 0)
    vl = DataLoader(Pairs(gva, statics, SEED + 1), batch_size=B, num_workers=2)
    model = Sep().to(dev)
    print(f"causal={model.causal}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    step0 = 0
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ck.get("steps_total") == STEPS:
            sch.load_state_dict(ck["sch"])
        else:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(STEPS - step0, 1))
            print(f"NOTICE: horizon changed, cosine rebuilt for {STEPS - step0} steps")
        print(f"resumed from step {step0}", flush=True)
    t0, best, vit = time.time(), math.inf, iter(vl)
    for step, (x, tgt, gi, ok, h) in enumerate(dl, start=step0):
        if step >= STEPS or (time.time() - t0) / 3600 > HOURS: break
        x, tgt, gi, ok, h = (t.to(dev, non_blocking=True)
                             for t in (x, tgt, gi, ok, h))
        s, p = model(x, h)
        Lp = neg_snr(p, tgt).mean()
        Lr = route_loss(s, p, gi, ok)
        w = min(1.0, step / max(WARM, 1))
        loss = Lp + w * IMUW * Lr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[{step}] snr_p {-Lp.item():.2f} dB  route {Lr.item():.3f}  "
                  f"sRMS {s.pow(2).mean().sqrt().item():.3f} "
                  f"pRMS {p.pow(2).mean().sqrt().item():.3f}  "
                  f"lr {sch.get_last_lr()[0]:.2e}  {(time.time()-t0)/3600:.2f}h",
                  flush=True)
        if step % 2000 == 0 and step > step0:
            model.eval(); acc = []
            with torch.no_grad():
                for _ in range(20):
                    try: vx, vt, vg, vo, vh = next(vit)
                    except StopIteration:
                        vit = iter(vl); vx, vt, vg, vo, vh = next(vit)
                    vx, vt, vg, vo, vh = (t.to(dev) for t in (vx, vt, vg, vo, vh))
                    vs, vp = model(vx, vh)
                    acc.append((neg_snr(vp, vt).mean()
                                + IMUW * route_loss(vs, vp, vg, vo)).item())
            v = float(np.mean(acc)); model.train()
            print(f"  VAL {v:.4f} {'(best)' if v < best else ''}", flush=True)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sch": sch.state_dict(), "step": step, "steps_total": STEPS,
                  "xpred": True, "cfg": {"DEG": DEG, "WIN": WIN, "IMUW": IMUW,
                                         "causal": model.causal, "HINT": HINT}}
            torch.save(ck, f"{RUNS}/last.pt")
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "val": v,
                            "xpred": True, "cfg": ck["cfg"]}, f"{RUNS}/best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step, "steps_total": STEPS,
                "xpred": True}, f"{RUNS}/last.pt")
    print(f"DONE step {step}, {(time.time()-t0)/3600:.2f} h, best val {best:.4f}",
          flush=True)

if __name__ == "__main__":
    main()
