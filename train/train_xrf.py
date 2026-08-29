#!/usr/bin/env python3
"""train_xrf -- cross-prediction separator on prep_xrf streams, carrying the
three post-mortem upgrades from the failed OctoNet/PA gates:

  SELF-CONDITIONING (replaces both memorisation and the rejected site hint):
    the input is [x, causal running mean of x] (264+264=528 ch). The room
    reference is DERIVED FROM THE RECORDING ITSELF at inference -- nothing to
    memorise, nothing to calibrate. The stress-test failure mode (hallucinating
    a memorised room) is structurally excluded: the model is trained to read
    its room off the conditioning channels.
  OBJECTIVE (unchanged family): private-side SNR against morph-corrected pair
    targets. Pairs from one (scene, rx) group = same room seen by the same
    radio, across takes and subjects. Morphs per block: real Chebyshev on the
    3 amp sub-blocks (30), complex Chebyshev on the 3 island sub-blocks (29).
  IMU ROUTING with lag-corrected envelopes (prep guarantees alignment; the
    release's raw alignment is +-0.5-1 s and would have nulled this loss).

  MIXIT_RUNS=~/zerdani/buffer/octonet/xrf_runs nohup python3 train/train_xrf.py &
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

OUT    = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_xrf"))
RUNS   = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/xrf_runs"))
B      = int(os.environ.get("B", "24"))
WIN    = int(os.environ.get("WIN", "512"))       # 10.24 s @ 50 Hz; (WIN-16)%8==0
DEG    = int(os.environ.get("DEG", "6"))
LR     = float(os.environ.get("LR", "2e-4"))
STEPS  = int(os.environ.get("STEPS", "40000"))
HOURS  = float(os.environ.get("HOURS", "2"))
IMUW   = float(os.environ.get("IMUW", "10.0"))   # body-level routing weight
LIMBW  = float(os.environ.get("LIMBW", "10.0"))  # per-limb assignment weight
SPW    = float(os.environ.get("SPW", "1.0"))     # per-slot norm penalty (Prop 4:
# the sum is constrained, individual slots are not -- without this they balloon
# into mutually-cancelling giants; observed coreRMS 6x at step 15k)
SNRMAX = float(os.environ.get("SNRMAX", "30.0"))
WARM   = int(os.environ.get("WARM", "2000"))
SEED   = int(os.environ.get("SEED", "0"))
NW     = int(os.environ.get("NW", "6"))
SELF   = int(os.environ.get("SELF", "0"))        # user rule: NO statistic inputs
LIMB   = int(os.environ.get("LIMB", "1"))        # slots: room|core|5 limbs
AUG    = int(os.environ.get("AUG", "0"))         # user rule: no augmentation
M      = 2 + (5 if LIMB else 0)
INIT   = os.environ.get("INIT", "")              # warm-start ckpt (model only)
GROUPBY = os.environ.get("GROUPBY", "scene,rx").split(",")
C = 264
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
CH30 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 30), DEG)
CH29 = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, 29), DEG)
CA3 = CH30[:, 1:4]
CI3 = CH29[:, 1:4]

def morph(sb, sa):
    """closed-form block morph mapping static_B -> static_A (264 real)."""
    out = np.empty(C, np.float32)
    for a in range(3):                                    # amp blocks, real
        sl = slice(a * 30, (a + 1) * 30)
        Bm = CH30 * sb[sl][:, None]
        cf, *_ = np.linalg.lstsq(Bm, sa[sl], rcond=None)
        out[sl] = Bm @ cf
    zb = sb[90:177] + 1j * sb[177:264]
    za = sa[90:177] + 1j * sa[177:264]
    for a in range(3):                                    # island blocks, complex
        sl = slice(a * 29, (a + 1) * 29)
        Bm = CH29 * zb[sl][:, None]
        cf, *_ = np.linalg.lstsq(Bm, za[sl], rcond=None)
        fit = Bm @ cf
        out[90 + sl.start:90 + sl.stop] = fit.real
        out[177 + sl.start:177 + sl.stop] = fit.imag
    return out

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

class Pairs(Dataset):
    def __init__(self, groups, statics, seed):
        self.groups, self.statics = groups, statics
        w = np.array([len(g) for g in groups], float)
        self.w = w / w.sum()
        self.rng = np.random.default_rng(seed)
    def __len__(self): return 10 ** 9
    def __getitem__(self, _):
        g = self.groups[self.rng.choice(len(self.groups), p=self.w)]
        ia, ib = self.rng.choice(len(g), 2, replace=False)
        (ra, na, ioka), (rb, _, _) = g[ia], g[ib]
        s0 = int(self.rng.integers(0, na - WIN + 1))
        x = np.asarray(np.load(f"{OUT}/streams/{ra:06d}.npy",
                               mmap_mode="r")[s0:s0 + WIN], np.float32)
        sa, sb = self.statics[ra], self.statics[rb]
        if AUG:
            # group-consistent synthetic room deformation: identical draw on
            # the window and BOTH statics (a data transform, not a statistic
            # input). Amp blocks: smooth log-gain; island blocks: constant
            # per-antenna phase (a delay shift's exact island signature) +
            # smooth magnitude reshape; common subcarrier roll.
            prof = np.ones(C, np.float32)
            phz = np.zeros(87, np.float32)
            for a_ in range(3):
                prof[a_*30:(a_+1)*30] = np.exp(CA3 @ self.rng.normal(0, .15, 3))
                prof[90+a_*29:90+(a_+1)*29] = prof[177+a_*29:177+(a_+1)*29] = \
                    np.exp(CI3 @ self.rng.normal(0, .1, 3))
                phz[a_*29:(a_+1)*29] = self.rng.uniform(-np.pi, np.pi)
            r_ = int(self.rng.integers(-4, 5))
            def ap(v):
                v = v * prof
                zc = (v[..., 90:177] + 1j*v[..., 177:264]) * np.exp(1j*phz)
                out = np.concatenate([
                    np.roll(v[..., :30], r_, -1), np.roll(v[..., 30:60], r_, -1),
                    np.roll(v[..., 60:90], r_, -1),
                    np.roll(zc.real[..., :29], r_, -1)[..., :29],
                    np.roll(zc.real[..., 29:58], r_, -1),
                    np.roll(zc.real[..., 58:], r_, -1),
                    np.roll(zc.imag[..., :29], r_, -1),
                    np.roll(zc.imag[..., 29:58], r_, -1),
                    np.roll(zc.imag[..., 58:], r_, -1)], -1)
                return out.astype(np.float32)
            x, sa, sb = ap(x), ap(sa), ap(sb)
            sc = 1.0 / max(float(np.sqrt((x ** 2).mean())), 1e-12)
            x, sa, sb = x * sc, sa * sc, sb * sc
        tgt = x - morph(sb, sa)[None, :]
        if ioka:
            gi = np.asarray(np.load(f"{OUT}/imu/{ra:06d}.npy",
                                    mmap_mode="r")[s0:s0 + WIN], np.float32)
            # residualize each limb's envelope against the other four: raw
            # envelopes are heavily collinear (whole-body motion), which makes
            # the limbs-x-slots assignment weakly identified -- probe 17's
            # measured fix, applied to the anchors the loss consumes.
            g2 = gi.copy()
            for i_ in range(5):
                oth = [j for j in range(5) if j != i_]
                A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
                beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
                g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
            gi = np.c_[gi.sum(1), g2]        # ch0 = raw total (body term),
        else:                                 # ch1..5 = residualized limbs
            gi = np.zeros((WIN, 6), np.float32)
        return (torch.from_numpy(x.T), torch.from_numpy(tgt.T),
                torch.from_numpy(gi.T), float(ioka))

def make_groups(meta):
    out = []
    for _, g in meta.groupby(GROUPBY):
        rows = [(int(r.rid), int(r.nsamp), int(r.imu_ok))
                for r in g.itertuples() if r.nsamp >= WIN]
        if len(rows) > 1: out.append(rows)
    return out

from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    """slots: 0 room | 1 body-core (statics + unattributed; residual lands
    here) | 2..6 limb dynamics [LW RW LP RP GL] when LIMB=1."""
    def __init__(self, cin=C, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(cin * (1 + SELF), nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=M, out_chan=nf, n_blocks=8,
                                n_repeats=4, bn_chan=192, hid_chan=512,
                                skip_chan=192, mask_act="linear",
                                causal=True, norm_type="cLN")
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x):                        # x (B, 264, T)
        T = x.shape[-1]
        if SELF:
            rm = torch.cumsum(x, -1) / torch.arange(1, T + 1, device=x.device)
            xin = torch.cat([x, rm], 1)
        else:
            xin = x
        e = self.enc(xin)
        y = self.dec((self.masker(e) * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, M, x.shape[1], T)
        res = x - y.sum(1)
        y = torch.cat([y[:, :1], (y[:, 1] + res).unsqueeze(1), y[:, 2:]], 1)
        return y                                  # (B, M, C, T)

def neg_snr(est, ref):
    num = ref.pow(2).sum((-2, -1)) + 1e-10
    den = (ref - est).pow(2).sum((-2, -1)) + 1e-10
    return -torch.clamp(10 * torch.log10(num / den), max=SNRMAX)

def _env(z):
    d = z - z.mean(-1, keepdim=True)
    return F.avg_pool1d(d.pow(2).sum(1, keepdim=True), 16, 8)[:, 0]

def _zs(x):
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)

def route_loss(y, imu6, ok):
    """imu6 (B,6,T): ch0 raw total motion, ch1..5 residualized limb envelopes.
    Body term: total motion out of the room slot. Limb term: each limb's
    OWN motion (residualized) into its own slot. Scale/static-blind."""
    er = _env(y[:, 0])
    ep = _env(y[:, 1:].sum(1))
    gt = F.avg_pool1d(imu6[:, :1], 16, 8)[:, 0]
    live = ((er.var(-1) > 1e-12) & (ep.var(-1) > 1e-12)
            & (gt.var(-1) > 1e-12) & (ok > 0.5))
    if live.sum() == 0: return y.new_zeros(()), y.new_zeros(())
    rs = (_zs(er) * _zs(gt)).mean(-1).clamp_min(0)
    rp = (_zs(ep) * _zs(gt)).mean(-1).clamp_min(0)
    body_term = (rs / (rs + rp + 1e-8))[live].mean()
    if not LIMB: return body_term, y.new_zeros(())
    gl = F.avg_pool1d(imu6[:, 1:], 16, 8)                # (B,5,F)
    envs = torch.stack([_env(y[:, 2 + i]) for i in range(5)], 1)  # (B,5,F)
    ez, gz, rz = _zs(envs), _zs(gl), _zs(er).unsqueeze(1)
    A = (ez.unsqueeze(2) * gz.unsqueeze(1)).mean(-1).clamp_min(0)  # slot i x limb j
    Ar = (rz * gz).mean(-1).clamp_min(0)                  # room x limb j
    own = torch.diagonal(A, dim1=1, dim2=2)
    other = A.sum(1) - own
    limb_live = (gl.var(-1) > 1e-12) & live.unsqueeze(1)
    frac = (other + Ar) / (own + other + Ar + 1e-8)
    if limb_live.sum() == 0: return body_term, y.new_zeros(())
    return body_term, frac[limb_live].mean()

def main():
    os.makedirs(RUNS, exist_ok=True)
    meta = pd.read_csv(f"{OUT}/meta.csv")
    if "split" in meta.columns:
        meta = meta[meta.split == "train"]
    meta = meta[meta.nsamp >= WIN].reset_index(drop=True)
    model = Sep()
    for attempt in range(10):
        try:
            model = model.to(dev); break
        except RuntimeError as e:
            print(f"to({dev}) failed, retry {attempt+1}/10 in 60s", flush=True)
            time.sleep(60)
    statics = build_statics(meta)
    rng = np.random.default_rng(SEED)
    val_names = set(rng.choice(sorted(meta.name.unique()),
                               max(4, meta.name.nunique() // 20), replace=False))
    gtr = make_groups(meta[~meta.name.isin(val_names)])
    gva = make_groups(meta[meta.name.isin(val_names)])
    print(f"{len(gtr)} train groups, {len(gva)} val groups | dev={dev} | "
          f"SELF={SELF} DEG={DEG} IMUW={IMUW} WIN={WIN} B={B}", flush=True)
    dl = DataLoader(Pairs(gtr, statics, SEED), batch_size=B, num_workers=NW,
                    pin_memory=(dev == "cuda"), persistent_workers=NW > 0)
    vl = DataLoader(Pairs(gva, statics, SEED + 1), batch_size=B, num_workers=2)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    step0 = 0
    if INIT and not os.path.exists(f"{RUNS}/last.pt"):
        cki = torch.load(os.path.expanduser(INIT), map_location=dev,
                         weights_only=False)
        model.load_state_dict(cki["model"])
        print(f"warm-start from {INIT} (step {cki.get('step')})", flush=True)
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        if ck.get("steps_total") == STEPS:
            sch.load_state_dict(ck["sch"])
        else:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(STEPS - step0, 1))
            print("NOTICE: horizon changed, cosine rebuilt")
        print(f"resumed from step {step0}", flush=True)
    t0, best, vit = time.time(), math.inf, iter(vl)
    for step, (x, tgt, gi, ok) in enumerate(dl, start=step0):
        if step >= STEPS or (time.time() - t0) / 3600 > HOURS: break
        x, tgt, gi, ok = (t.to(dev, non_blocking=True) for t in (x, tgt, gi, ok))
        y = model(x)
        s, p = y[:, 0], y[:, 1:].sum(1)
        Lp = neg_snr(p, tgt).mean()
        Lb, Ll = route_loss(y, gi, ok)
        Ls = y[:, 1:].pow(2).mean((-2, -1)).clamp_min(1e-12).sqrt().mean()
        w = min(1.0, step / max(WARM, 1))
        loss = Lp + w * (IMUW * Lb + LIMBW * Ll) + SPW * Ls
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            lrms = y[:, 2:].pow(2).mean().sqrt().item() if LIMB else 0.0
            print(f"[{step}] snr_p {-Lp.item():.2f} dB  routeB {Lb.item():.3f} "
                  f"routeL {Ll.item():.3f}  sRMS {s.pow(2).mean().sqrt().item():.3f} "
                  f"coreRMS {y[:,1].pow(2).mean().sqrt().item():.3f} "
                  f"limbRMS {lrms:.3f}  lr {sch.get_last_lr()[0]:.2e}  "
                  f"{(time.time()-t0)/3600:.2f}h", flush=True)
        if step % 2000 == 0 and step > step0:
            model.eval(); acc = []
            with torch.no_grad():
                for _ in range(20):
                    try: vx, vt, vg, vo = next(vit)
                    except StopIteration:
                        vit = iter(vl); vx, vt, vg, vo = next(vit)
                    vx, vt, vg, vo = (t.to(dev) for t in (vx, vt, vg, vo))
                    vy = model(vx)
                    vLb, vLl = route_loss(vy, vg, vo)
                    acc.append((neg_snr(vy[:, 1:].sum(1), vt).mean()
                                + IMUW * vLb + LIMBW * vLl).item())
            v = float(np.mean(acc)); model.train()
            print(f"  VAL {v:.4f} {'(best)' if v < best else ''}", flush=True)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sch": sch.state_dict(), "step": step, "steps_total": STEPS,
                  "xrf": True, "cfg": {"DEG": DEG, "WIN": WIN, "IMUW": IMUW,
                                       "LIMBW": LIMBW, "SELF": SELF,
                                       "LIMB": LIMB, "M": M, "AUG": AUG}}
            torch.save(ck, f"{RUNS}/last.pt")
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "val": v,
                            "xrf": True, "cfg": ck["cfg"]}, f"{RUNS}/best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step, "steps_total": STEPS,
                "xrf": True}, f"{RUNS}/last.pt")
    print(f"DONE step {step}, {(time.time()-t0)/3600:.2f} h, best val {best:.4f}",
          flush=True)

if __name__ == "__main__":
    main()
