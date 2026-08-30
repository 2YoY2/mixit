#!/usr/bin/env python3
"""PHASE 3: true MixIT on additive sanitized CSI (the unrun crown jewel of
the actor campaign, run with phase-2's data machinery and gate).

  raw .mat -> sanitize (STO ramp fit + CFO de-rotation; ADDITIVE) -> 400 Hz
  -> real layout (342, T) -> TCN masker, N slots, softmax partition
  losses: MixIT (MoM of two same-node recordings; greedy assignment +
          mixture-consistency) + limb-envelope PIT aux (keypoint GT)
  gate: rooms 4/5 slot envelopes vs GT limbs (matched/null/win)

  HOURS=12 python3 phase3/train_actor_mixit.py
"""
import os, time
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("OUT", "~/zerdani/buffer/octonet/actor_mixit_runs"))
HOURS = float(os.environ.get("HOURS", "12"))
STEPS = int(os.environ.get("STEPS", "400000"))
B = int(os.environ.get("B", "8"))            # pairs per step
LR = float(os.environ.get("LR", "3e-4"))
NSLOTS = int(os.environ.get("NSLOTS", "8"))
WIDTH = int(os.environ.get("WIDTH", "256"))
NBLOCKS = int(os.environ.get("NBLOCKS", "8"))
LIMBW = float(os.environ.get("LIMBW", "0.5"))
TAU = float(os.environ.get("TAU", "1e-3"))
EVERY = int(os.environ.get("EVERY", "2500"))
NEVAL = int(os.environ.get("NEVAL", "120"))
SEED = int(os.environ.get("SEED", "0"))
FS, WIN = 400.0, 1024
WINF, HOPF = 256, 128
NW_ = int(os.environ.get("NW_", "6"))
CIN = 2 * 3 * 57
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

def sanitize(c):
    """(3,57,T) raw -> hardware phase removed, ADDITIVITY preserved."""
    A, S, T = c.shape
    k = np.arange(S) - (S - 1) / 2.0
    ph = np.unwrap(np.angle(c.mean(0)), axis=0)
    slope = (k[:, None] * ph).sum(0) / (k ** 2).sum()
    c = c * np.exp(-1j * slope[None, None, :] * k[None, :, None])
    ref = c.mean(-1, keepdims=True)
    proj = (c * ref.conj()).sum((0, 1))
    c = c * np.exp(-1j * np.angle(proj))[None, None, :]
    return c

def load_400(f):
    """raw mat -> sanitized additive CSI on the 400 Hz grid, (T,3,57) c64."""
    try:
        return _load_400(f)
    except Exception:
        return None

def _load_400(f):
    with h5py.File(os.path.join(ROOT, f), "r") as h:
        c = h["csi/csi"][...]
        ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    x = (c["real"] + 1j * c["imag"]).astype(np.complex64)
    dt = float(np.median(np.diff(ts)))
    rate, t = None, None
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
    if rate is None:
        rate = 810.0; t = np.arange(x.shape[-1]) / rate
    keep = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., keep], t[keep]
    if float(t[-1]) < 3.0: return None
    x = sanitize(x)
    x = np.moveaxis(x, -1, 0).reshape(len(t), -1)          # (T,171)
    nb = int(float(t[-1]) * FS)
    if nb < WIN + 64: return None
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, x.shape[1]), np.complex64)
    np.add.at(s.real, idx, x.real)
    np.add.at(s.imag, idx, x.imag)
    m = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.mean() > 0.35: return None
    if bad.any():
        good = np.where(~bad)[0]
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(0, len(good) - 1)]
        m[bad] = m[near]
    m = m - m.mean(0)                                       # additive static off
    sc = np.sqrt((np.abs(m) ** 2).mean()) + 1e-9
    return (m / sc).reshape(nb, 3, 57)

def to_real(c):
    """(T,3,57) c64 -> (342, T) f32"""
    v = c.reshape(len(c), -1).T
    return np.concatenate([v.real, v.imag], 0).astype(np.float32)

class Pairs(Dataset):
    def __init__(self, meta, seed):
        self.by = {k: g[["rid", "file"]].values for k, g in
                   meta.groupby("node") if len(g) > 1}
        self.keys = sorted(self.by)
        self.gt = {int(r): f"{TOK}/imu/{int(r):06d}.npy"
                   for r in meta.rid.values
                   if os.path.exists(f"{TOK}/imu/{int(r):06d}.npy")}
        self.seed = seed
    def __len__(self): return 10 ** 9
    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        for _ in range(12):
            g = self.by[self.keys[rng.integers(len(self.keys))]]
            ia, ib = rng.choice(len(g), 2, replace=False)
            (ra, fa), (rb, fb) = g[ia], g[ib]
            xa, xb = load_400(fa), load_400(fb)
            if xa is None or xb is None: continue
            s0a = rng.integers(0, len(xa) - WIN)
            s0b = rng.integers(0, len(xb) - WIN)
            Xa = to_real(xa[s0a:s0a + WIN])
            Xb = to_real(xb[s0b:s0b + WIN])
            gi = np.zeros((WIN, 5), np.float32); ok = 0.0
            gf = self.gt.get(int(ra))
            if gf:
                gg = np.asarray(np.load(gf), np.float32)
                if len(gg) >= s0a + WIN:
                    gi = gg[s0a:s0a + WIN, :5]; ok = 1.0
            return Xa, Xb, gi, ok
        z = np.zeros((CIN, WIN), np.float32)
        return z, z, np.zeros((WIN, 5), np.float32), 0.0

class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(w, w, 3, padding=d, dilation=d), nn.PReLU(),
            nn.GroupNorm(8, w), nn.Conv1d(w, w, 1), nn.PReLU())
    def forward(self, x):
        return x + self.net(x)

class Sep(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv1d(CIN, WIDTH, 16, stride=8, padding=4)
        self.tcn = nn.Sequential(*[Block(WIDTH, 2 ** (i % 4))
                                   for i in range(NBLOCKS)])
        self.mask = nn.Conv1d(WIDTH, NSLOTS * WIDTH, 1)
        self.dec = nn.ConvTranspose1d(WIDTH, CIN, 16, stride=8, padding=4)
    def forward(self, x):
        L = x.shape[-1]
        z = self.enc(x)
        h = self.tcn(z)
        m = self.mask(h).reshape(x.shape[0], NSLOTS, WIDTH, -1)
        m = torch.softmax(m, dim=1)
        y = (m * z.unsqueeze(1)).reshape(-1, WIDTH, z.shape[-1])
        y = self.dec(y).reshape(x.shape[0], NSLOTS, CIN, -1)
        return y[..., :L]

def neg_thresh_snr(ref, est, tau=TAU):
    e = ((ref - est) ** 2).sum((-2, -1))
    p = (ref ** 2).sum((-2, -1))
    return -10.0 * torch.log10(p / (e + tau * p + 1e-12) + 1e-12)

def mixit_loss(out, x1, x2):
    c1 = (out * x1.unsqueeze(1)).sum((-2, -1))
    c2 = (out * x2.unsqueeze(1)).sum((-2, -1))
    a = (c1 >= c2).float().unsqueeze(-1).unsqueeze(-1)
    e1 = (out * a).sum(1)
    e2 = (out * (1 - a)).sum(1)
    sep = (neg_thresh_snr(x1, e1) + neg_thresh_snr(x2, e2)).mean() / 2.0
    cons = neg_thresh_snr(x1 + x2, out.sum(1)).mean()
    return sep + 0.1 * cons

def slot_envs_t(y):
    """(B,N,C,T) -> (B,N,nw) window energies"""
    e = y.pow(2).sum(2)
    return torch.nn.functional.avg_pool1d(e, WINF, HOPF)

def tcorr(x, y_):
    x = x - x.mean(-1, keepdim=True); y_ = y_ - y_.mean(-1, keepdim=True)
    return (x * y_).sum(-1) / (x.norm(dim=-1) * y_.norm(dim=-1) + 1e-8)

def pit_env_loss(y, gi, ok):
    env = slot_envs_t(y)                                    # (B,N,nw)
    G = torch.nn.functional.avg_pool1d(gi.transpose(1, 2), WINF, HOPF)
    live = (ok > 0.5) & (G.var(-1).sum(1) > 1e-10)
    if live.sum() == 0: return y.new_zeros(())
    env, G = env[live], G[live]
    order = G.mean(-1).argsort(dim=1, descending=True)
    li = order[:, 0]; lj = order[:, 1]
    gi1 = torch.gather(G, 1, li[:, None, None].expand(-1, 1, G.shape[-1]))[:, 0]
    gj1 = torch.gather(G, 1, lj[:, None, None].expand(-1, 1, G.shape[-1]))[:, 0]
    Ci = tcorr(env, gi1.unsqueeze(1))
    Cj = tcorr(env, gj1.unsqueeze(1))
    best = None
    for m1 in range(NSLOTS):
        for m2 in range(NSLOTS):
            if m1 == m2: continue
            v = (Ci[:, m1] + Cj[:, m2]) / 2
            best = v if best is None else torch.maximum(best, v)
    return (1.0 - best).mean()

def main():
    os.makedirs(RUNS, exist_ok=True)
    model = Sep()
    for at in range(10):
        try:
            model = model.to(dev); break
        except RuntimeError:
            print("gpu retry", flush=True); time.sleep(60)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"dev={dev}", flush=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    tr = man[man.scene.isin([1, 2, 3])].reset_index(drop=True)
    te = man[man.scene.isin([4, 5])].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    # preload eval recordings (rooms 4/5 with GT)
    ev = []
    for r in rng.permutation(te[te.rid.isin(
            [int(x[:6]) for x in os.listdir(f"{TOK}/imu")])].values.tolist()
            if False else te.rid.values):
        rid = int(r)
        gf = f"{TOK}/imu/{rid:06d}.npy"
        if not os.path.exists(gf): continue
        row = te[te.rid == rid].iloc[0]
        x = load_400(row.file)
        if x is None: continue
        gg = np.asarray(np.load(gf), np.float32)[:len(x), :5]
        if len(gg) < WIN: continue
        ev.append((to_real(x[:min(len(x), 2048)]), gg[:min(len(x), 2048)]))
        if len(ev) >= NEVAL: break
    print(f"eval recordings: {len(ev)}", flush=True)
    dl = DataLoader(Pairs(tr, SEED), batch_size=B, num_workers=NW_,
                    pin_memory=(dev == "cuda"), persistent_workers=NW_ > 0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = -1e9
    step0 = 0
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step0 = ck["step"]; best = ck.get("best", -1e9)
        print(f"resumed step {step0}", flush=True)
    t0 = time.time()
    for step, (Xa, Xb, gi, ok) in enumerate(dl, start=step0):
        if step >= STEPS or (time.time() - t0) / 3600 > HOURS: break
        Xa, Xb = Xa.to(dev, non_blocking=True), Xb.to(dev, non_blocking=True)
        gi, ok = gi.to(dev), ok.to(dev)
        mom = Xa + Xb
        y = model(mom)
        Lm = mixit_loss(y, Xa, Xb)
        ya = model(Xa)
        Lp = pit_env_loss(ya, gi, ok)
        loss = Lm + LIMBW * Lp
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 200 == 0:
            print(f"[{step}] mixit {Lm.item():+.2f} pit {Lp.item():.3f} "
                  f"{(time.time()-t0)/3600:.2f}h", flush=True)
        if step % EVERY == 0 and step > step0:
            model.eval()
            mm, nn_ = [], []
            with torch.no_grad():
                for X, gg in ev:
                    y1 = model(torch.from_numpy(X)[None].to(dev))
                    env = slot_envs_t(y1)[0].cpu().numpy()
                    nw = env.shape[-1]
                    G = np.stack([gg[w * HOPF:w * HOPF + WINF].mean(0)
                                  for w in range(nw)])
                    o = np.argsort(-G.mean(0)); li, lj = int(o[0]), int(o[1])
                    def corr(a, b):
                        if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
                        return float(np.corrcoef(a, b)[0, 1])
                    def sc(Gm):
                        C = np.array([[corr(env[m], Gm[:, li]),
                                       corr(env[m], Gm[:, lj])]
                                      for m in range(NSLOTS)])
                        return max((C[a1, 0] + C[a2, 1]) / 2
                                   for a1 in range(NSLOTS)
                                   for a2 in range(NSLOTS) if a1 != a2)
                    mm.append(sc(G))
                    nn_.append(sc(np.roll(G, nw // 2, 0)))
            mm, nn_ = np.array(mm), np.array(nn_)
            v = float(np.median(mm))
            print(f"  GATE [{step}] rooms45: matched {v:+.3f} null "
                  f"{np.median(nn_):+.3f} win {np.mean(mm > nn_)*100:.0f}% "
                  f"(n={len(mm)}) {'(best)' if v > best else ''}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best, "actor": True},
                       f"{RUNS}/last.pt")
            if v > best:
                best = v
                torch.save({"model": model.state_dict(), "step": step,
                            "best": best, "actor": True}, f"{RUNS}/best.pt")
            model.train()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best": best, "actor": True}, f"{RUNS}/last.pt")
    print(f"DONE step {step} best gate {best:+.3f}", flush=True)

if __name__ == "__main__":
    main()
