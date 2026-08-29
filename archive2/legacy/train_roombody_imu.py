#!/usr/bin/env python3
"""Room/body Sem-MixIT on OctoNet v2 streams, with IMU-supervised routing.

PORT of train_roombody.py (PerceptAlign) onto mixit_data_v2, plus one new loss.
Five losses, one job each:
  L_mix   structured MixIT: M=4 slots, fixed groups (room_a, body_a |
          room_b, body_b); assignment search only over which group
          reconstructs which mixture (2 perms). Does the separation.
  L_dop   lag-ACF motion routing (PROXY: "25-200 ms coherence = body").
  L_imu   NEW. Same routing objective, proxy replaced by a MEASUREMENT:
          "energy that rises and falls WHEN THE WEARER MOVES is body".
          Paired form: the mixture holds TWO bodies, so each body slot is
          scored against its OWN recording's IMU.
  L_swap  room-exchangeability across the pair. Evicts recording-specific
          static (= body) energy from the room slots.
  L_sp    Sem-MixIT sparsity on BODY slots only. Blocks the all-into-body
          collapse that L_swap alone permits.

Inference readout: room = s0 + s2, body = s1 + s3.

ABLATION: IMUW=0 reproduces the no-IMU model exactly. Both arms draw from the
SAME pool (recordings with imu_ok=1), so the only difference is the loss.

NO GAIN AUGMENTATION on the pair -- a random gain on one side breaks room
exchangeability (train_roombody.py's note; v2's Pairs does apply one, so this
port deliberately drops it).

  MIXIT_DATA=~/zerdani/buffer/octonet/mixit_data_v2 \
  MIXIT_RUNS=~/zerdani/buffer/octonet/roombody_imu_runs \
  IMUW=10 HOURS=14 nohup python3 train_roombody_imu.py > rb_imu.log 2>&1 &
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from asteroid.masknn import TDConvNet
from imu_loss import imu_pair_loss

torch.manual_seed(0)
D    = os.path.expanduser(os.environ.get("MIXIT_DATA", "~/zerdani/buffer/octonet/mixit_data_v2"))
CKPT = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/roombody_imu_runs"))
os.makedirs(CKPT, exist_ok=True)
M, B, LR, WIN, dev = 4, 32, 3e-4, 400, "cuda"
STEPS = int(os.environ.get("STEPS", "60000"))
HOURS = float(os.environ.get("HOURS", "14"))
DOPW  = float(os.environ.get("DOPW",  "10.0"))
IMUW  = float(os.environ.get("IMUW",  "10.0"))
SWAPW = float(os.environ.get("SWAPW", "0.5"))
SPW   = float(os.environ.get("SPW",   "1.0"))
WARM  = int(os.environ.get("WARM", "3000"))
LAGS  = (3, 5, 10, 20)          # 30-200 ms @100 Hz (WiDetect band; 200 Hz used 5,10,20,40)
meta = None

def grab(rid):
    row = meta.iloc[rid]
    a = np.load(f"{D}/streams/{row.rid:06d}.npy", mmap_mode="r")
    e = np.load(f"{D}/imu_env/{row.rid:06d}.npy", mmap_mode="r")
    n = min(row.nsamp, len(e))
    tries = 1 if np.random.rand() < 0.2 else 4
    w = ev = None; best = -1.0
    for _ in range(tries):
        s = np.random.randint(0, max(n - WIN, 1))
        c = np.array(a[s:s + WIN])
        d = c - c.mean(0, keepdims=True)
        dyn = float((np.abs(d) ** 2).sum() / ((np.abs(c) ** 2).sum() + 1e-12))
        if dyn > best: best, w, ev = dyn, c, np.array(e[s:s + WIN], np.float32)
    x = torch.view_as_real(torch.from_numpy(w)).permute(1, 2, 0).reshape(228, WIN)
    return x, torch.from_numpy(ev)

class Pairs(Dataset):
    """Same node (a receiver never sees cross-node mixtures), different action."""
    def __init__(self, ids):
        self.by = {}
        for r in ids: self.by.setdefault(meta.iloc[r]["node"], []).append(r)
        self.nodes = [n for n, v in self.by.items() if len(v) > 1]
        w = np.array([len(self.by[n]) for n in self.nodes], float); self.w = w / w.sum()
    def __len__(self): return 10 ** 9
    def __getitem__(self, _):
        pool = self.by[self.nodes[np.random.choice(len(self.nodes), p=self.w)]]
        i = pool[np.random.randint(len(pool))]; j = i
        for _ in range(20):
            j = pool[np.random.randint(len(pool))]
            if j != i and meta.iloc[j]["act"] != meta.iloc[i]["act"]: break
        x1, e1 = grab(i); x2, e2 = grab(j)
        return x1, x2, e1, e2                       # NO gain aug (breaks L_swap)

class Sep(nn.Module):
    def __init__(self, cin=228, nf=512, M=4, L=16, S=8):
        super().__init__()
        self.M, self.cin = M, cin
        self.enc = nn.Conv1d(cin, nf, L, stride=S)
        try:
            self.masker = TDConvNet(in_chan=nf, n_src=M, out_chan=nf, n_blocks=8,
                                    n_repeats=4, bn_chan=192, hid_chan=512,
                                    skip_chan=192, mask_act="linear",
                                    causal=True, norm_type="cLN")
            print("CAUSAL separator (streaming-capable)")
        except TypeError:
            self.masker = TDConvNet(in_chan=nf, n_src=M, out_chan=nf, n_blocks=8,
                                    n_repeats=4, bn_chan=192, hid_chan=512,
                                    skip_chan=192, mask_act="linear")
            print("WARNING: asteroid lacks causal arg -> NON-causal fallback")
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x):
        T = x.shape[-1]; e = self.enc(x); m = self.masker(e)
        y = self.dec((m * e.unsqueeze(1)).flatten(0, 1))[..., :T].view(-1, self.M, self.cin, T)
        return y + (x - y.sum(1)).unsqueeze(1) / self.M

def as_c(x):
    s = x.shape; xr = x.reshape(*s[:-2], 114, 2, s[-1])
    return torch.complex(xr[..., 0, :], xr[..., 1, :])

def nsnrc(est, ref):
    err = (ref - est).abs().pow(2).sum((-2, -1)) + 1e-8
    p = ref.abs().pow(2).sum((-2, -1)) + 1e-8
    return -10 * torch.log10(p / err)

def motion_stat(c):
    d = c - c.mean(-1, keepdim=True); a = 0.
    for L in LAGS:
        a = a + (d[..., :-L] * d[..., L:].conj()).mean(-1).abs().mean(-1)
    return a / len(LAGS)

def cproj(room, resid):
    num = (resid * room.conj()).sum((-2, -1))
    den = room.abs().pow(2).sum((-2, -1)).clamp_min(1e-8)
    return (num / den)[..., None, None] * room

def losses(x1, x2, e1, e2, model):
    s = model(x1 + x2)
    cs, cx1, cx2 = as_c(s), as_c(x1), as_c(x2)
    r1, r2 = cs[:, 0] + cs[:, 1], cs[:, 2] + cs[:, 3]
    l0 = nsnrc(r1, cx1) + nsnrc(r2, cx2)
    l1 = nsnrc(r1, cx2) + nsnrc(r2, cx1)
    a = (l1 < l0)
    L_mix = torch.where(a, l1, l0).mean() / 2
    A = motion_stat(cs)
    L_dop = ((A[:, 0] + A[:, 2]) / (A.sum(1) + 1e-12)).mean()
    L_imu = imu_pair_loss(cs, e1, e2, a) if IMUW > 0 else cs.real.new_zeros(())
    tA = torch.where(a[:, None, None], cx2, cx1)
    tB = torch.where(a[:, None, None], cx1, cx2)
    estA = cproj(cs[:, 2], tA - cs[:, 1]) + cs[:, 1]
    estB = cproj(cs[:, 0], tB - cs[:, 3]) + cs[:, 3]
    L_swap = (nsnrc(estA, tA) + nsnrc(estB, tB)).mean() / 2
    L_sp = cs[:, (1, 3)].abs().pow(2).mean((-2, -1)).clamp_min(1e-12).sqrt().mean()
    return L_mix, L_dop, L_imu, L_swap, L_sp

def main():
    global meta
    m = pd.read_csv(f"{D}/meta.csv")
    im = pd.read_csv(f"{D}/imu_meta.csv")
    meta = m.merge(im[["rid", "imu_ok", "node", "act"]], on="rid", how="inner")
    meta = meta[meta.imu_ok == 1].reset_index(drop=True)
    meta["rid"] = meta["rid"].astype(int)
    print(f"{len(meta)} recordings with IMU | nodes {sorted(meta.node.unique())} "
          f"| {meta.act.nunique()} activities | IMUW={IMUW} DOPW={DOPW}")
    ids = np.arange(len(meta)); rng = np.random.default_rng(0); rng.shuffle(ids)
    nv = max(int(0.05 * len(ids)), 8); va, tr = ids[:nv], ids[nv:]
    dl = DataLoader(Pairs(tr), batch_size=B, num_workers=8, pin_memory=True, persistent_workers=True)
    vl = DataLoader(Pairs(va), batch_size=B, num_workers=2)
    model = Sep(M=M).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    step0 = 0
    if os.path.exists(f"{CKPT}/last.pt"):
        ck = torch.load(f"{CKPT}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); step0 = ck["step"]
        print(f"resumed from step {step0}")
    t0, best = time.time(), math.inf
    vit = iter(vl)
    for step, (x1, x2, e1, e2) in enumerate(dl, start=step0):
        if step >= STEPS or (time.time() - t0) / 3600 > HOURS: break
        x1, x2, e1, e2 = [t.to(dev, non_blocking=True) for t in (x1, x2, e1, e2)]
        Lm, Ld, Li, Lw, Ls = losses(x1, x2, e1, e2, model)
        w = min(1.0, max(step - 0, 0) / max(WARM, 1))
        loss = Lm + w * (DOPW * Ld + IMUW * Li + SWAPW * Lw + SPW * Ls)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[{step}] loss {loss.item():.4f} | mix {Lm.item():.4f} dop {Ld.item():.4f} "
                  f"imu {Li.item():.4f} swap {Lw.item():.4f} sp {Ls.item():.4f} "
                  f"| lr {sch.get_last_lr()[0]:.2e} | {(time.time()-t0)/3600:.2f}h", flush=True)
        if step % 2000 == 0 and step > step0:
            model.eval(); acc = []
            with torch.no_grad():
                for _ in range(20):
                    try: p, q, f1, f2 = next(vit)
                    except StopIteration: vit = iter(vl); p, q, f1, f2 = next(vit)
                    p, q, f1, f2 = [t.to(dev) for t in (p, q, f1, f2)]
                    a1, a2, a3, a4, a5 = losses(p, q, f1, f2, model)
                    acc.append((a1 + DOPW*a2 + IMUW*a3 + SWAPW*a4 + SPW*a5).item())
            v = float(np.mean(acc)); model.train()
            print(f"  VAL {v:.4f} {'(best)' if v < best else ''}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sch": sch.state_dict(), "step": step}, f"{CKPT}/last.pt")
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step, "val": v}, f"{CKPT}/best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step}, f"{CKPT}/last.pt")
    print(f"DONE step {step}, {(time.time()-t0)/3600:.2f} h, best val {best:.4f}")

if __name__ == "__main__":
    main()
