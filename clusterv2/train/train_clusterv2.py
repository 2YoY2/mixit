#!/usr/bin/env python3
"""Clusterer v2: v1 objectives + PURITY as a first-class loss.

Same identity-free SetSep as v1 (M slots, per-token softmax).  Losses:
  base (per dataset, ported from v1):
    pa      PIT-envelope (top-2 residualized limb envelopes, best ordered
            slot pair) [+ MixIT-origin via bench mixes, see ownership]
    wimans  PIT-activity (Hungarian slot<->user CE, co-trained head)
  margin (new): exclusivity trained, not just measured —
    pa      penalize the wrong-permutation pair correlation
    wimans  hinge on (matched CE vs cyclic mis-assignment CE)
  ownership (new, the purity loss): bench mixes with per-token owner GT;
    each slot gets an origin by energy-majority (detached); tokens pay
    BCE for sitting in a wrong-origin slot; plus slot-origin entropy
    weighted by slot energy (a slot mixing origins is directly charged).

  DATASET=pa    M=8 HOURS=3 python3 clusterv2/train/train_clusterv2.py
  DATASET=wimans M=8 HOURS=2 python3 clusterv2/train/train_clusterv2.py
"""
import os, time, math, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

DATASET = os.environ.get("DATASET", "pa")
BENCHD = os.path.expanduser(os.environ.get(
    "BENCHD", "~/zerdani/buffer/clusterv2/bench"))
OUTD = os.path.expanduser(os.environ.get(
    "OUT", f"~/zerdani/buffer/clusterv2/runs/clusterv2.{DATASET}.r1"))
TOKPA = os.path.expanduser("~/zerdani/buffer/cluster/tok/pa-v1")
TOKWM = os.path.expanduser("~/zerdani/buffer/cluster/tok/wimans-v1")
M = int(os.environ.get("M", "8"))
D = int(os.environ.get("DIM", "256"))
NL = int(os.environ.get("LAYERS", "6"))
LR = float(os.environ.get("LR", "1e-4"))
WARM = int(os.environ.get("WARM", "1000"))
STEPS = int(os.environ.get("STEPS", "60000"))
HOURS = float(os.environ.get("HOURS", "3"))
EVERY = int(os.environ.get("EVERY", "1000"))
BP = int(os.environ.get("BP", "8"))       # base-loss recordings per step
BO = int(os.environ.get("BO", "6"))       # bench mixes per step
OWNW = float(os.environ.get("OWNW", "1.0"))
PURW = float(os.environ.get("PURW", "0.5"))
MARGW = float(os.environ.get("MARGW", "0.3"))
MAXT = int(os.environ.get("MAXT", "1400"))
SEED = int(os.environ.get("SEED", "0"))
HOPF, WINF = 128, 256
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x):
        h = self.enc(self.inp(x))
        return torch.softmax(self.head(h), -1), h

class ActHead(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(),
                                 nn.Linear(D, nc))
    def forward(self, s): return self.net(s)

def feats(t, nw):
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    return X, (10.0 ** le).astype(np.float32), t[:, 0].astype(np.int64)

# ---------------------------------------------------------------- data
def load_bench(split):
    d = f"{BENCHD}/{DATASET}_{split}"
    fs = sorted(glob.glob(f"{d}/*.npz"))
    out = []
    for f in fs:
        z = np.load(f)
        t, own, nw = z["toks"], z["own"], int(z["nw"])
        if len(t) < 32 or (own >= 0).sum() < 16: continue
        if len(t) > MAXT:
            k = np.argsort(-t[:, 4])[:MAXT]
            t, own = t[k], own[k]
        out.append((t, own.astype(np.int64), nw))
    print(f"bench {split}: {len(out)} mixes", flush=True)
    return out

if DATASET == "pa":
    man = pd.read_csv(f"{TOKPA}/manifest.csv")
    tr = man[man.split == "train"]
    GTD = f"{TOKPA}/limbenv"
    gtset = {int(f[:6]) for f in os.listdir(GTD)}
    PIT_IDS = [int(r) for r in tr.rid.values if int(r) in gtset]
    TE_IDS = [int(r) for r in man[man.split == "test"].rid.values
              if int(r) in gtset]
    rng.shuffle(PIT_IDS); rng.shuffle(TE_IDS)
    TE_IDS = TE_IDS[:200]
    def residual_pool(rid, nw):
        gi = np.asarray(np.load(f"{GTD}/{rid:06d}.npy"), np.float32)
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        n = min(nw, (len(g2) - WINF) // HOPF + 1)
        return np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                         for w in range(n)])
    def load_rec(rid):
        f = f"{TOKPA}/tokens/{rid:06d}.npz"
        if not os.path.exists(f): return None
        z = np.load(f); t = z["toks"]; nw = int(z["nw"])
        if len(t) < 32: return None
        if len(t) > MAXT: t = t[np.argsort(-t[:, 4])[:MAXT]]
        return t, nw
else:
    an = pd.read_csv(f"{TOKWM}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    VOCAB = sorted({str(getattr(r, f"user_{k}_activity")).strip()
                    for r in an.itertuples() for k in range(1, 6)
                    if isinstance(getattr(r, f"user_{k}_activity"), str)})
    V2I = {v: i for i, v in enumerate(VOCAB)}
    ITEMS = []
    for r in an.itertuples():
        n = int(r.number_of_users)
        if n < 1: continue
        f = f"{TOKWM}/tokens/{r.label}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); t = z["toks"]; nw = int(z["nw"])
        if len(t) < 8: continue
        if len(t) > 768: t = t[np.argsort(-t[:, 4])[:768]]
        acts = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_activity")
            if isinstance(v, str) and v.strip() in V2I:
                acts.append(V2I[v.strip()])
        if len(acts) != n: continue
        ITEMS.append((t, nw, acts, r.environment))
    TRENVS = {"classroom", "meeting_room"}
    IN_TR = [i for i, it in enumerate(ITEMS) if it[3] in TRENVS]
    XROOM = [i for i, it in enumerate(ITEMS) if it[3] not in TRENVS]
    rng.shuffle(IN_TR)
    HO = IN_TR[:int(len(IN_TR) * 0.1)]
    TRI = IN_TR[int(len(IN_TR) * 0.1):]
    print(f"wimans items {len(ITEMS)} train {len(TRI)} ho {len(HO)}",
          flush=True)

# ---------------------------------------------------------------- losses
def slot_envs(a, widx, e, nw):
    env = a.new_zeros(M, nw)
    env.index_add_(1, widx, (a * e[:, None]).T)
    return env

def tcorr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

def pit_env_loss(model, rid):
    r = load_rec(rid)
    if r is None: return None
    t, nw = r
    G = residual_pool(rid, nw)
    if len(G) < 8: return None
    X, e, widx = feats(t, nw)
    a, _ = model(torch.from_numpy(X)[None].to(dev))
    a = a[0]
    env = slot_envs(a, torch.from_numpy(widx).to(dev),
                    torch.from_numpy(e).to(dev), nw)
    T = len(G)
    order = np.argsort(-G.mean(0))
    li, lj = int(order[0]), int(order[1])
    Gt = torch.from_numpy(G).to(dev)
    C = torch.stack([torch.stack([tcorr(env[m_, :T], Gt[:, li]),
                                  tcorr(env[m_, :T], Gt[:, lj])])
                     for m_ in range(M)])
    best, bwrong = None, None
    for m1 in range(M):
        for m2 in range(M):
            if m1 == m2: continue
            v = (C[m1, 0] + C[m2, 1]) / 2
            if best is None or v > best:
                best = v
                bwrong = (C[m1, 1] + C[m2, 0]) / 2
    loss = 1.0 - best
    if MARGW > 0:
        loss = loss + MARGW * torch.relu(bwrong)     # exclusivity margin
    return loss

def slot_sigs(a, h, e):
    w = a * e[:, None]
    s = w.T @ h
    return s / (w.sum(0)[:, None] + 1e-6)

def pit_act_loss(model, head, item):
    t, nw, acts, _ = item
    X, e, widx = feats(t, nw)
    a, h = model(torch.from_numpy(X)[None].to(dev))
    a, h = a[0], h[0]
    sig = slot_sigs(a, h, torch.from_numpy(e).to(dev))
    lg = head(sig)
    y = torch.tensor(acts, device=dev)
    ce = torch.stack([nn.functional.cross_entropy(
        lg, y[u].expand(M), reduction="none")
        for u in range(len(acts))], 1)
    r_, c_ = linear_sum_assignment(ce.detach().cpu().numpy())
    matched = ce[r_, c_].mean()
    loss = matched
    if MARGW > 0 and len(acts) > 1:
        rr = np.roll(r_, 1)
        wrong = ce[rr, c_].mean()
        loss = loss + MARGW * torch.relu(matched - wrong + 0.5)
    return loss

def ownership_loss(model, mix):
    """the purity loss: token-BCE vs slot-majority origin + slot-origin
    entropy weighted by slot energy (valid-owner tokens only)."""
    t, own, nw = mix
    X, e, widx = feats(t, nw)
    a, _ = model(torch.from_numpy(X)[None].to(dev))
    a = a[0]
    ev = torch.from_numpy(e).to(dev)
    ow = torch.from_numpy(own).to(dev)
    val = ow >= 0
    if val.sum() < 16: return None
    av, ev_, owv = a[val], ev[val], ow[val].float()
    w_ = av * ev_[:, None]                       # (n, M)
    num = (w_ * owv[:, None]).sum(0)
    den = w_.sum(0) + 1e-8
    q = num / den                                # slot origin-1 fraction
    ohat = (q > 0.5).float().detach()
    s_i = (av * ohat[None, :]).sum(1).clamp(1e-6, 1 - 1e-6)
    bce = -(owv * torch.log(s_i) + (1 - owv) * torch.log(1 - s_i))
    l_tok = (bce * ev_).sum() / ev_.sum()
    qc = q.clamp(1e-6, 1 - 1e-6)
    ent = -(qc * torch.log(qc) + (1 - qc) * torch.log(1 - qc))
    wm = den / den.sum()
    l_pure = (wm * ent).sum()
    return l_tok + PURW * l_pure

# ---------------------------------------------------------------- eval
def quick_eval(model, head=None):
    model.eval()
    with torch.no_grad():
        if DATASET == "pa":
            ms, ws = [], []
            for rid in TE_IDS[:120]:
                r = load_rec(rid)
                if r is None: continue
                t, nw = r
                G = residual_pool(rid, nw)
                if len(G) < 8: continue
                X, e, widx = feats(t, nw)
                a, _ = model(torch.from_numpy(X)[None].to(dev))
                env = slot_envs(a[0], torch.from_numpy(widx).to(dev),
                                torch.from_numpy(e).to(dev), nw
                                ).cpu().numpy()
                T = len(G)
                order = np.argsort(-G.mean(0))
                li, lj = int(order[0]), int(order[1])
                def cc(x, y):
                    if x.std() < 1e-9 or y.std() < 1e-9: return 0.0
                    return float(np.corrcoef(x, y)[0, 1])
                C = np.array([[cc(env[m_, :T], G[:, li]),
                               cc(env[m_, :T], G[:, lj])]
                              for m_ in range(M)])
                best, bw = -9, 0
                for m1 in range(M):
                    for m2 in range(M):
                        if m1 == m2: continue
                        v = (C[m1, 0] + C[m2, 1]) / 2
                        if v > best: best, bw = v, (C[m1, 1] + C[m2, 0]) / 2
                ms.append(best); ws.append(bw)
            model.train()
            return float(np.median(ms)), float(np.median(ws))
        else:
            accs, wr = [], []
            for i in HO[:200]:
                t, nw, acts, _ = ITEMS[i]
                X, e, widx = feats(t, nw)
                a, h = model(torch.from_numpy(X)[None].to(dev))
                sig = slot_sigs(a[0], h[0], torch.from_numpy(e).to(dev))
                lg = head(sig)
                y = torch.tensor(acts, device=dev)
                ce = torch.stack([nn.functional.cross_entropy(
                    lg, y[u].expand(M), reduction="none")
                    for u in range(len(acts))], 1).cpu().numpy()
                r_, c_ = linear_sum_assignment(ce)
                pred = lg.argmax(1).cpu().numpy()
                accs.append(float((pred[r_] == y.cpu().numpy()[c_]).mean()))
                if len(acts) > 1:
                    rr = np.roll(r_, 1)
                    wr.append(float((pred[rr] == y.cpu().numpy()[c_]
                                     ).mean()))
            model.train()
            return float(np.mean(accs)), float(np.mean(wr) if wr else 0)

def bench_purity(model, val):
    """energy-weighted mean slot purity on bench val (fixed M)."""
    model.eval()
    ps = []
    with torch.no_grad():
        for t, own, nw in val[:150]:
            X, e, widx = feats(t, nw)
            a, _ = model(torch.from_numpy(X)[None].to(dev))
            hard = a[0].argmax(1).cpu().numpy()
            val_m = own >= 0
            tot, pure = 0.0, 0.0
            for m_ in range(M):
                sel = (hard == m_) & val_m
                if not sel.any(): continue
                e0 = e[sel][own[sel] == 0].sum()
                e1 = e[sel][own[sel] == 1].sum()
                tot += e0 + e1
                pure += max(e0, e1)
            if tot > 0: ps.append(pure / tot)
    model.train()
    return float(np.mean(ps))

# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTD, exist_ok=True)
    bt = load_bench("train")
    bv = load_bench("val")
    model = SetSep().to(dev)
    head = ActHead(len(VOCAB)).to(dev) if DATASET == "wimans" else None
    params = list(model.parameters()) + (
        list(head.parameters()) if head else [])
    print(f"clusterv2 {DATASET} M={M} params "
          f"{sum(p.numel() for p in params)/1e6:.1f}M OWNW={OWNW} "
          f"PURW={PURW} MARGW={MARGW}", flush=True)
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARM) *
        0.5 * (1.0 + math.cos(math.pi * s / STEPS)))
    best = -1e9
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        losses = []
        for _ in range(BP):
            if DATASET == "pa":
                l = pit_env_loss(model, PIT_IDS[rng.integers(len(PIT_IDS))])
            else:
                l = pit_act_loss(model, head,
                                 ITEMS[TRI[rng.integers(len(TRI))]])
            if l is not None: losses.append(l)
        for _ in range(BO):
            if not bt: break
            l = ownership_loss(model, bt[rng.integers(len(bt))])
            if l is not None: losses.append(OWNW * l)
        if not losses: continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % EVERY == 0:
            m_, w_ = quick_eval(model, head)
            pu = bench_purity(model, bv)
            score = m_ - max(w_, 0) + pu
            ck = {"model": model.state_dict(),
                  "cfg": {"M": M, "D": D, "NL": NL, "dataset": DATASET},
                  "step": step}
            if head: ck["head"] = head.state_dict()
            torch.save(ck, f"{OUTD}/last.pt")
            if score > best:
                best = score
                torch.save(ck, f"{OUTD}/best.pt")
            print(f"[{step}] loss {loss.item():.3f}  matched {m_:.3f} "
                  f"wrong {w_:.3f}  PURITY {pu:.3f}  "
                  f"{(time.time()-t0)/3600:.2f}h", flush=True)
    torch.save({"model": model.state_dict(),
                "cfg": {"M": M, "D": D, "NL": NL, "dataset": DATASET},
                "step": step,
                **({"head": head.state_dict()} if head else {})},
               f"{OUTD}/last.pt")
    print("done", flush=True)

if __name__ == "__main__":
    main()
