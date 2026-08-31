#!/usr/bin/env python3
"""Probe 48: PA-trained limbtok12 ZERO-SHOT on WiMANS multi-person tokens.

What does the PA limb clusterer do with multiple PEOPLE?  Three readings:
  1. Slot usage: energy share per slot vs n_users (does slot structure
     react to occupancy? emergent room slot s3 was the PA finding).
  2. Count from slot shares alone: ridge 6-way (8-dim input) — pooled and
     cross-env, vs probe-45 scalar floor and probe-46 transformer 74.8%.
  3. Separation (probe-47 protocol, slots instead of k-means): top-n
     energy slots as person clusters -> activity classifier -> multiset
     score vs rand-split / whole-topn / shuffle null.  Probe-47 refs at
     n=2: kmeans-SPAT 0.233, whole 0.206, rand 0.178, null 0.135.

  python3 diagnostics/48_wimans_limbtok.py
"""
import os, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
CKPT = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/archive/limbtok12_best_step43k_gate71.pt"))
ASTEPS = int(os.environ.get("ASTEPS", "3000"))
TESTENV = os.environ.get("TESTENV", "")
B = int(os.environ.get("B", "24"))
LR = float(os.environ.get("LR", "5e-4"))
MAXT = int(os.environ.get("MAXT", "768"))
M, D, NL = 8, 256, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        return torch.softmax(self.head(h), -1)

def load_all():
    an = pd.read_csv(f"{OUT}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    vocab = sorted({str(getattr(r, f"user_{k}_activity")).strip()
                    for r in an.itertuples() for k in range(1, 6)
                    if isinstance(getattr(r, f"user_{k}_activity"), str)})
    v2i = {v: i for i, v in enumerate(vocab)}
    items = []
    for r in an.itertuples():
        f = f"{OUT}/tokens/{r.label}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        t, nw = z["toks"], int(z["nw"])
        if len(t) < 8: continue
        if len(t) > MAXT:
            t = t[np.argsort(-t[:, 4])[:MAXT]]
        n = int(r.number_of_users)
        acts = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_activity")
            if isinstance(v, str) and v.strip() in v2i:
                acts.append(v2i[v.strip()])
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        # limbtok12 feature order (train_limbtok.feats):
        Xl = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]),
                   np.sin(t[:, 3]), np.cos(t[:, 3]),
                   t[:, 1] / 150.0, t[:, 0] / max(nw - 1, 1),
                   zle].astype(np.float32)
        # probe-46/47 activity-classifier feature order:
        Xa = np.stack([t[:, 0] / nw, t[:, 1] / 150.0,
                       np.sin(t[:, 2]), np.cos(t[:, 2]),
                       np.sin(t[:, 3]), np.cos(t[:, 3]),
                       le, np.full(len(t), float(r.wifi_band == 5.0))],
                      1).astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        items.append(dict(Xl=Xl, X=Xa, e=e, n=n, env=r.environment,
                          acts=acts))
    lez = np.concatenate([it["X"][:, 6] for it in items])
    mu, sd = float(lez.mean()), float(lez.std()) + 1e-9
    for it in items: it["X"][:, 6] = (it["X"][:, 6] - mu) / sd
    return items, vocab

def slot_assign(sep, items):
    with torch.no_grad():
        for i0 in range(0, len(items), 32):
            its = items[i0:i0 + 32]
            n = max(len(it["Xl"]) for it in its)
            X = torch.zeros(len(its), n, 7)
            mask = torch.ones(len(its), n, dtype=torch.bool)
            for k, it in enumerate(its):
                X[k, :len(it["Xl"])] = torch.from_numpy(it["Xl"])
                mask[k, :len(it["Xl"])] = False
            a = sep(X.to(dev), mask.to(dev)).cpu().numpy()
            for k, it in enumerate(its):
                it["slot"] = a[k, :len(it["Xl"])].argmax(1)
                sh = np.zeros(M)
                for j in range(M):
                    sh[j] = it["e"][it["slot"] == j].sum()
                it["share"] = sh / (sh.sum() + 1e-12)

class TokCls(nn.Module):
    def __init__(self, nc, H=128):
        super().__init__()
        self.inp = nn.Linear(8, H)
        lay = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.out = nn.Linear(H, nc)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        h = (h * (~mask)[:, :, None]).sum(1) / (~mask).sum(1, keepdim=True)
        return self.out(h)

def batch(items, ixb):
    its = [items[i] for i in ixb]
    n = max(len(it["X"]) for it in its)
    X = torch.zeros(len(its), n, 8)
    mask = torch.ones(len(its), n, dtype=torch.bool)
    for k, it in enumerate(its):
        X[k, :len(it["X"])] = torch.from_numpy(it["X"])
        mask[k, :len(it["X"])] = False
    return X.to(dev), mask.to(dev)

def train(items, labels, nc, steps, tag):
    net = TokCls(nc).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(47)
    bycls = {}
    for i, y in enumerate(labels): bycls.setdefault(y, []).append(i)
    keys = sorted(bycls)
    t0 = time.time()
    for step in range(steps):
        ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
               for c in rng.integers(0, len(keys), B)]
        X, mask = batch(items, ixb)
        y = torch.tensor([labels[i] for i in ixb]).to(dev)
        loss = nn.functional.cross_entropy(net(X, mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  [{tag} {step}] CE {loss.item():.3f} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
    net.eval()
    return net

def logits(net, items, ix):
    P = []
    with torch.no_grad():
        for i0 in range(0, len(ix), 32):
            X, mask = batch(items, ix[i0:i0 + 32])
            P.append(net(X, mask).cpu().numpy())
    return np.concatenate(P)

def setscore(pred, gt):
    a, b_ = Counter(pred), Counter(gt)
    return sum((a & b_).values()) / max(len(gt), 1)

def ridge_acc(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(6)[ytr]
    Wm = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                         Xtr.T @ Y)
    return float(((Xte @ Wm).argmax(1) == yte).mean())

def main():
    items, vocab = load_all()
    print(f"{len(items)} samples", flush=True)
    sep = SetSep().to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    sep.load_state_dict(ck["model"] if "model" in ck else ck)
    sep.eval()
    print(f"limbtok12 loaded ({CKPT.split('/')[-1]})", flush=True)
    slot_assign(sep, items)

    print("\n=== 1. slot energy share vs n_users (rows: n, cols: s0..s7)",
          flush=True)
    for n in range(6):
        sel = [it for it in items if it["n"] == n]
        sh = np.mean([it["share"] for it in sel], 0)
        na = np.mean([(it["share"] > 0.10).sum() for it in sel])
        print(f"  n={n} (N={len(sel)}): " +
              " ".join(f"{v:.2f}" for v in sh) +
              f"   active(>10%) {na:.2f}", flush=True)

    print("\n=== 2. count from 8-dim slot shares (ridge)", flush=True)
    env = np.array([it["env"] for it in items])
    yu = np.array([it["n"] for it in items])
    Xs = np.array([it["share"] for it in items])
    rng = np.random.default_rng(48)
    ix = rng.permutation(len(items))
    h = int(len(ix) * 0.8)
    acc = ridge_acc(Xs[ix[:h]], yu[ix[:h]], Xs[ix[h:]], yu[ix[h:]])
    print(f"  pooled heldout: {acc*100:.0f}%  (probe-45 scalars ~34%, "
          f"probe-46 transformer 74.8%, majority 30%)", flush=True)
    for e_ in np.unique(env):
        acc = ridge_acc(Xs[env != e_], yu[env != e_],
                        Xs[env == e_], yu[env == e_])
        print(f"  cross ->{e_:12s}: {acc*100:.0f}%", flush=True)

    print("\n=== 3. separation: slots as person clusters (probe-47 protocol)",
          flush=True)
    one = [i for i, it in enumerate(items)
           if it["n"] == 1 and len(it["acts"]) == 1]
    rng = np.random.default_rng(47)
    ixp = rng.permutation(one)
    hp = int(len(ixp) * 0.9)
    sub = [items[i] for i in ixp[:hp]]
    net = train(sub, [it["acts"][0] for it in sub], len(vocab), ASTEPS, "act")
    groups = [("ALL", None)] if not TESTENV else \
        [("TRAIN-ENVS", False), (f"TESTROOM {TESTENV}", True)]
    for gname, iste in groups:
      print(f"  --- {gname}", flush=True)
      for nU in (2, 3, 4, 5):
        mi = [i for i, it in enumerate(items)
              if it["n"] == nU and len(it["acts"]) == nU
              and (iste is None or (it["env"] == TESTENV) == iste)]
        if len(mi) < 30: continue
        sc_sl, dis_sl, gts = [], [], []
        for i in mi:
            it = items[i]
            order = np.argsort(-it["share"])
            cl = [j for j in order if (it["slot"] == j).sum() >= 8][:nU]
            if len(cl) < nU: continue
            preds = [int(logits(net, [dict(X=it["X"][it["slot"] == j])],
                                [0]).argmax()) for j in cl]
            sc_sl.append(setscore(preds, it["acts"]))
            dis_sl.append(len(set(preds)) > 1)
            gts.append(it["acts"])
        null = []
        for _ in range(100):
            pm = rng.permutation(len(gts))
            null.append(np.mean([setscore(gts[k], gts[pm[k]])
                                 for k in range(len(gts))]))
        print(f"  n={nU} (N={len(sc_sl)}): slots {np.mean(sc_sl):.3f}"
              f"(d{np.mean(dis_sl):.2f})  null {np.mean(null):.3f}"
              + ("   [p47 refs: kmeans 0.233 whole 0.206 rand 0.178]"
                 if nU == 2 else ""), flush=True)
    print("probe 48 done", flush=True)

if __name__ == "__main__":
    main()
