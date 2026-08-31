#!/usr/bin/env python3
"""Probe 47: do WiMANS tokens GROUP BY PERSON?  (activity-multiset test)

Users act independently (per-user activity labels).  If tokens of the same
person are closer than tokens of different people, then k-means clusters of
a multi-user sample should recover the per-person ACTIVITY multiset.

  1. Train activity classifier on 1-user samples (all envs pooled — activity
     is a TF signature; rooms only shift phi per the room-map law).
  2. On n>=2-user samples: k-means tokens into n clusters under three
     feature sets — SPAT (phi,psi), TF (time,freq), FULL — classify each
     cluster, score predicted activity multiset vs GT.  Controls: random
     token split (same classifier), whole-sample top-n logits, shuffle null.

  ASTEPS=3000 python3 diagnostics/47_wimans_person_tokens.py
"""
import os, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
ASTEPS = int(os.environ.get("ASTEPS", "3000"))
B = int(os.environ.get("B", "24"))
LR = float(os.environ.get("LR", "5e-4"))
MAXT = int(os.environ.get("MAXT", "768"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

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
        t, nw = z["toks"], float(z["nw"])
        if len(t) < 8: continue
        if len(t) > MAXT:
            t = t[np.argsort(-t[:, 4])[:MAXT]]
        n = int(r.number_of_users)
        acts = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_activity")
            if isinstance(v, str) and v.strip() in v2i:
                acts.append(v2i[v.strip()])
        X = np.stack([t[:, 0] / nw, t[:, 1] / 150.0,
                      np.sin(t[:, 2]), np.cos(t[:, 2]),
                      np.sin(t[:, 3]), np.cos(t[:, 3]),
                      t[:, 4], np.full(len(t), float(r.wifi_band == 5.0))],
                     1).astype(np.float32)
        items.append(dict(X=X, n=n, env=r.environment, acts=acts))
    le = np.concatenate([it["X"][:, 6] for it in items])
    mu, sd = float(le.mean()), float(le.std()) + 1e-9
    for it in items: it["X"][:, 6] = (it["X"][:, 6] - mu) / sd
    return items, vocab

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

def kmeans(X, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = Xs[rng.choice(len(Xs), k, replace=False)]
    lab = np.zeros(len(Xs), int)
    for _ in range(iters):
        d = ((Xs[:, None] - C[None]) ** 2).sum(-1)
        nl = d.argmin(1)
        if (nl == lab).all(): break
        lab = nl
        for j in range(k):
            if (lab == j).any(): C[j] = Xs[lab == j].mean(0)
    return lab

def setscore(pred, gt):
    a, b_ = Counter(pred), Counter(gt)
    return sum((a & b_).values()) / max(len(gt), 1)

ARMS = {"SPAT": [2, 3, 4, 5], "TF": [0, 1], "FULL": [0, 1, 2, 3, 4, 5]}

def main():
    items, vocab = load_all()
    print(f"{len(items)} samples, activities: {vocab}", flush=True)
    multi = [it for it in items if it["n"] >= 2 and len(it["acts"]) == it["n"]]
    het = np.mean([len(set(it["acts"])) > 1 for it in multi])
    print(f"multi-user samples {len(multi)}, distinct-activity fraction "
          f"{het:.2f} (do they act independently: yes if <1 and >0)",
          flush=True)

    one = [i for i, it in enumerate(items)
           if it["n"] == 1 and len(it["acts"]) == 1]
    rng = np.random.default_rng(47)
    ix = rng.permutation(one)
    h = int(len(ix) * 0.9)
    tr, ho = list(ix[:h]), list(ix[h:])
    sub = [items[i] for i in tr]
    net = train(sub, [it["acts"][0] for it in sub], len(vocab), ASTEPS, "act")
    lg = logits(net, items, ho)
    yh = np.array([items[i]["acts"][0] for i in ho])
    ph = lg.argmax(1)
    print(f"1-user activity acc (heldout, all envs) {np.mean(ph == yh):.3f} "
          f"(chance {1/len(vocab):.3f}, n={len(ho)})", flush=True)
    for k, v in enumerate(vocab):
        m = yh == k
        if m.any():
            print(f"    {v:12s} {np.mean(ph[m] == k):.2f} (n={m.sum()})",
                  flush=True)

    print("\n=== token clusters vs per-person activity multiset", flush=True)
    for nU in (2, 3, 4, 5):
        mi = [i for i, it in enumerate(items)
              if it["n"] == nU and len(it["acts"]) == nU]
        if len(mi) < 30: continue
        res = {a: [] for a in ARMS}
        dis = {a: [] for a in ARMS}
        sc_rs, sc_top, gts = [], [], []
        for i in mi:
            it = items[i]
            full = logits(net, [it], [0])[0]
            sc_top.append(setscore(list(np.argsort(-full)[:nU]), it["acts"]))
            rl = rng.permutation(len(it["X"])) % nU
            rp = []
            okr = True
            for a, cols in ARMS.items():
                lab = kmeans(it["X"][:, cols], nU, seed=i)
                if min((lab == j).sum() for j in range(nU)) < 8:
                    res[a].append(None); dis[a].append(None); continue
                preds = [int(logits(net, [dict(X=it["X"][lab == j])],
                                    [0]).argmax()) for j in range(nU)]
                res[a].append(setscore(preds, it["acts"]))
                dis[a].append(len(set(preds)) > 1)
            if min((rl == j).sum() for j in range(nU)) >= 8:
                rp = [int(logits(net, [dict(X=it["X"][rl == j])],
                                 [0]).argmax()) for j in range(nU)]
                sc_rs.append(setscore(rp, it["acts"]))
            gts.append(it["acts"])
        null = []
        for _ in range(100):
            pm = rng.permutation(len(gts))
            null.append(np.mean([setscore(gts[k], gts[pm[k]])
                                 for k in range(len(gts))]))
        line = "  ".join(
            f"{a} {np.mean([v for v in res[a] if v is not None]):.3f}"
            f"(d{np.mean([v for v in dis[a] if v is not None]):.2f})"
            for a in ARMS)
        print(f"  n={nU} (N={len(mi)}): {line}  rand-split "
              f"{np.mean(sc_rs):.3f}  whole-top{nU} {np.mean(sc_top):.3f}  "
              f"null {np.mean(null):.3f}", flush=True)
    print("probe 47 done", flush=True)

if __name__ == "__main__":
    main()
