#!/usr/bin/env python3
"""Probe 46: WiMANS at TOKEN level — (A) people-count readout, (B) separation.

A. COUNT: token-set transformer (act_from_tokens arch) -> 6-way n_users.
   Pooled 80/20 heldout (is count info in the tokens at all?) and
   cross-env train-2-test-1 (does it transfer?).  Probe 45 floor to beat:
   5-scalar ridge ~ majority (27-42% vs 32%).

B. SEPARATION: users stand at fixed canonical locations a-e.  Train a 5-way
   LOCATION classifier on 1-user samples only (per env).  On n>=2-user
   samples: k-means the tokens into n clusters on the SPATIAL axes
   (sin/cos phi, psi), classify each cluster as a token set, score the
   predicted location multiset vs GT.  Controls: random token split (same
   classifier — does clustering add anything?), whole-sample top-n logits,
   and a shuffle null (GT sets permuted within (env, n)).

  CSTEPS=5000 XSTEPS=3000 LSTEPS=2500 python3 diagnostics/46_wimans_tokencount.py
"""
import os, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
CSTEPS = int(os.environ.get("CSTEPS", "5000"))
XSTEPS = int(os.environ.get("XSTEPS", "3000"))
LSTEPS = int(os.environ.get("LSTEPS", "2500"))
B = int(os.environ.get("B", "24"))
LR = float(os.environ.get("LR", "5e-4"))
MAXT = int(os.environ.get("MAXT", "768"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
LOCS = ["a", "b", "c", "d", "e"]

def load_all():
    an = pd.read_csv(f"{OUT}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
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
        locs = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_location")
            if isinstance(v, str) and v.strip() in LOCS:
                locs.append(LOCS.index(v.strip()))
        X = np.stack([t[:, 0] / nw, t[:, 1] / 150.0,
                      np.sin(t[:, 2]), np.cos(t[:, 2]),
                      np.sin(t[:, 3]), np.cos(t[:, 3]),
                      t[:, 4], np.full(len(t), float(r.wifi_band == 5.0))],
                     1).astype(np.float32)
        items.append(dict(X=X, n=n, env=r.environment, locs=locs))
    le = np.concatenate([it["X"][:, 6] for it in items])
    mu, sd = float(le.mean()), float(le.std()) + 1e-9
    for it in items: it["X"][:, 6] = (it["X"][:, 6] - mu) / sd
    return items

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
    rng = np.random.default_rng(46)
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

def count_eval(net, items, ix, tag):
    lg = logits(net, items, ix)
    y = np.array([items[i]["n"] for i in ix])
    p = lg.argmax(1)
    envs = np.array([items[i]["env"] for i in ix])
    base = max(np.bincount(y, minlength=6)) / len(y)
    print(f"[{tag}] acc {np.mean(p == y):.3f}  |err|<=1 "
          f"{np.mean(np.abs(p - y) <= 1):.3f}  multi(2+) "
          f"{np.mean((p >= 2) == (y >= 2)):.3f}  occ(1+) "
          f"{np.mean((p >= 1) == (y >= 1)):.3f}  (majority {base:.3f})",
          flush=True)
    for e_ in np.unique(envs):
        m = envs == e_
        print(f"    {e_:12s} acc {np.mean(p[m] == y[m]):.3f}  "
              f"multi {np.mean((p[m] >= 2) == (y[m] >= 2)):.3f}", flush=True)
    cm = np.zeros((6, 6), int)
    for a, b_ in zip(y, p): cm[a, b_] += 1
    print("    confusion (rows=true 0-5):", flush=True)
    for k in range(6):
        print("      " + " ".join(f"{v:4d}" for v in cm[k]), flush=True)

def kmeans(X, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(-1)
        nl = d.argmin(1)
        if (nl == lab).all(): break
        lab = nl
        for j in range(k):
            if (lab == j).any(): C[j] = X[lab == j].mean(0)
    return lab

def setscore(pred, gt):
    a, b_ = Counter(pred), Counter(gt)
    return sum((a & b_).values()) / max(len(gt), 1)

def separation(items):
    print("\n=== B. SEPARATION via 1-user-trained location classifier",
          flush=True)
    rng = np.random.default_rng(46)
    for e_ in sorted({it["env"] for it in items}):
        one = [i for i, it in enumerate(items)
               if it["env"] == e_ and it["n"] == 1 and len(it["locs"]) == 1]
        if len(one) < 80:
            print(f"  {e_}: only {len(one)} 1-user samples, skip", flush=True)
            continue
        ix = rng.permutation(one)
        h = int(len(ix) * 0.85)
        tr, ho = list(ix[:h]), list(ix[h:])
        sub = [items[i] for i in tr]
        suby = [items[i]["locs"][0] for i in tr]
        net = train(sub, suby, 5, LSTEPS, f"loc-{e_}")
        lg = logits(net, items, ho)
        yh = np.array([items[i]["locs"][0] for i in ho])
        print(f"  {e_}: 1-user loc acc (heldout) "
              f"{np.mean(lg.argmax(1) == yh):.3f} (chance ~0.2, n={len(ho)})",
              flush=True)
        for nU in (2, 3, 4, 5):
            multi = [i for i, it in enumerate(items)
                     if it["env"] == e_ and it["n"] == nU
                     and len(it["locs"]) == nU]
            if len(multi) < 30: continue
            sc_cl, sc_rs, sc_top, dis_cl, dis_rs, gts = [], [], [], [], [], []
            for i in multi:
                it = items[i]
                Xs = it["X"][:, 2:6]                     # spatial axes only
                lab = kmeans(Xs, nU, seed=i)
                ok = [j for j in range(nU) if (lab == j).sum() >= 8]
                if len(ok) < nU: continue
                preds, rpreds = [], []
                rl = rng.permutation(len(Xs)) % nU       # random split ctrl
                for j in range(nU):
                    for pl, sel in ((preds, lab == j), (rpreds, rl == j)):
                        sub_it = [dict(X=it["X"][sel])]
                        pl.append(int(logits(net, sub_it, [0]).argmax()))
                full = logits(net, [it], [0])[0]
                top = list(np.argsort(-full)[:nU])
                sc_cl.append(setscore(preds, it["locs"]))
                sc_rs.append(setscore(rpreds, it["locs"]))
                sc_top.append(setscore(top, it["locs"]))
                dis_cl.append(len(set(preds)) > 1)
                dis_rs.append(len(set(rpreds)) > 1)
                gts.append(it["locs"])
            if len(sc_cl) < 20: continue
            null = []
            for _ in range(100):
                pm = rng.permutation(len(gts))
                null.append(np.mean([setscore(gts[k], gts[pm[k]])
                                     for k in range(len(gts))]))
            print(f"    {e_} n={nU} (N={len(sc_cl)}): cluster "
                  f"{np.mean(sc_cl):.3f}  rand-split {np.mean(sc_rs):.3f}  "
                  f"whole-top{nU} {np.mean(sc_top):.3f}  shuffle-null "
                  f"{np.mean(null):.3f}  distinct cl/rs "
                  f"{np.mean(dis_cl):.2f}/{np.mean(dis_rs):.2f}", flush=True)

def main():
    items = load_all()
    envs = sorted({it["env"] for it in items})
    tab = {e_: np.bincount([it["n"] for it in items if it["env"] == e_],
                           minlength=6) for e_ in envs}
    print(f"{len(items)} samples  " +
          "  ".join(f"{e}:{list(tab[e])}" for e in envs), flush=True)

    print("\n=== A. token-level 6-way count readout", flush=True)
    rng = np.random.default_rng(46)
    ix = rng.permutation(len(items))
    h = int(len(ix) * 0.8)
    tr, ho = list(ix[:h]), list(ix[h:])
    sub = [items[i] for i in tr]
    net = train(sub, [it["n"] for it in sub], 6, CSTEPS, "count-pooled")
    count_eval(net, items, ho, "POOLED heldout")
    for e_ in envs:
        tri = [i for i, it in enumerate(items) if it["env"] != e_]
        tei = [i for i, it in enumerate(items) if it["env"] == e_]
        sub = [items[i] for i in tri]
        net = train(sub, [it["n"] for it in sub], 6, XSTEPS, f"x->{e_}")
        count_eval(net, items, tei, f"CROSS-ENV -> {e_}")

    separation(items)
    print("probe 46 done", flush=True)

if __name__ == "__main__":
    main()
