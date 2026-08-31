#!/usr/bin/env python3
"""WiMANS person-clusterer: the limbtok recipe trained ON WiMANS unions.

Same SetSep architecture and MixIT-origin loss as train_limbtok, but the
origins are PEOPLE: a training item is the UNION of two samples from the
same (env, band) with <=1 user each (0+1 or 1+1; real n>=2 samples are
NEVER seen).  The model must assign tokens to slots such that some 2-group
slot partition recovers each origin — i.e. slots must not mix people.

Eval (every EVERY steps, held-out unions): origin PURITY = energy fraction
correctly split under the best hard slot->origin assignment.  best.pt by
purity.  Downstream check on REAL multi-user samples = probe 48 with
CK=<this best.pt>.

  STEPS=25000 HOURS=3 python3 phase2/train_wimanstok.py
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
RUNS = os.path.expanduser(os.environ.get(
    "RUNS", "~/zerdani/buffer/octonet/wimanstok_runs"))
STEPS = int(os.environ.get("STEPS", "25000"))
HOURS = float(os.environ.get("HOURS", "3"))
BM = int(os.environ.get("BM", "8"))
LR = float(os.environ.get("LR", "3e-4"))
M = int(os.environ.get("M", "8"))
D = int(os.environ.get("DIM", "256"))
NL = int(os.environ.get("LAYERS", "6"))
MAXT = int(os.environ.get("MAXT", "512"))
EVERY = int(os.environ.get("EVERY", "500"))
SEED = int(os.environ.get("SEED", "0"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

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

PARTS = None
def mixit_loss(a, e, origin):
    global PARTS
    if PARTS is None:
        ms = []
        for b in range(1, 2 ** M - 1):
            ms.append([float((b >> k) & 1) for k in range(M)])
        PARTS = torch.tensor(ms, device=a.device)
    p = a @ PARTS.T
    w = e / (e.sum() + 1e-8)
    err = ((p - origin[:, None]) ** 2 * w[:, None]).sum(0)
    return err.min()

def load_items():
    an = pd.read_csv(f"{TOK}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    items = []
    for r in an.itertuples():
        if int(r.number_of_users) > 1: continue
        f = f"{TOK}/tokens/{r.label}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        t, nw = z["toks"], int(z["nw"])
        if len(t) < 8: continue
        if len(t) > MAXT:
            t = t[np.argsort(-t[:, 4])[:MAXT]]
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]),
                  np.sin(t[:, 3]), np.cos(t[:, 3]),
                  t[:, 1] / 150.0, t[:, 0] / max(nw - 1, 1),
                  zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        e = e / (e.mean() + 1e-12)          # per-origin energy scale out
        items.append(dict(X=X, e=e, n=int(r.number_of_users),
                          cell=(r.environment, float(r.wifi_band))))
    return items

def unions(items, pairs):
    recs = []
    for ia, ib in pairs:
        A, Bb = items[ia], items[ib]
        X = np.concatenate([A["X"], Bb["X"]])
        e = np.concatenate([A["e"], Bb["e"]])
        o = np.concatenate([np.zeros(len(A["X"]), np.float32),
                            np.ones(len(Bb["X"]), np.float32)])
        recs.append((X, e, o))
    n = max(len(r[0]) for r in recs)
    X = torch.zeros(len(recs), n, 7)
    E = torch.zeros(len(recs), n)
    O = torch.zeros(len(recs), n)
    mask = torch.ones(len(recs), n, dtype=torch.bool)
    for k, (x, e, o) in enumerate(recs):
        X[k, :len(x)] = torch.from_numpy(x)
        E[k, :len(x)] = torch.from_numpy(e)
        O[k, :len(x)] = torch.from_numpy(o)
        mask[k, :len(x)] = False
    return X.to(dev), E.to(dev), O.to(dev), mask.to(dev), \
        [len(r[0]) for r in recs]

def purity(a_hard, e, o):
    """energy fraction correctly split, best hard slot->origin map."""
    tot = e.sum() + 1e-12
    corr = 0.0
    for j in range(M):
        m = a_hard == j
        if not m.any(): continue
        eA = e[m][o[m] < 0.5].sum()
        eB = e[m][o[m] > 0.5].sum()
        corr += max(eA, eB)
    return float(corr / tot)

def main():
    os.makedirs(RUNS, exist_ok=True)
    items = load_items()
    rng = np.random.default_rng(SEED)
    ix = rng.permutation(len(items))
    ho = set(ix[:int(len(ix) * 0.1)].tolist())
    cells = {}
    for i, it in enumerate(items):
        if i in ho: continue
        cells.setdefault(it["cell"], []).append(i)
    cells = {k: v for k, v in cells.items() if len(v) >= 10}
    ckeys = list(cells)
    hocells = {}
    for i in ho:
        hocells.setdefault(items[i]["cell"], []).append(i)
    evpairs = []
    for k, v in hocells.items():
        v1 = [i for i in v if items[i]["n"] == 1]
        for _ in range(24):
            if len(v1) < 2: break
            a, b = rng.choice(v1, 2, replace=False)
            evpairs.append((int(a), int(b)))
    print(f"{len(items)} items (0/1-user), {len(cells)} cells, "
          f"{len(evpairs)} eval unions, dev={dev}", flush=True)

    model = SetSep().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    step0, best = 0, -math.inf
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev,
                        weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); step0 = ck["step"]; best = ck["best"]
        print(f"resumed from step {step0}", flush=True)

    def evaluate():
        model.eval()
        ps = []
        with torch.no_grad():
            for i0 in range(0, len(evpairs), 16):
                X, E, O, mask, lens = unions(items, evpairs[i0:i0 + 16])
                a = model(X, mask)
                for k, L in enumerate(lens):
                    ps.append(purity(a[k, :L].argmax(1).cpu().numpy(),
                                     E[k, :L].cpu().numpy(),
                                     O[k, :L].cpu().numpy()))
        model.train()
        return float(np.mean(ps))

    t0 = time.time()
    for step in range(step0, STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        pairs = []
        for _ in range(BM):
            c = cells[ckeys[rng.integers(len(ckeys))]]
            while True:
                a, b = rng.choice(c, 2, replace=False)
                if items[a]["n"] + items[b]["n"] >= 1: break
            pairs.append((int(a), int(b)))
        X, E, O, mask, lens = unions(items, pairs)
        a = model(X, mask)
        loss = torch.stack([mixit_loss(a[k, :L], E[k, :L], O[k, :L])
                            for k, L in enumerate(lens)]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % EVERY == 0:
            pu = evaluate()
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sch": sch.state_dict(), "step": step, "best": best}
            torch.save(ck, f"{RUNS}/last.pt")
            if pu > best:
                best = pu
                torch.save(ck, f"{RUNS}/best.pt")
            print(f"[{step}] mixit {loss.item():.4f}  purity {pu:.3f} "
                  f"(best {best:.3f})  {(time.time()-t0)/3600:.2f}h",
                  flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step, "best": best},
               f"{RUNS}/last.pt")
    print(f"done at step {step}, best purity {best:.3f}", flush=True)

if __name__ == "__main__":
    main()
