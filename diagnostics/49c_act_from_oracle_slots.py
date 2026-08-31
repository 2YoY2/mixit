#!/usr/bin/env python3
"""Probe 49c: PA action recognition from ORACLE-selected limb slots.

If the oracle (Hungarian on GT envelopes) tells us which limbtok12 slots
are the moving limbs, what 17-way action accuracy do those slots' tokens
give — vs ALL tokens (tokens-direct control, the reigning route at
49.3%/22.8%) and vs the COMPLEMENT (non-limb slots, should be poor)?
Answers: does solved selection make separated limb channels BEAT raw
tokens for movement recognition?

Arms (same classifier, same budget, balanced):
  ALL     every token                       (matched-budget control)
  ORACLE  tokens whose argmax slot is an oracle limb slot
  COMPL   tokens in the remaining slots
Train scenes 1-3 (10% heldout), test scene 4.

  NTR=4000 NTE=2000 STEPS=5000 python3 diagnostics/49c_act_from_oracle_slots.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
NTR = int(os.environ.get("NTR", "4000"))
NTE = int(os.environ.get("NTE", "2000"))
STEPS = int(os.environ.get("STEPS", "5000"))
ACTTH = float(os.environ.get("ACTTH", "0.3"))
HOPF, WINF = 128, 256
NC = 17
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

ck = torch.load(f"{RUNS}/best.pt", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x):
        return torch.softmax(self.head(self.enc(self.inp(x))), -1)

sep = SetSep()
sep.load_state_dict(ck["model"])
sep = sep.to(dev).eval()
print(f"limbtok12 step {ck['step']} loaded, dev={dev}", flush=True)

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def harvest(ids, r2a, tag):
    """-> items: dict(Xall, Xora, Xcom, act) with 7-dim token feats."""
    out = []
    t0 = time.time()
    for n_, rid in enumerate(ids):
        rid = int(rid)
        act_lbl = r2a.get(rid)
        if act_lbl is None: continue
        tf, gf = f"{TOK}/tokens/{rid:06d}.npz", f"{TOK}/imu/{rid:06d}.npy"
        if not (os.path.exists(tf) and os.path.exists(gf)): continue
        z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
        if len(t) < 16: continue
        if len(t) > 1024:
            t = t[np.argsort(-t[:, 4])[:1024]]
        gi = np.asarray(np.load(gf), np.float32)
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                      for w in range(nw)])
        if len(G) < 8: continue
        mu = G.mean(0)
        actl = np.where(mu >= ACTTH * mu.max())[0]
        if len(actl) < 1: continue
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                  np.cos(t[:, 3]), t[:, 1] / 150.0,
                  t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        widx = t[:, 0].astype(int)
        with torch.no_grad():
            a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
        Em = np.zeros((M, nw))
        for m in range(M):
            np.add.at(Em[m], widx, a[:, m] * e)
        C = np.zeros((M, len(actl)))
        for m in range(M):
            for j, lj in enumerate(actl):
                C[m, j] = corr(Em[m], G[:, lj])
        r, c = linear_sum_assignment(-C)
        ora = set(int(v) for v in r)
        hard = a.argmax(1)
        mo = np.isin(hard, list(ora))
        if mo.sum() < 8 or (~mo).sum() < 8: continue
        out.append(dict(Xall=X, Xora=X[mo], Xcom=X[~mo],
                        act=act_lbl - 1))
        if (n_ + 1) % 1000 == 0:
            print(f"  [{tag}] {n_+1}/{len(ids)} kept={len(out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out

class TokCls(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.inp = nn.Linear(7, H)
        lay = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.out = nn.Linear(H, NC)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        h = (h * (~mask)[:, :, None]).sum(1) / (~mask).sum(1, keepdim=True)
        return self.out(h)

def run_arm(key, tr, ho, te, rng):
    net = TokCls().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    bycls = {}
    for i, it in enumerate(tr): bycls.setdefault(it["act"], []).append(i)
    keys = sorted(bycls)
    def batch(items, ixb):
        its = [items[i] for i in ixb]
        n = max(len(it[key]) for it in its)
        X = torch.zeros(len(its), n, 7)
        mask = torch.ones(len(its), n, dtype=torch.bool)
        for k2, it in enumerate(its):
            X[k2, :len(it[key])] = torch.from_numpy(it[key])
            mask[k2, :len(it[key])] = False
        return X.to(dev), mask.to(dev)
    t0 = time.time()
    for step in range(STEPS):
        ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
               for c in rng.integers(0, len(keys), 16)]
        X, mask = batch(tr, ixb)
        y = torch.tensor([tr[i]["act"] for i in ixb]).to(dev)
        loss = nn.functional.cross_entropy(net(X, mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  [{key} {step}] CE {loss.item():.3f} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
    net.eval()
    def ev(ds):
        P, Y = [], []
        with torch.no_grad():
            for i0 in range(0, len(ds), 16):
                ix = list(range(i0, min(i0 + 16, len(ds))))
                X, mask = batch(ds, ix)
                P += list(net(X, mask).argmax(1).cpu().numpy())
                Y += [ds[i]["act"] for i in ix]
        return float(np.mean(np.array(P) == np.array(Y)))
    return ev(ho), ev(te)

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    r2a = {int(r.rid): int(r.act) for r in man.itertuples()}
    rng = np.random.default_rng(49)
    tr_ids = rng.permutation(np.array(
        man[man.split == "train"].rid.values))[:NTR]
    te_ids = rng.permutation(np.array(
        man[(man.split == "test") & (man.scene.astype(str).str.contains(
            "4"))].rid.values))[:NTE]
    tr_all = harvest(tr_ids, r2a, "train")
    te = harvest(te_ids, r2a, "test4")
    ix = rng.permutation(len(tr_all))
    h = int(len(ix) * 0.9)
    tr = [tr_all[i] for i in ix[:h]]
    ho = [tr_all[i] for i in ix[h:]]
    ora_frac = np.mean([len(it["Xora"]) / len(it["Xall"]) for it in tr_all])
    print(f"train {len(tr)} ho {len(ho)} test {len(te)}  "
          f"(oracle tokens = {ora_frac*100:.0f}% of cloud)", flush=True)
    print("\n=== 17-way action (chance 0.059)", flush=True)
    for key, nm in (("Xall", "ALL    "), ("Xora", "ORACLE "),
                    ("Xcom", "COMPL  ")):
        a_ho, a_te = run_arm(key, tr, ho, te, rng)
        print(f"  [{nm}] heldout {a_ho:.3f}   scene4 {a_te:.3f}", flush=True)
    print("probe 49c done", flush=True)

if __name__ == "__main__":
    main()
