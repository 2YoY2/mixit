#!/usr/bin/env python3
"""Probe 49b: can a head NAME the limb from limbtok12 slot content alone?

Oracle (Hungarian on GT envelopes) pairs slots<->limbs at 82-99% argmax
consistency (probe 49).  Deployment needs naming WITHOUT envelopes.  Here:
oracle-label slots on scenes 1-3, train slot-signature -> 5-way limb head,
test on rooms 4/5 oracle labels.  The old limb-ID program (probes 19-26)
closed at chance for laterality — this re-asks it at the slot level.
Report: overall acc, per-limb, and the laterality confusions (LW<->RW,
LP<->RP) — the aperture law's prediction is that THOSE stay confused.

  NTR=3000 NTE=2000 python3 diagnostics/49b_limb_name_head.py
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
NTR = int(os.environ.get("NTR", "3000"))
NTE = int(os.environ.get("NTE", "2000"))
STEPS = int(os.environ.get("STEPS", "4000"))
ACTTH = float(os.environ.get("ACTTH", "0.3"))
MINCORR = float(os.environ.get("MINCORR", "0.3"))
HOPF, WINF = 128, 256
DEV5 = ["LW", "RW", "LP", "RP", "HD"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

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
        h = self.enc(self.inp(x))
        return torch.softmax(self.head(h), -1), h

model = SetSep()
model.load_state_dict(ck["model"])
model = model.to(dev).eval()
print(f"limbtok12 step {ck['step']} loaded, dev={dev}", flush=True)

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def harvest(ids, tag):
    """-> list of (sig D, limb label) oracle-labeled slots."""
    out = []
    t0 = time.time()
    for n_, rid in enumerate(ids):
        rid = int(rid)
        tf, gf = f"{TOK}/tokens/{rid:06d}.npz", f"{TOK}/imu/{rid:06d}.npy"
        if not (os.path.exists(tf) and os.path.exists(gf)): continue
        z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
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
        act = np.where(mu >= ACTTH * mu.max())[0]
        if len(act) < 1: continue
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                  np.cos(t[:, 3]), t[:, 1] / 150.0,
                  t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        widx = t[:, 0].astype(int)
        with torch.no_grad():
            a, h = model(torch.from_numpy(X)[None].to(dev))
            a, h = a[0].cpu().numpy(), h[0].cpu().numpy()
        Em = np.zeros((M, nw))
        for m in range(M):
            np.add.at(Em[m], widx, a[:, m] * e)
        C = np.zeros((M, len(act)))
        for m in range(M):
            for j, lj in enumerate(act):
                C[m, j] = corr(Em[m], G[:, lj])
        r, c = linear_sum_assignment(-C)
        w = a * e[:, None]
        for i in range(len(act)):
            if C[r[i], c[i]] < MINCORR: continue
            ws = w[:, r[i]]
            if ws.sum() < 1e-9: continue
            sig = (ws[:, None] * h).sum(0) / ws.sum()
            out.append((sig.astype(np.float32), int(act[c[i]])))
        if (n_ + 1) % 1000 == 0:
            print(f"  [{tag}] {n_+1}/{len(ids)} slots={len(out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(49)
    tr_ids = rng.permutation(man[man.split == "train"].rid.values)[:NTR]
    te_ids = rng.permutation(man[man.split == "test"].rid.values)[:NTE]
    tr = harvest(tr_ids, "train")
    te = harvest(te_ids, "test45")
    ytr = np.array([l for _, l in tr])
    print(f"train slots {len(tr)} " +
          " ".join(f"{DEV5[k]}:{(ytr==k).sum()}" for k in range(5)) +
          f" | test slots {len(te)}", flush=True)

    head = nn.Sequential(nn.Linear(D, D), nn.GELU(),
                         nn.Linear(D, 5)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-5)
    bycls = {}
    for i, (_, l) in enumerate(tr): bycls.setdefault(l, []).append(i)
    keys = sorted(bycls)
    Xtr = torch.from_numpy(np.stack([s for s, _ in tr]))
    for step in range(STEPS):
        ix = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
              for c in rng.integers(0, len(keys), 64)]
        y = torch.tensor([tr[i][1] for i in ix]).to(dev)
        loss = nn.functional.cross_entropy(head(Xtr[ix].to(dev)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  [head {step}] CE {loss.item():.3f}", flush=True)
    head.eval()
    Xte = torch.from_numpy(np.stack([s for s, _ in te]))
    yte = np.array([l for _, l in te])
    with torch.no_grad():
        pred = head(Xte.to(dev)).argmax(1).cpu().numpy()
    print(f"\n=== slot->limb NAMING on rooms 4/5 (no envelopes at test)",
          flush=True)
    base = max(np.bincount(yte, minlength=5)) / len(yte)
    print(f"  overall acc {np.mean(pred == yte):.3f}  (chance 0.20, "
          f"majority {base:.3f}, n={len(yte)})", flush=True)
    print("  confusion rows=true:", flush=True)
    for k in range(5):
        m = yte == k
        if not m.any(): continue
        cnt = np.bincount(pred[m], minlength=5) / m.sum()
        print(f"    {DEV5[k]}: " +
              " ".join(f"{DEV5[j]} {cnt[j]*100:3.0f}%" for j in range(5)) +
              f"  (n={m.sum()})", flush=True)
    lat = [(0, 1), (2, 3)]
    for a_, b_ in lat:
        m = (yte == a_) | (yte == b_)
        if not m.any(): continue
        merged = np.mean((pred[m] == yte[m]) |
                         ((pred[m] == a_) & (yte[m] == b_)) |
                         ((pred[m] == b_) & (yte[m] == a_)))
        side = np.mean(pred[m] == yte[m])
        print(f"  {DEV5[a_]}/{DEV5[b_]}: side-exact {side:.3f} vs "
              f"either-of-pair {merged:.3f} -> laterality "
              f"{'RESOLVED' if side > 0.6 * merged else 'CONFUSED'}",
              flush=True)
    print("probe 49b done", flush=True)

if __name__ == "__main__":
    main()
