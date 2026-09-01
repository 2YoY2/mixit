#!/usr/bin/env python3
"""Probe 49d: FROZEN classifier on oracle limb-slot tokens, scene 1 vs 4.

Zero training.  Per clip (3 rx): run limbtok12 (the clusterer), Hungarian
on GT envelopes picks the best slot per active limb (the oracle), keep the
tokens of those slots, classify with the SAVED act_cls_fine2.pt exactly as
act_from_tokens built its inputs (7 feats + 8 slot posteriors + 3 rx hot).
Arms: ALL tokens (sanity = tokens-direct), ORACLE slots, COMPLEMENT.
Output: accuracy per arm + 17-way confusion matrix (oracle arm) for
scene 1 (in-domain) and scene 4 (unseen room).

  NCLIP=800 python3 diagnostics/49d_frozen_oracle_confusion.py
"""
import os
os.environ.setdefault("TOK", os.path.expanduser(
    "~/zerdani/buffer/octonet/pa_tokens_fine2"))
os.environ.setdefault("MIXIT_RUNS", os.path.expanduser(
    "~/zerdani/buffer/octonet/limbtok12_runs"))
import time, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "..", "bench",
                        "train_posetok.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

TOK, M, dev = ptk.TOK, ptk.M, ptk.dev
CKF = os.path.expanduser(os.environ.get(
    "CKF", "~/zerdani/buffer/octonet/act_cls_fine2.pt"))
NCLIP = int(os.environ.get("NCLIP", "800"))
ACTTH = float(os.environ.get("ACTTH", "0.3"))
TMAX = int(os.environ.get("TMAX", "3000"))
WINF, HOPF = 256, 32                      # fine2 grid for envelopes
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}

class TokCls(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.inp = nn.Linear(18, H)
        lay = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.out = nn.Linear(H, NC)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        h = (h * (~mask)[:, :, None]).sum(1) / (~mask).sum(1, keepdim=True)
        return self.out(h)

net = TokCls().to(dev)
net.load_state_dict(torch.load(CKF, map_location=dev,
                               weights_only=False)["model"])
net.eval()
print(f"frozen classifier {CKF} loaded (NO training)", flush=True)

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def oracle_mask(rid, X, nw):
    """-> (limb-oracle mask, ADAPT-k* mask) over X's tokens."""
    gf = f"{TOK}/imu/{rid:06d}.npy"
    if not os.path.exists(gf): return None
    z = np.load(f"{TOK}/tokens/{rid:06d}.npz")
    t = z["toks"]
    w = t[:, 0].astype(int)
    e = (10.0 ** t[:, 4]).astype(np.float64)
    a = X[:, 7:7 + M].astype(np.float64)
    gi = np.asarray(np.load(gf), np.float32)
    g2 = gi.copy()
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    nww = min(nw, (len(g2) - WINF) // HOPF + 1)
    if nww < 8: return None
    G = np.stack([g2[wi * HOPF:wi * HOPF + WINF].mean(0)
                  for wi in range(nww)])
    mu = G.mean(0)
    act = np.where(mu >= ACTTH * mu.max())[0]
    if len(act) < 1: return None
    hard = a.argmax(1)
    Em = np.zeros((M, nw))
    for m in range(M):
        np.add.at(Em[m], w, a[:, m] * e)
    C = np.zeros((M, len(act)))
    for m in range(M):
        for j, lj in enumerate(act):
            C[m, j] = corr(Em[m, :nww], G[:, lj])
    r, _ = linear_sum_assignment(-C)
    # ADAPT-k*: rank slots by corr with TOTAL motion (unresidualized),
    # keep the cumulative-union prefix that maximizes correlation
    Gt = np.array([gi.sum(1)[wi * HOPF:wi * HOPF + WINF].mean()
                   for wi in range(nww)])
    ems, cs = [], []
    for m in range(M):
        em = np.zeros(nw)
        np.add.at(em, w[hard == m], e[hard == m])
        ems.append(em); cs.append(corr(em[:nww], Gt))
    order = np.argsort(-np.nan_to_num(cs, nan=-2))
    accv = np.zeros(nw)
    rks = []
    for m in order:
        accv = accv + ems[m]
        rks.append(corr(accv[:nww], Gt))
    kstar = int(np.nanargmax(rks)) + 1
    keepset = set(int(m) for m in order[:kstar])
    return np.isin(hard, r), np.isin(hard, list(keepset))

def classify(items):
    P = []
    with torch.no_grad():
        for i0 in range(0, len(items), 16):
            its = items[i0:i0 + 16]
            n = max(len(x) for x in its)
            X = torch.zeros(len(its), n, 18)
            mask = torch.ones(len(its), n, dtype=torch.bool)
            for k, x in enumerate(its):
                X[k, :len(x)] = torch.from_numpy(x.astype(np.float32))
                mask[k, :len(x)] = False
            P += list(net(X.to(dev), mask.to(dev)).argmax(1).cpu().numpy())
    return np.array(P)

def cap(x):
    if len(x) > TMAX:
        x = x[np.argsort(-x[:, 6].astype(np.float32))[:TMAX]]
    return x

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    r2a = {int(r.rid): int(r.act) for r in man.itertuples()}
    rng = np.random.default_rng(49)
    for scene in (1, 4):
        ms = man[man.scene == scene].copy()
        ms["ckey"] = ms["name"].str.replace(r"_r\d$", "", regex=True)
        groups = [g for _, g in ms.groupby("ckey")
                  if len(g) == 3 and set(g.node) == {"r1", "r2", "r3"}]
        rng.shuffle(groups)
        arms = {"ALL": [], "ORACLE": [], "COMPL": [], "ADAPTK": []}
        Y = []
        t0 = time.time()
        for g in groups:
            if len(Y) >= NCLIP: break
            rids = [int(r) for r in g.sort_values("node").rid.values]
            al = r2a.get(rids[0])
            if al is None: continue
            xs, os_, cs, ad = [], [], [], []
            ok = True
            for i, rid in enumerate(rids):
                res = ptk.rec_tok(rid, i)
                if res is None: ok = False; break
                X, _, nw = res
                X = np.asarray(X, np.float32)
                masks = oracle_mask(rid, X, nw)
                if masks is None: ok = False; break
                mo, ma = masks
                if mo.sum() < 8 or (~mo).sum() < 8 or ma.sum() < 8:
                    ok = False; break
                xs.append(X); os_.append(X[mo]); cs.append(X[~mo])
                ad.append(X[ma])
            if not ok: continue
            arms["ALL"].append(cap(np.concatenate(xs)))
            arms["ORACLE"].append(cap(np.concatenate(os_)))
            arms["COMPL"].append(cap(np.concatenate(cs)))
            arms["ADAPTK"].append(cap(np.concatenate(ad)))
            Y.append(al - 1)
        Y = np.array(Y)
        print(f"\n===== SCENE {scene}: {len(Y)} clips "
              f"({(time.time()-t0)/60:.1f}min harvest)", flush=True)
        preds = {}
        for arm in ("ALL", "ORACLE", "ADAPTK", "COMPL"):
            P = classify(arms[arm])
            preds[arm] = P
            Pm = np.array([MIRROR.get(v + 1, v + 1) for v in P])
            Ym = np.array([MIRROR.get(v + 1, v + 1) for v in Y])
            print(f"  [{arm:6s}] 17-class {np.mean(P == Y):.3f}  merged "
                  f"{np.mean(Pm == Ym):.3f}  (chance 0.059)", flush=True)
        print(f"\n  confusion SCENE {scene} (ADAPTK arm, rows=true, top-3):",
              flush=True)
        P = preds["ADAPTK"]
        for k in range(NC):
            m = Y == k
            if not m.any(): continue
            cnt = np.bincount(P[m], minlength=NC) / m.sum()
            top = np.argsort(-cnt)[:3]
            row = "  ".join(f"{NAMES[t]} {cnt[t]*100:.0f}%"
                            for t in top if cnt[t] > 0)
            print(f"    {NAMES[k]:10s} (n={m.sum():3d}) -> {row}",
                  flush=True)
    print("probe 49d done", flush=True)

if __name__ == "__main__":
    main()
