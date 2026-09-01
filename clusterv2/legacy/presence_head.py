#!/usr/bin/env python3
"""Per-slot PERSON-PRESENCE confidence for persontok (frozen).

The Hungarian can pair n slots to n people when n is GIVEN; the model
cannot say which slots actually HOLD a person.  Fix: a small presence head
on frozen slot signatures.
  positives  Hungarian-matched slots of occupied train-room recordings
  negatives  every slot of 0-user recordings (train rooms)
  excluded   unmatched slots of occupied scenes (ambiguous spillover)
Outputs on heldout train-rooms AND the never-seen room:
  AUC person-slot vs empty-slot | count from confident slots (n-hat =
  #slots p>thr) vs true n | reliability curve (calibration).

  python3 multi-person/presence_head.py
"""
import os, math
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

# reuse the trainer's defs (SetSep, ActHead, load_items, batch, slot_sigs)
exec(compile(open(os.path.join(os.path.dirname(__file__),
                               "train_persontok.py")).read().split(
    'def main()')[0], "train_persontok.py", "exec"))

CK = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/persontok_runs/best.pt"))
TESTENV = os.environ.get("TESTENV", "empty_room")
PSTEPS = int(os.environ.get("PSTEPS", "2000"))
THR = float(os.environ.get("THR", "0.5"))

def load_all_incl_empty():
    import pandas as pd
    an = pd.read_csv(f"{TOK}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    vocab = sorted({str(getattr(r, f"user_{k}_activity")).strip()
                    for r in an.itertuples() for k in range(1, 6)
                    if isinstance(getattr(r, f"user_{k}_activity"), str)})
    v2i = {v: i for i, v in enumerate(vocab)}
    items = []
    for r in an.itertuples():
        n = int(r.number_of_users)
        f = f"{TOK}/tokens/{r.label}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        t, nw = z["toks"], int(z["nw"])
        if len(t) < 8: continue
        if len(t) > MAXT:
            t = t[np.argsort(-t[:, 4])[:MAXT]]
        acts = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_activity")
            if isinstance(v, str) and v.strip() in v2i:
                acts.append(v2i[v.strip()])
        if len(acts) != n: continue
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]),
                  np.sin(t[:, 3]), np.cos(t[:, 3]),
                  t[:, 1] / 150.0, t[:, 0] / max(nw - 1, 1),
                  zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        items.append(dict(X=X, e=e, n=n, env=r.environment, acts=acts))
    return items, vocab

def main():
    items, vocab = load_all_incl_empty()
    ck = torch.load(CK, map_location=dev, weights_only=False)
    model = SetSep().to(dev); model.load_state_dict(ck["model"])
    head = ActHead(len(vocab)).to(dev); head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    print(f"{len(items)} samples (incl. 0-user); frozen {CK}", flush=True)

    # forward all: slot sigs + energy shares + matched slot sets
    feats, labels, envs, ns = [], [], [], []   # per SLOT rows
    recs = []                                   # per recording bookkeeping
    with torch.no_grad():
        for i0 in range(0, len(items), 32):
            sub = list(range(i0, min(i0 + 32, len(items))))
            a, h, E, mask = forward_batch(model, items, sub)
            for k, i in enumerate(sub):
                it = items[i]
                sig = slot_sigs(a[k], h[k], E[k], mask[k])
                w = (a[k] * E[k][:, None] *
                     (~mask[k])[:, None].float()).sum(0).cpu().numpy()
                share = w / (w.sum() + 1e-12)
                matched = set()
                if it["n"] >= 1:
                    lg = head(sig)
                    y = torch.tensor(it["acts"], device=dev)
                    ce = torch.stack(
                        [nn.functional.cross_entropy(
                            lg, y[u].expand(M), reduction="none")
                         for u in range(len(it["acts"]))], 1)
                    r, c = linear_sum_assignment(ce.cpu().numpy())
                    matched = set(int(v) for v in r)
                row0 = len(feats)
                for m in range(M):
                    if it["n"] == 0:
                        lab = 0
                    elif m in matched:
                        lab = 1
                    else:
                        lab = -1                # ambiguous, excluded
                    feats.append(np.r_[sig[m].cpu().numpy(),
                                       share[m], math.log10(w[m] + 1e-9)])
                    labels.append(lab)
                    envs.append(it["env"]); ns.append(it["n"])
                recs.append((row0, it["env"], it["n"]))
    F = np.stack(feats).astype(np.float32)
    L = np.array(labels); EN = np.array(envs); NS = np.array(ns)
    print(f"slot rows {len(F)}: pos {(L==1).sum()} neg {(L==0).sum()} "
          f"excl {(L==-1).sum()}", flush=True)

    tr_m = (EN != TESTENV) & (L >= 0)
    rng = np.random.default_rng(46)
    ix = rng.permutation(np.where(tr_m)[0])
    nho = int(len(ix) * 0.1)
    ho_ix, tr_ix = ix[:nho], ix[nho:]
    mu, sd = F[tr_ix].mean(0), F[tr_ix].std(0) + 1e-9
    Fz = (F - mu) / sd
    ph = nn.Sequential(nn.Linear(F.shape[1], 64), nn.GELU(),
                       nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(ph.parameters(), lr=1e-3)
    Xtr = torch.from_numpy(Fz[tr_ix]).to(dev)
    ytr = torch.from_numpy((L[tr_ix] == 1).astype(np.float32)).to(dev)
    for step in range(PSTEPS):
        bix = torch.randint(0, len(Xtr), (256,), device=dev)
        loss = nn.functional.binary_cross_entropy_with_logits(
            ph(Xtr[bix])[:, 0], ytr[bix])
        opt.zero_grad(); loss.backward(); opt.step()
    ph.eval()
    with torch.no_grad():
        P = torch.sigmoid(ph(torch.from_numpy(Fz).to(dev))[:, 0]
                          ).cpu().numpy()

    def auc(y, p):
        o = np.argsort(p)
        r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
        npos = y.sum(); nneg = len(y) - npos
        if npos == 0 or nneg == 0: return float("nan")
        return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                     / (npos * nneg))

    for gname, gm in (("HELDOUT train-rooms",
                       np.isin(np.arange(len(F)), ho_ix)),
                      (f"TESTROOM {TESTENV}", EN == TESTENV)):
        m = gm & (L >= 0)
        print(f"\n=== {gname}: slot AUC {auc((L[m]==1).astype(int), P[m]):.3f}"
              f"  (n={m.sum()})", flush=True)
        # reliability
        print("  reliability (conf bucket -> empirical person rate):",
              flush=True)
        for lo in np.arange(0, 1, 0.2):
            b = m & (P >= lo) & (P < lo + 0.2)
            if b.sum() < 20: continue
            print(f"    [{lo:.1f},{lo+0.2:.1f}): {np.mean(L[b]==1):.2f} "
                  f"(n={b.sum()})", flush=True)
        # count from confident slots (all recordings of the group)
        accs, off1, occ = [], [], []
        for row0, env, n in recs:
            if (env == TESTENV) != (gname.startswith("TESTROOM")): continue
            nhat = int((P[row0:row0 + M] > THR).sum())
            accs.append(nhat == n)
            off1.append(abs(nhat - n) <= 1)
            occ.append((nhat >= 1) == (n >= 1))
        print(f"  count from confident slots (thr {THR}): exact "
              f"{np.mean(accs):.3f}  |err|<=1 {np.mean(off1):.3f}  "
              f"occupancy {np.mean(occ):.3f}  (N={len(accs)}; p46 refs: "
              f"pooled 74.8%, x-env 38-42%)", flush=True)
    print("presence head done", flush=True)

if __name__ == "__main__":
    main()
