#!/usr/bin/env python3
"""Per-person movement readout from THEIR slot — zero new training.

For each user, take the slot the Hungarian pairs to them (the 'their slot'
of the person-per-slot result) and classify that ONE slot's content:
  HEAD   frozen persontok activity head (co-trained, no new training)
  PROTO  nearest class-mean of handcrafted Doppler stats of the slot's
         tokens (energy share, logE, Doppler centroid, 3 band fractions,
         envelope variability) — prototypes from train-room matched slots
Report per-activity recall (confusion) on the held-out room + accuracy
per n.  Chance 1/9.

  python3 multi-person/slot_activity_readout.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

exec(compile(open(os.path.join(os.path.dirname(__file__),
                               "train_persontok.py")).read().split(
    'def main()')[0], "train_persontok.py", "exec"))

CK = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/persontok_runs/best.pt"))
TESTENV = os.environ.get("TESTENV", "empty_room")

def proto_feats(it, slot_id, hard):
    m = hard == slot_id
    t_w = it["Xf"][m]                       # raw cols cache [w,f,logE]
    if m.sum() < 3:
        return None
    e = 10.0 ** t_w[:, 2].astype(np.float64)
    w = e / e.sum()
    f = t_w[:, 1]
    fc = float((w * f).sum())
    b1 = float(w[(f >= 2) & (f < 10)].sum())
    b2 = float(w[(f >= 10) & (f < 40)].sum())
    b3 = float(w[f >= 40].sum())
    wi = t_w[:, 0].astype(int)
    env = np.zeros(int(wi.max()) + 1)
    np.add.at(env, wi, e)
    ev = env / (env.mean() + 1e-12)
    etot = 10.0 ** it["Xf"][:, 2].astype(np.float64)
    return np.array([np.log10(e.sum() / (etot.sum() + 1e-12) + 1e-6),
                     np.log10(e.sum() + 1e-9), fc / 150.0, b1, b2, b3,
                     float(ev.std())], np.float64)

def main():
    items, vocab = load_items()
    NC = len(vocab)
    ck = torch.load(CK, map_location=dev, weights_only=False)
    model = SetSep().to(dev); model.load_state_dict(ck["model"])
    head = ActHead(NC).to(dev); head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    print(f"{len(items)} annotated samples; frozen {CK}", flush=True)

    # keep raw [w, f, logE] per item for prototype features
    import pandas as pd
    an = pd.read_csv(f"{TOK}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    lab2row = {r.label: r for r in an.itertuples()}
    # rebuild raw cols aligned with load_items' filtering order
    raws = []
    for r in an.itertuples():
        n = int(r.number_of_users)
        if n < 1: continue
        f = f"{TOK}/tokens/{r.label}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        t = z["toks"]
        if len(t) < 8: continue
        if len(t) > MAXT:
            t = t[np.argsort(-t[:, 4])[:MAXT]]
        acts = []
        for k in range(1, n + 1):
            v = getattr(r, f"user_{k}_activity")
            if isinstance(v, str): acts.append(v.strip())
        if len(acts) != n: continue
        raws.append(t[:, [0, 1, 4]])
    assert len(raws) == len(items), (len(raws), len(items))
    for it, rw in zip(items, raws): it["Xf"] = rw

    # forward all; per-user matched slot; collect rows
    rows = []   # (env, n, true_act, head_pred, proto_feat)
    with torch.no_grad():
        for i0 in range(0, len(items), 32):
            sub = list(range(i0, min(i0 + 32, len(items))))
            a, h, E, mask = forward_batch(model, items, sub)
            for k, i in enumerate(sub):
                it = items[i]
                sig = slot_sigs(a[k], h[k], E[k], mask[k])
                lg = head(sig)
                y = torch.tensor(it["acts"], device=dev)
                ce = torch.stack(
                    [nn.functional.cross_entropy(lg, y[u].expand(M),
                                                 reduction="none")
                     for u in range(len(it["acts"]))], 1)
                r, c = linear_sum_assignment(ce.cpu().numpy())
                hard = a[k].argmax(1).cpu().numpy()[:len(it["Xf"])]
                pred = lg.argmax(1).cpu().numpy()
                for j in range(len(it["acts"])):
                    pf = proto_feats(it, int(r[j]), hard)
                    rows.append((it["env"], it["n"],
                                 int(it["acts"][c[j]]),
                                 int(pred[r[j]]), pf))
    # prototypes from train-room rows with valid feats
    tr = [(y, pf) for env, n, y, hp, pf in rows
          if env != TESTENV and pf is not None]
    F = np.stack([pf for _, pf in tr])
    Yp = np.array([y for y, _ in tr])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    P = np.stack([((F[Yp == k] - mu) / sd).mean(0) if (Yp == k).any()
                  else np.zeros(F.shape[1]) for k in range(NC)])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)

    for gname, iste in (("TRAIN-ROOMS", False), (f"TESTROOM {TESTENV}",
                                                 True)):
        sel = [(n, y, hp, pf) for env, n, y, hp, pf in rows
               if (env == TESTENV) == iste]
        Y = np.array([y for _, y, _, _ in sel])
        Hp = np.array([hp for _, _, hp, _ in sel])
        pp = []
        for _, y, _, pf in sel:
            if pf is None: pp.append(-1); continue
            v = (pf - mu) / sd
            v = v / (np.linalg.norm(v) + 1e-9)
            pp.append(int((v @ Pn.T).argmax()))
        Pp = np.array(pp)
        vm = Pp >= 0
        print(f"\n=== {gname} ({len(Y)} person-slots, chance "
              f"{1/NC:.3f})", flush=True)
        print(f"  HEAD  acc {np.mean(Hp == Y):.3f}   "
              f"PROTO acc {np.mean(Pp[vm] == Y[vm]):.3f}", flush=True)
        for n in range(1, 6):
            m = np.array([s[0] for s in sel]) == n
            if not m.any(): continue
            print(f"    n={n}: HEAD {np.mean(Hp[m] == Y[m]):.3f}  "
                  f"PROTO {np.mean(Pp[m & vm] == Y[m & vm]):.3f} "
                  f"(N={m.sum()})", flush=True)
        print("  per-activity recall (HEAD | PROTO | n):", flush=True)
        for kcl in range(NC):
            m = Y == kcl
            if not m.any(): continue
            hr = np.mean(Hp[m] == kcl)
            pr = np.mean(Pp[m & vm] == kcl) if (m & vm).any() else float("nan")
            top = np.bincount(Hp[m], minlength=NC)
            alt = int(np.argsort(-top)[0])
            alt = vocab[alt] if alt != kcl else vocab[
                int(np.argsort(-top)[1])] if m.sum() > 1 else "-"
            print(f"    {vocab[kcl]:10s} {hr:.2f} | {pr:.2f} | "
                  f"n={m.sum():4d}  (top confusion: {alt})", flush=True)
    print("slot activity readout done", flush=True)

if __name__ == "__main__":
    main()
