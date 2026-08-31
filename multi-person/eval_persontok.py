#!/usr/bin/env python3
"""persontok gate — does slot k track human k (and only human k)?

On real multi-user samples, per group (train rooms / held-out room), per n:
  matched   Hungarian slot<->user activity accuracy (GT used ONLY to match)
  wrongperm cyclic mis-assignment control — slot k scored against human k+1;
            a real person-per-slot model scores HIGH matched, LOW wrongperm
  multiset  GT-free: top-n energy slots' predicted activities vs the true
            activity multiset (probe-47/48 protocol) + shuffle null
Plus the slot energy-share table by n (does slot structure react to count).

  CK=~/zerdani/buffer/octonet/persontok_runs/best.pt TESTENV=empty_room \
    python3 multi-person/eval_persontok.py
"""
import os
from collections import Counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# reuse the trainer's model/loader defs without running its main()
exec(compile(open(os.path.join(os.path.dirname(__file__),
                               "train_persontok.py")).read().split(
    'def main()')[0], "train_persontok.py", "exec"))

CK = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/persontok_runs/best.pt"))
TESTENV = os.environ.get("TESTENV", "empty_room")

def setscore(pred, gt):
    a, b = Counter(pred), Counter(gt)
    return sum((a & b).values()) / max(len(gt), 1)

def main():
    items, vocab = load_items()
    ck = torch.load(CK, map_location=dev, weights_only=False)
    model = SetSep().to(dev); model.load_state_dict(ck["model"])
    head = ActHead(len(vocab)).to(dev); head.load_state_dict(ck["head"])
    model.eval(); head.eval()
    print(f"loaded {CK} (step {ck['step']})", flush=True)
    rng = np.random.default_rng(49)

    print("\n=== slot energy share by n (all samples)", flush=True)
    shares = {i: None for i in range(len(items))}
    with torch.no_grad():
        for i0 in range(0, len(items), 32):
            sub = list(range(i0, min(i0 + 32, len(items))))
            a, h, E, mask = forward_batch(model, items, sub)
            for k, i in enumerate(sub):
                w = (a[k] * E[k][:, None] *
                     (~mask[k])[:, None].float()).sum(0).cpu().numpy()
                shares[i] = w / (w.sum() + 1e-12)
                items[i]["sig"] = slot_sigs(a[k], h[k], E[k], mask[k]).cpu()
    for n in range(1, 6):
        sel = [shares[i] for i, it in enumerate(items) if it["n"] == n]
        if not sel: continue
        sh = np.mean(sel, 0)
        print(f"  n={n}: " + " ".join(f"{v:.2f}" for v in sh) +
              f"   active(>10%) "
              f"{np.mean([(s > 0.10).sum() for s in sel]):.2f}", flush=True)

    for gname, iste in [("TRAIN-ROOMS", False), (f"TESTROOM {TESTENV}", True)]:
        print(f"\n=== {gname}", flush=True)
        for n in range(1, 6):
            mi = [i for i, it in enumerate(items)
                  if it["n"] == n and (it["env"] == TESTENV) == iste]
            if len(mi) < 30: continue
            accs, wrongs, msets, gts = [], [], [], []
            with torch.no_grad():
                for i in mi:
                    it = items[i]
                    lg = head(it["sig"].to(dev)).cpu()
                    y = np.array(it["acts"])
                    ce = np.stack(
                        [(-torch.log_softmax(lg, 1)[:, u]).numpy()
                         for u in y], 1)
                    r, c = linear_sum_assignment(ce)
                    pred = lg.argmax(1).numpy()
                    accs.append(float((pred[r] == y[c]).mean()))
                    if n > 1:
                        rr = np.roll(r, 1)
                        wrongs.append(float((pred[rr] == y[c]).mean()))
                    topn = np.argsort(-shares[i])[:n]
                    msets.append(setscore([int(pred[j]) for j in topn],
                                          list(y)))
                    gts.append(list(y))
            null = []
            for _ in range(100):
                pm = rng.permutation(len(gts))
                null.append(np.mean([setscore(gts[k], gts[pm[k]])
                                     for k in range(len(gts))]))
            w = f"{np.mean(wrongs):.3f}" if wrongs else "  -  "
            print(f"  n={n} (N={len(mi)}): matched {np.mean(accs):.3f}  "
                  f"wrongperm {w}  multiset {np.mean(msets):.3f}  "
                  f"null {np.mean(null):.3f}", flush=True)
    print("eval done", flush=True)

if __name__ == "__main__":
    main()
