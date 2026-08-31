#!/usr/bin/env python3
"""persontok — the phase-2 clusterer recipe with PEOPLE as the actors.

SetSep token clusterer (identical arch+features to limbtok12) + co-trained
activity head.  PIT-ACTIVITY loss: pool each slot's tokens into a signature
(energy-weighted hidden states), classify, Hungarian best slot<->user
assignment on the CE cost matrix — identity-free, exactly the role PA's
PIT-envelope loss played, with the 9-way per-user activity labels replacing
keypoint envelopes (the only per-person GT WiMANS has).

Train: real 1-5-user samples of TRENVS (default rooms 1-2).  Room 3 is
NEVER seen.  Eval each EVERY steps: matched activity acc + wrong-perm acc
on heldout (in-domain) and on the held-out room.  best.pt by in-domain
matched acc.

  TRENVS=classroom,meeting_room HOURS=2 python3 multi-person/train_persontok.py
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

TOK = os.path.expanduser(os.environ.get(
    "WTOK", "~/zerdani/buffer/octonet/wimans_tokens"))
RUNS = os.path.expanduser(os.environ.get(
    "RUNS", "~/zerdani/buffer/octonet/persontok_runs"))
STEPS = int(os.environ.get("STEPS", "25000"))
HOURS = float(os.environ.get("HOURS", "2"))
B = int(os.environ.get("B", "16"))
LR = float(os.environ.get("LR", "1e-4"))
WARM = int(os.environ.get("WARM", "1000"))
ENTW = float(os.environ.get("ENTW", "0.03"))
M = int(os.environ.get("M", "8"))
D = int(os.environ.get("DIM", "256"))
NL = int(os.environ.get("LAYERS", "6"))
MAXT = int(os.environ.get("MAXT", "768"))
EVERY = int(os.environ.get("EVERY", "500"))
SEED = int(os.environ.get("SEED", "0"))
TRENVS = set(e for e in os.environ.get(
    "TRENVS", "classroom,meeting_room").split(",") if e)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
FDIM = 7          # updated by load_items: +3 rx one-hot for v2 tokens

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(FDIM, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        return torch.softmax(self.head(h), -1), h

class ActHead(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(),
                                 nn.Linear(D, nc))
    def forward(self, s): return self.net(s)

def load_items():
    an = pd.read_csv(f"{TOK}/manifest.csv")
    an.columns = [c.strip("﻿") for c in an.columns]
    vocab = sorted({str(getattr(r, f"user_{k}_activity")).strip()
                    for r in an.itertuples() for k in range(1, 6)
                    if isinstance(getattr(r, f"user_{k}_activity"), str)})
    v2i = {v: i for i, v in enumerate(vocab)}
    items = []
    for r in an.itertuples():
        n = int(r.number_of_users)
        if n < 1: continue
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
        if t.shape[1] >= 6:                     # v2 tokens: rx one-hot
            hot = np.zeros((len(t), 3), np.float32)
            hot[np.arange(len(t)), t[:, 5].astype(int)] = 1
            X = np.c_[X, hot]
        e = (10.0 ** le).astype(np.float32)
        e = np.minimum(e, np.quantile(e, 0.95))
        e = e / (e.mean() + 1e-12)
        items.append(dict(X=X, e=e, n=n, env=r.environment, acts=acts))
    global FDIM
    if items: FDIM = items[0]["X"].shape[1]
    return items, vocab

def forward_batch(model, items, ix):
    n = max(len(items[i]["X"]) for i in ix)
    X = torch.zeros(len(ix), n, FDIM)
    E = torch.zeros(len(ix), n)
    mask = torch.ones(len(ix), n, dtype=torch.bool)
    for k, i in enumerate(ix):
        L = len(items[i]["X"])
        X[k, :L] = torch.from_numpy(items[i]["X"])
        E[k, :L] = torch.from_numpy(items[i]["e"])
        mask[k, :L] = False
    a, h = model(X.to(dev), mask.to(dev))
    return a, h, E.to(dev), mask.to(dev)

def slot_sigs(a, h, E, mask):
    """(N-token) -> (M, D) energy-weighted slot signatures, one sample."""
    w = a * E[:, None] * (~mask)[:, None].float()        # (N, M)
    s = w.T @ h                                          # (M, D)
    return s / (w.sum(0)[:, None] + 1e-6)

def pit_loss(head, sig, acts):
    lg = head(sig)                                       # (M, nc)
    y = torch.tensor(acts, device=lg.device)
    cost = torch.stack([nn.functional.cross_entropy(
        lg.expand(len(acts), -1, -1)[u], y[u].expand(M), reduction="none")
        for u in range(len(acts))], 1)                   # (M, n)
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    matched = cost[r, c].mean()
    pred = lg[r].argmax(1)
    acc = float((pred == y[c]).float().mean())
    wrong = float("nan")
    if len(acts) > 1:
        rr = np.roll(r, 1)
        wrong = float((lg[rr].argmax(1) == y[c]).float().mean())
    return matched, acc, wrong

def usage_entropy(a, E, mask):
    w = (E * (~mask).float())
    q = (a * w[:, :, None]).sum((0, 1))
    q = q / (q.sum() + 1e-8)
    return -(q * torch.log(q + 1e-8)).sum()

def main():
    os.makedirs(RUNS, exist_ok=True)
    items, vocab = load_items()
    rng = np.random.default_rng(SEED)
    intrain = [i for i, it in enumerate(items) if it["env"] in TRENVS]
    xroom = [i for i, it in enumerate(items) if it["env"] not in TRENVS]
    ix = rng.permutation(intrain)
    nho = int(len(ix) * 0.1)
    ho = list(ix[:nho])
    tr = list(ix[nho:])
    evx = list(rng.permutation(xroom)[:400])
    print(f"{len(items)} items | train {len(tr)} ho {len(ho)} "
          f"(envs {sorted(TRENVS)}) | xroom eval {len(evx)} | "
          f"acts {vocab} | dev={dev}", flush=True)

    model = SetSep().to(dev)
    head = ActHead(len(vocab)).to(dev)
    params = list(model.parameters()) + list(head.parameters())
    print(f"params {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARM) *
        0.5 * (1.0 + math.cos(math.pi * s / STEPS)))
    step0, best = 0, -math.inf
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev,
                        weights_only=False)
        model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
        step0 = ck["step"]; best = ck["best"]
        print(f"resumed from step {step0}", flush=True)

    def evaluate(pool):
        model.eval(); head.eval()
        accs, wrongs = [], []
        with torch.no_grad():
            for i0 in range(0, len(pool), 32):
                sub = pool[i0:i0 + 32]
                a, h, E, mask = forward_batch(model, items, sub)
                for k, i in enumerate(sub):
                    sig = slot_sigs(a[k], h[k], E[k], mask[k])
                    _, acc, wrong = pit_loss(head, sig, items[i]["acts"])
                    accs.append(acc)
                    if not math.isnan(wrong): wrongs.append(wrong)
        model.train(); head.train()
        return float(np.mean(accs)), \
            float(np.mean(wrongs)) if wrongs else float("nan")

    t0 = time.time()
    for step in range(step0, STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        bix = [tr[rng.integers(len(tr))] for _ in range(B)]
        a, h, E, mask = forward_batch(model, items, bix)
        losses = []
        ent = usage_entropy(a, E, mask)
        for k, i in enumerate(bix):
            sig = slot_sigs(a[k], h[k], E[k], mask[k])
            matched, _, _ = pit_loss(head, sig, items[i]["acts"])
            losses.append(matched)
        loss = torch.stack(losses).mean() + ENTW * (math.log(M) - ent)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % EVERY == 0:
            ai, wi = evaluate(ho)
            ax, wx = evaluate(evx)
            ck = {"model": model.state_dict(), "head": head.state_dict(),
                  "opt": opt.state_dict(), "sch": sch.state_dict(),
                  "step": step, "best": best, "vocab": vocab,
                  "M": M, "D": D, "NL": NL}
            torch.save(ck, f"{RUNS}/last.pt")
            if ai > best:
                best = ai
                torch.save(ck, f"{RUNS}/best.pt")
            print(f"[{step}] loss {loss.item():.3f}  IN acc {ai:.3f} "
                  f"wrong {wi:.3f} (best {best:.3f})  XROOM acc {ax:.3f} "
                  f"wrong {wx:.3f}  {(time.time()-t0)/3600:.2f}h",
                  flush=True)
    torch.save({"model": model.state_dict(), "head": head.state_dict(),
                "opt": opt.state_dict(), "sch": sch.state_dict(),
                "step": step, "best": best, "vocab": vocab,
                "M": M, "D": D, "NL": NL}, f"{RUNS}/last.pt")
    print(f"done at step {step}, best IN acc {best:.3f}", flush=True)

if __name__ == "__main__":
    main()
