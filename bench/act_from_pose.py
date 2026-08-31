#!/usr/bin/env python3
"""Movement classification FROM THE POSE MODEL'S OUTPUT (user's check):
small GRU on predicted skeleton sequences -> 17 actions, scenes 1-3 train,
scene 4 confusion matrix. Calibration arm: same classifier on GT skeletons
(the ceiling for window-rate pose->action). pred/GT gap = semantic fidelity
of the pose output.

Env must match the pose ckpt's architecture (SLOTQ/POSESLOTS/STATTOK/...).
  POSECKPT=~/zerdani/buffer/octonet/posetok_v7runs/best.pt \
  python3 bench/act_from_pose.py
"""
import os, time, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "train_posetok.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

POSECKPT = os.path.expanduser(os.environ.get(
    "POSECKPT", "~/zerdani/buffer/octonet/posetok_v7runs/best.pt"))
STEPS = int(os.environ.get("STEPS", "6000"))
TRSC = [int(v) for v in os.environ.get("TRSC", "1,2,3").split(",")]
TESC = [int(v) for v in os.environ.get("TESC", "4").split(",")]
INDOM = float(os.environ.get("INDOM", "0"))   # >0: heldout fraction of TRSC
                                              # clips as in-domain test set
                                              # (TESC ignored), seed-0 split
                                              # like act_from_tokens
B = int(os.environ.get("B", "64"))
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}
dev = ptk.dev

net = ptk.PoseTok().to(dev)
ck = torch.load(POSECKPT, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"pose model {POSECKPT} step {ck.get('step')}", flush=True)

man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
RID2ACT = {int(r.rid): int(r.act) for r in man.itertuples()}
RID2SC = {int(r.rid): int(r.scene) for r in man.itertuples()}

def pose_sets(scenes):
    ds = ptk.build(scenes)
    out = []
    with torch.no_grad():
        for it in ds:
            tok, P, nw, rids, S12, SPt = it
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            qs = torch.from_numpy(S12.astype(np.float32))[None].to(dev) \
                if ptk.SLOTQ else None
            spt = torch.from_numpy(SPt.astype(np.float32))[None].to(dev) \
                if ptk.STATTOK else None
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev) \
                if ptk.STATIC else None
            pr = net(X, mask, [nw], st, qs, spt)[0, :len(P)].cpu().numpy()
            act = RID2ACT.get(int(rids[0]))
            if act is None: continue
            gt = np.nan_to_num(P.reshape(len(P), -1))
            out.append((pr.reshape(len(pr), -1).astype(np.float32),
                        gt.astype(np.float32), act - 1,
                        RID2SC.get(int(rids[0]), 0)))
    return out

class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(45, 128, 2, batch_first=True)
        self.out = nn.Linear(128, NC)
    def forward(self, x, lens):
        h, _ = self.gru(x)
        p = (h * (torch.arange(h.shape[1], device=x.device)[None, :, None]
                  < lens[:, None, None]).float()).sum(1) / lens[:, None]
        return self.out(p)

def run_arm(tag, ai, tr, te):
    cls = Cls().to(dev)
    opt = torch.optim.Adam(cls.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    bycls = {}
    for i, it in enumerate(tr): bycls.setdefault(it[2], []).append(i)
    keys = sorted(bycls)
    for step in range(STEPS):
        ix = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
              for c in rng.integers(0, len(keys), B)]
        items = [tr[i] for i in ix]
        n = max(len(it[ai]) for it in items)
        X = torch.zeros(B, n, 45)
        L_ = torch.tensor([len(it[ai]) for it in items]).float()
        y = torch.tensor([it[2] for it in items])
        for k, it in enumerate(items):
            X[k, :len(it[ai])] = torch.from_numpy(it[ai])
        loss = nn.functional.cross_entropy(cls(X.to(dev), L_.to(dev)),
                                           y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    cls.eval()
    P, Y = [], []
    with torch.no_grad():
        for it in te:
            lg = cls(torch.from_numpy(it[ai])[None].to(dev),
                     torch.tensor([len(it[ai])]).float().to(dev))
            P.append(int(lg.argmax())); Y.append(it[2])
    P, Y = np.array(P), np.array(Y)
    acc = (P == Y).mean()
    Pm = np.array([MIRROR.get(v + 1, v + 1) for v in P])
    Ym = np.array([MIRROR.get(v + 1, v + 1) for v in Y])
    print(f"\n[{tag}] test: 17-class {acc:.3f}  "
          f"mirror-merged {np.mean(Pm == Ym):.3f}  (chance 0.059)", flush=True)
    return P, Y

print(f"building pose sets (train {TRSC}, test {TESC})", flush=True)
tr = pose_sets(TRSC)
if INDOM > 0:
    rng0 = np.random.default_rng(0)
    ixp = rng0.permutation(len(tr))
    ncut = int(len(ixp) * (1 - INDOM))
    te = [tr[i] for i in ixp[ncut:]]
    tr = [tr[i] for i in ixp[:ncut]]
    print(f"IN-DOMAIN mode: heldout {INDOM:.0%} of scenes {TRSC} clips "
          f"as test (TESC ignored)", flush=True)
else:
    te = pose_sets(TESC)
print(f"train {len(tr)} test {len(te)}", flush=True)
Pp, Yp = run_arm("PRED-pose", 0, tr, te)
Pg, Yg = run_arm("GT-pose ceiling", 1, tr, te)
if INDOM > 0:
    sc_arr = np.array([it[3] for it in te])
    print("\nper-scene (in-domain heldout):", flush=True)
    for sc in sorted(set(sc_arr.tolist())):
        m = sc_arr == sc
        print(f"  scene{sc}: PRED {np.mean(Pp[m] == Yp[m]):.3f}  "
              f"GT-ceiling {np.mean(Pg[m] == Yg[m]):.3f}  (n={m.sum()})",
              flush=True)
print("\nconfusion (PRED-pose arm, test scenes, rows=true, % of row, top-3 shown):")
for k in range(NC):
    m = Yp == k
    if not m.any(): continue
    cnt = np.bincount(Pp[m], minlength=NC) / m.sum()
    top = np.argsort(-cnt)[:3]
    row = "  ".join(f"{NAMES[t]} {cnt[t]*100:.0f}%" for t in top if cnt[t] > 0)
    print(f"  {NAMES[k]:10s} (n={m.sum():4d}) -> {row}", flush=True)
print("""
READ: PRED close to GT-ceiling -> the pose output preserves movement
semantics. PRED << ceiling -> pose output drops action-relevant detail.""")
