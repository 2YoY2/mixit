#!/usr/bin/env python3
"""Body-action estimator on the TOKENS THEMSELVES (v1 input: motion tokens
+ slot posteriors + rx one-hot; NO statistics). Token-set transformer ->
17 actions. Train scenes 1-3 (10% clip heldout), test scene 4. Confusion
matrix on the transfer split.

  python3 bench/act_from_tokens.py
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

STEPS = int(os.environ.get("STEPS", "8000"))
B = int(os.environ.get("B", "32"))
LR = float(os.environ.get("LR", "5e-4"))
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}
dev = ptk.dev

man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
RID2ACT = {int(r.rid): int(r.act) for r in man.itertuples()}

def sets(scenes):
    ds = ptk.build(scenes)
    out = []
    for it in ds:
        act = RID2ACT.get(int(it[3][0]))
        if act is None: continue
        out.append((it[0], act - 1))
    return out

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

print("building sets", flush=True)
tr_all = sets([1, 2, 3])
te = sets([4])
rng = np.random.default_rng(0)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
print(f"train {len(tr)} ho {len(ho)} test4 {len(te)}", flush=True)

net = TokCls().to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
# CKF: train-once/eval-many. If the file exists the classifier is LOADED
# and training is skipped (zero-retraining transfer protocol — the frozen
# scenes-1-3 model is evaluated on transformed test tokens); otherwise it
# trains as usual and saves there.
CKF = os.path.expanduser(os.environ.get("CKF", ""))
loaded = False
if CKF and os.path.exists(CKF):
    net.load_state_dict(torch.load(CKF, map_location=dev,
                                   weights_only=False)["model"])
    net.eval(); loaded = True
    print(f"loaded frozen classifier {CKF} (no training)", flush=True)
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
bycls = {}
for i, it in enumerate(tr): bycls.setdefault(it[1], []).append(i)
keys = sorted(bycls)
t0 = time.time()
for step in range(0 if loaded else STEPS):
    ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
           for c in rng.integers(0, len(keys), B)]
    items = [tr[i] for i in ixb]
    n = max(len(it[0]) for it in items)
    X = torch.zeros(B, n, 18)
    mask = torch.ones(B, n, dtype=torch.bool)
    y = torch.tensor([it[1] for it in items])
    for k, it in enumerate(items):
        X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
        mask[k, :len(it[0])] = False
    loss = nn.functional.cross_entropy(net(X.to(dev), mask.to(dev)),
                                       y.to(dev))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"[{step}] CE {loss.item():.3f} {(time.time()-t0)/3600:.2f}h",
              flush=True)
net.eval()
def ev(ds, tag):
    P, Y = [], []
    with torch.no_grad():
        for i0 in range(0, len(ds), 32):
            its = ds[i0:i0 + 32]
            n = max(len(it[0]) for it in its)
            X = torch.zeros(len(its), n, 18)
            mask = torch.ones(len(its), n, dtype=torch.bool)
            for k, it in enumerate(its):
                X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
                mask[k, :len(it[0])] = False
            lg = net(X.to(dev), mask.to(dev)).cpu().numpy()
            P += list(lg.argmax(1)); Y += [it[1] for it in its]
    P, Y = np.array(P), np.array(Y)
    Pm = np.array([MIRROR.get(v + 1, v + 1) for v in P])
    Ym = np.array([MIRROR.get(v + 1, v + 1) for v in Y])
    print(f"[{tag}] 17-class {np.mean(P == Y):.3f}  "
          f"merged {np.mean(Pm == Ym):.3f}  (chance 0.059)", flush=True)
    return P, Y
if CKF and not loaded:
    torch.save({"model": net.state_dict()}, CKF)
    print(f"saved classifier -> {CKF}", flush=True)
ev(ho, "heldout 1-3")
Pt, Yt = ev(te, "TEST scene4")
print("\nconfusion (scene 4, rows=true, top-3):")
for k in range(NC):
    m = Yt == k
    if not m.any(): continue
    cnt = np.bincount(Pt[m], minlength=NC) / m.sum()
    top = np.argsort(-cnt)[:3]
    row = "  ".join(f"{NAMES[t]} {cnt[t]*100:.0f}%" for t in top if cnt[t] > 0)
    print(f"  {NAMES[k]:10s} (n={m.sum():4d}) -> {row}", flush=True)
