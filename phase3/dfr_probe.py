#!/usr/bin/env python3
"""PHASE 3 (2): DFR probe (Kirichenko 2204.02937) on the v2 statics model:
freeze everything, retrain ONLY the output layer with balanced sampling.
If v2's posture knowledge is real-but-misweighted, scene-4 recovers toward
~116 while keeping PCK; if features themselves are room-bound, no head
reweighting can save it.

  POSECKPT=~/zerdani/buffer/octonet/posetok_runs/best.pt \
  STATIC=1 TEMPL=0 SLOTQ=0 STATTOK=0 python3 phase3/dfr_probe.py
"""
import os, time, importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "..", "bench",
                        "train_posetok.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

POSECKPT = os.path.expanduser(os.environ.get(
    "POSECKPT", "~/zerdani/buffer/octonet/posetok_runs/best.pt"))
STEPS = int(os.environ.get("STEPS", "3000"))
B = 12
dev = ptk.dev
np_ = np

net = ptk.PoseTok().to(dev)
ck = torch.load(POSECKPT, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"])
print(f"loaded {POSECKPT} step {ck.get('step')}", flush=True)
for p_ in net.parameters(): p_.requires_grad = False
for p_ in net.out.parameters(): p_.requires_grad = True

tr_all = ptk.build([1, 2, 3])
te = ptk.build([4])
rng = np.random.default_rng(0)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.95):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.95)]]
NJ, ROOTJ = ptk.NJ, ptk.ROOTJ
mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([it[1][:, j][np.isfinite(it[1][:, j]).all(-1)]
                         for it in tr if np.isfinite(it[1][:, j]).any()])
    if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3
MUt = torch.from_numpy(mu.astype(np.float32)).to(dev)
SDt = torch.from_numpy(sd.astype(np.float32)).to(dev)

def qev(ds, cap):
    rs = []
    with torch.no_grad():
        for it in ds[:cap]:
            tok, P, nw, rids = it[0], it[1], it[2], it[3]
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev)
            pr = net(X, mask, [nw], st)[0, :len(P)].cpu().numpy()
            pr = pr * sd + mu
            r = ptk.mpjpe_pck(pr, P)
            if r: rs.append(r)
    rs = np.array(rs)
    return rs[:, 0].mean(), rs[:, 1].mean() * 100, rs[:, 2].mean() * 100

net.eval()
h0 = qev(ho, len(ho)); t0_ = qev(te, len(te))
print(f"BEFORE: heldout {h0[0]:.0f}mm PCK20/50 {h0[1]:.1f}/{h0[2]:.1f} | "
      f"scene4 {t0_[0]:.0f}mm {t0_[1]:.1f}/{t0_[2]:.1f}", flush=True)

opt = torch.optim.Adam(net.out.parameters(), lr=1e-3)
t0 = time.time()
for step in range(STEPS):
    ixb = rng.choice(len(tr), B)
    items = [tr[i] for i in ixb]
    n = max(len(it[0]) for it in items)
    nws = [it[2] for it in items]
    X = torch.zeros(B, n, 18)
    mask = torch.ones(B, n, dtype=torch.bool)
    Y = torch.full((B, max(nws), NJ, 3), np.nan)
    S = torch.zeros(B, 3, 399)
    for k, it in enumerate(items):
        X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
        mask[k, :len(it[0])] = False
        Y[k, :len(it[1])] = torch.from_numpy(it[1])
        S[k] = torch.from_numpy(ptk.get_static(it[3]))
    X, mask, Y, S = X.to(dev), mask.to(dev), Y.to(dev), S.to(dev)
    pred = net(X, mask, nws, S)
    Z = (Y - MUt) / SDt
    msk = torch.isfinite(Y).all(-1, keepdim=True)
    msk[:, :, ROOTJ] = False
    loss = (torch.where(msk, (pred - torch.nan_to_num(Z)).abs(),
                        torch.zeros_like(pred)).sum()
            / msk.sum().clamp(min=1) / 3)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"[{step}] {loss.item():.3f}", flush=True)
net.eval()
h1 = qev(ho, len(ho)); t1_ = qev(te, len(te))
print(f"AFTER : heldout {h1[0]:.0f}mm PCK20/50 {h1[1]:.1f}/{h1[2]:.1f} | "
      f"scene4 {t1_[0]:.0f}mm {t1_[1]:.1f}/{t1_[2]:.1f}", flush=True)
print("""READ: scene4 recovering toward ~116 with PCK kept -> posture was
learned, head misweighted (DFR positive). No recovery -> features room-bound.""")
