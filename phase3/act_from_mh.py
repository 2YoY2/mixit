#!/usr/bin/env python3
"""Movement-classification battery on the MH-WTA head's skeletons.

Three arms through the same GRU classifier (act_from_pose protocol):
  ORACLE-pose    per-clip best-of-K hypothesis, picked WITH GT (diagnostic
                 upper bound: how much movement semantics the hypothesis
                 SET carries — if this nears the GT ceiling, solving
                 selection delivers PCK and semantics together)
  SELECTED-pose  the deployable selector pick
  GT ceiling     ground-truth skeletons

Env: TRSC/TESC scenes (default 1,2,3 -> 4), INDOM>0 = heldout fraction of
TRSC clips as in-domain test (TESC ignored).  CK = MH checkpoint.
LANDMINE: CKPT env consumed by train_posetok3 import — popped here.
"""
import os, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.pop("CKPT", None)
_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "mh", os.path.join(_dir, "train_posetok_mh.py"))
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)
ptk = mh.ptk
K, NJ, ROOTJ, dev = mh.K, ptk.NJ, ptk.ROOTJ, ptk.dev

CK = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/posetok_mh_runs/last.pt"))
STEPS = int(os.environ.get("STEPS", "6000"))
TRSC = [int(v) for v in os.environ.get("TRSC", "1,2,3").split(",")]
TESC = [int(v) for v in os.environ.get("TESC", "4").split(",")]
INDOM = float(os.environ.get("INDOM", "0"))
B = int(os.environ.get("B", "64"))
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}

net = mh.MHPoseTok().to(dev)
ck = torch.load(CK, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"MH model {CK} step {ck.get('step')} K={K}", flush=True)

# mu/sd from the training-split recipe (needed to de-normalize hypotheses)
# MUSC must match the scenes the pose ckpt was TRAINED on (s1-model: MUSC=1)
MUSC = [int(v) for v in os.environ.get("MUSC", "1,2,3").split(",")]
LIMBSEL = os.environ.get("LIMBSEL", "")
_smap = None
def _limbsel(items):
    """same filter as train_posetok_mh: predominant-limb slot tokens only."""
    global _smap
    if not LIMBSEL: return items
    if _smap is None:
        z = np.load(os.path.expanduser(LIMBSEL))
        _smap = {int(r): int(m) for r, m in zip(z["rids"], z["mask"])}
    out = []
    for it in items:
        tok = it[0]
        rx = tok[:, 15:18].astype(np.float32).argmax(1)
        sl = tok[:, 7:15].astype(np.float32).argmax(1)
        keep = np.zeros(len(tok), bool)
        for i_ in range(3):
            m_ = _smap.get(int(it[3][i_]), 0)
            keep |= (rx == i_) & (np.right_shift(m_, sl) & 1 > 0)
        if keep.sum() < 16: continue
        out.append((tok[keep],) + tuple(it[1:]))
    return out
_tr_all = _limbsel(ptk.build(MUSC))
_rng = np.random.default_rng(ptk.SEED)
_ix = _rng.permutation(len(_tr_all))
_tr = [_tr_all[i] for i in _ix[:int(len(_ix) * 0.95)]]
mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([it[1][:, j][np.isfinite(it[1][:, j]).all(-1)]
                         for it in _tr if np.isfinite(it[1][:, j]).any()])
    if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3

man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
RID2ACT = {int(r.rid): int(r.act) for r in man.itertuples()}
RID2SC = {int(r.rid): int(r.scene) for r in man.itertuples()}

def pose_sets(scenes):
    ds = _tr_all if scenes == MUSC else _limbsel(ptk.build(scenes))
    out = []
    with torch.no_grad():
        for it in ds:
            tok, P, nw, rids = it[0], it[1], it[2], it[3]
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev) \
                if ptk.STATIC else None
            pred, sl = net(X, mask, [nw], st)
            pz = pred[0, :len(P)].cpu().numpy()
            hyps = pz * sd[None, None] + mu[None, None]      # (T,K,NJ,3)
            rs = [ptk.mpjpe_pck(hyps[:, k_], P) for k_ in range(K)]
            if not any(rs): continue
            ora = int(np.argmin([r[0] if r else 1e9 for r in rs]))
            selk = int(sl[0].argmax())
            act = RID2ACT.get(int(rids[0]))
            if act is None: continue
            gt = np.nan_to_num(P.reshape(len(P), -1))
            out.append((hyps[:, ora].reshape(len(P), -1).astype(np.float32),
                        hyps[:, selk].reshape(len(P), -1).astype(np.float32),
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
    for i, it in enumerate(tr): bycls.setdefault(it[3], []).append(i)
    keys = sorted(bycls)
    for step in range(STEPS):
        ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
               for c in rng.integers(0, len(keys), B)]
        items = [tr[i] for i in ixb]
        n = max(len(it[ai]) for it in items)
        X = torch.zeros(B, n, 45)
        L_ = torch.tensor([len(it[ai]) for it in items]).float()
        y = torch.tensor([it[3] for it in items])
        for k_, it in enumerate(items):
            X[k_, :len(it[ai])] = torch.from_numpy(it[ai])
        loss = nn.functional.cross_entropy(cls(X.to(dev), L_.to(dev)),
                                           y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    cls.eval()
    P_, Y_ = [], []
    with torch.no_grad():
        for it in te:
            lg = cls(torch.from_numpy(it[ai])[None].to(dev),
                     torch.tensor([len(it[ai])]).float().to(dev))
            P_.append(int(lg.argmax())); Y_.append(it[3])
    P_, Y_ = np.array(P_), np.array(Y_)
    Pm = np.array([MIRROR.get(v + 1, v + 1) for v in P_])
    Ym = np.array([MIRROR.get(v + 1, v + 1) for v in Y_])
    print(f"[{tag}] test: 17-class {(P_ == Y_).mean():.3f}  "
          f"mirror-merged {np.mean(Pm == Ym):.3f}  (chance 0.059)",
          flush=True)
    return P_, Y_

print(f"building pose sets (train {TRSC}, test {TESC}, INDOM={INDOM})",
      flush=True)
tr = pose_sets(TRSC)
if INDOM > 0:
    rng0 = np.random.default_rng(0)
    ixp = rng0.permutation(len(tr))
    ncut = int(len(ixp) * (1 - INDOM))
    te = [tr[i] for i in ixp[ncut:]]
    tr = [tr[i] for i in ixp[:ncut]]
else:
    te = pose_sets(TESC)
print(f"train {len(tr)} test {len(te)}", flush=True)
Po, Yo = run_arm("ORACLE-pose", 0, tr, te)
run_arm("SELECTED-pose", 1, tr, te)
run_arm("GT ceiling", 2, tr, te)
print("\nconfusion (ORACLE-pose arm, rows=true, top-3):", flush=True)
for k_ in range(NC):
    m = Yo == k_
    if not m.any(): continue
    cnt = np.bincount(Po[m], minlength=NC) / m.sum()
    top = np.argsort(-cnt)[:3]
    row = "  ".join(f"{NAMES[t]} {cnt[t]*100:.0f}%" for t in top
                    if cnt[t] > 0)
    print(f"  {NAMES[k_]:10s} (n={m.sum():4d}) -> {row}", flush=True)
