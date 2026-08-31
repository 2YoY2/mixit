#!/usr/bin/env python3
"""Oracle-selection analysis for the MH-WTA head: is the oracle index
predictable from GT-free signals?  (user: "find a relation between the
data and the oracle")

Rankers scored per eval set (pick-acc vs oracle + MPJPE/PCK of the pick):
  sel      learned selector logits (the trained baseline)
  env-E    physics: corr(hypothesis speed profile, token energy envelope)
  env-F    physics: corr(speed profile, energy-weighted |Doppler| envelope)
  sel+env  z-scored sum of sel logits + env-E corr
  act-maj  DIAGNOSTIC ONLY (uses GT action): majority winner index of the
           clip's action on TRAIN clips -> how much of selection an action
           classifier could carry
  oracle   GT pick (ceiling); 'mean' row = shrunk control

Eval-only on a saved MH checkpoint (CK env, default last.pt).
LANDMINE: CKPT env consumed by train_posetok3 import — popped here.
"""
import os, importlib.util
import numpy as np
import torch

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
net = mh.MHPoseTok().to(dev)
ck = torch.load(CK, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"ckpt {CK} step {ck.get('step')}", flush=True)

tr_all = ptk.build([1, 2, 3])
te = ptk.build([4])
rng = np.random.default_rng(ptk.SEED)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.95):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.95)]]
mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([it[1][:, j][np.isfinite(it[1][:, j]).all(-1)]
                         for it in tr if np.isfinite(it[1][:, j]).any()])
    if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3

import pandas as pd
man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
R2A = {int(r.rid): int(r.act) for r in man.itertuples()}

def envelopes(S12, T):
    """(energy, |doppler|) per window from the cached window summaries"""
    s = S12[:T].astype(np.float64)
    E = (10.0 ** s[:, 0::6])                       # (T, nslots*nrx)
    f = np.abs(s[:, 1::6])
    w = E / np.maximum(E.sum(1, keepdims=True), 1e-12)
    return E.sum(1), (w * f).sum(1)

NSLOT = len(ptk.POSESLOTS)
def slot_envs(S12, T):
    """(NSLOT, T) per-slot energy summed over the 3 receivers.
    S12 layout: rx blocks of NSLOT*6 cols; col si*6 = log10 energy."""
    s = S12[:T].astype(np.float64)
    ns6 = NSLOT * 6
    return np.stack([sum(10.0 ** s[:, r * ns6 + si * 6] for r in range(3))
                     for si in range(NSLOT)])

from itertools import permutations
LIMBS = {"LW": [7], "RW": [4], "Lleg": [13, 14], "Rleg": [10, 11]}
def slot_score(hyp, sE):
    """best-assignment mean corr between slot envelopes and the
    hypothesis's per-limb speed profiles (phase-2 gate protocol)."""
    spd = np.linalg.norm(np.diff(hyp, axis=0), axis=-1)      # (T-1, NJ)
    le = [spd[:, js].mean(1) for js in LIMBS.values()]
    C = np.array([[zc(sE[si][1:], e) for e in le] for si in range(NSLOT)])
    return max(np.mean([C[si, p[si]] for si in range(NSLOT)])
               for p in permutations(range(len(le)), NSLOT))

def zc(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    if n < 4 or a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def clipstats(ds, tag):
    out = []
    with torch.no_grad():
        for it in ds:
            tok, P, nw, rids, S12 = it[0], it[1], it[2], it[3], it[4]
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev) \
                if ptk.STATIC else None
            pred, sl = net(X, mask, [nw], st)
            pz = pred[0, :len(P)].cpu().numpy()
            hyps = pz * sd[None, None] + mu[None, None]      # (T,K,NJ,3)
            rs = [ptk.mpjpe_pck(hyps[:, k_], P) for k_ in range(K)]
            if not any(rs): continue
            mp = np.array([r[0] if r else 1e9 for r in rs])
            spd = np.linalg.norm(np.diff(hyps, axis=0), axis=-1).mean(-1).T
            eE, eF = envelopes(S12, len(P))
            sE = slot_envs(S12, len(P))
            cE = np.array([zc(spd[k_], eE[1:]) for k_ in range(K)])
            cF = np.array([zc(spd[k_], eF[1:]) for k_ in range(K)])
            cS = np.array([slot_score(hyps[:, k_], sE) for k_ in range(K)])
            out.append(dict(
                rs=rs, mp=mp, oracle=int(mp.argmin()),
                sel=sl[0].cpu().numpy(), cE=cE, cF=cF, cS=cS,
                act=R2A.get(int(rids[0]), 0),
                mean_r=ptk.mpjpe_pck(hyps.mean(1), P)))
    print(f"{tag}: {len(out)} clips", flush=True)
    return out

def zs(v):
    v = np.asarray(v, np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)

def score(cs, name, pickfn):
    accs, rows = [], []
    for c in cs:
        k_ = pickfn(c)
        accs.append(k_ == c["oracle"])
        if c["rs"][k_]: rows.append(c["rs"][k_])
    rows = np.array(rows)
    print(f"  {name:8s} pick-acc {np.mean(accs)*100:4.1f}%  "
          f"MPJPE {rows[:,0].mean():6.1f}  PCK@20 {rows[:,1].mean()*100:5.1f}"
          f"  PCK@50 {rows[:,2].mean()*100:5.1f}", flush=True)

TRC = clipstats(tr[:3000], "train (act-maj fit)")
actmaj = {}
for a in set(c["act"] for c in TRC):
    ws = [c["oracle"] for c in TRC if c["act"] == a]
    actmaj[a] = int(np.bincount(ws, minlength=K).argmax())
print("winner-by-action concentration (train): " + "  ".join(
    f"a{a}:{np.bincount([c['oracle'] for c in TRC if c['act']==a], minlength=K).max()/max(1,len([1 for c in TRC if c['act']==a]))*100:.0f}%"
    for a in sorted(actmaj)), flush=True)

for ds, tag in ((ho, "heldout 1-3"), (te, "TEST scene4")):
    cs = clipstats(ds, tag)
    print(f"[{tag}] (n={len(cs)})", flush=True)
    score(cs, "sel", lambda c: int(c["sel"].argmax()))
    score(cs, "env-E", lambda c: int(c["cE"].argmax()))
    score(cs, "env-F", lambda c: int(c["cF"].argmax()))
    score(cs, "slotperm", lambda c: int(c["cS"].argmax()))
    score(cs, "sel+env", lambda c: int((zs(c["sel"]) + zs(c["cE"])).argmax()))
    score(cs, "sel+slot", lambda c: int((zs(c["sel"]) + zs(c["cS"])).argmax()))
    score(cs, "act-maj", lambda c: actmaj.get(c["act"], 0))
    score(cs, "oracle", lambda c: c["oracle"])
    mr = np.array([c["mean_r"] for c in cs if c["mean_r"]])
    print(f"  {'mean':8s} pick-acc  ---   MPJPE {mr[:,0].mean():6.1f}  "
          f"PCK@20 {mr[:,1].mean()*100:5.1f}  PCK@50 {mr[:,2].mean()*100:5.1f}",
          flush=True)
    orc = np.array([c["oracle"] for c in cs])
    print("  oracle-index histogram: " +
          " ".join(f"{k_}:{(orc==k_).mean()*100:.0f}%" for k_ in range(K)),
          flush=True)
print("rerank done", flush=True)
