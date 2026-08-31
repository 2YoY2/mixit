#!/usr/bin/env python3
"""Probe 40: WHAT does v2-BIG read from its raw-static input?

Eval-time static corruption on the frozen checkpoint (no training).
Modes, per eval set:
  true      statics as-is (reproduces battery part-1 numbers)
  roommean  per-(scene,node) mean static = room print only, zero
            per-recording info.  ~true -> pure session/room lookup.
  shufscene donor = another clip, same scene (room kept, person/stance
            destroyed, but a *real* person is still in the static)
  shufact   donor = same scene AND same action (action-typical stance
            kept; per-recording person/idiosyncrasy destroyed)
  zeros     static path silenced -> total contribution of the input
  wrongroom (scene-4 only) donor from scenes 1-3 -> poison direction

Reading: true==roommean==shufscene -> statics act as a room-ID
(adaptation-tier interpretation of v2 stands, nothing per-recording is
decoded).  true beats shufscene -> genuine per-recording posture/person
signal; shufact vs shufscene separates stance-of-action from person.

env must match ckpt arch (STATIC=1 TEMPL=0 SLOTQ=0 STATTOK=0 PERSTOK=0
TOKOFF=0 POSESLOTS=1,2,3 HDIM=384 ENCL=5 HEADS=8).  Pose ckpt via PCKPT.
LANDMINE: CKPT env is consumed by the train_posetok3 import (separator,
RUNS-relative) — this script deletes it before importing.
"""
import os, importlib.util
import numpy as np
import pandas as pd
import torch

os.environ.pop("CKPT", None)                 # see landmine note above
spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "..", "phase3", "legacy",
                        "train_posetok3.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

PCKPT = os.path.expanduser(os.environ.get(
    "PCKPT", "~/zerdani/buffer/octonet/posetok_v2big_runs/best.pt"))
dev = ptk.dev
NJ, ROOTJ = ptk.NJ, ptk.ROOTJ

net = ptk.PoseTok().to(dev)                  # claim GPU before heavy reads
for _try in range(60):
    try:
        torch.zeros(256, 1024, 1024, device=dev); torch.cuda.empty_cache()
        break
    except RuntimeError:
        import time; print("GPU busy, retry in 60s", flush=True)
        time.sleep(60)
ck = torch.load(PCKPT, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"ckpt {PCKPT} step {ck.get('step')}", flush=True)

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

man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
R2S = dict(zip(man.rid.astype(int), man.scene.astype(int)))
R2A = dict(zip(man.rid.astype(int), man.act))
prng = np.random.default_rng(40)

def donor_map(ds, keyfn):
    """cyclic derangement within groups of keyfn -> idx: donor idx"""
    groups = {}
    for i, it in enumerate(ds):
        groups.setdefault(keyfn(it), []).append(i)
    dm = {}
    for g in groups.values():
        order = prng.permutation(g)
        for a, b in zip(order, np.roll(order, 1)):
            dm[int(a)] = int(b)
    return dm

def room_means(ds):
    """per-(scene,node-slot) mean static over the whole set"""
    acc, cnt = {}, {}
    for it in ds:
        st = ptk.get_static(it[3])                      # (3, 399)
        sc = R2S[int(it[3][0])]
        for k in range(3):
            key = (sc, k)
            acc[key] = acc.get(key, 0) + st[k]
            cnt[key] = cnt.get(key, 0) + 1
    return {k: (acc[k] / cnt[k]).astype(np.float32) for k in acc}

def run(ds, tag, donors=None, zero=False, rmeans=None, pool=None):
    rs, sp, tc = [], [], []
    JS = [j for j in range(NJ) if j != ROOTJ]
    with torch.no_grad():
        for i, it in enumerate(ds):
            tok, P, nw, rids = it[0], it[1], it[2], it[3]
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            if zero:
                st = torch.zeros(1, 3, 399, device=dev)
            elif rmeans is not None:
                sc = R2S[int(rids[0])]
                st = torch.from_numpy(np.stack(
                    [rmeans[(sc, k)] for k in range(3)]))[None].to(dev)
            elif donors is not None:
                src = (pool if pool is not None else ds)[donors[i]][3]
                st = torch.from_numpy(ptk.get_static(src))[None].to(dev)
            else:
                st = torch.from_numpy(ptk.get_static(rids))[None].to(dev)
            pr = net(X, mask, [nw], st, None, None)[0, :len(P)].cpu().numpy()
            pr = pr * sd + mu
            r = ptk.mpjpe_pck(pr, P)
            if not r: continue
            rs.append(r)
            m = np.isfinite(P).all(-1); m[:, ROOTJ] = False
            if m.sum() > NJ:
                sp.append(float(np.nanmean(pr[:, JS].std(0)) * 100))
                pd_ = (pr - pr.mean(0))[m]
                gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
                den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
                tc.append(float((pd_ * gd).sum() / den))
    rs = np.array(rs)
    print(f"[{tag:>28}] MPJPE {rs[:,0].mean():6.1f} mm  PCK@20 "
          f"{rs[:,1].mean()*100:5.1f}  PCK@50 {rs[:,2].mean()*100:5.1f}  "
          f"pred-std {np.median(sp):4.1f} cm  traj-corr {np.median(tc):+.3f}"
          f"  (n={len(rs)})", flush=True)

for ds, name in ((ho, "heldout 1-3"), (te, "TEST scene4")):
    print(f"--- {name} ---", flush=True)
    rm = room_means(ds)
    run(ds, f"{name} true")
    run(ds, f"{name} roommean", rmeans=rm)
    run(ds, f"{name} shufscene",
        donors=donor_map(ds, lambda it: R2S[int(it[3][0])]))
    run(ds, f"{name} shufact",
        donors=donor_map(ds, lambda it: (R2S[int(it[3][0])],
                                         R2A[int(it[3][0])])))
    run(ds, f"{name} zeros", zero=True)
    if name == "TEST scene4":
        dm = {i: int(prng.integers(len(tr_all))) for i in range(len(ds))}
        run(ds, f"{name} wrongroom", donors=dm, pool=tr_all)
print("probe 40 done", flush=True)
