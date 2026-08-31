#!/usr/bin/env python3
"""Full battery, part 1, for a trained posetok3 checkpoint: full heldout +
scene-4 eval with collapse diagnostics (dist-to-meanpose, pred vs GT
temporal std, traj-corr, paired-beats-baseline).

  env must match the ckpt arch. CKPT=~/.../best.pt python3 phase3/eval_v2big.py
"""
import os, importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "legacy",
                        "train_posetok3.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

CKPT = os.path.expanduser(os.environ.get(
    "POSECK", "~/zerdani/buffer/octonet/posetok_v2big_runs/best.pt"))
dev = ptk.dev
NJ, ROOTJ = ptk.NJ, ptk.ROOTJ

net = ptk.PoseTok().to(dev)
ck = torch.load(CKPT, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"ckpt {CKPT} step {ck.get('step')}", flush=True)

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

def run(ds, tag):
    JS = [j for j in range(NJ) if j != ROOTJ]
    rs, dm, sp, sg, tc, wins = [], [], [], [], [], []
    with torch.no_grad():
        for it in ds:
            tok, P, nw, rids = it[0], it[1], it[2], it[3]
            X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev)
            qs = (torch.from_numpy(it[4].astype(np.float32))[None].to(dev)
                  if ptk.SLOTQ else None)
            spt = None
            pr = net(X, mask, [nw], st, qs, spt)[0, :len(P)].cpu().numpy()
            pr = pr * sd + mu
            r = ptk.mpjpe_pck(pr, P)
            if not r: continue
            rs.append(r)
            m = np.isfinite(P).all(-1); m[:, ROOTJ] = False
            if m.sum() > NJ:
                dm.append(float(np.linalg.norm((pr - mu)[m], axis=-1
                                               ).mean() * 100))
                sp.append(float(np.nanmean(pr[:, JS].std(0)) * 100))
                sg.append(float(np.nanmean(np.nanstd(P[:, JS], 0)) * 100))
                pd_ = (pr - pr.mean(0))[m]
                gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
                den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
                tc.append(float((pd_ * gd).sum() / den))
                wins.append(r[0] < ptk.mpjpe_pck(
                    np.broadcast_to(mu, P.shape), P)[0])
    rs = np.array(rs)
    print(f"[{tag}] MPJPE {rs[:,0].mean():.0f} mm  PCK@20 "
          f"{rs[:,1].mean()*100:.1f}  PCK@50 {rs[:,2].mean()*100:.1f}  "
          f"(n={len(rs)})", flush=True)
    print(f"  DIAG: dist-to-mean {np.median(dm):.1f} cm | pred-std "
          f"{np.median(sp):.1f} vs GT {np.median(sg):.1f} cm | traj-corr "
          f"{np.median(tc):+.3f} | beats-baseline {np.mean(wins)*100:.0f}%",
          flush=True)

run(ho, "heldout 1-3")
run(te, "TEST scene4")
