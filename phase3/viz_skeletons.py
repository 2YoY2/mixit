#!/usr/bin/env python3
"""Render predicted-vs-GT skeleton GIFs for chosen actions (mh3 oracle).

For each requested action: pick one clip (scene SC), run the MH pose model,
take the oracle-of-8 hypothesis (the skeletons the batteries classified),
animate GT (left) vs prediction (right) as BODY25[:15] stick figures.

  CK=~/.../posetok_mh3_runs/best.pt TOK=~/.../pa_tokens_fine2 \
  ACTS=15,14,13,11 SC=1 python3 phase3/viz_skeletons.py
"""
import os
os.environ.pop("CKPT", None)
import importlib.util
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

spec = importlib.util.spec_from_file_location(
    "mh", os.path.join(os.path.dirname(__file__), "train_posetok_mh.py"))
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)
ptk = mh.ptk

CK = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/octonet/posetok_mh3_runs/best.pt"))
OUTD = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/octonet/archive2"))
ACTS = [int(v) for v in os.environ.get("ACTS", "15,14,13,11").split(",")]
SC = int(os.environ.get("SC", "1"))
K = mh.K
NJ = ptk.NJ
dev = ptk.dev
ANAMES = {15: "squat", 14: "jumpjack", 13: "ccw-spin", 11: "pick-up",
          12: "cw-spin", 10: "jump", 8: "L-sid-lun", 9: "R-sid-lun"}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]

net = mh.MHPoseTok().to(dev)
ck = torch.load(CK, map_location=dev, weights_only=False)
net.load_state_dict(ck["model"]); net.eval()
print(f"MH model {CK} step {ck.get('step')}", flush=True)

items = ptk.build([1, 2, 3]) if SC in (1, 2, 3) else ptk.build([SC])
rng = np.random.default_rng(0)
ix = rng.permutation(len(items))
tr = [items[i] for i in ix[:int(len(ix) * 0.95)]]
mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([it[1][:, j][np.isfinite(it[1][:, j]).all(-1)]
                         for it in tr if np.isfinite(it[1][:, j]).any()])
    if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3

man = pd.read_csv(f"{ptk.TOK}/manifest.csv")
R2A = {int(r.rid): int(r.act) for r in man.itertuples()}
R2S = {int(r.rid): int(r.scene) for r in man.itertuples()}
ho = [items[i] for i in ix[int(len(ix) * 0.95):]]     # heldout clips only

def render(P, Q, title, path):
    """P GT (T,15,3), Q pred (T,15,3) -> side-by-side gif."""
    fin = np.isfinite(P).all((1, 2))
    P, Q = P[fin], Q[fin]
    if len(P) < 8: return False
    var = np.nanvar(P.reshape(-1, 3), 0)
    ax0, ax1 = np.argsort(-var)[:2]           # two most animated axes
    lo = np.nanpercentile(P.reshape(-1, 3), 2, 0) - 0.15
    hi = np.nanpercentile(P.reshape(-1, 3), 98, 0) + 0.15
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    fig.suptitle(title)
    arts = []
    for ax, nm, col in ((axes[0], "ground truth", "tab:green"),
                        (axes[1], "model (oracle hyp)", "tab:red")):
        ax.set_xlim(lo[ax0], hi[ax0]); ax.set_ylim(lo[ax1], hi[ax1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=9)
        lines = [ax.plot([], [], "-o", color=col, ms=2, lw=1.5)[0]
                 for _ in EDGES]
        arts.append(lines)
    def frame(t):
        for S, lines in ((P[t], arts[0]), (Q[t], arts[1])):
            for (a, b), ln in zip(EDGES, lines):
                ln.set_data([S[a, ax0], S[b, ax0]], [S[a, ax1], S[b, ax1]])
        return [l for ls in arts for l in ls]
    ani = FuncAnimation(fig, frame, frames=len(P), blit=True)
    ani.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return True

done = []
with torch.no_grad():
    for act in ACTS:
        got = False
        for it in ho:
            rid = int(it[3][0])
            if R2A.get(rid) != act or R2S.get(rid) != SC: continue
            tok, P, nw, rids = it[0], it[1], it[2], it[3]
            if not np.isfinite(P).any(): continue
            X = torch.from_numpy(np.asarray(tok, np.float32))[None].to(dev)
            mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
            st = torch.from_numpy(ptk.get_static(rids))[None].to(dev) \
                if ptk.STATIC else None
            pred, sl = net(X, mask, [nw], st)
            pz = pred[0, :len(P)].cpu().numpy()
            hyps = pz * sd[None, None] + mu[None, None]
            errs = []
            for k_ in range(K):
                d = np.linalg.norm(hyps[:, k_] - P, axis=-1)
                errs.append(np.nanmean(d))
            kbest = int(np.nanargmin(errs))
            nm = ANAMES.get(act, f"act{act}")
            path = f"{OUTD}/skel_{nm}.gif"
            if render(P, hyps[:, kbest],
                      f"{nm} (scene {SC}, MPJPE "
                      f"{errs[kbest]*1000:.0f}mm)", path):
                print(f"  {nm}: rid {rid} oracle-hyp {kbest} "
                      f"mpjpe {errs[kbest]*1000:.0f}mm -> {path}",
                      flush=True)
                done.append(path)
                got = True
                break
        if not got:
            print(f"  act {act}: no heldout clip found", flush=True)
print(f"rendered {len(done)} gifs", flush=True)
