#!/usr/bin/env python3
"""Render GT-vs-prediction GIFs for the slot-stream decoder.

  SC=1 ACTS=15,14,13,11 CK=.../slotstream.r1/best.pt \
  OUTG=~/zerdani/buffer/cluster/logs/slotstream_s1 \
  python3 train/viz_slotstream.py
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

exec(compile(open(os.path.join(os.path.dirname(__file__),
                               "train_slotstream.py")).read().split(
    "def main()")[0], "train_slotstream.py", "exec"))

CKF = os.path.expanduser(os.environ.get(
    "CK", "~/zerdani/buffer/cluster/runs/downstream/slotstream.r1/best.pt"))
OUTG = os.path.expanduser(os.environ.get(
    "OUTG", "~/zerdani/buffer/cluster/logs/slotstream_s1"))
SC = int(os.environ.get("SC", "1"))
ACTS = [int(v) for v in os.environ.get("ACTS", "15,14,13,11").split(",")]
ANAMES = {15: "squat", 14: "jumpjack", 13: "ccw-spin", 11: "pick-up"}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
os.makedirs(OUTG, exist_ok=True)

net = SlotStream().to(dev)
net.load_state_dict(torch.load(CKF, map_location=dev,
                               weights_only=False)["model"])
net.eval()
print(f"slotstream {CKF} loaded", flush=True)

def build_act(scene, act):
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[(man.scene == scene) & (man.act == act)].copy()
    man["ckl"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    for _, g in man.groupby("ckl"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        pf = f"{POSED}/{rids[0]:06d}.npy"
        gf = f"{LIMBD}/{rids[0]:06d}.npy"
        if not (os.path.exists(pf) and os.path.exists(gf)): continue
        P = np.asarray(np.load(pf), np.float32)
        gi = np.asarray(np.load(gf), np.float32)
        G = residual_limbs(gi, len(P))
        root = P[:, ROOTJ]
        fs = []
        for i, r in enumerate(rids):
            F = rec_streams(r, root, G)
            if F is None: break
            F[:, :, 16 + i] = 1.0
            fs.append(F)
        if len(fs) != 3: continue
        nw = min(min(f.shape[1] for f in fs), len(P))
        return np.concatenate([f[:, :nw] for f in fs], 0), P[:nw]
    return None

def render(P, Q, title, path):
    fin = np.isfinite(P).all((1, 2))
    P, Q = P[fin], Q[fin]
    if len(P) < 8: return False
    var = np.nanvar(P.reshape(-1, 3), 0)
    a0, a1 = np.argsort(-var)[:2]
    lo = np.nanpercentile(P.reshape(-1, 3), 2, 0) - 0.15
    hi = np.nanpercentile(P.reshape(-1, 3), 98, 0) + 0.15
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    fig.suptitle(title)
    arts = []
    for ax, nm, col in ((axes[0], "ground truth", "tab:green"),
                        (axes[1], "slot-stream model", "tab:red")):
        ax.set_xlim(lo[a0], hi[a0]); ax.set_ylim(lo[a1], hi[a1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=9)
        arts.append([ax.plot([], [], "-o", color=col, ms=2, lw=1.5)[0]
                     for _ in EDGES])
    def fr(t):
        for S_, ls in ((P[t], arts[0]), (Q[t], arts[1])):
            for (e0, e1), ln in zip(EDGES, ls):
                ln.set_data([S_[e0, a0], S_[e1, a0]],
                            [S_[e0, a1], S_[e1, a1]])
        return [l for ls in arts for l in ls]
    ani = FuncAnimation(fig, fr, frames=len(P), blit=True)
    ani.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)
    return True

with torch.no_grad():
    for act in ACTS:
        clip = build_act(SC, act)
        if clip is None:
            print(f"  act {act}: none", flush=True); continue
        F, P = clip
        pr, anc = net(torch.from_numpy(
            F.astype(np.float32))[None].to(dev))
        pr = pr[0].cpu().numpy()
        nm = ANAMES.get(act, f"act{act}")
        e = mpjpe_np(pr, P)
        path = f"{OUTG}/skel_{nm}.gif"
        if render(P, pr, f"{nm} (scene {SC}, MPJPE {e:.0f}mm)", path):
            print(f"  {nm}: mpjpe {e:.0f}mm -> {path}", flush=True)
print("viz done", flush=True)
