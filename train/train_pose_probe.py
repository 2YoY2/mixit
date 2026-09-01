#!/usr/bin/env python3
"""Pose-transfer probe: is the limb-cluster model's OUTPUT a universal
representation? Train a small pose head (root-relative 3D joints, what the
PerceptAlign paper predicts) on SCENE 1 only, test on SCENE 4 (unseen room).

Arms, same head architecture, same budget, only the input differs:
  model  frozen limbtok12 slot features: per window, per slot: log energy
         + 3 Doppler-band log energies (M*4 per rx, 3 rx concat)
  raw    no separator: per window, 8 Doppler-band log energies + total
         (9 per rx, 3 rx concat)
Baseline: scene-1 mean pose (the floor any transfer must beat).
Metric: MPJPE (cm, root-relative, masked) on scene-4 clips + scene-1 heldout.

  HOURS=1 python3 train/train_pose_probe.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
OUTD = os.path.expanduser(os.environ.get("OUT", "~/zerdani/buffer/octonet/pose_probe_runs"))
HOURS = float(os.environ.get("HOURS", "1.0"))
STEPS = int(os.environ.get("STEPS", "8000"))
B = int(os.environ.get("B", "32"))
LR = float(os.environ.get("LR", "1e-3"))
H = int(os.environ.get("H", "256"))
SEED = int(os.environ.get("SEED", "0"))
ARMS = os.environ.get("ARMS", "model,raw").split(",")
DIAG = int(os.environ.get("DIAG", "0"))
NJ = 15
BANDS = [(2, 10), (10, 40), (40, 150)]
RAWB = np.linspace(2, 150, 9)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x):
        return torch.softmax(self.head(self.enc(self.inp(x))), -1)

import time as _t
sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        print("gpu retry", flush=True); _t.sleep(60)

def rec_feats(rid):
    """(model_feat (nw, M*4), raw_feat (nw, 9), nw) or None."""
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int); f = t[:, 1]
    mf = np.zeros((nw, M * 4), np.float64)
    for m in range(M):
        np.add.at(mf[:, m * 4], w, a[:, m] * e)
        for bi, (lo, hi) in enumerate(BANDS):
            s = (f >= lo) & (f < hi)
            np.add.at(mf[:, m * 4 + 1 + bi], w[s], a[s, m] * e[s])
    rf = np.zeros((nw, 9), np.float64)
    np.add.at(rf[:, 0], w, e)
    for bi in range(8):
        s = (f >= RAWB[bi]) & (f < RAWB[bi + 1])
        np.add.at(rf[:, 1 + bi], w[s], e[s])
    def nz(v):
        v = np.log10(v + 1e-9)
        return ((v - v.mean(0)) / (v.std(0) + 1e-6)).astype(np.float32)
    return nz(mf), nz(rf), nw

def build(scene):
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene == scene].copy()
    man["ck"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for ckey, g in man.groupby("ck"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        pf = f"{TOK}/pose/{rids[0]:06d}.npy"
        if not os.path.exists(pf): continue
        fs = [rec_feats(r) for r in rids]
        if any(f is None for f in fs): continue
        nw = min(f[2] for f in fs)
        P = np.asarray(np.load(pf), np.float32)[:nw]
        if not np.isfinite(P).any(): continue
        out.append((np.concatenate([f[0][:nw] for f in fs], 1),
                    np.concatenate([f[1][:nw] for f in fs], 1), P,
                    int(g.act.values[0])))
        if len(out) % 500 == 0: print(f"  scene{scene}: {len(out)}", flush=True)
    return out

class Head(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.gru = nn.GRU(fin, H, 2, batch_first=True)
        self.out = nn.Linear(H, NJ * 3)
    def forward(self, x):
        return self.out(self.gru(x)[0]).view(x.shape[0], -1, NJ, 3)

def mpjpe(pred, gt):
    """masked mean per-joint error (cm), root joint excluded. (nw,NJ,3)."""
    m = np.isfinite(gt).all(-1)
    m[:, 8] = False
    if not m.any(): return np.nan
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)
    return float(d[m].mean() * 100)

def pck2(pred, gt):
    m = np.isfinite(gt).all(-1)
    m[:, 8] = False
    if not m.any(): return np.nan, np.nan
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)[m]
    return float((d < 0.02).mean() * 100), float((d < 0.05).mean() * 100)

def run_arm(name, ai, tr, ho, te, mu):
    fin = tr[0][ai].shape[1]
    head = Head(fin).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS / 2: break
        ix = rng.choice(len(tr), B)
        nw = max(len(tr[i][2]) for i in ix)
        X = torch.zeros(B, nw, fin); Y = torch.full((B, nw, NJ, 3), np.nan)
        for k, i in enumerate(ix):
            F, P = tr[i][ai], tr[i][2]
            X[k, :len(F)] = torch.from_numpy(F)
            Y[k, :len(P)] = torch.from_numpy(P)
        X, Y = X.to(dev), Y.to(dev)
        pred = head(X)
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        msk[:, :, 8] = False
        loss = (torch.where(msk, (pred - torch.nan_to_num(Y)).abs(),
                            torch.zeros_like(pred)).sum()
                / msk.sum().clamp(min=1) / 3)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [{name} {step}] L1 {loss.item()*100:.2f} cm", flush=True)
    def ev(ds):
        errs, p20, p50 = [], [], []
        with torch.no_grad():
            for F_, R_, P, _a in ds:
                F = F_ if ai == 0 else R_
                pr = head(torch.from_numpy(F)[None].to(dev))[0].cpu().numpy()
                errs.append(mpjpe(pr, P))
                a_, b_ = pck2(pr, P)
                p20.append(a_); p50.append(b_)
        return (float(np.nanmedian(errs)), float(np.nanmean(p20)),
                float(np.nanmean(p50)))
    os.makedirs(OUTD, exist_ok=True)
    torch.save({"model": head.state_dict(), "fin": fin, "arm": name},
               f"{OUTD}/{name}.pt")
    if DIAG:
        JS = [j for j in range(NJ) if j != 8]
        dm, sp, sg, tc, wins = [], [], [], [], []
        with torch.no_grad():
            for F_, R_, P, _a in te:
                F = F_ if ai == 0 else R_
                pr = head(torch.from_numpy(F)[None].to(dev))[0].cpu().numpy()
                m = np.isfinite(P).all(-1); m[:, 8] = False
                if not m.any(): continue
                dm.append(float(np.linalg.norm((pr - mu)[m], axis=-1).mean() * 100))
                sp.append(float(pr[:, JS].std(0).mean() * 100))
                sg.append(float(np.nanmean(np.nanstd(P[:, JS], 0)) * 100))
                pd_ = (pr - pr.mean(0))[m]
                gd = np.nan_to_num((P - np.nanmean(P, 0)))[m]
                den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
                tc.append(float((pd_ * gd).sum() / den))
                wins.append(mpjpe(pr, P) < mpjpe(np.broadcast_to(mu, P.shape), P))
        print(f"  DIAG[{name}] scene{TESC}: dist-to-meanpose {np.median(dm):.2f} cm"
              f" | pred temporal-std {np.median(sp):.2f} vs GT {np.median(sg):.2f} cm"
              f" | traj-corr {np.median(tc):+.3f}"
              f" | paired-beats-baseline {np.mean(wins)*100:.0f}%", flush=True)
    return ev(ho), ev(te)

TRSC = [int(s) for s in os.environ.get("TRSC", "1").split(",")]
TESC = int(os.environ.get("TESC", "4"))
print(f"frozen sep: {CKPT} step {ck['step']} | building clip sets "
      f"(train scenes {TRSC} -> test scene {TESC})", flush=True)
tr_all = []
for s in TRSC:
    tr_all += build(s)
te = build(TESC)
rng = np.random.default_rng(SEED)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
print(f"scene1 train {len(tr)} / heldout {len(ho)} | scene4 test {len(te)}",
      flush=True)
mu = np.zeros((NJ, 3))
for j in range(NJ):
    vs = np.concatenate([P[:, j][np.isfinite(P[:, j]).all(-1)]
                         for _, _, P, _ in tr if np.isfinite(P[:, j]).any()])
    mu[j] = vs.mean(0) if len(vs) else 0
base_ho = float(np.nanmedian([mpjpe(np.broadcast_to(mu, P.shape), P)
                           for _, _, P, _ in ho]))
base_te = float(np.nanmedian([mpjpe(np.broadcast_to(mu, P.shape), P)
                           for _, _, P, _ in te]))
print(f"\nmean-pose baseline: scene1-ho {base_ho:.1f} cm | scene4 {base_te:.1f} cm",
      flush=True)
for name, ai in (("model", 0), ("raw", 1)):
    if name not in ARMS: continue
    (h, h20, h50), (t, t20, t50) = run_arm(name, ai, tr, ho, te, mu)
    print(f"[{name:5s}] ho: MPJPE {h:.1f} cm PCK@20 {h20:.1f} PCK@50 {h50:.1f}"
          f" | scene4: MPJPE {t:.1f} cm PCK@20 {t20:.1f} PCK@50 {t50:.1f}",
          flush=True)
print("""
READ: model < raw < baseline on scene 4 = the separator's output transfers
pose information to an unseen room better than raw Doppler statistics --
the output is (that much) universal. model ~ baseline = motion features
alone don't carry posture; expected partial ceiling, see caveat in report.""")


RENDER_ACTS = [int(v) for v in os.environ.get("RENDER_ACTS", "").split(",")
               if v]
if RENDER_ACTS and os.path.exists(f"{OUTD}/model.pt"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    ANAMES = {15: "squat", 14: "jumpjack", 13: "ccw-spin", 11: "pick-up"}
    EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
             (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
    ckm = torch.load(f"{OUTD}/model.pt", map_location=dev, weights_only=False)
    hd = Head(ckm["fin"]).to(dev)
    hd.load_state_dict(ckm["model"]); hd.eval()
    GIFD = os.path.expanduser("~/zerdani/buffer/cluster/logs")
    for pool, tag in ((ho, "ho"), (te, f"s{TESC}")):
        for act in RENDER_ACTS:
            clip = next((it for it in pool if it[3] == act), None)
            if clip is None: continue
            F, _, P, _ = clip
            with torch.no_grad():
                pr = hd(torch.from_numpy(F)[None].to(dev))[0].cpu().numpy()
            fin_ = np.isfinite(P).all((1, 2))
            Pg, Qg = P[fin_], pr[fin_]
            if len(Pg) < 8: continue
            var = np.nanvar(Pg.reshape(-1, 3), 0)
            a0, a1 = np.argsort(-var)[:2]
            lo = np.nanpercentile(Pg.reshape(-1, 3), 2, 0) - 0.15
            hi = np.nanpercentile(Pg.reshape(-1, 3), 98, 0) + 0.15
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
            nm = ANAMES.get(act, f"act{act}")
            fig.suptitle(f"{nm} [{tag}] slot-output model, moving GT "
                         f"(MPJPE {mpjpe(pr, P):.0f}cm)")
            arts = []
            for ax, t_, c_ in ((axes[0], "ground truth", "tab:green"),
                               (axes[1], "prediction", "tab:red")):
                ax.set_xlim(lo[a0], hi[a0]); ax.set_ylim(lo[a1], hi[a1])
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(t_, fontsize=9)
                arts.append([ax.plot([], [], "-o", color=c_, ms=2,
                                     lw=1.5)[0] for _ in EDGES])
            def fr(t2):
                for S, ls in ((Pg[t2], arts[0]), (Qg[t2], arts[1])):
                    for (e0, e1), ln in zip(EDGES, ls):
                        ln.set_data([S[e0, a0], S[e1, a0]],
                                    [S[e0, a1], S[e1, a1]])
                return [l for ls in arts for l in ls]
            ani = FuncAnimation(fig, fr, frames=len(Pg), blit=True)
            ani.save(f"{GIFD}/skelabs_{nm}_{tag}.gif",
                     writer=PillowWriter(fps=6))
            plt.close(fig)
            print(f"gif: {nm} {tag}", flush=True)
