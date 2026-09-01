#!/usr/bin/env python3
"""Learned limb clustering on MUSIC tokens (the deep-clustering road,
probes 27/28 feasibility-positive). Input = token sets from prep/tokenize_pa
(doppler f, angle phi, delay psi per TF bin). Model = set transformer ->
softmax slot assignment per token (M slots, identity-free).

Losses (both permutation/identity-free; phase-1's identity-pinned routing is
exactly what this replaces):
  PIT-envelope  recordings with keypoint GT (PA pose GT, legacy imu/ dirs of
                the old preps, joined on file path): slot energy envelopes,
                best ordered slot-pair vs the top-2 active limbs' residualized
                envelopes, loss = 1 - matched corr.
  MixIT-origin  two recordings, same receiver, token sets UNIONED: the model
                must be able to split its slots into two groups that carry
                each origin's energy. min over all 2-group slot partitions of
                energy-weighted MSE on token origin.

Gate metric (rooms 4/5, never trained): probe-27 battery -- matched corr vs
rolled null, must beat the zero-learning doppler k-means (+0.16 med / 59%).

  MIXIT_RUNS=~/zerdani/buffer/octonet/limbtok_runs nohup python3 \
    train/train_limbtok.py &
"""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TOK   = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS  = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok_runs"))
STEPS = int(os.environ.get("STEPS", "20000"))
HOURS = float(os.environ.get("HOURS", "2"))
BP    = int(os.environ.get("BP", "8"))       # PIT recordings per step
BM    = int(os.environ.get("BM", "8"))       # MixIT pairs per step
LR    = float(os.environ.get("LR", "3e-4"))
M     = int(os.environ.get("M", "6"))
D     = int(os.environ.get("DIM", "128"))
NL    = int(os.environ.get("LAYERS", "4"))
PITW  = float(os.environ.get("PITW", "1.0"))
MIXW  = float(os.environ.get("MIXW", "1.0"))
EVERY = int(os.environ.get("EVERY", "500"))
NEVAL = int(os.environ.get("NEVAL", "200"))
SEED  = int(os.environ.get("SEED", "0"))
CANON = int(os.environ.get("CANON", "0"))
HOPF, WINF = 128, 256
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

class SetSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(7, D)
        lay = nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.head = nn.Linear(D, M)
    def forward(self, x, mask):                  # x (B,N,7) mask (B,N) True=pad
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        return torch.softmax(self.head(h), -1)   # (B,N,M)

# claim the GPU before any heavy reads (GB10 landmine)
model = SetSep()
for attempt in range(10):
    try:
        model = model.to(dev); break
    except RuntimeError:
        print(f"to({dev}) failed, retry {attempt+1}/10 in 60s", flush=True)
        time.sleep(60)

PEAKS = {}
if CANON:
    _pk = np.load(f"{TOK}/static_peaks.npz")
    PEAKS = {int(r): (float(a), float(b)) for r, a, b in
             zip(_pk["rids"], _pk["phi0"], _pk["psi0"])}
    print(f"canonical mode: {len(PEAKS)} static peaks", flush=True)

def feats(z, rid=None):
    """npz -> (n,7) float32 features + aux (w idx, energy, nw)."""
    t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    phi, psi = t[:, 2], t[:, 3]
    if CANON and rid is not None and rid in PEAKS:
        p0, q0 = PEAKS[rid]
        phi = np.arctan2(np.sin(phi - p0), np.cos(phi - p0))
        psi = np.arctan2(np.sin(psi - q0), np.cos(psi - q0))
    X = np.c_[np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi),
              t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    return X, t[:, 0].astype(np.int64), (10.0 ** le).astype(np.float32), nw

def residual_pool(gpath, nw):
    gi = np.asarray(np.load(gpath), np.float32)
    g2 = gi.copy()
    for i_ in range(5):
        oth = [j for j in range(5) if j != i_]
        A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
        beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
        g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
    G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                  for w in range(min(nw, (len(g2) - WINF) // HOPF + 1))])
    return G

def slot_envs(a, widx, e, nw):
    """a (n,M) soft assign, e (n,) energy -> (M, nw) differentiable envs."""
    env = a.new_zeros(M, nw)
    env.index_add_(1, widx, (a * e[:, None]).T)
    return env

def tcorr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

def pit_loss(a, widx, e, nw, G):
    """best ordered slot pair vs top-2 limbs; 1 - matched corr."""
    env = slot_envs(a, widx, e, nw)
    T = min(nw, len(G))
    if T < 8: return None
    Gt = torch.from_numpy(G[:T]).to(a.device)
    order = torch.argsort(-Gt.mean(0))
    li, lj = int(order[0]), int(order[1])
    C = torch.stack([torch.stack([tcorr(env[m, :T], Gt[:, li]),
                                  tcorr(env[m, :T], Gt[:, lj])])
                     for m in range(M)])         # (M,2)
    vs = [(C[m1, 0] + C[m2, 1]) / 2 for m1 in range(M) for m2 in range(M)
          if m1 != m2]
    return 1.0 - torch.stack(vs).max()

PARTS = None
def mixit_loss(a, e, origin):
    """min over 2-group slot partitions of energy-weighted origin MSE."""
    global PARTS
    if PARTS is None:
        ms = []
        for b in range(1, 2 ** M - 1):
            ms.append([float((b >> k) & 1) for k in range(M)])
        PARTS = torch.tensor(ms, device=a.device)          # (P, M)
    p = a @ PARTS.T                                        # (n, P)
    w = e / (e.sum() + 1e-8)
    err = ((p - origin[:, None]) ** 2 * w[:, None]).sum(0) # (P,)
    return err.min()

def load_rec(rid):
    try:
        return np.load(f"{TOK}/tokens/{rid:06d}.npz")
    except Exception:
        return None

def main():
    os.makedirs(RUNS, exist_ok=True)
    man = pd.read_csv(f"{TOK}/manifest.csv")
    have = {int(f[:6]) for f in os.listdir(f"{TOK}/tokens")}
    man = man[man.rid.isin(have)]
    gtset = {int(f[:6]) for f in os.listdir(f"{TOK}/imu")} \
        if os.path.isdir(f"{TOK}/imu") else set()
    tr_all = man[man.split == "train"].reset_index(drop=True)
    gt_tr = {int(r): f"{TOK}/imu/{int(r):06d}.npy"
             for r in tr_all.rid.values if int(r) in gtset}
    gt_te = {int(r): f"{TOK}/imu/{int(r):06d}.npy"
             for r in man[man.split == "test"].rid.values if int(r) in gtset}
    print(f"tokens on disk: {len(man)} | train {len(tr_all)} "
          f"(with GT {len(gt_tr)}) | test with GT {len(gt_te)} | dev={dev}",
          flush=True)
    rng = np.random.default_rng(SEED)
    pit_ids = np.array(sorted(gt_tr))
    bynode = {k: g.rid.values for k, g in tr_all.groupby("node")}
    ev_ids = rng.permutation(np.array(sorted(gt_te)))[:NEVAL]
    Gcache = {}
    def getG(rid, path, nw):
        if rid not in Gcache: Gcache[rid] = residual_pool(path, nw)
        return Gcache[rid]

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    step0, best = 0, -math.inf
    if os.path.exists(f"{RUNS}/last.pt"):
        ck = torch.load(f"{RUNS}/last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); step0 = ck["step"]; best = ck["best"]
        print(f"resumed from step {step0}", flush=True)

    def forward_recs(recs):
        """list of (X, widx, e, nw) -> list of (a, widx, e, nw)."""
        n = max(len(r[0]) for r in recs)
        X = torch.zeros(len(recs), n, 7)
        mask = torch.ones(len(recs), n, dtype=torch.bool)
        for k, r in enumerate(recs):
            X[k, :len(r[0])] = torch.from_numpy(r[0])
            mask[k, :len(r[0])] = False
        a = model(X.to(dev), mask.to(dev))
        return [(a[k, :len(r[0])], r[1], r[2], r[3]) for k, r in enumerate(recs)]

    t0 = time.time()
    for step in range(step0, STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        model.train()
        loss = torch.zeros((), device=dev)
        # PIT stream
        picks, recs = [], []
        for rid in rng.choice(pit_ids, BP, replace=False):
            z = load_rec(int(rid))
            if z is None: continue
            recs.append(feats(z, int(rid))); picks.append(int(rid))
        npit = 0
        if recs:
            for (a, widx, e, nw), rid in zip(forward_recs(recs), picks):
                G = getG(rid, gt_tr[rid], nw)
                lp = pit_loss(a, torch.from_numpy(widx).to(dev),
                              torch.from_numpy(e).to(dev), nw, G)
                if lp is not None: loss = loss + PITW * lp; npit += 1
        # MixIT stream -- all pairs through ONE padded forward
        nmix = 0
        pairs = []
        for _ in range(BM):
            node = list(bynode)[rng.integers(len(bynode))]
            ra, rb = rng.choice(bynode[node], 2, replace=False)
            za, zb = load_rec(int(ra)), load_rec(int(rb))
            if za is None or zb is None: continue
            Xa, wa, ea, nwa = feats(za, int(ra)); Xb, wb, eb, nwb = feats(zb, int(rb))
            X = np.r_[Xa, Xb]
            le = X[:, 6]                      # re-z-score energy over union
            X[:, 6] = (le - le.mean()) / (le.std() + 1e-6)
            org = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))].astype(np.float32)
            pairs.append((X, org, np.r_[ea, eb]))
        if pairs:
            n = max(len(p[0]) for p in pairs)
            Xb_ = torch.zeros(len(pairs), n, 7)
            msk = torch.ones(len(pairs), n, dtype=torch.bool)
            for k, p in enumerate(pairs):
                Xb_[k, :len(p[0])] = torch.from_numpy(p[0])
                msk[k, :len(p[0])] = False
            Am = model(Xb_.to(dev), msk.to(dev))
            for k, (Xp, org, ee) in enumerate(pairs):
                loss = loss + MIXW * mixit_loss(
                    Am[k, :len(Xp)], torch.from_numpy(ee).to(dev),
                    torch.from_numpy(org).to(dev))
                nmix += 1
        if npit + nmix == 0: continue
        loss = loss / max(npit + nmix, 1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sch.step()
        if step % 50 == 0:
            print(f"[{step}] loss {loss.item():.4f} (pit {npit} mix {nmix}) "
                  f"lr {sch.get_last_lr()[0]:.2e} "
                  f"{(time.time()-t0)/3600:.2f}h", flush=True)
        if step % EVERY == 0 and step > step0:
            model.eval()
            mm, nn_ = [], []
            cons = np.zeros((5, M))
            with torch.no_grad():
                for rid in ev_ids:
                    z = load_rec(int(rid))
                    if z is None: continue
                    X, widx, e, nw = feats(z, int(rid))
                    a = model(torch.from_numpy(X)[None].to(dev),
                              torch.zeros(1, len(X), dtype=torch.bool,
                                          device=dev))[0]
                    env = slot_envs(a, torch.from_numpy(widx).to(dev),
                                    torch.from_numpy(e).to(dev), nw
                                    ).cpu().numpy()
                    G = getG(int(rid), gt_te[int(rid)], nw)
                    T = min(nw, len(G))
                    if T < 8: continue
                    order = np.argsort(-G[:T].mean(0))
                    def sc(Gm):
                        C = np.zeros((M, 2))
                        for m in range(M):
                            for c, l in enumerate(order[:2]):
                                x1 = env[m, :T] - env[m, :T].mean()
                                y1 = Gm[:T, l] - Gm[:T, l].mean()
                                d = np.linalg.norm(x1) * np.linalg.norm(y1)
                                C[m, c] = (x1 @ y1) / d if d > 1e-9 else 0.0
                        bv, bp = -2, (0, 1)
                        for m1 in range(M):
                            for m2 in range(M):
                                if m1 == m2: continue
                                v_ = (C[m1, 0] + C[m2, 1]) / 2
                                if v_ > bv: bv, bp = v_, (m1, m2)
                        return bv, bp
                    v_, (bm1, bm2) = sc(G)
                    mm.append(v_)
                    cons[int(order[0]), bm1] += 1
                    cons[int(order[1]), bm2] += 1
                    nn_.append(sc(np.roll(G[:T], T // 2, 0))[0])
            if not mm:
                print(f"  EVAL step {step}: no test tokens yet", flush=True)
                model.train(); continue
            mm, nn_ = np.array(mm), np.array(nn_)
            v = float(np.median(mm))
            rows = cons.sum(1)
            ok_ = rows >= 20
            maj = float((cons.max(1)[ok_] / rows[ok_]).mean()) if ok_.any() else 0
            print(f"  EVAL step {step}: matched {v:+.3f}  null "
                  f"{np.median(nn_):+.3f}  win {np.mean(mm > nn_)*100:.0f}% "
                  f"slot-consist {maj*100:.0f}% (chance {100/M:.0f}%) "
                  f"(n={len(mm)}) {'(best)' if v > best else ''}", flush=True)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sch": sch.state_dict(), "step": step, "best": best,
                  "limbtok": True,
                  "cfg": {"M": M, "D": D, "NL": NL, "PITW": PITW,
                          "MIXW": MIXW}}
            torch.save(ck, f"{RUNS}/last.pt")
            if v > best:
                best = v; ck["best"] = best
                torch.save(ck, f"{RUNS}/best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sch": sch.state_dict(), "step": step, "best": best,
                "limbtok": True}, f"{RUNS}/last.pt")
    print(f"DONE step {step} best eval {best:+.3f}", flush=True)

if __name__ == "__main__":
    main()
