#!/usr/bin/env python3
"""Pose from GROUPED TOKENS, fast version: token-set transformer with
per-window queries -> root-relative 15-joint skeleton. Train scenes 1-3,
test scene 4. Uses existing pa_tokens/pose GT. No geometry, no statics.
Reports MPJPE (mm) + PCK@20/50 (root-relative) vs mean-pose baseline.

  HOURS=1 python3 bench/train_posetok.py
"""
import os, time, pickle, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
HOURS = float(os.environ.get("HOURS", "1.0"))
STEPS = int(os.environ.get("STEPS", "40000"))
B = int(os.environ.get("B", "12"))
LR = float(os.environ.get("LR", "5e-4"))
TMAX = int(os.environ.get("TMAX", "1600"))
STATIC = int(os.environ.get("STATIC", "1"))
VELW = float(os.environ.get("VELW", "0.5"))
SMARK = os.path.expanduser(os.environ.get(
    "SMARK", "~/zerdani/buffer/octonet/archive2/statics_done.marker"))
SEED = int(os.environ.get("SEED", "0"))
NJ, ROOTJ = 15, 8
HOPF, WINF = 128, 256
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]
FTOK = 7 + M + 3

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

sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        print("gpu retry", flush=True); time.sleep(60)

def rec_tok(rid, rx):
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X7 = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
               np.cos(t[:, 3]), t[:, 1] / 150.0,
               t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X7)[None].to(dev))[0].cpu().numpy()
    hot = np.zeros((len(t), 3), np.float32); hot[:, rx] = 1
    return np.c_[X7, a.astype(np.float32), hot].astype(np.float16), nw

def build(scenes):
    cf = f"{TOK}/tokpose2_{'-'.join(map(str, scenes))}.pkl"
    if os.path.exists(cf):
        return pickle.load(open(cf, "rb"))
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin(scenes)].copy()
    man["ckey"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for ckey, g in man.groupby("ckey"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        pf = f"{TOK}/pose/{rids[0]:06d}.npy"
        if not os.path.exists(pf): continue
        ts = [rec_tok(r, i) for i, r in enumerate(rids)]
        if any(t is None for t in ts): continue
        nw = min(t[1] for t in ts)
        tok = np.concatenate([t[0] for t in ts], 0)
        if len(tok) > TMAX:
            keep = np.argsort(-tok[:, 6].astype(np.float32))[:TMAX]
            tok = tok[keep]
        P = np.asarray(np.load(pf), np.float32)[:nw]
        if not np.isfinite(P).any(): continue
        out.append((tok, P, nw, rids))
        if len(out) % 1000 == 0: print(f"  s{scenes}: {len(out)}", flush=True)
    pickle.dump(out, open(cf, "wb"), protocol=4)
    return out

SCACHE = {}
def get_static(rids):
    vs = []
    for r in rids:
        if r not in SCACHE:
            f = f"{TOK}/statics/{r:06d}.npy"
            v = np.load(f).astype(np.float32) if os.path.exists(f) \
                else np.zeros(399, np.float32)
            a = v[:171]; c = v[171:]
            a = (a - a.mean()) / (a.std() + 1e-6)
            SCACHE[r] = np.r_[a, c].astype(np.float32)
        vs.append(SCACHE[r])
    return np.stack(vs)                      # (3, 399)

class PoseTok(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.inp = nn.Linear(FTOK, H)
        self.sinp = nn.Linear(399, H)
        self.type_emb = nn.Parameter(torch.zeros(2, H))
        lay = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.qproj = nn.Linear(64, H)
        self.att = nn.MultiheadAttention(H, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(),
                                nn.Linear(2 * H, H))
        self.out = nn.Linear(H, NJ * 3)
    def forward(self, x, mask, nws, st=None):
        h = self.inp(x) + self.type_emb[0]
        if st is not None:
            sh = self.sinp(st) + self.type_emb[1]        # (B, 3, H)
            h = torch.cat([h, sh], 1)
            mask = torch.cat([mask, torch.zeros(
                x.shape[0], 3, dtype=torch.bool, device=mask.device)], 1)
        h = self.enc(h, src_key_padding_mask=mask)
        B_, nq = x.shape[0], max(nws)
        tt = torch.arange(nq, device=x.device).float()[None, :, None]
        k = torch.arange(32, device=x.device).float()[None, None, :]
        q = torch.cat([torch.sin(tt / 20 * (k + 1)), torch.cos(tt / 20 * (k + 1))],
                      -1).expand(B_, -1, -1)
        q = self.qproj(q)
        o, _ = self.att(q, h, h, key_padding_mask=mask)
        o = o + self.ff(o)
        return self.out(o).view(B_, nq, NJ, 3)

def mpjpe_pck(pred, gt):
    m = np.isfinite(gt).all(-1); m[:, ROOTJ] = False
    if not m.any(): return None
    d = np.linalg.norm(np.nan_to_num(pred - gt), axis=-1)[m] * 1000
    return d.mean(), (d < 20).mean(), (d < 50).mean()

def main():
    print("building sets", flush=True)
    tr_all = build([1, 2, 3])
    te = build([4])
    rng = np.random.default_rng(SEED)
    ix = rng.permutation(len(tr_all))
    ho = [tr_all[i] for i in ix[int(len(ix) * 0.95):]]
    tr = [tr_all[i] for i in ix[:int(len(ix) * 0.95)]]
    print(f"train {len(tr)} / ho {len(ho)} / test-scene4 {len(te)}", flush=True)
    if STATIC:
        while not os.path.exists(SMARK):
            print("waiting for statics ...", flush=True); time.sleep(60)
    mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
    for j in range(NJ):
        vs = np.concatenate([P[:, j][np.isfinite(P[:, j]).all(-1)]
                             for _, P, _, _ in tr if np.isfinite(P[:, j]).any()])
        if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3
    MUt = torch.from_numpy(mu.astype(np.float32)).to(dev)
    SDt = torch.from_numpy(sd.astype(np.float32)).to(dev)
    bt = [mpjpe_pck(np.broadcast_to(mu, P.shape), P) for _, P, _, _ in te]
    bt = np.array([r for r in bt if r])
    print(f"mean-pose baseline scene4: MPJPE {bt[:,0].mean():.0f} mm  "
          f"PCK@20 {bt[:,1].mean()*100:.1f}  PCK@50 {bt[:,2].mean()*100:.1f}",
          flush=True)
    net = PoseTok().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    OUTD = os.path.expanduser("~/zerdani/buffer/octonet/posetok_runs")
    os.makedirs(OUTD, exist_ok=True)
    def qev(ds, cap=150):
        rs, ratio, tc = [], [], []
        mu_np, sd_np = MUt.cpu().numpy(), SDt.cpu().numpy()
        with torch.no_grad():
            for tok, P, nw, rids in ds[:cap]:
                X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
                mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
                st = torch.from_numpy(get_static(rids))[None].to(dev) \
                    if STATIC else None
                pr = net(X, mask, [nw], st)[0, :len(P)].cpu().numpy()
                pr = pr * sd_np + mu_np                   # de-normalize
                r = mpjpe_pck(pr, P)
                if r: rs.append(r)
                m = np.isfinite(P).all(-1); m[:, ROOTJ] = False
                if m.sum() > NJ:
                    js = [j for j in range(NJ) if j != ROOTJ]
                    gs = np.nanmean(np.nanstd(P[:, js], 0))
                    if np.isfinite(gs) and gs > 1e-9:
                        ratio.append(float(np.nanmean(pr[:, js].std(0)) / gs))
                    pd_ = (pr - pr.mean(0))[m]
                    gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
                    den = np.linalg.norm(pd_) * np.linalg.norm(gd) + 1e-9
                    tc.append(float((pd_ * gd).sum() / den))
        rs = np.array(rs)
        return (rs[:, 0].mean(), rs[:, 1].mean() * 100, rs[:, 2].mean() * 100,
                np.median(ratio), np.median(tc))
    best = 1e9
    if os.path.exists(f"{OUTD}/last.pt"):
        ckr = torch.load(f"{OUTD}/last.pt", map_location=dev,
                         weights_only=False)
        net.load_state_dict(ckr["model"])
        if "opt" in ckr: opt.load_state_dict(ckr["opt"])
        best = ckr.get("best", 1e9)
        print(f"resumed from step {ckr['step']} (best {best:.0f})", flush=True)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS: break
        ixb = rng.choice(len(tr), B)
        items = [tr[i] for i in ixb]
        n = max(len(it[0]) for it in items)
        nws = [it[2] for it in items]
        X = torch.zeros(B, n, FTOK)
        mask = torch.ones(B, n, dtype=torch.bool)
        Y = torch.full((B, max(nws), NJ, 3), np.nan)
        S = torch.zeros(B, 3, 399)
        for k, it in enumerate(items):
            X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
            mask[k, :len(it[0])] = False
            Y[k, :len(it[1])] = torch.from_numpy(it[1])
            if STATIC: S[k] = torch.from_numpy(get_static(it[3]))
        X, mask, Y, S = X.to(dev), mask.to(dev), Y.to(dev), S.to(dev)
        pred = net(X, mask, nws, S if STATIC else None)   # in Z-space
        Z = (Y - MUt) / SDt
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        msk[:, :, ROOTJ] = False
        loss = (torch.where(msk, (pred - torch.nan_to_num(Z)).abs(),
                            torch.zeros_like(pred)).sum()
                / msk.sum().clamp(min=1) / 3)
        dz = pred[:, 1:] - pred[:, :-1]
        dg = torch.nan_to_num(Z[:, 1:] - Z[:, :-1])
        mv = msk[:, 1:] & msk[:, :-1]
        loss = loss + VELW * (torch.where(mv, (dz - dg).abs(),
                              torch.zeros_like(dz)).sum()
                              / mv.sum().clamp(min=1) / 3)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % 500 == 0:
            print(f"[{step}] L1z {loss.item():.3f} (do-nothing=1.0) "
                  f"{(time.time()-t0)/3600:.2f}h", flush=True)
        if step % 2000 == 0 and step > 0:
            net.eval()
            h = qev(ho)
            print(f"  EVAL [{step}] heldout: MPJPE {h[0]:.0f} mm  "
                  f"PCK@20 {h[1]:.1f}%  PCK@50 {h[2]:.1f}%  "
                  f"std-ratio {h[3]:.2f}  traj-corr {h[4]:+.2f}"
                  f"{'  (best)' if h[0] < best else ''}", flush=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best}, f"{OUTD}/last.pt")
            if h[0] < best:
                best = h[0]
                torch.save({"model": net.state_dict(), "step": step},
                           f"{OUTD}/best.pt")
            if step % 5000 == 0:
                t = qev(te, cap=len(te))
                print(f"  TEST4 [{step}] scene4 FULL (n={len(te)}): "
                      f"MPJPE {t[0]:.0f} mm  PCK@20 {t[1]:.1f}%  "
                      f"PCK@50 {t[2]:.1f}%  std-ratio {t[3]:.2f}  "
                      f"traj-corr {t[4]:+.2f}   [paper x-scene: 181.5/44.2/79.5]",
                      flush=True)
            net.train()
    net.eval()
    for tag, ds in (("heldout 1-3", ho), ("TEST scene4", te)):
        r = qev(ds, cap=len(ds))
        print(f"[{tag}] MPJPE {r[0]:.0f} mm  PCK@20 {r[1]:.1f}  "
              f"PCK@50 {r[2]:.1f}  std-ratio {r[3]:.2f}  "
              f"traj-corr {r[4]:+.2f}  (n={len(ds)})", flush=True)
    print("""paper Table 3 (ABSOLUTE frame, geometry-conditioned; ours is
root-relative, geometry-free -- context, not same protocol):
  PerceptAlign cross-scene: MPJPE 181.5  PCK@20 44.2  PCK@50 79.5
  PerceptAlign in-domain  : MPJPE 137.2  PCK@20 55.2  PCK@50 88.7""", flush=True)

if __name__ == "__main__":
    main()
