#!/usr/bin/env python3
"""Multi-hypothesis (relaxed winner-takes-all) pose decoding — lever 2
against the PCK gap.

Diagnosis (battery on v2-BIG): mean-regression shrinks motion to 0.6 cm vs
GT 2.8 cm — the L1/L2-optimal move under uncertainty.  MPJPE tolerates the
shrinkage; PCK@20/50 cannot (paper x-scene target 181.5/44.2/79.5: we beat
their MPJPE and lose PCK 10x).  Mode-seeking fix: the head emits K full
trajectory hypotheses; training scores each clip on its BEST hypothesis
(clip-level WTA, losers get EPSW), so hypotheses stay on the pose manifold
at full motion amplitude instead of collapsing to the conditional mean.
A selector head (CE on the winner index) picks at test time.

Reports three rows per eval: selected (deployable), oracle (headroom),
mean-hyp (should reproduce the shrunk baseline).  Success signature:
selected std-ratio -> ~1 and PCK@20/50 up vs the flat-sweep small model
(~116 mm / PCK@50 ~24-28 scene4); oracle >> selected = selector is the
bottleneck, not the hypotheses.

Small tokens-only model for the test (env defaults: HDIM=128 ENCL=3
STATIC=0 SLOTQ=0).  POSESLOTS=1,2,3 reuses the v2-BIG build caches.
LANDMINE: CKPT env is consumed by the train_posetok3 import (separator,
RUNS-relative) — deleted here before importing.
"""
import os, time, importlib.util
import numpy as np
import torch
import torch.nn as nn

os.environ.pop("CKPT", None)
spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(os.path.dirname(__file__), "legacy",
                        "train_posetok3.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)

K = int(os.environ.get("K", "8"))
EPSW = float(os.environ.get("EPSW", "0.05"))
SELW = float(os.environ.get("SELW", "0.1"))
# soft-PCK objective (user hypothesis: MPJPE-style loss punishes bold
# trajectories linearly -> frozen postures win; threshold reward fixes the
# incentive).  SOFTPCK>0 adds weight*(1 - soft-PCK@20/50) to each
# hypothesis loss AND to winner selection; L1W scales the L1 anchor
# (needed: saturating loss alone has dead gradients far from GT).
SOFTPCK = float(os.environ.get("SOFTPCK", "0"))
L1W = float(os.environ.get("L1W", "1.0"))
TAU = float(os.environ.get("TAU", "0.01"))       # sigmoid temp (m)
OUTD = os.path.expanduser(os.environ.get(
    "OUT", "~/zerdani/buffer/octonet/posetok_mh_runs"))
dev = ptk.dev
NJ, ROOTJ, FTOK = ptk.NJ, ptk.ROOTJ, ptk.FTOK
B, LR, STEPS, HOURS = ptk.B, ptk.LR, ptk.STEPS, ptk.HOURS
EVERY, TEVERY, SHIFTS, VELW = ptk.EVERY, ptk.TEVERY, ptk.SHIFTS, ptk.VELW
STATIC, SDROP, SWAP = ptk.STATIC, ptk.SDROP, ptk.SWAP
HDIM, HEADS, ENCL = ptk.HDIM, ptk.HEADS, ptk.ENCL

class MHPoseTok(nn.Module):
    def __init__(self):
        super().__init__()
        H = HDIM
        self.inp = nn.Linear(FTOK, H)
        self.sinp = nn.Linear(399, H)
        self.type_emb = nn.Parameter(torch.zeros(2, H))
        lay = nn.TransformerEncoderLayer(H, HEADS, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, ENCL)
        self.qproj = nn.Linear(64, H)
        self.att = nn.MultiheadAttention(H, HEADS, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(),
                                nn.Linear(2 * H, H))
        self.out = nn.Linear(H, K * NJ * 3)
        self.sel = nn.Linear(H, K)
    def forward(self, x, mask, nws, st=None):
        h = self.inp(x) + self.type_emb[0]
        if st is not None:
            sh = self.sinp(st) + self.type_emb[1]
            h = torch.cat([h, sh], 1)
            mask = torch.cat([mask, torch.zeros(
                x.shape[0], 3, dtype=torch.bool, device=mask.device)], 1)
        h = self.enc(h, src_key_padding_mask=mask)
        B_, nq = x.shape[0], max(nws)
        tt = torch.arange(nq, device=x.device).float()[None, :, None]
        k = torch.arange(32, device=x.device).float()[None, None, :]
        q = torch.cat([torch.sin(tt / 20 * (k + 1)),
                       torch.cos(tt / 20 * (k + 1))], -1).expand(B_, -1, -1)
        q = self.qproj(q)
        o, _ = self.att(q, h, h, key_padding_mask=mask)
        o = o + self.ff(o)
        pred = self.out(o).view(B_, nq, K, NJ, 3)
        wm = (torch.arange(nq, device=x.device)[None, :]
              < torch.tensor(nws, device=x.device)[:, None]).float()
        sl = (self.sel(o) * wm[:, :, None]).sum(1) / wm.sum(1)[:, None]
        return pred, sl                       # (B,nq,K,NJ,3), (B,K)

def hyp_losses(pred, Y, MUt, SDt):
    """per-sample per-hypothesis loss: min-over-shifts L1 + VELW*velocity"""
    Z = (Y - MUt) / SDt
    msk = torch.isfinite(Y).all(-1, keepdim=True)
    msk[:, :, ROOTJ] = False
    cands = []
    for sh in range(-SHIFTS, SHIFTS + 1):
        if sh == 0:
            Zs, Ms = Z, msk
        elif sh > 0:
            Zs = torch.cat([Z[:, sh:], torch.full_like(Z[:, :sh],
                                                       np.nan)], 1)
            Ms = torch.cat([msk[:, sh:], torch.zeros_like(msk[:, :sh])], 1)
        else:
            Zs = torch.cat([torch.full_like(Z[:, :(-sh)], np.nan),
                            Z[:, :sh]], 1)
            Ms = torch.cat([torch.zeros_like(msk[:, :(-sh)]),
                            msk[:, :sh]], 1)
        d = (pred - torch.nan_to_num(Zs)[:, :, None]).abs()      # B,nq,K,NJ,3
        l_ = (torch.where(Ms[:, :, None], d, torch.zeros_like(d))
              .sum(dim=(1, 3, 4)) / Ms.sum(dim=(1, 2, 3))[:, None]
              .clamp(min=1) / 3)                                 # (B,K)
        l_ = L1W * l_
        if SOFTPCK > 0:
            dr = ((pred - torch.nan_to_num(Zs)[:, :, None]) * SDt) \
                .norm(dim=-1)                                    # B,nq,K,NJ (m)
            hit = 0.5 * torch.sigmoid((0.02 - dr) / TAU) \
                + 0.5 * torch.sigmoid((0.05 - dr) / TAU)
            Msq = Ms.squeeze(-1)[:, :, None]                     # B,nq,1,NJ
            sc = ((hit * Msq).sum(dim=(1, 3))
                  / Msq.sum(dim=(1, 3)).clamp(min=1))            # (B,K)
            l_ = l_ + SOFTPCK * (1.0 - sc)
        cands.append(l_)
    L = torch.stack(cands, -1).min(-1).values                    # (B,K)
    Z0 = torch.nan_to_num(Z)
    dz = pred[:, 1:] - pred[:, :-1]
    dg = (Z0[:, 1:] - Z0[:, :-1])[:, :, None]
    mv = (msk[:, 1:] & msk[:, :-1])[:, :, None]
    vel = (torch.where(mv, (dz - dg).abs(), torch.zeros_like(dz))
           .sum(dim=(1, 3, 4)) / mv.sum(dim=(1, 2, 3, 4))[:, None]
           .clamp(min=1) / 3)
    return L + VELW * vel

def main():
    print(f"MH-WTA pose head: K={K} EPSW={EPSW} SELW={SELW} HDIM={HDIM} "
          f"ENCL={ENCL} STATIC={STATIC} B={B} STEPS={STEPS} HOURS={HOURS} "
          f"SOFTPCK={SOFTPCK} L1W={L1W} TAU={TAU}", flush=True)
    tr_all = ptk.build([1, 2, 3])
    te = ptk.build([4])
    rng = np.random.default_rng(ptk.SEED)
    ix = rng.permutation(len(tr_all))
    ho = [tr_all[i] for i in ix[int(len(ix) * 0.95):]]
    tr = [tr_all[i] for i in ix[:int(len(ix) * 0.95)]]
    print(f"train {len(tr)} / ho {len(ho)} / test-scene4 {len(te)}",
          flush=True)
    mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
    for j in range(NJ):
        vs = np.concatenate([it[1][:, j][np.isfinite(it[1][:, j]).all(-1)]
                             for it in tr if np.isfinite(it[1][:, j]).any()])
        if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3
    MUt = torch.from_numpy(mu.astype(np.float32)).to(dev)
    SDt = torch.from_numpy(sd.astype(np.float32)).to(dev)
    net = MHPoseTok().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    os.makedirs(OUTD, exist_ok=True)

    def qev(ds, cap=150):
        rows = {k: [] for k in ("sel", "ora", "mean")}
        stat = {k: ([], []) for k in ("sel", "mean")}   # ratio, traj-corr
        with torch.no_grad():
            for it in ds[:cap]:
                tok, P, nw, rids = it[0], it[1], it[2], it[3]
                X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
                mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
                st = torch.from_numpy(ptk.get_static(rids))[None].to(dev) \
                    if STATIC else None
                pred, sl = net(X, mask, [nw], st)
                pz = pred[0, :len(P)].cpu().numpy()          # (T,K,NJ,3)
                hyps = pz * sd[None, None] + mu[None, None]
                cand = {"sel": hyps[:, int(sl[0].argmax())],
                        "mean": hyps.mean(1)}
                rs_k = [ptk.mpjpe_pck(hyps[:, k_], P) for k_ in range(K)]
                ok = [r for r in rs_k if r]
                if not ok: continue
                cand["ora"] = hyps[:, int(np.argmin(
                    [r[0] if r else 1e9 for r in rs_k]))]
                for name, pr in cand.items():
                    r = ptk.mpjpe_pck(pr, P)
                    if r: rows[name].append(r)
                    if name in stat:
                        m = np.isfinite(P).all(-1); m[:, ROOTJ] = False
                        if m.sum() > NJ:
                            js = [j for j in range(NJ) if j != ROOTJ]
                            gs = np.nanmean(np.nanstd(P[:, js], 0))
                            if np.isfinite(gs) and gs > 1e-9:
                                stat[name][0].append(
                                    float(np.nanmean(pr[:, js].std(0)) / gs))
                            pd_ = (pr - pr.mean(0))[m]
                            gd = np.nan_to_num(P - np.nanmean(P, 0))[m]
                            den = (np.linalg.norm(pd_) * np.linalg.norm(gd)
                                   + 1e-9)
                            stat[name][1].append(
                                float((pd_ * gd).sum() / den))
        out = {}
        for name, rs in rows.items():
            rs = np.array(rs)
            out[name] = (rs[:, 0].mean(), rs[:, 1].mean() * 100,
                         rs[:, 2].mean() * 100)
        for name in stat:
            out[name] = out[name] + (np.median(stat[name][0]),
                                     np.median(stat[name][1]))
        return out

    def report(tag, r):
        print(f"  {tag} selected: MPJPE {r['sel'][0]:.0f}  "
              f"PCK@20 {r['sel'][1]:.1f}  PCK@50 {r['sel'][2]:.1f}  "
              f"std-ratio {r['sel'][3]:.2f}  traj-corr {r['sel'][4]:+.2f}",
              flush=True)
        print(f"  {tag} oracle-K: MPJPE {r['ora'][0]:.0f}  "
              f"PCK@20 {r['ora'][1]:.1f}  PCK@50 {r['ora'][2]:.1f}   "
              f"mean-hyp: MPJPE {r['mean'][0]:.0f}  "
              f"PCK@20 {r['mean'][1]:.1f}  PCK@50 {r['mean'][2]:.1f}  "
              f"std-ratio {r['mean'][3]:.2f}", flush=True)

    best = -1.0
    if os.path.exists(f"{OUTD}/last.pt"):
        ckr = torch.load(f"{OUTD}/last.pt", map_location=dev,
                         weights_only=False)
        net.load_state_dict(ckr["model"])
        if "opt" in ckr: opt.load_state_dict(ckr["opt"])
        best = ckr.get("best", -1.0)
        print(f"resumed from step {ckr['step']} (best {best:.1f})",
              flush=True)
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
        for k_, it in enumerate(items):
            X[k_, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
            mask[k_, :len(it[0])] = False
            Y[k_, :len(it[1])] = torch.from_numpy(it[1])
            if STATIC:
                u = rng.random()
                if u < SDROP: pass
                elif u < SDROP + SWAP:
                    S[k_] = torch.from_numpy(
                        ptk.get_static(tr[rng.integers(len(tr))][3]))
                else:
                    S[k_] = torch.from_numpy(ptk.get_static(it[3]))
        X, mask, Y = X.to(dev), mask.to(dev), Y.to(dev)
        pred, sl = net(X, mask, nws, S.to(dev) if STATIC else None)
        L = hyp_losses(pred, Y, MUt, SDt)                        # (B,K)
        win = L.argmin(1)
        lw = L.gather(1, win[:, None])[:, 0]
        lose = (L.sum(1) - lw) / max(K - 1, 1)
        loss = (lw + EPSW * lose).mean() \
            + SELW * nn.functional.cross_entropy(sl, win.detach())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                selacc = (sl.argmax(1) == win).float().mean().item()
                nun = len(torch.unique(win))
            print(f"[{step}] wta {lw.mean().item():.3f} sel-acc {selacc:.2f}"
                  f" live-hyps {nun}/{K} {(time.time()-t0)/3600:.2f}h",
                  flush=True)
        if step % EVERY == 0 and step > 0:
            net.eval()
            h = qev(ho)
            tag = "(best)" if h["sel"][2] > best else ""
            print(f"EVAL [{step}] heldout {tag}", flush=True)
            report("ho", h)
            torch.save({"model": net.state_dict(),
                        "opt": opt.state_dict(), "step": step,
                        "best": best}, f"{OUTD}/last.pt")
            if h["sel"][2] > best:
                best = h["sel"][2]
                torch.save({"model": net.state_dict(), "step": step},
                           f"{OUTD}/best.pt")
            if step % TEVERY == 0:
                t = qev(te, cap=len(te))
                print(f"TEST4 [{step}] scene4 FULL (n={len(te)})   "
                      f"[paper x-scene: 181.5/44.2/79.5]", flush=True)
                report("s4", t)
            net.train()
    net.eval()
    print("FINAL (current weights):", flush=True)
    for tag, ds in (("heldout 1-3", ho), ("TEST scene4", te)):
        r = qev(ds, cap=len(ds))
        print(f"[{tag}] (n={len(ds)})", flush=True)
        report(tag, r)
    print("paper x-scene: MPJPE 181.5  PCK@20 44.2  PCK@50 79.5 "
          "(absolute-frame protocol; ours root-relative)", flush=True)

if __name__ == "__main__":
    main()
