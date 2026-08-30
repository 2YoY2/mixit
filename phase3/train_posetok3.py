#!/usr/bin/env python3
"""PHASE 3 (1): ComBat-style statics harmonization (the literature branch
we skipped): per-(scene, receiver) LOCATION template estimated on
ACTION-BALANCED subsets (so it cannot absorb pose signal) subtracted, and
per-(scene, receiver) SCALE divided out (ComBat's second step). No swap, no
dropout -- decomposition, not destruction. Refs: Johnson 2007 (ComBat),
Solomonoff 2005 (NAP), STAP, AdaBN.

Pose from GROUPED TOKENS, fast version: token-set transformer with
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
STATIC = int(os.environ.get("STATIC", "0"))
SLOTQ = int(os.environ.get("SLOTQ", "1"))
HDIM = int(os.environ.get("HDIM", "128"))
LAGFIX = int(os.environ.get("LAGFIX", "0"))
STATTOK = int(os.environ.get("STATTOK", "0"))
PERSTOK = int(os.environ.get("PERSTOK", "0"))
NPP = 4
NSP = 8
SHIFTS = int(os.environ.get("SHIFTS", "2"))
ENCL = int(os.environ.get("ENCL", "3"))
HEADS = int(os.environ.get("HEADS", "4"))
POSESLOTS = [int(v) for v in os.environ.get("POSESLOTS", "1,2").split(",")]
VELW = float(os.environ.get("VELW", "0.5"))
SDROP = float(os.environ.get("SDROP", "0.3"))
SWAP = float(os.environ.get("SWAP", "0.2"))
TEMPL = int(os.environ.get("TEMPL", "1"))
OUTDIR = os.path.expanduser(os.environ.get("OUT", "~/zerdani/buffer/octonet/posetok_runs"))
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
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int)
    st = np.zeros((nw, len(POSESLOTS) * 6), np.float64)
    for si, m in enumerate(POSESLOTS):
        wm = a[:, m] * e
        den = np.zeros(nw); np.add.at(den, w, wm)
        np.add.at(st[:, si * 6], w, wm)
        for ci, v in enumerate((t[:, 1] / 150.0, np.sin(t[:, 2]),
                                np.cos(t[:, 2]), np.sin(t[:, 3]),
                                np.cos(t[:, 3]))):
            acc = np.zeros(nw); np.add.at(acc, w, wm * v)
            st[:, si * 6 + 1 + ci] = acc / np.maximum(den, 1e-9)
    st[:, ::6] = np.log10(st[:, ::6] + 1e-9)
    return (np.c_[X7, a.astype(np.float32), hot].astype(np.float16),
            st.astype(np.float16), nw)

def build(scenes):
    slottag = "".join(map(str, POSESLOTS))
    cf = f"{TOK}/tokpose5s{slottag}_{'-'.join(map(str, scenes))}.pkl"
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
        nw = min(t[2] for t in ts)
        tok = np.concatenate([t[0] for t in ts], 0)
        if len(tok) > TMAX:
            keep = np.argsort(-tok[:, 6].astype(np.float32))[:TMAX]
            tok = tok[keep]
        S12 = np.concatenate([t[1][:nw] for t in ts], 1)
        P = np.asarray(np.load(pf), np.float32)
        if not np.isfinite(P).any(): continue
        if LAGFIX and len(P) >= nw:
            eng = (10.0 ** S12[:, 0::6].astype(np.float64)).sum(1)
            spd = np.linalg.norm(np.diff(np.nan_to_num(P), axis=0), axis=-1
                                 ).sum(-1)
            spd = np.r_[spd[:1], spd]
            best_l, best_c = 0, -2
            for lag in range(-3, 4):
                a = eng[max(0, -lag):nw - max(0, lag)]
                b = spd[max(0, lag):len(spd) - max(0, -lag)][:len(a)]
                if len(a) > 8 and a.std() > 1e-9 and b.std() > 1e-9:
                    c = np.corrcoef(a[:len(b)], b)[0, 1]
                    if c > best_c: best_c, best_l = c, lag
            P = P[max(0, best_l):][:nw] if best_l >= 0 else \
                np.r_[np.full((-best_l, *P.shape[1:]), np.nan,
                              np.float32), P[:nw + best_l]]
        P = P[:nw]
        if len(P) < nw:
            P = np.r_[P, np.full((nw - len(P), *P.shape[1:]), np.nan,
                                 np.float32)]
        out.append((tok, P.astype(np.float32), nw, rids, S12,
                    static_path_tokens(rids).astype(np.float16)))
        if len(out) % 1000 == 0: print(f"  s{scenes}: {len(out)}", flush=True)
    pickle.dump(out, open(cf, "wb"), protocol=4)
    return out

_L, _NPH, _NPS = 20, 37, 37
_PH = np.linspace(-np.pi, np.pi, _NPH, endpoint=False)
_PS = np.linspace(-np.pi, np.pi, _NPS, endpoint=False)
_AA = np.stack([np.ones(_NPH), np.exp(1j * _PH)], 1)
_AS = np.exp(1j * np.outer(_PS, np.arange(_L)))
_ST = (_AA[:, None, :, None] * _AS[None, :, None, :]).reshape(
    _NPH * _NPS, 2 * _L)
_ST = (_ST / np.sqrt(2 * _L)).astype(np.complex64)
_IPH, _IPS = np.unravel_index(np.arange(_NPH * _NPS), (_NPH, _NPS))

def static_path_tokens(rids):
    """(3*NSP, 18) tokens: top-NSP Bartlett peaks of each rx static."""
    out = []
    for ri, r in enumerate(rids):
        f = f"{TOK}/statics/{r:06d}.npy"
        toks = np.zeros((NSP, 18), np.float32)
        if os.path.exists(f):
            v = np.load(f)
            y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
            sb = np.stack([y[:, k:k + _L].reshape(-1)
                           for k in range(57 - _L + 1)], 0)
            P = (np.abs(sb @ _ST.conj().T) ** 2).mean(0)
            order = np.argsort(-P)
            picks = []
            for j in order:
                if all(abs(int(_IPH[j]) - int(_IPH[k])) > 2 or
                       abs(int(_IPS[j]) - int(_IPS[k])) > 2 for k in picks):
                    picks.append(j)
                if len(picks) >= NSP: break
            lp = np.log10(P[picks] + 1e-12)
            zlp = (lp - lp.mean()) / (lp.std() + 1e-6)
            for n, j in enumerate(picks):
                phi, psi = _PH[_IPH[j]], _PS[_IPS[j]]
                toks[n, :7] = [np.sin(phi), np.cos(phi), np.sin(psi),
                               np.cos(psi), -0.05, 0.5, zlp[n]]
                toks[n, 15 + ri] = 1.0
        out.append(toks)
    return np.concatenate(out, 0)                        # (3*NSP, 18)

PCACHE = {}
def person_tokens(rids):
    """(3*NPP, 18): Bartlett peaks of the TEMPLATE-SUBTRACTED product static
    = the person's standing paths, parameterized (phi, psi, logp)."""
    out = []
    for ri, r in enumerate(rids):
        if r in PCACHE:
            t = PCACHE[r].copy()
        else:
            t = np.zeros((NPP, 18), np.float32)
            f = f"{TOK}/statics/{r:06d}.npy"
            if os.path.exists(f) and r in RIDMETA and RIDMETA[r] in TEMPLATES:
                v = np.load(f)
                loc = TEMPLATES[RIDMETA[r]][0]
                res = v - loc
                y = (res[171:285] + 1j * res[285:]).reshape(2, 57
                                                            ).astype(np.complex64)
                sb = np.stack([y[:, k:k + _L].reshape(-1)
                               for k in range(57 - _L + 1)], 0)
                P = (np.abs(sb @ _ST.conj().T) ** 2).mean(0)
                order = np.argsort(-P)
                picks = []
                for j in order:
                    if all(abs(int(_IPH[j]) - int(_IPH[k])) > 2 or
                           abs(int(_IPS[j]) - int(_IPS[k])) > 2 for k in picks):
                        picks.append(j)
                    if len(picks) >= NPP: break
                lp = np.log10(P[picks] + 1e-12)
                zlp = (lp - lp.mean()) / (lp.std() + 1e-6)
                for n, j in enumerate(picks):
                    phi, psi = _PH[_IPH[j]], _PS[_IPS[j]]
                    t[n, :7] = [np.sin(phi), np.cos(phi), np.sin(psi),
                                np.cos(psi), -0.05, 0.5, zlp[n]]
            PCACHE[r] = t.copy()
        t2 = t.copy(); t2[:, 15 + ri] = 1.0
        out.append(t2)
    return np.concatenate(out, 0)

SCACHE, RIDMETA, TEMPLATES = {}, {}, {}
def _raw_static(r):
    f = f"{TOK}/statics/{r:06d}.npy"
    v = np.load(f).astype(np.float32) if os.path.exists(f) \
        else np.zeros(399, np.float32)
    a = v[:171]; c = v[171:]
    a = (a - a.mean()) / (a.std() + 1e-6)
    return np.r_[a, c].astype(np.float32)

def build_templates():
    """ComBat location+scale per (scene, node), ACTION-BALANCED estimates."""
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin([1, 2, 3, 4])]
    for r in man.itertuples():
        RIDMETA[int(r.rid)] = (int(r.scene), r.node)
    rng_ = np.random.default_rng(1)
    for (sc, nd), g in man.groupby(["scene", "node"]):
        locs, scs = [], []
        for act, ga in g.groupby("act"):
            rs = rng_.permutation(ga.rid.values)[:40]
            vs = np.stack([_raw_static(int(r)) for r in rs])
            locs.append(np.median(vs, 0)); scs.append(vs.std(0))
        loc = np.mean(locs, 0).astype(np.float32)
        scale = np.maximum(np.mean(scs, 0), 1e-2).astype(np.float32)
        TEMPLATES[(int(sc), nd)] = (loc, scale)
    print(f"ComBat templates (loc+scale, act-balanced) for "
          f"{sorted(TEMPLATES)}", flush=True)

def get_static(rids):
    vs = []
    for r in rids:
        if r not in SCACHE:
            v = _raw_static(r)
            if TEMPL and r in RIDMETA and RIDMETA[r] in TEMPLATES:
                loc, scale = TEMPLATES[RIDMETA[r]]
                v = (v - loc) / scale
            SCACHE[r] = v.astype(np.float32)
        vs.append(SCACHE[r])
    return np.stack(vs)                      # (3, 399)

class PoseTok(nn.Module):
    def __init__(self, H=None):
        super().__init__()
        H = H or HDIM
        self.inp = nn.Linear(FTOK, H)
        self.sinp = nn.Linear(399, H)
        self.type_emb = nn.Parameter(torch.zeros(2, H))
        lay = nn.TransformerEncoderLayer(H, HEADS, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, ENCL)
        self.qproj = nn.Linear(64 + (len(POSESLOTS) * 6 * 3 if SLOTQ else 0), H)
        self.att = nn.MultiheadAttention(H, HEADS, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(),
                                nn.Linear(2 * H, H))
        self.out = nn.Linear(H, NJ * 3)
    def forward(self, x, mask, nws, st=None, qs=None, sp=None):
        h = self.inp(x) + self.type_emb[0]
        if sp is not None:
            hp = self.inp(sp) + self.type_emb[1]
            h = torch.cat([h, hp], 1)
            mask = torch.cat([mask, torch.zeros(
                x.shape[0], sp.shape[1], dtype=torch.bool,
                device=mask.device)], 1)
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
        if SLOTQ and qs is not None:
            q = torch.cat([q, qs], -1)
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
    if STATIC or PERSTOK:
        while not os.path.exists(SMARK):
            print("waiting for statics ...", flush=True); time.sleep(60)
        if TEMPL or PERSTOK: build_templates()
    mu = np.zeros((NJ, 3)); sd = np.ones((NJ, 3))
    for j in range(NJ):
        vs = np.concatenate([P[:, j][np.isfinite(P[:, j]).all(-1)]
                             for it in tr for P in [it[1]] if np.isfinite(P[:, j]).any()])
        if len(vs): mu[j] = vs.mean(0); sd[j] = vs.std(0) + 1e-3
    MUt = torch.from_numpy(mu.astype(np.float32)).to(dev)
    SDt = torch.from_numpy(sd.astype(np.float32)).to(dev)
    bt = [mpjpe_pck(np.broadcast_to(mu, it[1].shape), it[1]) for it in te for P in [it[1]]]
    bt = np.array([r for r in bt if r])
    print(f"mean-pose baseline scene4: MPJPE {bt[:,0].mean():.0f} mm  "
          f"PCK@20 {bt[:,1].mean()*100:.1f}  PCK@50 {bt[:,2].mean()*100:.1f}",
          flush=True)
    net = PoseTok().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    OUTD = OUTDIR
    os.makedirs(OUTD, exist_ok=True)
    def qev(ds, cap=150):
        rs, ratio, tc = [], [], []
        mu_np, sd_np = MUt.cpu().numpy(), SDt.cpu().numpy()
        with torch.no_grad():
            for tok, P, nw, rids, S12, SPt in ds[:cap]:
                X = torch.from_numpy(tok.astype(np.float32))[None].to(dev)
                mask = torch.zeros(1, len(tok), dtype=torch.bool, device=dev)
                st = torch.from_numpy(get_static(rids))[None].to(dev) \
                    if STATIC else None
                qs = torch.from_numpy(S12.astype(np.float32))[None].to(dev) \
                    if SLOTQ else None
                spt = (torch.from_numpy(person_tokens(rids).astype(
                    np.float32))[None].to(dev) if PERSTOK else
                    (torch.from_numpy(SPt.astype(np.float32))[None].to(dev)
                     if STATTOK else None))
                pr = net(X, mask, [nw], st, qs, spt)[0, :len(P)].cpu().numpy()
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
    WARM = os.path.expanduser(os.environ.get("WARM", ""))
    if WARM and os.path.exists(WARM) and not os.path.exists(f"{OUTD}/last.pt"):
        wck = torch.load(WARM, map_location=dev, weights_only=False)
        net.load_state_dict(wck["model"])
        print(f"warm-start from {WARM} (step {wck.get('step')})", flush=True)
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
        QS = torch.zeros(B, max(nws), len(POSESLOTS) * 6 * 3)
        nsp = 3 * (NPP if PERSTOK else NSP)
        SP = torch.zeros(B, nsp, 18)
        for k, it in enumerate(items):
            SP[k] = torch.from_numpy(
                person_tokens(it[3]).astype(np.float32)) if PERSTOK \
                else torch.from_numpy(it[5].astype(np.float32))
            X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
            mask[k, :len(it[0])] = False
            Y[k, :len(it[1])] = torch.from_numpy(it[1])
            QS[k, :len(it[4])] = torch.from_numpy(it[4].astype(np.float32))
            if STATIC:
                u = rng.random()
                if u < SDROP:
                    pass                                  # zeros = dropout
                elif u < SDROP + SWAP:
                    oth = tr[rng.integers(len(tr))]
                    S[k] = torch.from_numpy(get_static(oth[3]))
                else:
                    S[k] = torch.from_numpy(get_static(it[3]))
        X, mask, Y, S = X.to(dev), mask.to(dev), Y.to(dev), S.to(dev)
        QS, SP = QS.to(dev), SP.to(dev)
        pred = net(X, mask, nws, S if STATIC else None,
                   QS if SLOTQ else None,
                   SP if (STATTOK or PERSTOK) else None)  # in Z-space
        Z = (Y - MUt) / SDt
        msk = torch.isfinite(Y).all(-1, keepdim=True)
        msk[:, :, ROOTJ] = False
        cands = []
        for sh in range(-SHIFTS, SHIFTS + 1):
            if sh == 0:
                Zs, Ms = Z, msk
            elif sh > 0:
                Zs = torch.cat([Z[:, sh:], torch.full_like(Z[:, :sh], np.nan)], 1)
                Ms = torch.cat([msk[:, sh:], torch.zeros_like(msk[:, :sh])], 1)
            else:
                Zs = torch.cat([torch.full_like(Z[:, :(-sh)], np.nan),
                                Z[:, :sh]], 1)
                Ms = torch.cat([torch.zeros_like(msk[:, :(-sh)]),
                                msk[:, :sh]], 1)
            l_ = (torch.where(Ms, (pred - torch.nan_to_num(Zs)).abs(),
                              torch.zeros_like(pred)).sum(dim=(1, 2, 3))
                  / Ms.sum(dim=(1, 2, 3)).clamp(min=1) / 3)
            cands.append(l_)
        loss = torch.stack(cands, 1).min(1).values.mean()
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
