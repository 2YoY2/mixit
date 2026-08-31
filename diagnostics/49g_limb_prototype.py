#!/usr/bin/env python3
"""Probe 49g: ZERO-TRAINING action classification from limb-named slots.

'A slot is a pixel': per recording, name each slot's predominant limb by
correlation (keep only clear predominance), then lay the recording out as
a CANONICAL limb-indexed feature grid — limb x (energy share, envelope
shape, Doppler stats).  Classification = nearest class prototype (class
mean on scenes 1-3, cosine).  No gradients anywhere.

Arms:
  LIMB  features ordered by limb identity  (the design: canonical grid)
  SLOT  same features ordered by slot energy rank (no naming — control)
Report: heldout scenes 1-3, scene 4; mirror-twin pair-hit vs side-exact
(laterality is where limb naming must show up).

  python3 diagnostics/49g_limb_prototype.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
NTR = int(os.environ.get("NTR", "4000"))
NTE = int(os.environ.get("NTE", "2000"))
ACTTH = float(os.environ.get("ACTTH", "0.3"))
MARG = float(os.environ.get("MARG", "0.1"))
CMIN = float(os.environ.get("CMIN", "0.2"))
HOPF, WINF = 128, 256
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR_PAIRS = [(0, 1), (3, 4), (5, 6), (7, 8), (11, 12), (15, 16)]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUNS}/best.pt", map_location="cpu", weights_only=False)
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

sep = SetSep(); sep.load_state_dict(ck["model"])
sep = sep.to(dev).eval()
print(f"limbtok12 loaded, dev={dev}", flush=True)

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def slot_feats(msk, t, e, env, etot):
    """deterministic 7-dim features of one slot's tokens + envelope."""
    if msk.sum() < 3 or env.sum() < 1e-9:
        return np.zeros(7, np.float32)
    es = e[msk]; f = t[msk, 1]
    w = es / es.sum()
    fc = float((w * f).sum())
    b1 = float(w[(f >= 2) & (f < 10)].sum())
    b2 = float(w[(f >= 10) & (f < 40)].sum())
    b3 = float(w[f >= 40].sum())
    ev = env / (env.mean() + 1e-12)
    return np.array([np.log10(es.sum() / (etot + 1e-12) + 1e-6),
                     fc / 150.0, b1, b2, b3,
                     float(ev.std()),
                     float(np.abs(np.diff(ev)).mean())], np.float32)

def harvest(ids, r2a, tag):
    out = []
    t0 = time.time()
    for n_, rid in enumerate(ids):
        rid = int(rid)
        al = r2a.get(rid)
        if al is None: continue
        tf, gf = f"{TOK}/tokens/{rid:06d}.npz", f"{TOK}/imu/{rid:06d}.npy"
        if not (os.path.exists(tf) and os.path.exists(gf)): continue
        z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
        if len(t) < 16: continue
        gi = np.asarray(np.load(gf), np.float32)
        g2 = gi.copy()
        for i_ in range(5):
            oth = [j for j in range(5) if j != i_]
            A_ = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
            beta, *_ = np.linalg.lstsq(A_, gi[:, i_], rcond=None)
            g2[:, i_] = np.clip(gi[:, i_] - A_ @ beta, 0, None)
        G = np.stack([g2[w * HOPF:w * HOPF + WINF].mean(0)
                      for w in range(nw)])
        if len(G) < 8: continue
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                  np.cos(t[:, 3]), t[:, 1] / 150.0,
                  t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float32)
        widx = t[:, 0].astype(int)
        with torch.no_grad():
            a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
        Em = np.zeros((M, nw))
        for m in range(M):
            np.add.at(Em[m], widx, a[:, m] * e)
        hard = a.argmax(1)
        etot = float(e.sum())
        # per-slot predominant limb, predominance-filtered
        slotlimb = {}
        slotcorr = {}
        for m in range(M):
            cs = np.array([corr(Em[m], G[:, j]) for j in range(5)])
            o = np.argsort(-cs)
            if cs[o[0]] >= CMIN and cs[o[0]] - cs[o[1]] >= MARG:
                slotlimb[m] = int(o[0]); slotcorr[m] = cs[o[0]]
        if len(slotlimb) < 1: continue
        # LIMB grid: for each limb, its best-corr named slot (or zeros)
        FL = np.zeros((5, 7), np.float32)
        for lb in range(5):
            cand = [m for m, l_ in slotlimb.items() if l_ == lb]
            if not cand: continue
            m = max(cand, key=lambda mm: slotcorr[mm])
            FL[lb] = slot_feats(hard == m, t, e, Em[m], etot)
        # SLOT grid control: top-5 slots by energy, same features
        order = np.argsort(-Em.sum(1))[:5]
        FS = np.stack([slot_feats(hard == m, t, e, Em[m], etot)
                       for m in order])
        out.append(dict(FL=FL.ravel(), FS=FS.ravel(), act=al - 1))
        if (n_ + 1) % 1000 == 0:
            print(f"  [{tag}] {n_+1}/{len(ids)} kept={len(out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out

def proto_classify(tr, te, key):
    Xtr = np.stack([it[key] for it in tr])
    ytr = np.array([it["act"] for it in tr])
    Xte = np.stack([it[key] for it in te])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    P = np.stack([Xtr[ytr == k].mean(0) if (ytr == k).any()
                  else np.zeros(Xtr.shape[1]) for k in range(NC)])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    return (Xn @ Pn.T).argmax(1)

def report(tag, P, Y):
    twins = []
    for a_, b_ in MIRROR_PAIRS:
        m = (Y == a_) | (Y == b_)
        if not m.any(): continue
        twins.append((np.mean((P[m] == a_) | (P[m] == b_)),
                      np.mean(P[m] == Y[m])))
    print(f"    {tag}: acc {np.mean(P == Y):.3f}  "
          f"twin-pair-hit {np.mean([t[0] for t in twins]):.3f}  "
          f"twin-side-exact {np.mean([t[1] for t in twins]):.3f}",
          flush=True)

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    r2a = {int(r.rid): int(r.act) for r in man.itertuples()}
    rng = np.random.default_rng(49)
    tr_ids = rng.permutation(np.array(
        man[man.split == "train"].rid.values))[:NTR]
    te_ids = rng.permutation(np.array(
        man[(man.split == "test") &
            (man.scene.astype(str).str.contains("4"))].rid.values))[:NTE]
    tr_all = harvest(tr_ids, r2a, "train")
    te = harvest(te_ids, r2a, "test4")
    ix = rng.permutation(len(tr_all))
    h = int(len(ix) * 0.85)
    tr = [tr_all[i] for i in ix[:h]]
    ho = [tr_all[i] for i in ix[h:]]
    print(f"prototype pool {len(tr)}  heldout {len(ho)}  scene4 {len(te)}",
          flush=True)
    for key, nm in (("FL", "LIMB-grid"), ("FS", "SLOT-grid")):
        print(f"  === {nm} (nearest class prototype, chance 0.059)",
              flush=True)
        Yh = np.array([it["act"] for it in ho])
        report("heldout", proto_classify(tr, ho, key), Yh)
        Yt = np.array([it["act"] for it in te])
        report("scene4 ", proto_classify(tr, te, key), Yt)
    print("probe 49g done", flush=True)

if __name__ == "__main__":
    main()
