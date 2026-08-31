#!/usr/bin/env python3
"""Probe 49f: does PER-RECORDING limb annotation of tokens improve HAR?

Per recording: limbtok12 slot envelopes vs residualized GT limb envelopes
-> each slot's PREDOMINANT limb (kept only if top corr beats runner-up by
MARG and is >= CMIN).  Tokens of kept slots get a 5-dim limb one-hot.
Arms (same classifier arch/budget):
  ALL   all tokens, 7-dim                  (tokens-direct control)
  LID   labeled-slot tokens, 7+5-dim       (the design under test)
  SHUF  same tokens, limb labels permuted  (annotation-correctness control)
Train scenes 1-3 (10% heldout), test scene 4.  Mirror-twin table per arm —
laterality is where correct limb naming must show up.
NB the labels come from GT envelopes at train AND test: this measures the
ORACLE-ANNOTATION ceiling (deployment needs an anchor: IMU / vision).

  NTR=4000 NTE=2000 STEPS=5000 python3 diagnostics/49f_limbid_classifier.py
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
STEPS = int(os.environ.get("STEPS", "5000"))
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
torch.manual_seed(0)

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

def harvest(ids, r2a, rng, tag):
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
        if len(t) > 1024:
            t = t[np.argsort(-t[:, 4])[:1024]]
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
        # per-slot predominant limb (per recording, predominance-filtered)
        slotlimb = {}
        for m in range(M):
            cs = np.array([corr(Em[m], G[:, j]) for j in range(5)])
            o = np.argsort(-cs)
            if cs[o[0]] >= CMIN and cs[o[0]] - cs[o[1]] >= MARG:
                slotlimb[m] = int(o[0])
        if len(slotlimb) < 1: continue
        hard = a.argmax(1)
        lab = np.array([slotlimb.get(h, -1) for h in hard])
        keep = lab >= 0
        if keep.sum() < 8: continue
        hot = np.zeros((keep.sum(), 5), np.float32)
        hot[np.arange(keep.sum()), lab[keep]] = 1
        perm = rng.permutation(5)
        hots = np.zeros_like(hot)
        hots[np.arange(keep.sum()), perm[lab[keep]]] = 1
        out.append(dict(Xall=X,
                        Xlid=np.c_[X[keep], hot],
                        Xshu=np.c_[X[keep], hots],
                        act=al - 1))
        if (n_ + 1) % 1000 == 0:
            print(f"  [{tag}] {n_+1}/{len(ids)} kept={len(out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out

class TokCls(nn.Module):
    def __init__(self, fin, H=128):
        super().__init__()
        self.inp = nn.Linear(fin, H)
        lay = nn.TransformerEncoderLayer(H, 4, 2 * H, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.out = nn.Linear(H, NC)
    def forward(self, x, mask):
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        h = (h * (~mask)[:, :, None]).sum(1) / (~mask).sum(1, keepdim=True)
        return self.out(h)

def run_arm(key, fin, tr, ho, te, rng):
    net = TokCls(fin).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
    bycls = {}
    for i, it in enumerate(tr): bycls.setdefault(it["act"], []).append(i)
    keys = sorted(bycls)
    def batch(items, ixb):
        its = [items[i] for i in ixb]
        n = max(len(it[key]) for it in its)
        X = torch.zeros(len(its), n, fin)
        mask = torch.ones(len(its), n, dtype=torch.bool)
        for k2, it in enumerate(its):
            X[k2, :len(it[key])] = torch.from_numpy(it[key])
            mask[k2, :len(it[key])] = False
        return X.to(dev), mask.to(dev)
    t0 = time.time()
    for step in range(STEPS):
        ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
               for c in rng.integers(0, len(keys), 16)]
        X, mask = batch(tr, ixb)
        y = torch.tensor([tr[i]["act"] for i in ixb]).to(dev)
        loss = nn.functional.cross_entropy(net(X, mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  [{key} {step}] CE {loss.item():.3f} "
                  f"{(time.time()-t0)/60:.0f}min", flush=True)
    net.eval()
    def ev(ds):
        P, Y = [], []
        with torch.no_grad():
            for i0 in range(0, len(ds), 16):
                ix = list(range(i0, min(i0 + 16, len(ds))))
                X, mask = batch(ds, ix)
                P += list(net(X, mask).argmax(1).cpu().numpy())
                Y += [ds[i]["act"] for i in ix]
        return np.array(P), np.array(Y)
    return ev(ho), ev(te)

def report(tag, P, Y):
    acc = np.mean(P == Y)
    twins = []
    for a_, b_ in MIRROR_PAIRS:
        m = (Y == a_) | (Y == b_)
        if not m.any(): continue
        pairhit = (P[m] == a_) | (P[m] == b_)
        side = np.mean(P[m] == Y[m])
        twins.append((np.mean(pairhit), side))
    ph = np.mean([t[0] for t in twins])
    se = np.mean([t[1] for t in twins])
    print(f"    {tag}: acc {acc:.3f}  twin-pair-hit {ph:.3f}  "
          f"twin-side-exact {se:.3f}", flush=True)

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    r2a = {int(r.rid): int(r.act) for r in man.itertuples()}
    rng = np.random.default_rng(49)
    tr_ids = rng.permutation(np.array(
        man[man.split == "train"].rid.values))[:NTR]
    te_ids = rng.permutation(np.array(
        man[(man.split == "test") &
            (man.scene.astype(str).str.contains("4"))].rid.values))[:NTE]
    tr_all = harvest(tr_ids, r2a, rng, "train")
    te = harvest(te_ids, r2a, rng, "test4")
    ix = rng.permutation(len(tr_all))
    h = int(len(ix) * 0.9)
    tr = [tr_all[i] for i in ix[:h]]
    ho = [tr_all[i] for i in ix[h:]]
    lid_frac = np.mean([len(it["Xlid"]) / len(it["Xall"])
                        for it in tr_all])
    print(f"train {len(tr)} ho {len(ho)} test4 {len(te)}  "
          f"(labeled tokens = {lid_frac*100:.0f}% of cloud)", flush=True)
    for key, fin, nm in (("Xall", 7, "ALL "), ("Xlid", 12, "LID "),
                         ("Xshu", 12, "SHUF")):
        (Ph, Yh), (Pt, Yt) = run_arm(key, fin, tr, ho, te, rng)
        print(f"  === {nm} (chance 0.059)", flush=True)
        report("heldout", Ph, Yh)
        report("scene4 ", Pt, Yt)
    print("probe 49f done", flush=True)

if __name__ == "__main__":
    main()
