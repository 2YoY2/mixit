#!/usr/bin/env python3
"""Purity evaluation on the superposition bench: v1 vs v2, same M.

Per mix: run clusterer, hard-assign tokens; per slot: energy split by true
owner (valid tokens only).  Report:
  purity     energy-weighted mean of max-owner fraction per slot
  slot-SIR   10*log10(pure/impure) per slot, energy-weighted mean (dB)
  tok-acc    token ownership accuracy under slot-majority mapping
  excluded   fraction of token energy with own=-1 (collisions/ambiguous)
Reported at FIXED M; purity inflates with slot count, so cross-M numbers
are not comparable (classic clustering-purity caveat).

  DATASET=pa CKV2=~/zerdani/buffer/clusterv2/runs/clusterv2.pa.r1/best.pt \
  CKV1=~/zerdani/buffer/cluster/runs/clusterer/limbtok.pa-v1.r12/best.pt \
  python3 clusterv2/eval/eval_purity.py
"""
import os, glob
import numpy as np
import torch
import torch.nn as nn

DATASET = os.environ.get("DATASET", "pa")
BENCHD = os.path.expanduser(os.environ.get(
    "BENCHD", "~/zerdani/buffer/clusterv2/bench"))
CKV1 = os.path.expanduser(os.environ.get("CKV1", ""))
CKV2 = os.path.expanduser(os.environ.get("CKV2", ""))
NMIX = int(os.environ.get("NMIX", "400"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(ck):
    cfg = ck.get("cfg", {})
    M = cfg.get("M", 8); D = cfg.get("D", 256); NL = cfg.get("NL", 6)
    class SetSep(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(7, D)
            lay = nn.TransformerEncoderLayer(
                D, 4, 2 * D, batch_first=True, norm_first=True,
                dropout=0.0)
            self.enc = nn.TransformerEncoder(lay, NL)
            self.head = nn.Linear(D, M)
        def forward(self, x):
            return torch.softmax(self.head(self.enc(self.inp(x))), -1)
    m = SetSep()
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), M

def feats(t, nw):
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    return np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                 np.cos(t[:, 3]), t[:, 1] / 150.0,
                 t[:, 0] / max(nw - 1, 1), zle].astype(np.float32), \
        (10.0 ** le).astype(np.float64)

def evaluate(tag, ckpath, mixes):
    ck = torch.load(ckpath, map_location=dev, weights_only=False)
    model, M = build_model(ck)
    purs, sirs, accs, excl = [], [], [], []
    soft = []
    with torch.no_grad():
        for t, own, nw, ratio in mixes:
            X, e = feats(t, nw)
            a = model(torch.from_numpy(X)[None].to(dev))[0]
            hard = a.argmax(1).cpu().numpy()
            v = own >= 0
            etot_all = e.sum()
            excl.append(1 - e[v].sum() / max(etot_all, 1e-12))
            tot = pure = 0.0
            acc_n = acc_d = 0.0
            for m_ in range(M):
                sel = (hard == m_) & v
                if not sel.any(): continue
                e0 = e[sel][own[sel] == 0].sum()
                e1 = e[sel][own[sel] == 1].sum()
                tot += e0 + e1
                pure += max(e0, e1)
                maj = 0 if e0 >= e1 else 1
                acc_n += e[sel][own[sel] == maj].sum()
                acc_d += e0 + e1
            if tot <= 0: continue
            p = pure / tot
            purs.append(p)
            sirs.append(10 * np.log10(min(p, 1 - 1e-4)
                                      / max(1 - p, 1e-4)))
            accs.append(acc_n / max(acc_d, 1e-12))
            if ratio is not None:
                alive = np.isfinite(ratio)
                st = sp = 0.0
                for m_ in range(M):
                    sel = (hard == m_) & alive
                    if not sel.any(): continue
                    eA = (e[sel] * ratio[sel]).sum()
                    eB = (e[sel] * (1 - ratio[sel])).sum()
                    st += eA + eB
                    sp += max(eA, eB)
                if st > 0: soft.append(sp / st)
    sf = f"  SOFT-purity {np.mean(soft):.3f}" if soft else ""
    print(f"[{tag:18s}] M={M}  purity {np.mean(purs):.3f}  "
          f"slot-SIR {np.mean(sirs):5.1f} dB  tok-acc {np.mean(accs):.3f}"
          f"  excluded {np.mean(excl):.2f}{sf}  (N={len(purs)})",
          flush=True)

def main():
    d = f"{BENCHD}/{DATASET}_val" + os.environ.get("SUFF", "")
    fs = sorted(glob.glob(f"{d}/*.npz"))[:NMIX]
    mixes = []
    for f in fs:
        z = np.load(f)
        t, own, nw = z["toks"], z["own"].astype(np.int64), int(z["nw"])
        ratio = z["ratio"].astype(np.float64) if "ratio" in z.files else None
        if len(t) < 32 or (own >= 0).sum() < 16: continue
        mixes.append((t, own, nw, ratio))
    print(f"{DATASET} bench val: {len(mixes)} mixes", flush=True)
    if CKV1: evaluate("v1", CKV1, mixes)
    if CKV2: evaluate("v2", CKV2, mixes)

if __name__ == "__main__":
    main()
