#!/usr/bin/env python3
"""Which slot carries the pose information? (user's probe)
Per slot m: per-window state [log energy, mean Doppler, mean sin/cos phi,
mean sin/cos psi] from the FROZEN separator's soft assignment. Ridge from
ONE slot's state -> z-scored pose deviations (per joint coord), trained on
scenes 1-3 windows, scored on scene 4 (unseen room) as mean |corr| per
joint group. + ALL row (all slots concat).

  python3 diagnostics/31_slot_pose_probe.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
NTR = int(os.environ.get("NTR", "500"))
NTE = int(os.environ.get("NTE", "250"))
LAM = float(os.environ.get("LAM", "100"))
NJ, ROOTJ, HOPF, WINF = 15, 8, 128, 256
GROUPS = {"wrists": [4, 7], "elbows": [3, 6], "knees": [10, 13],
          "ankles": [11, 14], "head": [0, 1]}
dev = "cuda" if torch.cuda.is_available() else "cpu"

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

import time
sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        time.sleep(60)

def slot_state(rid):
    tf, pf = f"{TOK}/tokens/{rid:06d}.npz", f"{TOK}/pose/{rid:06d}.npy"
    if not (os.path.exists(tf) and os.path.exists(pf)): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X7 = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
               np.cos(t[:, 3]), t[:, 1] / 150.0,
               t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X7)[None].to(dev))[0].cpu().numpy()
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int)
    F = np.zeros((nw, M, 6))
    for m in range(M):
        wm = a[:, m] * e
        den = np.zeros(nw); np.add.at(den, w, wm)
        np.add.at(F[:, m, 0], w, wm)
        for ci, v in enumerate((t[:, 1] / 150.0, np.sin(t[:, 2]),
                                np.cos(t[:, 2]), np.sin(t[:, 3]),
                                np.cos(t[:, 3]))):
            acc = np.zeros(nw); np.add.at(acc, w, wm * v)
            F[:, m, 1 + ci] = acc / np.maximum(den, 1e-9)
    F[:, :, 0] = np.log10(F[:, :, 0] + 1e-9)
    P = np.asarray(np.load(pf), np.float32)[:nw]
    return F.astype(np.float32), P

def gather(rids, cap):
    Xs, Ys = [], []
    n = 0
    for rid in rids:
        r = slot_state(int(rid))
        if r is None: continue
        F, P = r
        Xs.append(F); Ys.append(P)
        n += 1
        if n >= cap: break
    return np.concatenate(Xs), np.concatenate(Ys)

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
tr_ids = rng.permutation(man[man.scene.isin([1, 2, 3])].rid.values)
te_ids = rng.permutation(man[man.scene == 4].rid.values)
Xtr, Ptr = gather(tr_ids, NTR)
Xte, Pte = gather(te_ids, NTE)
print(f"train windows {len(Xtr)} | test windows {len(Xte)}", flush=True)
mu = np.nanmean(Ptr, 0); sd = np.nanstd(Ptr, 0) + 1e-3
Ztr = (Ptr - mu) / sd; Zte = (Pte - mu) / sd
JS = [j for j in range(NJ) if j != ROOTJ]

def probe(cols):
    A = Xtr[:, cols].reshape(len(Xtr), -1)
    B = Xte[:, cols].reshape(len(Xte), -1)
    am, asd = A.mean(0), A.std(0) + 1e-9
    A = (A - am) / asd; B = (B - am) / asd
    Y = np.nan_to_num(Ztr[:, JS].reshape(len(Ztr), -1))
    W = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ Y)
    Pr = (B @ W).reshape(len(B), len(JS), 3)
    out = {}
    for g, js in GROUPS.items():
        cs = []
        for j in js:
            ji = JS.index(j)
            for c in range(3):
                y = Zte[:, j, c]; p = Pr[:, ji, c]
                m = np.isfinite(y)
                if m.sum() > 50 and y[m].std() > 1e-6 and p[m].std() > 1e-6:
                    cs.append(abs(np.corrcoef(p[m], y[m])[0, 1]))
        out[g] = np.mean(cs) if cs else np.nan
    allc = []
    for ji, j in enumerate(JS):
        for c in range(3):
            y = Zte[:, j, c]; p = Pr[:, ji, c]
            m = np.isfinite(y)
            if m.sum() > 50 and y[m].std() > 1e-6 and p[m].std() > 1e-6:
                allc.append(abs(np.corrcoef(p[m], y[m])[0, 1]))
    out["ALL"] = np.mean(allc)
    return out

hdr = ["wrists", "elbows", "knees", "ankles", "head", "ALL"]
print(f"\nscene-4 |corr| per slot (rows) x joint group:")
print("slot   " + "".join(f"{h:>8s}" for h in hdr))
for m in range(M):
    r = probe([m])
    print(f"  s{m}  " + "".join(f"{r[h]:8.3f}" for h in hdr))
r = probe(list(range(M)))
print(f"  ALL " + "".join(f"{r[h]:8.3f}" for h in hdr))
print("""
READ: rows far above ~0.05 carry pose info that TRANSFERS. If the room slot
(s3) is flat while motion slots are hot, the separation is semantically
aligned with pose. ALL >> best single slot -> pose info is distributed.""")
