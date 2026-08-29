#!/usr/bin/env python3
"""Per-action limb test (user's design): PA actions are scripted, so the
action LABEL is clean limb ground truth. Step 1: map each action to its limb
usage profile from the keypoint envelopes. Step 2: classify limb-defined
action pairs from CMN-STFT features on held-out clips:

  pair L-R   : most LW-dominant action vs most RW-dominant  (pure laterality)
  pair A-L   : most wrist-dominant vs most leg-dominant     (limb class)
  pair CTRL  : most-motion vs least-motion action           (instrument check)

Accuracy >> 50% on held-out clips (and rooms 4/5) = limb info IS in the
spectra and the earlier pooled-regression null was the instrument's fault.

  python3 diagnostics/21_action_limb.py
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

TR = os.path.expanduser(os.environ.get("PREP_TR", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
TE = os.path.expanduser(os.environ.get("PREP_TE", "~/zerdani/buffer/octonet/prep_pa_xrf400t"))
WINF, HOPF = 256, 128
DEV = ["LW", "RW", "LP", "RP", "HD"]
freqs = np.fft.fftfreq(WINF, 1 / 400.0)
FSEL = (np.abs(freqs) >= 2) & (np.abs(freqs) <= 150)
NPA = int(os.environ.get("NPA", "12"))       # recordings per action for the map

def limb_profile(root, rids):
    prof, tot = [], []
    for rid in rids:
        try:
            gi = np.asarray(np.load(f"{root}/imu/{int(rid):06d}.npy"), np.float32)
        except Exception:
            continue
        g2 = gi.copy()
        for i in range(5):
            oth = [j for j in range(5) if j != i]
            A = np.c_[gi[:, oth], np.ones(len(gi), np.float32)]
            b, *_ = np.linalg.lstsq(A, gi[:, i], rcond=None)
            g2[:, i] = np.clip(gi[:, i] - A @ b, 0, None)
        prof.append(g2.mean(0)); tot.append(float(gi.sum(1).mean()))
    if not prof: return None, 0.0
    return np.mean(prof, 0), float(np.mean(tot))

def clip_feat(root, rid):
    x = np.asarray(np.load(f"{root}/streams/{int(rid):06d}.npy"), np.float32)
    T = len(x)
    if T < WINF + HOPF: return None
    z = x[:, 90:177] + 1j * x[:, 177:264]
    zb = z.mean(0)
    gz = np.abs(zb); thr = 0.05 * np.median(gz) + 1e-9
    zb = np.where(gz < thr, thr + 0j, zb)
    z = z / zb - 1.0
    nw = (T - WINF) // HOPF + 1
    han = np.hanning(WINF)[:, None]
    S = np.empty((nw, int(FSEL.sum())), np.float32)
    for w in range(nw):
        F = np.fft.fft(z[w * HOPF:w * HOPF + WINF] * han, axis=0)
        S[w] = np.log10((np.abs(F[FSEL]) ** 2).mean(1) + 1e-12)
    return np.r_[S.mean(0), S.std(0)]

def classify(root_tr, ids_a, ids_b, root_te=None, te_a=None, te_b=None):
    rng = np.random.default_rng(0)
    def feats(root, ids):
        out = []
        for rid in ids:
            f = clip_feat(root, rid)
            if f is not None: out.append(f)
        return np.array(out)
    Fa, Fb = feats(root_tr, ids_a), feats(root_tr, ids_b)
    na, nb = len(Fa), len(Fb)
    ia, ib = rng.permutation(na), rng.permutation(nb)
    ka, kb = int(na * 0.7), int(nb * 0.7)
    Xtr = np.r_[Fa[ia[:ka]], Fb[ib[:kb]]]
    ytr = np.r_[np.ones(ka), -np.ones(kb)]
    Xte = np.r_[Fa[ia[ka:]], Fb[ib[kb:]]]
    yte = np.r_[np.ones(na - ka), -np.ones(nb - kb)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    G = ((Xtr - mu) / sd)
    B = np.linalg.solve(G.T @ G + 100 * np.eye(G.shape[1]), G.T @ ytr)
    acc = float((np.sign(((Xte - mu) / sd) @ B) == yte).mean())
    acc45 = np.nan
    if root_te is not None and te_a is not None and len(te_a) + len(te_b) > 10:
        Ga, Gb = feats(root_te, te_a), feats(root_te, te_b)
        if len(Ga) > 3 and len(Gb) > 3:
            Xe = np.r_[Ga, Gb]; ye = np.r_[np.ones(len(Ga)), -np.ones(len(Gb))]
            acc45 = float((np.sign(((Xe - mu) / sd) @ B) == ye).mean())
    return acc, acc45, na, nb

mtr = pd.read_csv(f"{TR}/meta.csv"); mtr = mtr[mtr.imu_ok == 1]
mte = pd.read_csv(f"{TE}/meta.csv")
rng = np.random.default_rng(0)
print("== action -> limb usage map (residual envelope share, from GT):")
rows = []
for act, g in mtr.groupby("act"):
    rids = rng.permutation(g.rid.values)[:NPA]
    p, tot = limb_profile(TR, rids)
    if p is None: continue
    dom = int(np.argmax(p))
    rows.append((act, DEV[dom], p / (p.sum() + 1e-9), tot, len(g)))
rows.sort(key=lambda r: r[0])
for act, dom, p, tot, n in rows:
    print(f"  act {act:3d}: dom {dom}  [" +
          " ".join(f"{DEV[i]}:{p[i]:.2f}" for i in range(5)) +
          f"]  motion {tot:6.1f}  n={n}")

def top_act(dev_i, exclude=()):
    best, ba = -1, None
    for act, dom, p, tot, n in rows:
        if act in exclude or n < 30: continue
        if p[dev_i] > best: best, ba = p[dev_i], act
    return ba

aLW, aRW = top_act(0), top_act(1)
leg = max(rows, key=lambda r: (r[2][2] + r[2][3]) if r[4] >= 30 else -1)[0]
wrist = max(rows, key=lambda r: (r[2][0] + r[2][1]) if r[4] >= 30 else -1)[0]
hi = max(rows, key=lambda r: r[3] if r[4] >= 30 else -1)[0]
lo = min(rows, key=lambda r: r[3] if r[4] >= 30 else 1e9)[0]
pairs = [("L-vs-R hand", aLW, aRW), ("wrist-vs-leg", wrist, leg),
         ("CTRL hi-vs-lo motion", hi, lo)]
print()
for nm, a, b in pairs:
    if a is None or b is None or a == b:
        print(f"{nm}: no valid pair"); continue
    ids_a = mtr[mtr.act == a].rid.values
    ids_b = mtr[mtr.act == b].rid.values
    te_a = mte[mte.act == a].rid.values
    te_b = mte[mte.act == b].rid.values
    acc, acc45, na, nb = classify(TR, ids_a, ids_b, TE, te_a, te_b)
    print(f"{nm} (act {a} vs {b}, n={na}/{nb}): heldout acc {acc:.3f}  "
          f"rooms45 acc {acc45 if acc45 == acc45 else float('nan'):.3f}  (chance 0.5)")
print("\nREAD: CTRL must be >>0.5 (instrument works). L-vs-R >>0.5 -> laterality"
      "\nis in the spectra and the pooled-null was instrument failure."
      "\nwrist-vs-leg >>0.5 only -> limb-class info yes, laterality no.")
