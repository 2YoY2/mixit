#!/usr/bin/env python3
"""Confirmation runs for the limb-recognition pipeline (plan step 1+2):

  A. MOTION-MATCHED wrist-vs-leg (act 5 vs 17, motion 5.4 vs 5.7): removes the
     intensity shortcut. Above chance here = genuine limb-class information.
  B. LATERALITY, per-receiver view (act 12 LW vs act 11 RW, n~759 each,
     motion-matched 4.6 vs 4.2): features = the 3 receivers' spectra
     CONCATENATED per clip (spatial cue kept) vs single-receiver (destroyed).
     The gap between those two accuracies isolates the spatial contribution.

  python3 diagnostics/22_limb_confirm.py
"""
import os
import numpy as np
import pandas as pd

TR = os.path.expanduser(os.environ.get("PREP_TR", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
TE = os.path.expanduser(os.environ.get("PREP_TE", "~/zerdani/buffer/octonet/prep_pa_xrf400t"))
WINF, HOPF = 256, 128
freqs = np.fft.fftfreq(WINF, 1 / 400.0)
FSEL = (np.abs(freqs) >= 2) & (np.abs(freqs) <= 150)

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

def ridge_cls(Xa, Xb, Ea=None, Eb=None):
    rng = np.random.default_rng(0)
    na, nb = len(Xa), len(Xb)
    ia, ib = rng.permutation(na), rng.permutation(nb)
    ka, kb = int(na * .7), int(nb * .7)
    Xtr = np.r_[Xa[ia[:ka]], Xb[ib[:kb]]]
    ytr = np.r_[np.ones(ka), -np.ones(kb)]
    Xte = np.r_[Xa[ia[ka:]], Xb[ib[kb:]]]
    yte = np.r_[np.ones(na - ka), -np.ones(nb - kb)]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    G = (Xtr - mu) / sd
    B = np.linalg.solve(G.T @ G + 100 * np.eye(G.shape[1]), G.T @ ytr)
    acc = float((np.sign(((Xte - mu) / sd) @ B) == yte).mean())
    acc2 = np.nan
    if Ea is not None and len(Ea) > 3 and len(Eb) > 3:
        Xe = np.r_[Ea, Eb]; ye = np.r_[np.ones(len(Ea)), -np.ones(len(Eb))]
        acc2 = float((np.sign(((Xe - mu) / sd) @ B) == ye).mean())
    return acc, acc2

def per_rid(root, meta, act, cap=400):
    rids = meta[meta.act == act].rid.values[:cap]
    out = [f for f in (clip_feat(root, r) for r in rids) if f is not None]
    return np.array(out) if out else np.empty((0, 1))

def per_clip3(root, meta, act, cap=400):
    m = meta[meta.act == act].copy()
    m["ck"] = m.name.str.replace(r"_r\d$", "", regex=True)
    out = []
    for ck, g in m.groupby("ck"):
        if len(g) != 3: continue
        fs = [clip_feat(root, r) for r in g.sort_values("node").rid.values]
        if any(f is None for f in fs): continue
        out.append(np.concatenate(fs))
        if len(out) >= cap: break
    return np.array(out) if out else np.empty((0, 1))

mtr = pd.read_csv(f"{TR}/meta.csv")
mte = pd.read_csv(f"{TE}/meta.csv")

print("A) motion-matched wrist(act5, 5.4) vs leg(act17, 5.7):")
acc, acc45 = ridge_cls(per_rid(TR, mtr, 5), per_rid(TR, mtr, 17),
                       per_rid(TE, mte, 5), per_rid(TE, mte, 17))
print(f"   heldout {acc:.3f}   rooms45 {acc45:.3f}   (chance 0.5)")

print("B) laterality act12(LW) vs act11(RW), motion-matched:")
acc1, a145 = ridge_cls(per_rid(TR, mtr, 12), per_rid(TR, mtr, 11),
                       per_rid(TE, mte, 12), per_rid(TE, mte, 11))
print(f"   single-receiver (spatial cue destroyed): heldout {acc1:.3f}  rooms45 {a145:.3f}")
acc3, a345 = ridge_cls(per_clip3(TR, mtr, 12), per_clip3(TR, mtr, 11),
                       per_clip3(TE, mte, 12), per_clip3(TE, mte, 11))
print(f"   3-receiver concat (spatial cue kept)   : heldout {acc3:.3f}  rooms45 {a345:.3f}")
print("""
READ: A >> 0.5 -> limb-class info is real, not intensity. B: 3-rx >> 1-rx ->
laterality lives in the spatial view, as physics demands; both ~0.5 ->
laterality absent at this aperture.""")
