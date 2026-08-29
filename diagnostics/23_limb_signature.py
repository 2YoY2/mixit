#!/usr/bin/env python3
"""Feasibility test for per-limb CSI (user's design): separation will live in
the STFT, but limb IDENTIFICATION must live in the subcarrier x antenna
domain. CMN removes the static room print; this probe asks whether the
DYNAMIC signature -- which islands light up when a given limb moves -- is
(a) present within a room and (b) stable across rooms.

Per clip: CMN modulation -> 0.64 s STFT -> +-2-150 Hz band power, averaged
over windows, kept PER ISLAND (87 = 3 ant x 29 subc, untransformed), log,
per-receiver mean removed (intensity out, pattern only), 3 receivers
concatenated = 261-dim spatial signature. Motion-matched pairs from probe 22:
act12(LW) vs act11(RW) = laterality; act5(wrist) vs act17(leg) = limb class.

Rooms available now: scene 1 (train prep) + scenes 4, 5 (test prep).
  A. per-(act,room) prototype cosine matrix    -- eyeball the invariance
  B. within-room 70/30 accuracy                -- is the signature there at all
  C. leave-one-room-out accuracy, 3 rotations  -- does it survive the room
  D. label-shuffle null + intensity-only control (motion shortcut killer)

READ: C >> null -> room-invariant limb ID possible, project is GO cross-room.
B >> null but C ~ null -> per-site calibration tier only. Both ~ null ->
probe 22's closed verdict stands for real.

  python3 diagnostics/23_limb_signature.py
"""
import os
import numpy as np
import pandas as pd

TR = os.path.expanduser(os.environ.get("PREP_TR", "~/zerdani/buffer/octonet/prep_pa_xrf400"))
TE = os.path.expanduser(os.environ.get("PREP_TE", "~/zerdani/buffer/octonet/prep_pa_xrf400t"))
LAM = float(os.environ.get("LAM", "100"))
NSH = int(os.environ.get("NSH", "20"))
WINF, HOPF = 256, 128
freqs = np.fft.fftfreq(WINF, 1 / 400.0)
FSEL = (np.abs(freqs) >= 2) & (np.abs(freqs) <= 150)
PAIRS = [(n, int(a), int(b)) for n, a, b in
         (p.split(":") for p in os.environ.get(
             "PAIRS", "hand-vs-otherhand:1:2").split(";"))]

def sig_rx(root, rid):
    """one receiver's spatial signature: per-island motion-band log energy,
    mean-removed (pattern, not intensity) + the removed intensity scalar."""
    x = np.asarray(np.load(f"{root}/streams/{int(rid):06d}.npy"), np.float32)
    T = len(x)
    if T < WINF + HOPF: return None, None
    z = x[:, 90:177] + 1j * x[:, 177:264]
    zb = z.mean(0)
    gz = np.abs(zb); thr = 0.05 * np.median(gz) + 1e-9
    zb = np.where(gz < thr, thr + 0j, zb)
    z = z / zb - 1.0
    nw = (T - WINF) // HOPF + 1
    han = np.hanning(WINF)[:, None]
    e = np.zeros(87, np.float64)
    for w in range(nw):
        F = np.fft.fft(z[w * HOPF:w * HOPF + WINF] * han, axis=0)
        e += (np.abs(F[FSEL]) ** 2).sum(0)
    e = np.log10(e / nw + 1e-12)
    return (e - e.mean()).astype(np.float32), np.float32(e.mean())

def clips(root, meta, act):
    """(n,261) signatures + (n,3) intensity scalars for 3-rx-complete clips."""
    m = meta[meta.act == act].copy()
    m["ck"] = m["name"].str.replace(r"_r\d$", "", regex=True)
    S, I = [], []
    for ck, g in m.groupby("ck"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        fs = [sig_rx(root, r) for r in g.sort_values("node").rid.values]
        if any(f[0] is None for f in fs): continue
        S.append(np.concatenate([f[0] for f in fs]))
        I.append(np.array([f[1] for f in fs]))
    return np.array(S), np.array(I)

def ridge_acc(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    G = (Xtr - mu) / sd
    B = np.linalg.solve(G.T @ G + LAM * np.eye(G.shape[1]), G.T @ ytr)
    return float((np.sign(((Xte - mu) / sd) @ B) == yte).mean())

def within(X, y, rng):
    ix = rng.permutation(len(y))
    k = int(len(y) * 0.7)
    return ridge_acc(X[ix[:k]], y[ix[:k]], X[ix[k:]], y[ix[k:]])

mtr = pd.read_csv(f"{TR}/meta.csv")
mte = pd.read_csv(f"{TE}/meta.csv")
rng = np.random.default_rng(0)

for pname, aA, aB in PAIRS:
    print(f"\n================ {pname} (act {aA} vs {aB}) ================")
    room, X, I, y = {}, {}, {}, {}
    for rm, root, meta in [(1, TR, mtr), (4, TE, mte[mte.scene == 4]),
                           (5, TE, mte[mte.scene == 5])]:
        Sa, Ia = clips(root, meta, aA)
        Sb, Ib = clips(root, meta, aB)
        if not len(Sa) or not len(Sb):
            print(f"  room {rm}: missing clips ({len(Sa)}/{len(Sb)})"); continue
        X[rm] = np.r_[Sa, Sb]; I[rm] = np.r_[Ia, Ib]
        y[rm] = np.r_[np.ones(len(Sa)), -np.ones(len(Sb))]
        print(f"  room {rm}: n={len(Sa)}/{len(Sb)}")
    rooms = sorted(X)

    print("A) prototype cosines (rows/cols = act@room):")
    protos, labs = [], []
    for rm in rooms:
        for av, an in ((aA, 1), (aB, -1)):
            p = X[rm][y[rm] == an].mean(0)
            protos.append(p / (np.linalg.norm(p) + 1e-12))
            labs.append(f"a{av}@r{rm}")
    P = np.array(protos)
    cm = P @ P.T
    print("      " + "".join(f"{l:>8s}" for l in labs))
    for i, l in enumerate(labs):
        print(f"{l:>6s}" + "".join(f"{cm[i, j]:8.3f}" for j in range(len(labs))))

    print("B) within-room 70/30:")
    for rm in rooms:
        print(f"  room {rm}: acc {within(X[rm], y[rm], rng):.3f}")

    print("C) leave-one-room-out (signature) vs D) controls:")
    for rm in rooms:
        tr = [r for r in rooms if r != rm]
        Xt = np.concatenate([X[r] for r in tr]); yt = np.concatenate([y[r] for r in tr])
        acc = ridge_acc(Xt, yt, X[rm], y[rm])
        nul = np.mean([ridge_acc(Xt, rng.permutation(yt), X[rm], y[rm])
                       for _ in range(NSH)])
        It = np.concatenate([I[r] for r in tr])
        ic = ridge_acc(It, yt, I[rm], y[rm])
        print(f"  test room {rm}: acc {acc:.3f}   shuffle-null {nul:.3f}   "
              f"intensity-only {ic:.3f}")
print("\nREAD: C >> null (and intensity ~ null) on the hand pair = laterality "
      "signature survives the room -> per-limb CSI is possible; separation in "
      "STFT + ID in subcarrier/antenna is the architecture.")
