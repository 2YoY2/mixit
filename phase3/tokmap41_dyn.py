#!/usr/bin/env python3
"""Probe 41g: room map estimated from REPEATED MOVEMENTS (user's anchor).

Every scripted action repeats in every room, so for a fixed (action, node)
the token (phi,psi) distributions of room 1 vs room SRC differ only by the
room effect (+ subject/position noise).  Estimator: energy-weighted
phi/psi histograms on the 37-bin tokenizer grid, per action; circular
cross-correlation alignment -> per-action (dphi,dpsi); aggregate = energy-
weighted circular mean over actions.  Per-action spread printed — small
spread = the rigid-shift room model itself is validated by dynamics.

Same operator class as 41d (rigid grid shift), different anchor:
statics (41d) vs repeated movements (41g).  Scene-SRC tokens are shifted
and written to TOK3 with the usual symlink overlay.

  SRC=2 NCLIP=60 python3 phase3/tokmap41_dyn.py
"""
import os, glob
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
SRC = int(os.environ.get("SRC", "2"))
TOK3 = os.path.expanduser(os.environ.get(
    "TOK3", f"~/zerdani/buffer/octonet/pa_tokens_dynmap{SRC}1"))
NCLIP = int(os.environ.get("NCLIP", "60"))
NODES = ["r1", "r2", "r3"]
NPH, NPS = 37, 37
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
GB = 2 * np.pi / NPH

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def hists(rids, rng):
    """energy-weighted (phi,psi) histograms over clips' tokens"""
    hp, hs, tot = np.zeros(NPH), np.zeros(NPS), 0.0
    rids = list(rids); rng.shuffle(rids)
    for r in rids[:NCLIP]:
        f = f"{TOK}/tokens/{int(r):06d}.npz"
        if not os.path.exists(f): continue
        t = np.load(f)["toks"]
        w = 10.0 ** t[:, 4].astype(np.float64)
        ip = np.searchsorted(PH, t[:, 2] - 1e-4).clip(0, NPH - 1)
        is_ = np.searchsorted(PS, t[:, 3] - 1e-4).clip(0, NPS - 1)
        np.add.at(hp, ip, w); np.add.at(hs, is_, w)
        tot += w.sum()
    return hp, hs, tot

def best_shift(h1, h2):
    """bins to ADD to h2's variable so it aligns with h1 (circular)"""
    sc = [float((h1 * np.roll(h2, s)).sum()) for s in range(NPH)]
    s = int(np.argmax(sc))
    return s if s <= NPH // 2 else s - NPH

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    os.makedirs(f"{TOK3}/tokens", exist_ok=True)
    rng = np.random.default_rng(41)
    DP = {}
    for nd in NODES:
        vphi, vpsi, wts, peracts = [], [], [], []
        for act in sorted(man[man.scene == 1].act.unique()):
            g1 = man[(man.scene == 1) & (man.node == nd) & (man.act == act)]
            g2 = man[(man.scene == SRC) & (man.node == nd) & (man.act == act)]
            if len(g1) < 5 or len(g2) < 5: continue
            hp1, hs1, t1 = hists(g1.rid, rng)
            hp2, hs2, t2 = hists(g2.rid, rng)
            if t1 <= 0 or t2 <= 0: continue
            dphi = best_shift(hp1, hp2) * GB
            dpsi = best_shift(hs1, hs2) * GB
            w = min(t1, t2)
            vphi.append(dphi); vpsi.append(dpsi); wts.append(w)
            peracts.append((int(act), dphi, dpsi))
        wts = np.array(wts)
        mphi = np.angle(np.sum(wts * np.exp(1j * np.array(vphi))))
        mpsi = np.angle(np.sum(wts * np.exp(1j * np.array(vpsi))))
        DP[nd] = (mphi, mpsi)
        sp = np.abs(wrap(np.array(vphi) - mphi))
        print(f"  {nd}: per-act dphi " +
              " ".join(f"a{a}:{d:+.2f}" for a, d, _ in peracts), flush=True)
        print(f"  dynmap{SRC}->1 {nd}: dphi {mphi:+.2f}  dpsi {mpsi:+.2f}  "
              f"(spread med {np.median(sp):.2f}, n_act={len(vphi)})",
              flush=True)

    for name in ("manifest.csv", "pose", "imu", "statics", "statics_add",
                 "tenv", "static_peaks.npz"):
        src, dst = f"{TOK}/{name}", f"{TOK3}/{name}"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for cf in (glob.glob(f"{TOK}/*_1-2-3.pkl")
               + glob.glob(f"{TOK}/tokpose5s123_1.pkl")):
        dst = f"{TOK3}/{os.path.basename(cf)}"
        if not os.path.lexists(dst): os.symlink(cf, dst)
    nlink = ndone = 0
    for r in man.itertuples():
        rid, sc = int(r.rid), int(r.scene)
        src = f"{TOK}/tokens/{rid:06d}.npz"
        dst = f"{TOK3}/tokens/{rid:06d}.npz"
        if not os.path.exists(src) or os.path.lexists(dst): continue
        if sc != SRC:
            os.symlink(src, dst); nlink += 1
            continue
        z = np.load(src)
        toks = z["toks"].copy()
        dphi, dpsi = DP[r.node]
        toks[:, 2] = wrap(toks[:, 2] + dphi)
        toks[:, 3] = wrap(toks[:, 3] + dpsi)
        np.savez(dst, toks=toks.astype(np.float32), nw=z["nw"])
        ndone += 1
    print(f"dynmap done: {ndone} scene-{SRC} recs mapped, {nlink} "
          f"symlinked -> {TOK3}", flush=True)

if __name__ == "__main__":
    main()
