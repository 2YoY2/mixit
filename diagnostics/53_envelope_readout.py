#!/usr/bin/env python3
"""Probe 53: can you SEE the movement in what each representation explains?

Per recording (fine grid): reconstruct the 5 limb envelope curves by
least-squares from each representation's 8 basis curves (RAW-8 blocks,
CLEAN-8 Doppler bands, SLOT-8), then classify the action from the
reconstructed limb-labeled curves — nearest class prototype, ZERO training.
GT arm = classify from the true limb envelopes (the ceiling of this
readout).  Laterality is the headline: the curve basis is limb-LABELED
(L-wrist vs R-wrist row), so mirror twins should resolve if the
reconstruction is faithful.

  NTR=1500 NTE=1000 python3 diagnostics/53_envelope_readout.py
"""
import os, time
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get(
    "TOK", "~/zerdani/buffer/octonet/pa_tokens_fine2"))
ROOT = os.path.expanduser(os.environ.get(
    "ROOT", "~/zerdani/buffer/PerceptAlign"))
RUNS = os.path.expanduser(os.environ.get(
    "MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
NTR = int(os.environ.get("NTR", "1500"))
NHO = int(os.environ.get("NHO", "400"))
NTE = int(os.environ.get("NTE", "1000"))
FS, WINF = 400.0, 256
HOPF = int(os.environ.get("HOPF", "32"))
NB8, T0, NC = 8, 24, 17
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

def raw_groups(path, nw):
    try:
        with h5py.File(path, "r") as h:
            c = h["csi/csi"][...]
            ts = h["csi/timestamp"][...].ravel().astype(np.float64)
    except Exception:
        return None
    x = np.abs(c["real"] + 1j * c["imag"]).astype(np.float32)
    dt = float(np.median(np.diff(ts)))
    t = None
    for unit in (1.0, 1e-3, 1e-6, 1e-9):
        if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
            t = (ts - ts[0]) * unit; break
    if t is None:
        t = np.arange(x.shape[-1]) / 810.0
    keep = np.concatenate([[True], np.diff(t) > 0])
    x, t = x[..., keep], t[keep]
    x = np.moveaxis(x, -1, 0)
    T = len(x)
    d = np.abs(x[1:] - x[:-1]).reshape(T - 1, -1)
    nch = d.shape[1]
    bounds = np.linspace(0, nch, NB8 + 1).astype(int)
    dg = np.stack([d[:, bounds[i]:bounds[i + 1]].sum(1)
                   for i in range(NB8)], 1)
    tb = t[1:]
    nb = int(float(t[-1]) * FS)
    if nb < WINF + 2 * HOPF: return None
    idx = np.minimum((tb * FS).astype(int), nb - 1)
    acc = np.zeros((nb, NB8)); cnt = np.zeros(nb)
    np.add.at(acc, idx, dg); np.add.at(cnt, idx, 1)
    e = acc / np.maximum(cnt, 1)[:, None]
    nww = min(nw, (nb - WINF) // HOPF + 1)
    return np.stack([e[w * HOPF:w * HOPF + WINF].mean(0)
                     for w in range(nww)])

def recon(E, Y):
    A = np.c_[E, np.ones(len(E))]
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return A @ beta

def resamp(Y):
    """(nww, 5) -> (T0*5,) flattened, per-curve unit-RMS."""
    idx = np.linspace(0, len(Y) - 1, T0)
    out = np.stack([np.interp(idx, np.arange(len(Y)), Y[:, j])
                    for j in range(5)], 1)
    out = out / (np.sqrt((out ** 2).mean(0)) + 1e-9)
    return out.ravel()

def harvest(ids, r2a, r2f, tag):
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
        R = raw_groups(os.path.join(ROOT, r2f[rid]), nw)
        if R is None or len(R) < 16: continue
        nww = len(R)
        Y = np.stack([[gi[:, j][w * HOPF:w * HOPF + WINF].mean()
                       for w in range(nww)] for j in range(5)], 1)
        if Y.std() < 1e-9: continue
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                  np.cos(t[:, 3]), t[:, 1] / 150.0,
                  t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
        e = (10.0 ** le).astype(np.float64)
        widx = t[:, 0].astype(int)
        with torch.no_grad():
            a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
        hard = a.argmax(1)
        S8 = np.zeros((nww, M))
        for m in range(M):
            em = np.zeros(nw)
            np.add.at(em, widx[hard == m], e[hard == m])
            S8[:, m] = em[:nww]
        fb = np.linspace(2, 150, NB8 + 1)
        C8 = np.zeros((nww, NB8))
        bi = np.clip(np.searchsorted(fb, t[:, 1]) - 1, 0, NB8 - 1)
        for b in range(NB8):
            em = np.zeros(nw)
            mb = bi == b
            np.add.at(em, widx[mb], e[mb])
            C8[:, b] = em[:nww]
        out.append(dict(GT=resamp(Y), SLOT=resamp(recon(S8, Y)),
                        RAW=resamp(recon(R, Y)),
                        CLEAN=resamp(recon(C8, Y)), act=al - 1))
        if (n_ + 1) % 500 == 0:
            print(f"  [{tag}] {n_+1}/{len(ids)} kept={len(out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out

def proto(tr, te, key):
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
          f"twin-pair {np.mean([t[0] for t in twins]):.3f}  "
          f"twin-side {np.mean([t[1] for t in twins]):.3f}", flush=True)

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    r2a = {int(r.rid): int(r.act) for r in man.itertuples()}
    r2f = {int(r.rid): r.file for r in man.itertuples()}
    rng = np.random.default_rng(53)
    tr_ids = rng.permutation(np.array(
        man[man.split == "train"].rid.values))[:NTR + NHO]
    te_ids = rng.permutation(np.array(
        man[(man.split == "test") &
            (man.scene.astype(str).str.contains("4"))].rid.values))[:NTE]
    allitems = harvest(tr_ids, r2a, r2f, "train")
    te = harvest(te_ids, r2a, r2f, "test4")
    ix = rng.permutation(len(allitems))
    h = max(len(allitems) - NHO, int(len(allitems) * 0.8))
    tr = [allitems[i] for i in ix[:h]]
    ho = [allitems[i] for i in ix[h:]]
    print(f"proto pool {len(tr)}  heldout {len(ho)}  scene4 {len(te)}",
          flush=True)
    Yh = np.array([it["act"] for it in ho])
    Yt = np.array([it["act"] for it in te])
    for key in ("GT", "SLOT", "RAW", "CLEAN"):
        print(f"  === {key} envelopes (nearest prototype, chance 0.059)",
              flush=True)
        report("heldout", proto(tr, ho, key), Yh)
        report("scene4 ", proto(tr, te, key), Yt)
    print("probe 53 done", flush=True)

if __name__ == "__main__":
    main()
