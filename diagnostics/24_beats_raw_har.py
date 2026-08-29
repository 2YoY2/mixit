#!/usr/bin/env python3
"""Track 2 "beats raw" HAR probe: clip-level action recognition on the SAME
features and SAME classifier, fed four different channels of each recording:

  raw   mean-removed complex islands of the raw stream (room-colored baseline)
  cmn   CMN modulation z/z_bar - 1 (normalization alone, no model)
  body  CMN separator's dynamic channel  y[1:].sum  (the product)
  room  CMN separator's room slot        y[0]       (must carry ~no action info)

Train scenes 1-3 (70% of clip-keys; 30% = room-familiar heldout), test rooms
4/5 never trained on -- neither by the separator nor by the classifier.
Featurizer = the validated micro-Doppler basis (probes 20-22): 0.64 s STFT on
islands, +-2-150 Hz two-sided log power pooled over channels, clip feature =
per-freq mean+std over windows. Classifier = one-vs-rest ridge, lam=100.
The cmn arm attributes any body-over-raw win between normalization and
separation. Flagship read: body >= raw AND room ~ chance.

  PYTHONPATH=~/zerdani/buffer/octonet/ref_asteroid \
  PREP_TR=~/zerdani/buffer/octonet/prep_pa_xrf400,~/zerdani/buffer/octonet/prep_pa_xrf400b \
  python3 diagnostics/23_beats_raw_har.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TRS   = [os.path.expanduser(p) for p in os.environ.get(
         "PREP_TR", "~/zerdani/buffer/octonet/prep_pa_xrf400").split(",") if p]
TE    = os.path.expanduser(os.environ.get("PREP_TE", "~/zerdani/buffer/octonet/prep_pa_xrf400t"))
RUNS  = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/archive2/pa400_cmn_runs"))
CKPT  = os.environ.get("CKPT", "best.pt")
CACHE = os.path.expanduser(os.environ.get("CACHE", "~/zerdani/buffer/octonet/har23_runs"))
MAXTR = int(os.environ.get("MAXTR", "0"))    # per-root caps, smoke only
MAXTE = int(os.environ.get("MAXTE", "0"))
BATCH = int(os.environ.get("BATCH", "16"))
LAM   = float(os.environ.get("LAM", "100"))
C, WINF, HOPF = 264, 256, 128                # 0.64 s STFT @ 400 Hz
ARMS = ("raw", "cmn", "body", "room")
dev = "cuda" if torch.cuda.is_available() else "cpu"
freqs = np.fft.fftfreq(WINF, 1 / 400.0)
FSEL = (np.abs(freqs) >= 2) & (np.abs(freqs) <= 150)
NF = int(FSEL.sum())

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
assert ck.get("xrf") and int(ck["cfg"].get("CMN", 0)), "need a CMN train_xrf ckpt"
SELF, M = int(ck["cfg"].get("SELF", 0)), int(ck["cfg"]["M"])
print(f"ckpt {CKPT} step {ck['step']} cfg {ck['cfg']} dev={dev}", flush=True)

from asteroid.masknn import TDConvNet
class Sep(nn.Module):
    def __init__(self, cin=C, nf=512, L=16, S=8):
        super().__init__()
        self.enc = nn.Conv1d(cin * (1 + SELF), nf, L, stride=S)
        self.masker = TDConvNet(in_chan=nf, n_src=M, out_chan=nf, n_blocks=8,
                                n_repeats=4, bn_chan=192, hid_chan=512,
                                skip_chan=192, mask_act="linear",
                                causal=True, norm_type="cLN")
        self.dec = nn.ConvTranspose1d(nf, cin, L, stride=S)
    def forward(self, x):
        T = x.shape[-1]
        if SELF:
            rm = torch.cumsum(x, -1) / torch.arange(1, T + 1, device=x.device)
            x = torch.cat([x, rm], 1)
        e = self.enc(x)
        y = self.dec((self.masker(e) * e.unsqueeze(1)).flatten(0, 1))[..., :T]
        y = y.view(-1, M, C, T)
        res = x[:, :C] - y.sum(1)
        return torch.cat([y[:, :1], (y[:, 1] + res).unsqueeze(1), y[:, 2:]], 1)

# claim the GPU before any heavy file reads (GB10 page-cache landmine)
net = Sep()
net.load_state_dict(ck["model"])
for attempt in range(10):
    try:
        net = net.to(dev).eval(); break
    except RuntimeError:
        print(f"to({dev}) failed, retry {attempt+1}/10 in 60s", flush=True)
        time.sleep(60)

def cmn_in(x, sa):
    aa = np.abs(sa[:90])
    ga = np.maximum(aa, 0.05 * np.median(aa) + 1e-9)
    zb = sa[90:177] + 1j * sa[177:264]
    gz = np.abs(zb)
    thr = 0.05 * np.median(gz) + 1e-9
    zb = np.where(gz < thr, thr + 0j, zb)
    va = x[:, :90] / ga - 1.0
    vz = (x[:, 90:177] + 1j * x[:, 177:264]) / zb - 1.0
    return np.concatenate([va, vz.real, vz.imag], -1).astype(np.float32)

def stft_feat(z):
    """z (T,87) complex modulation -> [mean_w, std_w] of two-sided log power."""
    T = len(z)
    if T < WINF + HOPF: return None
    nw = (T - WINF) // HOPF + 1
    han = np.hanning(WINF)[:, None]
    S = np.empty((nw, NF), np.float32)
    for w in range(nw):
        F = np.fft.fft(z[w * HOPF:w * HOPF + WINF] * han, axis=0)
        S[w] = np.log10((np.abs(F[FSEL]) ** 2).mean(1) + 1e-12)
    return np.r_[S.mean(0), S.std(0)]

def isl(v):
    return v[:, 90:177] + 1j * v[:, 177:264]

def featurize(root, rids):
    """all four arms for a list of rids; net batched over length-sorted pads."""
    out = {a: {} for a in ARMS}
    order = []
    for rid in rids:
        try:
            x = np.asarray(np.load(f"{root}/streams/{int(rid):06d}.npy"), np.float32)
        except Exception:
            continue
        T = len(x) - ((len(x) - 16) % 8)
        if T < WINF + HOPF: continue
        order.append((int(rid), T))
    order.sort(key=lambda r: r[1])
    t0 = time.time()
    for b0 in range(0, len(order), BATCH):
        chunk = order[b0:b0 + BATCH]
        Tm = max(t for _, t in chunk)
        xb = np.zeros((len(chunk), Tm, C), np.float32)
        xins = []
        for k, (rid, T) in enumerate(chunk):
            x = np.asarray(np.load(f"{root}/streams/{rid:06d}.npy"),
                           np.float32)[:T]
            xin = cmn_in(x, x.mean(0))
            xb[k, :T] = xin
            xins.append(xin)
            zr = isl(x)
            out["raw"][rid] = stft_feat(zr - zr.mean(0))
            out["cmn"][rid] = stft_feat(isl(xin))
        with torch.no_grad():
            y = net(torch.from_numpy(xb.transpose(0, 2, 1)).to(dev)).cpu().numpy()
        for k, (rid, T) in enumerate(chunk):
            out["body"][rid] = stft_feat(isl(y[k, 1:].sum(0).T[:T]))
            out["room"][rid] = stft_feat(isl(y[k, 0].T[:T]))
        if (b0 // BATCH) % 50 == 0:
            print(f"  {b0 + len(chunk)}/{len(order)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out

def load_or_make(root, meta, cap):
    rids = meta.rid.values
    if cap and len(rids) > cap:
        rids = np.random.default_rng(0).permutation(rids)[:cap]
    tag = os.path.basename(root.rstrip("/")) + (f"_cap{cap}" if cap else "")
    cf = f"{CACHE}/feats_{tag}.npz"
    if os.path.exists(cf):
        z = np.load(cf)
        return {a: dict(zip(z["rids"].tolist(),
                            np.asarray(z[a], np.float32))) for a in ARMS}
    print(f"featurizing {root} ({len(rids)} recs) ...", flush=True)
    out = featurize(root, rids)
    ok = [r for r in out["raw"] if all(out[a][r] is not None for a in ARMS)]
    np.savez(cf, rids=np.array(ok, np.int64),
             **{a: np.stack([out[a][r] for r in ok]) for a in ARMS})
    return {a: {r: out[a][r] for r in ok} for a in ARMS}

def table(meta, feats, arm):
    """align meta rows with cached features; returns X, y(0-based), clipkey."""
    m = meta[meta.rid.isin(feats[arm])].copy()
    X = np.stack([feats[arm][int(r)] for r in m.rid])
    return X, (m.act.values - 1).astype(int), \
        m.name.str.replace(r"_r\d$", "", regex=True).values

def fit_ovr(X, y, K):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    G = (X - mu) / sd
    Y = -np.ones((len(y), K), np.float32)
    Y[np.arange(len(y)), y] = 1.0
    B = np.linalg.solve(G.T @ G + LAM * np.eye(G.shape[1]), G.T @ Y)
    return mu, sd, B

def bal(y, p, K):
    return float(np.mean([np.mean(p[y == k] == k) for k in range(K) if (y == k).any()]))

os.makedirs(CACHE, exist_ok=True)
mtr = pd.concat([pd.read_csv(f"{r}/meta.csv").assign(_root=r) for r in TRS])
mtr = mtr[(mtr.split == "train") & (mtr.nsamp >= WINF + HOPF)].reset_index(drop=True)
mte = pd.read_csv(f"{TE}/meta.csv").assign(_root=TE)
mte = mte[(mte.split == "test") & (mte.nsamp >= WINF + HOPF)].reset_index(drop=True)
K = int(max(mtr.act.max(), mte.act.max()))
print(f"train {len(mtr)} recs scenes {sorted(mtr.scene.unique())} | "
      f"test {len(mte)} recs scenes {sorted(mte.scene.unique())} | K={K}", flush=True)

ftr = {a: {} for a in ARMS}
for r in TRS:
    f = load_or_make(r, mtr[mtr._root == r], MAXTR)
    for a in ARMS: ftr[a].update(f[a])
fte = load_or_make(TE, mte, MAXTE)

rng = np.random.default_rng(0)
keys = mtr.name.str.replace(r"_r\d$", "", regex=True)
uk = rng.permutation(sorted(keys.unique()))
hold = set(uk[int(len(uk) * 0.7):])
print(f"{len(uk)} train clip-keys, {len(hold)} held out | chance {1/K:.3f} | "
      f"maj45 {mte.act.value_counts().max()/len(mte):.3f}\n", flush=True)
print(f"{'arm':6s}{'heldout':>9s}{'ho-bal':>8s}{'rooms45':>9s}{'r45-bal':>9s}"
      f"{'r45-clip':>10s}{'clip-bal':>10s}")
print("-" * 61)
for a in ARMS:
    X, y, ckey = table(mtr, ftr, a)
    ho = np.isin(ckey, list(hold))
    mu, sd, B = fit_ovr(X[~ho], y[~ho], K)
    pho = (((X[ho] - mu) / sd) @ B).argmax(1)
    Xe, ye, ce = table(mte, {a: fte[a]}, a)
    S45 = ((Xe - mu) / sd) @ B
    p45 = S45.argmax(1)
    cs = pd.DataFrame(S45).groupby(ce).sum()
    cy = pd.Series(ye, index=ce).groupby(level=0).first()
    pc, yc = cs.values.argmax(1), cy.loc[cs.index].values
    print(f"{a:6s}{np.mean(pho == y[ho]):9.3f}{bal(y[ho], pho, K):8.3f}"
          f"{np.mean(p45 == ye):9.3f}{bal(ye, p45, K):9.3f}"
          f"{np.mean(pc == yc):10.3f}{bal(yc, pc, K):10.3f}", flush=True)

print("""
READ: flagship = body >= raw on rooms45 with room ~ chance. body ~ cmn >> raw
-> the win is CMN normalization, separator adds nothing for HAR. room >> chance
-> action info leaks into the room slot; separation impure (cf. leak 0.81).""")
