#!/usr/bin/env python3
"""Per-recording predominant-limb slot sets for pose training input filter.

For every rid (scenes 1-4, coarse pa_tokens): limbtok12 slot envelopes vs
residualized GT limb envelopes; a slot is NAMED iff its best limb corr
>= CMIN and beats the runner-up by >= MARG.  Output: limbsel_slots.npz
(rids, mask) — mask bit m set iff slot m is a named limb slot.

  python3 phase3/build_limbsel.py
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
OUTF = os.path.expanduser(os.environ.get(
    "OUTF", "~/zerdani/buffer/octonet/limbsel_slots.npz"))
CMIN = float(os.environ.get("CMIN", "0.2"))
MARG = float(os.environ.get("MARG", "0.1"))
SELMODE = os.environ.get("SELMODE", "limb")
GAIN = float(os.environ.get("GAIN", "0.01"))
HOPF, WINF = 128, 256
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

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin([1, 2, 3, 4])]
    rids_out, mask_out = [], []
    t0 = time.time()
    for n_, rid in enumerate(man.rid.values):
        rid = int(rid)
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
        msk = 0
        if SELMODE == "explain":
            # greedy forward selection: slot admitted only if it adds
            # >= GAIN explained variance of the 5 raw limb envelopes
            Yt = np.stack([[gi[:, j][wi * HOPF:wi * HOPF + WINF].mean()
                            for wi in range(len(G))] for j in range(5)], 1)
            Emat = Em[:, :len(G)].T
            sst = ((Yt - Yt.mean(0)) ** 2).sum()
            chosen, r2 = [], 0.0
            for _ in range(M):
                bg, bm, br = 0.0, None, r2
                for m in range(M):
                    if m in chosen: continue
                    A2 = np.c_[Emat[:, chosen + [m]], np.ones(len(Yt))]
                    beta2, *_ = np.linalg.lstsq(A2, Yt, rcond=None)
                    r2n = 1 - ((Yt - A2 @ beta2) ** 2).sum() / max(sst, 1e-12)
                    if r2n - r2 > bg: bg, bm, br = r2n - r2, m, r2n
                if bm is None or bg < GAIN: break
                chosen.append(bm); r2 = br
            if not chosen:
                chosen = [int(np.argmax(Em.sum(1)))]
            for m in chosen: msk |= (1 << m)
        else:
            for m in range(M):
                cs = np.array([corr(Em[m], G[:, j]) for j in range(5)])
                o = np.argsort(-cs)
                if cs[o[0]] >= CMIN and cs[o[0]] - cs[o[1]] >= MARG:
                    msk |= (1 << m)
        rids_out.append(rid); mask_out.append(msk)
        if (n_ + 1) % 5000 == 0:
            print(f"  {n_+1}/{len(man)} named={len(rids_out)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    np.savez(OUTF, rids=np.array(rids_out, np.int64),
             mask=np.array(mask_out, np.int64))
    nb = np.array([bin(m).count('1') for m in mask_out])
    print(f"saved {OUTF}: {len(rids_out)} rids, mean named slots "
          f"{nb.mean():.2f}, zero-named {(nb==0).mean()*100:.0f}%",
          flush=True)

if __name__ == "__main__":
    main()
