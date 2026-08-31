#!/usr/bin/env python3
"""Probe 49e phase 1: do limbtok12 slots have STABLE limb identities?

Per recording: slot x limb correlation matrix (envelopes, residualized GT).
Aggregate per slot over many recordings: mean correlation profile over the
5 limbs + how often each limb is that slot's argmax.  A slot is NAMED only
if one limb predominates (argmax share >= SHARE and mean-corr margin >=
MARG).  Run separately on train scenes (1-3) and scene 4: does the map
exist, and does the SAME map hold in the unseen room?
This is the no-GT-at-test design: name slots once offline, annotate tokens
by the fixed map forever.

  N=600 python3 diagnostics/49e_slot_limb_map.py
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
N = int(os.environ.get("N", "600"))
SHARE = float(os.environ.get("SHARE", "0.4"))
MARG = float(os.environ.get("MARG", "0.05"))
HOPF, WINF = 128, 256
DEV5 = ["LW", "RW", "LP", "RP", "HD"]
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
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def profile(ids, tag):
    """-> mean corr (M,5), argmax share (M,5), n used."""
    S = np.zeros((M, 5)); C = np.zeros((M, 5)); nrec = 0
    cnt = np.zeros((M, 5))
    t0 = time.time()
    for rid in ids:
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
        Cm = np.zeros((M, 5))
        for m in range(M):
            for j in range(5):
                Cm[m, j] = corr(Em[m], G[:len(G), j])
        Cm = np.nan_to_num(Cm)
        C += Cm
        for m in range(M):
            cnt[m, int(np.argmax(Cm[m]))] += 1
        nrec += 1
    C /= max(nrec, 1)
    share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    print(f"\n=== {tag} ({nrec} recordings, "
          f"{(time.time()-t0)/60:.1f}min)", flush=True)
    print("  slot | mean corr per limb (LW RW LP RP HD) | argmax share | "
          "verdict", flush=True)
    named = {}
    for m in range(M):
        mc = " ".join(f"{v:+.2f}" for v in C[m])
        sh = " ".join(f"{v:.2f}" for v in share[m])
        j = int(np.argmax(share[m]))
        srt = np.sort(C[m])[::-1]
        ok = share[m, j] >= SHARE and (srt[0] - srt[1]) >= MARG \
            and j == int(np.argmax(C[m]))
        v = f"-> {DEV5[j]}" if ok else "(no predominant limb)"
        if ok: named[m] = j
        print(f"   s{m}  | {mc} | {sh} | {v}", flush=True)
    return named

def main():
    man = pd.read_csv(f"{TOK}/manifest.csv")
    rng = np.random.default_rng(49)
    tr_ids = rng.permutation(np.array(
        man[man.split == "train"].rid.values))[:N]
    s4_ids = rng.permutation(np.array(
        man[(man.split == "test") &
            (man.scene.astype(str).str.contains("4"))].rid.values))[:N]
    map_tr = profile(tr_ids, "TRAIN scenes 1-3")
    map_s4 = profile(s4_ids, "SCENE 4 (unseen)")
    agree = {m: (map_tr.get(m), map_s4.get(m)) for m in
             set(map_tr) | set(map_s4)}
    print("\n=== map stability train -> scene4:", flush=True)
    for m, (a, b) in sorted(agree.items()):
        an = DEV5[a] if a is not None else "-"
        bn = DEV5[b] if b is not None else "-"
        print(f"   s{m}: train {an}  scene4 {bn}  "
              f"{'AGREE' if a == b and a is not None else 'differ'}",
              flush=True)
    print("probe 49e done", flush=True)

if __name__ == "__main__":
    main()
