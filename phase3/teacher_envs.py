#!/usr/bin/env python3
"""Teacher slot envelopes: frozen limbtok12 over cached tokens -> per-slot
energy envelopes with FIXED slot identity. -> pa_tokens/tenv/{rid}.npy
(NSLOTS, nw) float16.
  python3 phase3/teacher_envs.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
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

import time
sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        time.sleep(60)

os.makedirs(f"{TOK}/tenv", exist_ok=True)
man = pd.read_csv(f"{TOK}/manifest.csv")
done = 0
with torch.no_grad():
    for i, r in enumerate(man.itertuples()):
        rid = int(r.rid)
        of = f"{TOK}/tenv/{rid:06d}.npy"
        if os.path.exists(of): done += 1; continue
        tf = f"{TOK}/tokens/{rid:06d}.npz"
        if not os.path.exists(tf): continue
        z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
        le = t[:, 4]
        zle = (le - le.mean()) / (le.std() + 1e-6)
        X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                  np.cos(t[:, 3]), t[:, 1] / 150.0,
                  t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
        e = (10.0 ** le).astype(np.float64)
        w = t[:, 0].astype(int)
        env = np.zeros((M, nw))
        for m in range(M):
            np.add.at(env[m], w, a[:, m] * e)
        np.save(of, env.astype(np.float16))
        done += 1
        if (i + 1) % 5000 == 0: print(f"{i+1}/{len(man)}", flush=True)
print(f"{done} teacher envelopes", flush=True)
