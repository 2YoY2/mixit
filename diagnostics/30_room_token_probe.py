#!/usr/bin/env python3
"""Where does the trained separator PUT the room? (user's probe)
Inject synthetic tokens carrying the ROOM's signature -- the static's MUSIC
peak (phi0, psi0), low Doppler, static-grade energy -- into real recordings'
token sets, forward through the FROZEN 12h separator, and read:
  1. which slots catch the room tokens (posterior distribution)
  2. is that routing consistent across recordings / across ROOMS
  3. do injected tokens disturb the real tokens' assignments (flip rate)
Control: body-like injections (random coords, 10-40 Hz band).

  python3 diagnostics/30_room_token_probe.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
NREC = int(os.environ.get("NREC", "200"))
K = int(os.environ.get("K", "12"))
L, NPH, NPS = 20, 37, 37
PH = np.linspace(-np.pi, np.pi, NPH, endpoint=False)
PS = np.linspace(-np.pi, np.pi, NPS, endpoint=False)
A_ANT = np.stack([np.ones(NPH), np.exp(1j * PH)], 1)
A_SUB = np.exp(1j * np.outer(PS, np.arange(L)))
STEER = (A_ANT[:, None, :, None] * A_SUB[None, :, None, :]).reshape(
    NPH * NPS, 2 * L)
STEER = (STEER / np.sqrt(2 * L)).astype(np.complex64).conj()
IPH, IPS = np.unravel_index(np.arange(NPH * NPS), (NPH, NPS))
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

def static_peak(rid):
    f = f"{TOK}/statics/{rid:06d}.npy"
    if not os.path.exists(f): return None
    v = np.load(f)
    y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
    sb = np.stack([y[:, k:k + L].reshape(-1) for k in range(57 - L + 1)], 0)
    R = sb.conj().T @ sb
    ew, ev = np.linalg.eigh(R)
    P = 1.0 / np.maximum(1.0 - np.abs(STEER @ ev[:, -1]) ** 2, 1e-6)
    j = int(P.argmax())
    return PH[IPH[j]], PS[IPS[j]]

def feats7(t, nw):
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    return np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
                 np.cos(t[:, 3]), t[:, 1] / 150.0,
                 t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)

def run_split(name, scenes, rng, man):
    sub = man[man.scene.isin(scenes)]
    rids = rng.permutation(sub.rid.values)
    post_room, post_body, flips, majslot = [], [], [], {}
    scmaj = {}
    n = 0
    for rid in rids:
        rid = int(rid)
        pk = static_peak(rid)
        tf = f"{TOK}/tokens/{rid:06d}.npz"
        if pk is None or not os.path.exists(tf): continue
        z = np.load(tf); t = z["toks"].astype(np.float64); nw = int(z["nw"])
        e90 = np.percentile(t[:, 4], 90)
        wgrid = np.linspace(0, nw - 1, K).round()
        room = np.c_[wgrid, rng.uniform(2.4, 5.5, K),
                     np.full(K, pk[0]), np.full(K, pk[1]),
                     np.full(K, e90)]
        body = np.c_[wgrid, rng.uniform(10, 40, K),
                     rng.uniform(-np.pi, np.pi, K),
                     rng.uniform(-np.pi, np.pi, K),
                     np.full(K, np.median(t[:, 4]))]
        with torch.no_grad():
            a0 = sep(torch.from_numpy(feats7(t, nw))[None].to(dev))[0]
            arm = sep(torch.from_numpy(feats7(np.r_[t, room], nw)
                                       )[None].to(dev))[0]
            abd = sep(torch.from_numpy(feats7(np.r_[t, body], nw)
                                       )[None].to(dev))[0]
        pr = arm[len(t):].mean(0).cpu().numpy()
        pb = abd[len(t):].mean(0).cpu().numpy()
        post_room.append(pr); post_body.append(pb)
        flips.append(float((arm[:len(t)].argmax(1) != a0.argmax(1))
                           .float().mean()))
        ms = int(pr.argmax())
        majslot[ms] = majslot.get(ms, 0) + 1
        sc = int(man[man.rid == rid].scene.iloc[0])
        scmaj.setdefault(sc, {}).setdefault(ms, 0)
        scmaj[sc][ms] += 1
        n += 1
        if n >= NREC: break
    PR, PB = np.stack(post_room), np.stack(post_body)
    print(f"\n== {name} (n={n})")
    print("  room-token mean posterior: " +
          " ".join(f"s{m}:{PR[:, m].mean():.2f}" for m in range(M)))
    print("  body-token mean posterior: " +
          " ".join(f"s{m}:{PB[:, m].mean():.2f}" for m in range(M)))
    tot = sum(majslot.values())
    top = sorted(majslot.items(), key=lambda kv: -kv[1])
    print(f"  room-token majority slot share: "
          f"{top[0][1]/tot*100:.0f}% on slot {top[0][0]} "
          f"(chance {100/M:.0f}%)")
    for sc in sorted(scmaj):
        d = scmaj[sc]; t2 = sum(d.values())
        b = max(d.items(), key=lambda kv: kv[1])
        print(f"    scene {sc}: slot {b[0]} {b[1]/t2*100:.0f}%")
    print(f"  real-token flip rate under injection: "
          f"{np.mean(flips)*100:.2f}%")

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
run_split("scenes123", [1, 2, 3], rng, man)
run_split("rooms45", [4, 5], rng, man)
print("""
READ: one dominant slot for room tokens, SAME across scenes -> the model has
an emergent room concept with stable identity. Scattered/scene-dependent ->
no stable room slot. Low flip rate -> assignments robust to injection.""")
