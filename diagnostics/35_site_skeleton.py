#!/usr/bin/env python3
"""Site-skeleton static decomposition: per (scene, node) pool ALL sampled
statics' smoothed snapshots -> one FB-MUSIC -> K fixed site paths (the
skeleton). Per recording: LS amplitudes on the frozen skeleton. Then A/B/C
on the amplitude vectors (no path matching, no per-recording delay noise).

  python3 diagnostics/35_site_skeleton.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
NPER, NUSR = int(os.environ.get("NPER", "8")), int(os.environ.get("NUSR", "5"))
LA, K = int(os.environ.get("LA", "44")), int(os.environ.get("K", "6"))
NGRID = 512
EXCL = float(os.environ.get("EXCL", "0.10"))
PSI = np.linspace(-np.pi, np.pi, NGRID, endpoint=False)
AG = np.exp(1j * np.outer(PSI, np.arange(LA))) / np.sqrt(LA)

def snaps(v57):
    nsh = 57 - LA + 1
    fwd = np.stack([v57[k:k + LA] for k in range(nsh)], 0)
    return np.concatenate([fwd, np.conj(fwd[:, ::-1])], 0)

def load_static(rid):
    f = f"{TOK}/statics/{rid:06d}.npy"
    if not os.path.exists(f): return None
    v = np.load(f)
    return (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
recs = {}
for (sc, nd), g in man.groupby(["scene", "node"]):
    for u in rng.permutation(g.subject.unique())[:NUSR]:
        for r in rng.permutation(g[g.subject == u].rid.values)[:NPER]:
            y = load_static(int(r))
            if y is not None:
                recs.setdefault((sc, nd), []).append((u, y))

# --- site skeletons from POOLED covariance ---
SKEL = {}
for site, lst in recs.items():
    amps_all = {}
    for p in range(2):
        S = np.concatenate([snaps(y[p]) for _, y in lst], 0)
        R = S.conj().T @ S / len(S)
        ew, ev = np.linalg.eigh(R)
        En = ev[:, :LA - K]
        P = 1.0 / np.maximum((np.abs(AG.conj() @ En) ** 2).sum(1), 1e-12)
        order = np.argsort(-P)
        picks = []
        for j in order:
            if EXCL > 0 and abs(PSI[j]) < EXCL: continue
            if all(min(abs(j - k), NGRID - abs(j - k)) > 6 for k in picks):
                picks.append(int(j))
            if len(picks) >= K: break
        SKEL.setdefault(site, []).append(PSI[picks])

def amp_vec(y):
    """per rec: LS amplitudes on the site skeleton, both pairs, |.| stacked."""
    out = []
    for p in range(2):
        psis = SK[p]
        A = np.exp(1j * np.outer(np.arange(57), psis))
        a, *_ = np.linalg.lstsq(A, y[p], rcond=None)
        out.append(np.abs(a))
    v = np.concatenate(out)
    return v / (np.linalg.norm(v) + 1e-12)

V = {}
for site, lst in recs.items():
    global SK
    SK = SKEL[site]
    for u, y in lst:
        V.setdefault((site[0], site[1], u), []).append(amp_vec(y))

def cos(a, b): return float(a @ b)
def comp(mode):
    cs = []
    keys = list(V)
    if mode == "A":
        for k in keys:
            lst = V[k]
            cs += [cos(lst[a], lst[b]) for a in range(len(lst))
                   for b in range(a + 1, len(lst))]
    else:
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                okB = ki[0] == kj[0] and ki[1] == kj[1] and ki[2] != kj[2]
                okC = ki[0] != kj[0] and ki[1] == kj[1]
                if (mode == "B" and not okB) or (mode == "C" and not okC):
                    continue
                cs += [cos(a, b) for a in V[ki][:3] for b in V[kj][:3]]
    return np.mean(cs), len(cs)

# cross-room: amplitude vectors live on DIFFERENT skeletons -> compare via
# sorted-magnitude profile (skeleton-free summary)
a = comp("A"); b = comp("B"); c = comp("C")
print(f"{'comparison (site-skeleton amps)':36s}{'cos':>8s}{'pairs':>8s}")
print(f"{'A same room+rx+user':36s}{a[0]:8.3f}{a[1]:8d}")
print(f"{'B same room+rx, diff user':36s}{b[0]:8.3f}{b[1]:8d}")
print(f"{'C diff room (sorted-profile)':36s}{c[0]:8.3f}{c[1]:8d}")
print(f"\nstance signal (A-B): {a[0]-b[0]:+.3f}   room specificity (B-C): "
      f"{b[0]-c[0]:+.3f}")
print("""
READ: with frozen site skeletons, A-B = per-path amplitude stance signal
free of delay-estimation noise; B-C = how room-bound the amplitude patterns
are. High A-B -> stance atoms = (site path id, amplitude) are real.""")
