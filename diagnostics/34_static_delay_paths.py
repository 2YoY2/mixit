#!/usr/bin/env python3
"""Proper delay super-resolution on the STATIC (user's claim: walls/DC etc
sit at different delays even at 20 MHz). Per recording, per antenna-pair:
57-point static frequency response -> forward-backward subcarrier-smoothed
MUSIC (aperture L=44, model order D=4) -> path list (delay psi_k, complex
amp via LS fit). Verify:
  A same room+rx+user   -- repeatability
  B same room+rx, diff user -- are path DELAYS actor-invariant?
  C diff room           -- are they room-specific?
  + per-path AMPLITUDE correlation within/cross user (the stance signal)

  python3 diagnostics/34_static_delay_paths.py
"""
import os
import numpy as np
import pandas as pd

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
NPER, NUSR = int(os.environ.get("NPER", "8")), int(os.environ.get("NUSR", "5"))
LA = int(os.environ.get("LA", "44"))
MORD = int(os.environ.get("MORD", "4"))
NGRID = 512
SIG = float(os.environ.get("SIG", "0.03"))
PSI = np.linspace(-np.pi, np.pi, NGRID, endpoint=False)
AG = np.exp(1j * np.outer(PSI, np.arange(LA))) / np.sqrt(LA)   # (NGRID, LA)

def paths(v57):
    """(57,) complex -> (delays psi[D], amps complex[D]) via FB-smoothed MUSIC."""
    nsh = 57 - LA + 1
    fwd = np.stack([v57[k:k + LA] for k in range(nsh)], 0)
    bwd = np.conj(fwd[:, ::-1])
    sb = np.concatenate([fwd, bwd], 0)
    R = sb.conj().T @ sb / len(sb)
    ew, ev = np.linalg.eigh(R)
    En = ev[:, :LA - MORD]
    P = 1.0 / np.maximum((np.abs(AG.conj() @ En) ** 2).sum(1), 1e-12)
    order = np.argsort(-P)
    picks = []
    for j in order:
        if all(min(abs(j - k), NGRID - abs(j - k)) > 6 for k in picks):
            picks.append(int(j))
        if len(picks) >= MORD: break
    psis = PSI[picks]
    A = np.exp(1j * np.outer(np.arange(57), psis))            # (57, D)
    amps, *_ = np.linalg.lstsq(A, v57, rcond=None)
    return psis, amps

def soft_spec(psis, amps):
    d = np.abs(PSI[:, None] - psis[None, :])
    d = np.minimum(d, 2 * np.pi - d)
    s = (np.exp(-0.5 * (d / SIG) ** 2) * (np.abs(amps) ** 2)[None, :]).sum(1)
    return s / (np.linalg.norm(s) + 1e-12)

def rec_repr(rid):
    f = f"{TOK}/statics/{rid:06d}.npy"
    if not os.path.exists(f): return None
    v = np.load(f)
    y = (v[171:285] + 1j * v[285:]).reshape(2, 57).astype(np.complex64)
    out = []
    for p in range(2):
        ps, am = paths(y[p])
        out.append((ps, am, soft_spec(ps, am)))
    return out

def cos(a, b): return float(a @ b)

def match_amp_corr(ra, rb):
    """match paths by delay (tol), correlate matched amplitude magnitudes."""
    cs = []
    for p in range(2):
        pa_, aa, _ = ra[p]; pb_, ab, _ = rb[p]
        for i, ps in enumerate(pa_):
            d = np.abs(pb_ - ps)
            d = np.minimum(d, 2 * np.pi - d)
            j = int(np.argmin(d))
            if d[j] < 0.06:
                cs.append((np.abs(aa[i]), np.abs(ab[j])))
    if len(cs) < 3: return np.nan, 0
    a, b = np.array(cs).T
    if a.std() < 1e-9 or b.std() < 1e-9: return np.nan, len(cs)
    return float(np.corrcoef(a, b)[0, 1]), len(cs)

rng = np.random.default_rng(0)
man = pd.read_csv(f"{TOK}/manifest.csv")
S = {}
for (sc, nd), g in man.groupby(["scene", "node"]):
    for u in rng.permutation(g.subject.unique())[:NUSR]:
        for r in rng.permutation(g[g.subject == u].rid.values)[:NPER]:
            rp = rec_repr(int(r))
            if rp is not None:
                S.setdefault((sc, nd, u), []).append(rp)

def compare(sel):
    cs, ac, nm = [], [], 0
    keys = list(S)
    for i, ki in enumerate(keys):
        for kj in keys[i:]:
            if ki == kj:
                lst = S[ki]
                pr = [(lst[a], lst[b]) for a in range(len(lst))
                      for b in range(a + 1, len(lst))]
            elif sel(ki, kj):
                pr = [(a, b) for a in S[ki][:3] for b in S[kj][:3]]
            else:
                continue
            if ki == kj and sel != SAME: continue
            if ki != kj and sel == SAME: continue
            for ra, rb in pr:
                cs.append(np.mean([cos(ra[p][2], rb[p][2]) for p in range(2)]))
                c, n = match_amp_corr(ra, rb)
                if np.isfinite(c): ac.append(c)
                nm += 1
    return np.mean(cs), (np.mean(ac) if ac else np.nan), nm

SAME = "same"
def compare2(mode):
    cs, ac, nm = [], [], 0
    keys = list(S)
    if mode == "A":
        for k in keys:
            lst = S[k]
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    ra, rb = lst[a], lst[b]
                    cs.append(np.mean([cos(ra[p][2], rb[p][2]) for p in range(2)]))
                    c, n = match_amp_corr(ra, rb)
                    if np.isfinite(c): ac.append(c)
                    nm += 1
    else:
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                okB = ki[0] == kj[0] and ki[1] == kj[1] and ki[2] != kj[2]
                okC = ki[0] != kj[0] and ki[1] == kj[1]
                if (mode == "B" and not okB) or (mode == "C" and not okC):
                    continue
                for ra in S[ki][:3]:
                    for rb in S[kj][:3]:
                        cs.append(np.mean([cos(ra[p][2], rb[p][2])
                                           for p in range(2)]))
                        c, n = match_amp_corr(ra, rb)
                        if np.isfinite(c): ac.append(c)
                        nm += 1
    return np.mean(cs), (np.mean(ac) if ac else np.nan), nm

a = compare2("A"); b = compare2("B"); c = compare2("C")
print(f"{'comparison':34s}{'delay-spec':>11s}{'amp-corr':>10s}{'pairs':>7s}")
print(f"{'A same room+rx+user':34s}{a[0]:11.3f}{a[1]:10.3f}{a[2]:7d}")
print(f"{'B same room+rx, diff user':34s}{b[0]:11.3f}{b[1]:10.3f}{b[2]:7d}")
print(f"{'C diff room':34s}{c[0]:11.3f}{c[1]:10.3f}{c[2]:7d}")
print(f"\ndelay skeleton: actor-invariance (B) {b[0]:.3f} vs room-specificity"
      f" gap (B-C) {b[0]-c[0]:+.3f}")
print(f"stance in amplitudes: within-user {a[1]:.3f} vs cross-user {b[1]:.3f}"
      f" (gap {a[1]-b[1]:+.3f})")
print("""
READ: B high & C low on delay-spec -> path DELAYS are the room skeleton
(geometry, actor-invariant, room-specific) -- the user's claim confirmed
with a proper estimator. A>>B on amp-corr -> per-path amplitudes carry the
person: stance atoms = (room path id, amplitude delta).""")
