#!/usr/bin/env python3
"""Movement classifier on the CLUSTER MODEL'S OUTPUT (user's request): the
frozen limbtok12 separator (12h run, gate 71%) assigns every token to one
of M=8 identity-free slots; the classifier reads ONLY the slot-state
sequences (per-window, per-slot energy + Doppler/angle means, 3 rx
concatenated) — no raw tokens, no pose.

Reference rows: old slot-summary head 21.0% x-room; tokens-direct (which
already includes per-token slot probs as features) 49.3/22.8.

  TRSC=1,2,3 TESC=4 python3 phase3/act_from_slots.py
  INDOM=0.1 ...     in-domain heldout variant
"""
import os, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.pop("CKPT", None)
_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ptk", os.path.join(_dir, "legacy", "train_posetok3.py"))
ptk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptk)
dev = ptk.dev
TOK = ptk.TOK

RUNS = os.path.expanduser(os.environ.get(
    "SEPRUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
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
print(f"separator {RUNS}/best.pt step {ck.get('step')} M={M}", flush=True)

STEPS = int(os.environ.get("STEPS", "6000"))
TRSC = [int(v) for v in os.environ.get("TRSC", "1,2,3").split(",")]
TESC = [int(v) for v in os.environ.get("TESC", "4").split(",")]
INDOM = float(os.environ.get("INDOM", "0"))
B = int(os.environ.get("B", "64"))
NC = 17
NAMES = ["L-arm-str", "R-arm-str", "both-str", "L-lat-rai", "R-lat-rai",
         "L-fwd-lun", "R-fwd-lun", "L-sid-lun", "R-sid-lun", "jump",
         "pick-up", "cw-spin", "ccw-spin", "jumpjack", "squat",
         "L-rot", "R-rot"]
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}
FD = 3 * M * 6

def slot_state(rid, nwcap):
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = min(int(z["nw"]), nwcap)
    if nw < 4 or len(t) < 8: return None
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X7 = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
               np.cos(t[:, 3]), t[:, 1] / 150.0,
               t[:, 0] / max(int(z["nw"]) - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X7)[None].to(dev))[0].cpu().numpy()
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int)
    keep = w < nw
    a, e, w, t = a[keep], e[keep], w[keep], t[keep]
    F = np.zeros((nw, M, 6))
    for m in range(M):
        wm = a[:, m] * e
        den = np.zeros(nw); np.add.at(den, w, wm)
        np.add.at(F[:, m, 0], w, wm)
        for ci, v in enumerate((t[:, 1] / 150.0, np.sin(t[:, 2]),
                                np.cos(t[:, 2]), np.sin(t[:, 3]),
                                np.cos(t[:, 3]))):
            acc = np.zeros(nw); np.add.at(acc, w, wm * v)
            F[:, m, 1 + ci] = acc / np.maximum(den, 1e-9)
    F[:, :, 0] = np.log10(F[:, :, 0] + 1e-9)
    return F.reshape(nw, -1).astype(np.float32)

man = pd.read_csv(f"{TOK}/manifest.csv")
RID2ACT = {int(r.rid): int(r.act) for r in man.itertuples()}

def sets(scenes):
    ds = ptk.build(scenes)
    out = []
    for it in ds:
        nw, rids = it[2], it[3]
        act = RID2ACT.get(int(rids[0]))
        if act is None: continue
        Fs = [slot_state(int(r), nw) for r in rids]
        if any(f is None for f in Fs): continue
        nwm = min(len(f) for f in Fs)
        out.append((np.concatenate([f[:nwm] for f in Fs], 1), act - 1))
        if len(out) % 2000 == 0: print(f"  {len(out)}", flush=True)
    return out

class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(FD, 128, 2, batch_first=True)
        self.out = nn.Linear(128, NC)
    def forward(self, x, lens):
        h, _ = self.gru(x)
        p = (h * (torch.arange(h.shape[1], device=x.device)[None, :, None]
                  < lens[:, None, None]).float()).sum(1) / lens[:, None]
        return self.out(p)

def main():
    print(f"building slot-state sets (train {TRSC}, test {TESC}, "
          f"INDOM={INDOM})", flush=True)
    tr = sets(TRSC)
    if INDOM > 0:
        rng0 = np.random.default_rng(0)
        ixp = rng0.permutation(len(tr))
        ncut = int(len(ixp) * (1 - INDOM))
        te = [tr[i] for i in ixp[ncut:]]
        tr = [tr[i] for i in ixp[:ncut]]
    else:
        te = sets(TESC)
    print(f"train {len(tr)} test {len(te)}", flush=True)
    cls = Cls().to(dev)
    opt = torch.optim.Adam(cls.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    bycls = {}
    for i, it in enumerate(tr): bycls.setdefault(it[1], []).append(i)
    keys = sorted(bycls)
    for step in range(STEPS):
        ixb = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
               for c in rng.integers(0, len(keys), B)]
        items = [tr[i] for i in ixb]
        n = max(len(it[0]) for it in items)
        X = torch.zeros(B, n, FD)
        L_ = torch.tensor([len(it[0]) for it in items]).float()
        y = torch.tensor([it[1] for it in items])
        for k_, it in enumerate(items):
            X[k_, :len(it[0])] = torch.from_numpy(it[0])
        loss = nn.functional.cross_entropy(cls(X.to(dev), L_.to(dev)),
                                           y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"[{step}] CE {loss.item():.3f}", flush=True)
    cls.eval()
    P_, Y_ = [], []
    with torch.no_grad():
        for it in te:
            lg = cls(torch.from_numpy(it[0])[None].to(dev),
                     torch.tensor([len(it[0])]).float().to(dev))
            P_.append(int(lg.argmax())); Y_.append(it[1])
    P_, Y_ = np.array(P_), np.array(Y_)
    Pm = np.array([MIRROR.get(v + 1, v + 1) for v in P_])
    Ym = np.array([MIRROR.get(v + 1, v + 1) for v in Y_])
    print(f"[SLOT-STATE cls] test: 17-class {(P_ == Y_).mean():.3f}  "
          f"mirror-merged {np.mean(Pm == Ym):.3f}  (chance 0.059)",
          flush=True)
    print("\nconfusion (rows=true, top-3):", flush=True)
    for k_ in range(NC):
        m = Y_ == k_
        if not m.any(): continue
        cnt = np.bincount(P_[m], minlength=NC) / m.sum()
        top = np.argsort(-cnt)[:3]
        row = "  ".join(f"{NAMES[t]} {cnt[t]*100:.0f}%" for t in top
                        if cnt[t] > 0)
        print(f"  {NAMES[k_]:10s} (n={m.sum():4d}) -> {row}", flush=True)

if __name__ == "__main__":
    main()
