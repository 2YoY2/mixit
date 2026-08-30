#!/usr/bin/env python3
"""Movement-class head on the frozen separator output: 17-way scripted-action
recognition (the PA taxonomy: arm stretches, lateral raises, lunges, jump,
spins, squat, rotations -- with left/right MIRROR TWINS). Train scenes 1-3,
test scenes 4+5 (unseen rooms). Arms: model (frozen limbtok12 slot features)
vs raw Doppler bands, same head, same budget.

Pre-registered read: motion CLASS should transfer; mirror twins should
confuse (laterality identity is absent from the signal -- probes 23/25/26).
Reported: 17-class acc, mirror-merged 11-class acc, per-twin confusion.

  python3 train/train_act_probe.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
STEPS = int(os.environ.get("STEPS", "6000"))
HOURS = float(os.environ.get("HOURS", "1.0"))
B = int(os.environ.get("B", "64"))
LR = float(os.environ.get("LR", "1e-3"))
H = int(os.environ.get("H", "128"))
SEED = int(os.environ.get("SEED", "0"))
NC = 17
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}
BANDS = [(2, 10), (10, 40), (40, 150)]
RAWB = np.linspace(2, 150, 9)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

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

sep = SetSep(); sep.load_state_dict(ck["model"])
for at in range(10):
    try:
        sep = sep.to(dev).eval(); break
    except RuntimeError:
        print("gpu retry", flush=True); time.sleep(60)

def rec_feats(rid):
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
              np.cos(t[:, 3]), t[:, 1] / 150.0,
              t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X)[None].to(dev))[0].cpu().numpy()
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int); f = t[:, 1]
    mf = np.zeros((nw, M * 4), np.float64)
    for m in range(M):
        np.add.at(mf[:, m * 4], w, a[:, m] * e)
        for bi, (lo, hi) in enumerate(BANDS):
            s = (f >= lo) & (f < hi)
            np.add.at(mf[:, m * 4 + 1 + bi], w[s], a[s, m] * e[s])
    rf = np.zeros((nw, 9), np.float64)
    np.add.at(rf[:, 0], w, e)
    for bi in range(8):
        s = (f >= RAWB[bi]) & (f < RAWB[bi + 1])
        np.add.at(rf[:, 1 + bi], w[s], e[s])
    def nz(v):
        v = np.log10(v + 1e-9)
        return ((v - v.mean(0)) / (v.std(0) + 1e-6)).astype(np.float32)
    return nz(mf), nz(rf), nw

def build(scenes):
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin(scenes)].copy()
    man["ckey"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for ckey, g in man.groupby("ckey"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        fs = [rec_feats(r) for r in rids]
        if any(f is None for f in fs): continue
        nw = min(f[2] for f in fs)
        out.append((np.concatenate([f[0][:nw] for f in fs], 1),
                    np.concatenate([f[1][:nw] for f in fs], 1),
                    int(g.act.iloc[0]) - 1))
        if len(out) % 1000 == 0:
            print(f"  scenes{scenes}: {len(out)}", flush=True)
    return out

class Cls(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.gru = nn.GRU(fin, H, 2, batch_first=True)
        self.out = nn.Linear(H, NC)
    def forward(self, x, lens):
        h, _ = self.gru(x)
        idx = (lens - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
        pooled = (h * (torch.arange(h.shape[1], device=x.device)[None, :, None]
                       < lens[:, None, None]).float()).sum(1) / lens[:, None]
        return self.out(pooled)

def run_arm(name, ai, tr, ho, te):
    fin = tr[0][ai].shape[1]
    net = Cls(fin).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS / 2: break
        ix = rng.choice(len(tr), B)
        nw = max(len(tr[i][ai]) for i in ix)
        X = torch.zeros(B, nw, fin)
        L_ = torch.tensor([len(tr[i][ai]) for i in ix])
        y = torch.tensor([tr[i][2] for i in ix])
        for k, i in enumerate(ix):
            X[k, :len(tr[i][ai])] = torch.from_numpy(tr[i][ai])
        logits = net(X.to(dev), L_.to(dev))
        loss = nn.functional.cross_entropy(logits, y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  [{name} {step}] CE {loss.item():.3f}", flush=True)
    def ev(ds):
        P, Y = [], []
        with torch.no_grad():
            for it in ds:
                F = it[ai]
                lg = net(torch.from_numpy(F)[None].to(dev),
                         torch.tensor([len(F)]).to(dev))[0].cpu().numpy()
                P.append(int(lg.argmax())); Y.append(it[2])
        return np.array(P), np.array(Y)
    return net, ev(ho), ev(te)

def merge(y):
    return np.array([MIRROR.get(v + 1, v + 1) for v in y])

def report(tag, P, Y):
    acc = float((P == Y).mean())
    bal = float(np.mean([np.mean(P[Y == k] == k) for k in np.unique(Y)]))
    Pm, Ym = merge(P), merge(Y)
    accm = float((Pm == Ym).mean())
    balm = float(np.mean([np.mean(Pm[Ym == k] == k) for k in np.unique(Ym)]))
    print(f"    {tag}: 17-class acc {acc:.3f} (bal {bal:.3f}) | "
          f"mirror-merged 11-class acc {accm:.3f} (bal {balm:.3f})", flush=True)
    return acc, accm

print(f"frozen sep {CKPT} step {ck['step']} | building sets", flush=True)
tr_all = build([1, 2, 3])
te = build([4, 5])
rng = np.random.default_rng(SEED)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
yte = np.array([t[2] for t in te])
maj = max(np.bincount(yte)) / len(yte)
print(f"train {len(tr)} / ho {len(ho)} / test45 {len(te)} | "
      f"chance {1/NC:.3f} majority {maj:.3f}", flush=True)
for name, ai in (("model", 0), ("raw", 1)):
    net, (Ph, Yh), (Pt, Yt) = run_arm(name, ai, tr, ho, te)
    print(f"[{name}]", flush=True)
    report("heldout(1-3)", Ph, Yh)
    _, _ = report("test rooms 4/5", Pt, Yt)
    print("    mirror twins on rooms 4/5 (true->pred rates):", flush=True)
    for b, a in sorted(MIRROR.items()):
        ai_, bi_ = a - 1, b - 1
        ma, mb = Yt == ai_, Yt == bi_
        if not (ma.any() and mb.any()): continue
        print(f"      act{a}/act{b}: correct {np.mean(Pt[ma]==ai_):.2f}/"
              f"{np.mean(Pt[mb]==bi_):.2f}  swapped "
              f"{np.mean(Pt[ma]==bi_):.2f}/{np.mean(Pt[mb]==ai_):.2f}", flush=True)
print("""
READ: merged >> full with twins confused at ~coin-flip = motion class
transfers, laterality identity absent (as pre-registered). model >> raw on
merged = the separator output is the better universal action representation.""")
