#!/usr/bin/env python3
"""HAR on GROUPED TOKENS (user's plan, pre-reconstruction): a small model
trained on the separator's grouped tokens must surpass other methods.

Arms (same data, same budget, same balanced sampling, rooms 4/5 never seen):
  grouped   token set transformer; per token [sin/cos phi, sin/cos psi,
            f, t, zlogE, slot posterior a_1..a_M (frozen limbtok12), rx 1-hot]
  ungrouped same model, slot posteriors zeroed -- isolates the grouping
  specCNN   classic WiFi-HAR baseline: channel-pooled Doppler spectrogram
            (rebuilt from token energies) -> small 2D CNN, 3 rx as channels
  rawGRU    Doppler band energies -> GRU (the earlier control)

  python3 train/train_har_grouped.py
"""
import os, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TOK = os.path.expanduser(os.environ.get("TOK", "~/zerdani/buffer/octonet/pa_tokens"))
RUNS = os.path.expanduser(os.environ.get("MIXIT_RUNS", "~/zerdani/buffer/octonet/limbtok12_runs"))
CKPT = os.environ.get("CKPT", "best.pt")
STEPS = int(os.environ.get("STEPS", "15000"))
HOURS = float(os.environ.get("HOURS", "2.0"))
B = int(os.environ.get("B", "48"))
LR = float(os.environ.get("LR", "5e-4"))
SEED = int(os.environ.get("SEED", "0"))
TMAX = int(os.environ.get("TMAX", "2048"))
NC, NF = 17, 95
MIRROR = {2: 1, 5: 4, 7: 6, 9: 8, 13: 12, 17: 16}
NAMES = ["L-arm-stretch", "R-arm-stretch", "both-arms-stretch",
         "L-lateral-raise", "R-lateral-raise", "L-fwd-lunge", "R-fwd-lunge",
         "L-side-lunge", "R-side-lunge", "jump", "pick-up", "cw-spin",
         "ccw-spin", "jumping-jack", "squat", "L-rotation", "R-rotation"]
ARMS = os.environ.get("ARMS", "grouped,ungrouped,specCNN,rawGRU").split(",")
RAWB = np.linspace(2, 150, 9)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

ck = torch.load(f"{RUNS}/{CKPT}", map_location="cpu", weights_only=False)
M, D, NL = ck["cfg"]["M"], ck["cfg"]["D"], ck["cfg"]["NL"]
FTOK = 7 + M + 3

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

FI = np.searchsorted(np.linspace(2, 150, NF + 1), 0)  # not used; keep simple
FGRID = np.linspace(2, 150, NF)

def rec_parts(rid, rx):
    tf = f"{TOK}/tokens/{rid:06d}.npz"
    if not os.path.exists(tf): return None
    z = np.load(tf); t = z["toks"]; nw = int(z["nw"])
    le = t[:, 4]
    zle = (le - le.mean()) / (le.std() + 1e-6)
    X7 = np.c_[np.sin(t[:, 2]), np.cos(t[:, 2]), np.sin(t[:, 3]),
               np.cos(t[:, 3]), t[:, 1] / 150.0,
               t[:, 0] / max(nw - 1, 1), zle].astype(np.float32)
    with torch.no_grad():
        a = sep(torch.from_numpy(X7)[None].to(dev))[0].cpu().numpy()
    hot = np.zeros((len(t), 3), np.float32); hot[:, rx] = 1
    tok = np.c_[X7, a.astype(np.float32), hot]
    e = (10.0 ** le).astype(np.float64)
    w = t[:, 0].astype(int)
    fi = np.clip(np.searchsorted(FGRID, t[:, 1]) - 1, 0, NF - 1)
    S = np.full((nw, NF), -9.0, np.float32)
    np.add.at(S, (w, fi), 0)  # placeholder; fill below
    G = np.zeros((nw, NF), np.float64)
    np.add.at(G, (w, fi), e)
    S = np.log10(G + 1e-9).astype(np.float32)
    S = (S - S.mean()) / (S.std() + 1e-6)
    rf = np.zeros((nw, 9), np.float64)
    np.add.at(rf[:, 0], w, e)
    for bi in range(8):
        s_ = (t[:, 1] >= RAWB[bi]) & (t[:, 1] < RAWB[bi + 1])
        np.add.at(rf[:, 1 + bi], w[s_], e[s_])
    rf = np.log10(rf + 1e-9)
    rf = ((rf - rf.mean(0)) / (rf.std(0) + 1e-6)).astype(np.float32)
    return tok, S, rf, nw

def build(scenes):
    cf = f"{TOK}/harcache_{'-'.join(map(str, scenes))}.pkl"
    if os.path.exists(cf):
        return pickle.load(open(cf, "rb"))
    man = pd.read_csv(f"{TOK}/manifest.csv")
    man = man[man.scene.isin(scenes)].copy()
    man["ckey"] = man["name"].str.replace(r"_r\d$", "", regex=True)
    out = []
    for ckey, g in man.groupby("ckey"):
        if len(g) != 3 or set(g.node) != {"r1", "r2", "r3"}: continue
        rids = [int(r) for r in g.sort_values("node").rid.values]
        ps = [rec_parts(r, i) for i, r in enumerate(rids)]
        if any(p is None for p in ps): continue
        nw = min(p[3] for p in ps)
        tok = np.concatenate([p[0] for p in ps], 0).astype(np.float16)
        if len(tok) > TMAX:
            keep = np.argsort(-tok[:, 6].astype(np.float32))[:TMAX]
            tok = tok[keep]
        spec = np.stack([p[1][:nw] for p in ps], 0).astype(np.float16)
        raw = np.concatenate([p[2][:nw] for p in ps], 1).astype(np.float16)
        out.append((tok, spec, raw, int(g.act.iloc[0]) - 1))
        if len(out) % 1000 == 0: print(f"  scenes{scenes}: {len(out)}", flush=True)
    pickle.dump(out, open(cf, "wb"), protocol=4)
    return out

class TokCls(nn.Module):
    def __init__(self, zero_slots=False):
        super().__init__()
        self.zero = zero_slots
        self.inp = nn.Linear(FTOK, 96)
        lay = nn.TransformerEncoderLayer(96, 4, 192, batch_first=True,
                                         norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(lay, 3)
        self.out = nn.Linear(96, NC)
    def forward(self, x, mask):
        if self.zero:
            x = x.clone(); x[:, :, 7:7 + M] = 0
        h = self.enc(self.inp(x), src_key_padding_mask=mask)
        h = (h * (~mask)[:, :, None]).sum(1) / (~mask).sum(1, keepdim=True)
        return self.out(h)

class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)))
        self.out = nn.Linear(64 * 32, NC)
    def forward(self, x):
        return self.out(self.net(x).flatten(1))

class RawGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(27, 128, 2, batch_first=True)
        self.out = nn.Linear(128, NC)
    def forward(self, x, lens):
        h, _ = self.gru(x)
        p = (h * (torch.arange(h.shape[1], device=x.device)[None, :, None]
                  < lens[:, None, None]).float()).sum(1) / lens[:, None]
        return self.out(p)

def batchify(items, arm):
    if arm in ("grouped", "ungrouped"):
        n = max(len(it[0]) for it in items)
        X = torch.zeros(len(items), n, FTOK)
        mask = torch.ones(len(items), n, dtype=torch.bool)
        for k, it in enumerate(items):
            X[k, :len(it[0])] = torch.from_numpy(it[0].astype(np.float32))
            mask[k, :len(it[0])] = False
        return (X.to(dev), mask.to(dev))
    if arm == "specCNN":
        n = max(it[1].shape[1] for it in items)
        X = torch.zeros(len(items), 3, n, NF)
        for k, it in enumerate(items):
            X[k, :, :it[1].shape[1]] = torch.from_numpy(it[1].astype(np.float32))
        return (X.to(dev),)
    n = max(len(it[2]) for it in items)
    X = torch.zeros(len(items), n, 27)
    L_ = torch.tensor([len(it[2]) for it in items])
    for k, it in enumerate(items):
        X[k, :len(it[2])] = torch.from_numpy(it[2].astype(np.float32))
    return (X.to(dev), L_.to(dev))

def run_arm(arm, tr, ho, te):
    net = {"grouped": lambda: TokCls(False), "ungrouped": lambda: TokCls(True),
           "specCNN": SpecCNN, "rawGRU": RawGRU}[arm]().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    bycls = {}
    for i, it in enumerate(tr): bycls.setdefault(it[3], []).append(i)
    keys = sorted(bycls)
    t0 = time.time()
    for step in range(STEPS):
        if (time.time() - t0) / 3600 > HOURS / len(ARMS): break
        ix = [bycls[keys[c]][rng.integers(len(bycls[keys[c]]))]
              for c in rng.integers(0, len(keys), B)]
        items = [tr[i] for i in ix]
        y = torch.tensor([it[3] for it in items]).to(dev)
        logits = net(*batchify(items, arm))
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  [{arm} {step}] CE {loss.item():.3f}", flush=True)
    net.eval()
    def ev(ds):
        P, Y = [], []
        with torch.no_grad():
            for i0 in range(0, len(ds), 32):
                its = ds[i0:i0 + 32]
                lg = net(*batchify(its, arm)).cpu().numpy()
                P += list(lg.argmax(1)); Y += [it[3] for it in its]
        return np.array(P), np.array(Y)
    return ev(ho), ev(te)

def merge(y):
    return np.array([MIRROR.get(v + 1, v + 1) for v in y])

def acc_line(P, Y):
    acc = float((P == Y).mean())
    bal = float(np.mean([np.mean(P[Y == k] == k) for k in np.unique(Y)]))
    Pm, Ym = merge(P), merge(Y)
    am = float((Pm == Ym).mean())
    bm = float(np.mean([np.mean(Pm[Ym == k] == k) for k in np.unique(Ym)]))
    return acc, bal, am, bm

print(f"building sets (frozen sep {CKPT} step {ck['step']})", flush=True)
tr_all = build([1, 2, 3])
te = build([4, 5])
rng = np.random.default_rng(SEED)
ix = rng.permutation(len(tr_all))
ho = [tr_all[i] for i in ix[int(len(ix) * 0.9):]]
tr = [tr_all[i] for i in ix[:int(len(ix) * 0.9)]]
print(f"train {len(tr)} / ho {len(ho)} / test45 {len(te)} | chance {1/NC:.3f}",
      flush=True)
print(f"{'arm':10s}{'ho-17':>7s}{'ho-bal':>8s}{'te-17':>7s}{'te-bal':>8s}"
      f"{'te-m11':>8s}{'te-m11b':>9s}")
for arm in ARMS:
    (Ph, Yh), (Pt, Yt) = run_arm(arm, tr, ho, te)
    ah, bh, _, _ = acc_line(Ph, Yh)
    at, bt, am, bm = acc_line(Pt, Yt)
    print(f"{arm:10s}{ah:7.3f}{bh:8.3f}{at:7.3f}{bt:8.3f}{am:8.3f}{bm:9.3f}",
          flush=True)
print("""
READ: grouped > ungrouped = the separator's grouping itself carries the win.
grouped > specCNN (the field's standard spectrogram baseline) = 'surpasses
other methods' on unseen rooms.""")
