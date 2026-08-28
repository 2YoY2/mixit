#!/usr/bin/env python3
"""prep_pa_xrf: PerceptAlign -> the XRF-matched island representation, so the
XRF-pretrained model fine-tunes without any input-spec change.

Per .mat (= one receiver, 3 antennas, 57 subc, ~900 Hz):
  1. complex CSI, timestamps-derived rate (no fps assumptions)
  2. AGC: per-packet per-receiver gain norm
  3. bin-average to FS=50 Hz (matching XRF's rate exactly)
  4. subcarrier axis 57 -> 30 by linear combination (a fixed linear filter --
     preserves the additive channel model)
  5. Hampel on amplitudes; SFO/PDD island detrend; SNR damping
  6. stream = [90 amp | 87 island re | 87 island im] = (T, 264) float16,
     block-normalised exactly like prep_xrf. STATICS ARE KEPT -- division is
     the model's job.
meta: rid,file,name,node,date,scene,subject,act,trial,nsamp,rate,imu_ok=0,
split (scenes 1-3 train / 4-5 test). Group key for training: (node,date) =
same receiver + same layout.

  LIMIT=12 python3 prep/prep_pa_xrf.py    # smoke
"""
import os, glob, re
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import median_filter

ROOT  = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/PerceptAlign"))
OUT   = os.path.expanduser(os.environ.get("PREP_OUT", "~/zerdani/buffer/octonet/prep_pa_xrf"))
FS    = float(os.environ.get("FS", "50"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NPROC = int(os.environ.get("NPROC", "6"))
PAT = re.compile(r"Scene(\d+)/user(\d+)/action(\d+)/([^/]+)/csi_mat/(\d+)-r(\d)\.mat$")
src, dst = np.linspace(0, 1, 57), np.linspace(0, 1, 30)
Wm = np.zeros((30, 57), np.float32)
for i, d in enumerate(dst):
    j = min(max(np.searchsorted(src, d), 1), 56)
    w = (d - src[j - 1]) / (src[j] - src[j - 1])
    Wm[i, j - 1], Wm[i, j] = 1 - w, w

def bin_mean_c(t, y, nb):
    idx = np.minimum((t * FS).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float32)
    s = np.zeros((nb, y.shape[1]), np.complex64)
    np.add.at(s.real, idx, y.real.astype(np.float32))
    np.add.at(s.imag, idx, y.imag.astype(np.float32))
    m = s / np.maximum(cnt, 1)[:, None]
    bad = cnt == 0
    if bad.any():
        good = np.where(~bad)[0]
        if len(good) == 0: return None, 1.0
        near = good[np.searchsorted(good, np.where(bad)[0]).clip(0, len(good)-1)]
        m[bad] = m[near]
    return m, float(bad.mean())

def one(job):
    rid, f = job
    try:
        mm = PAT.search(f)
        with h5py.File(os.path.join(ROOT, f), "r") as h:
            c = h["csi/csi"][...]
            ts = h["csi/timestamp"][...].ravel().astype(np.float64)
        x = (c["real"] + 1j * c["imag"]).astype(np.complex64)   # (3,57,T)
        dt = float(np.median(np.diff(ts)))
        rate, t = None, None
        for unit in (1.0, 1e-3, 1e-6, 1e-9):
            if dt > 0 and 100 <= 1.0 / (dt * unit) <= 5000:
                rate = 1.0 / (dt * unit); t = (ts - ts[0]) * unit; break
        if rate is None:
            rate = 810.0; t = np.arange(x.shape[-1]) / rate
        keep = np.concatenate([[True], np.diff(t) > 0])
        x, t = x[..., keep], t[keep]
        dur = float(t[-1])
        if dur < 2.0: return None
        x = np.moveaxis(x, -1, 0)                               # (T,3,57)
        g = np.sqrt((np.abs(x) ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
        x = x / g                                               # AGC
        nb = int(dur * FS)
        if nb < 100: return None
        y, fill = bin_mean_c(t, x.reshape(len(x), -1), nb)
        if y is None or fill > 0.35: return None
        y = y.reshape(nb, 3, 57) @ Wm.T                         # (nb,3,30)
        amp = np.abs(y)
        a2 = amp.reshape(nb, -1)                                # Hampel
        med = median_filter(a2, size=(31, 1))
        dev = np.abs(a2 - med)
        mad = median_filter(dev, size=(31, 1)) * 1.4826 + 1e-9
        a2 = np.where(dev > 3 * mad, med, a2)
        amp = a2.reshape(nb, 3, 30)
        ph = np.angle(y)
        cc = amp * np.exp(1j * ph)
        z = cc[..., :-1] * np.conj(cc[..., 1:])
        u = z / (np.abs(z) + 1e-12)                             # SFO/PDD detrend
        z = z * np.exp(-1j * np.angle(u.mean(-1, keepdims=True)))
        st = amp.reshape(nb, -1).mean(0)
        wgt = st / (st + np.percentile(st, 25))                 # SNR damping
        aw = amp.reshape(nb, -1) * wgt[None, :]
        zw = np.sqrt(wgt.reshape(3, 30)[:, :-1] * wgt.reshape(3, 30)[:, 1:]).reshape(-1)
        zf = z.reshape(nb, -1) * zw[None, :]
        aw = aw / (np.sqrt((aw ** 2).mean()) + 1e-12)
        zs = np.sqrt((np.abs(zf) ** 2).mean()) + 1e-12
        out = np.c_[aw, zf.real / zs, zf.imag / zs].astype(np.float16)
        np.save(f"{OUT}/streams/{rid:06d}.npy", out)
        scene, user = int(mm.group(1)), int(mm.group(2))
        act, clip, take, rx = int(mm.group(3)), mm.group(4), int(mm.group(5)), int(mm.group(6))
        return (rid, f, f"s{scene}_u{user}_a{act}_{clip}_t{take}_r{rx}",
                f"r{rx}", f"s{scene}_u{user}", scene, user, act, take,
                nb, round(rate, 1), 0, "train" if scene <= 3 else "test")
    except Exception:
        if os.environ.get("DEBUG"):
            import traceback; traceback.print_exc()
        return None

def main():
    os.makedirs(f"{OUT}/streams", exist_ok=True)
    files = sorted(os.path.relpath(f, ROOT) for f in
                   glob.glob(f"{ROOT}/Scene*/user*/action*/*/csi_mat/*.mat"))
    jobs = list(enumerate(files))
    if LIMIT: jobs = jobs[:LIMIT]
    print(f"{len(jobs)} mats -> {OUT} (FS={FS})", flush=True)
    rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            if r is not None: rows.append(r)
            if (i + 1) % 2000 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows, columns=["rid", "file", "name", "node", "date",
                                     "scene", "subject", "act", "trial",
                                     "nsamp", "rate", "imu_ok", "split"]
                      ).sort_values("rid")
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print(f"\n{len(df)} kept / {len(jobs)}")
    print(df.groupby("scene").agg(recs=("rid", "size"), users=("subject", "nunique"),
                                  nsamp_med=("nsamp", "median")))

if __name__ == "__main__":
    main()
