#!/usr/bin/env python3
"""Manifest + census for OctoNet-upload. Read-only. Run BEFORE any prep decision.

The raw layout is messy: wifi pickles under node_*/wifi/exp-*/, imu pickles flat
under imu/, no scene field anywhere, metadata only in directory names. This
script builds one table and answers the questions the prep design depends on:

  - what exists: recordings per node / subject / activity / trial, chunks per dir
  - sessions: recording DATES per subject (the paper's 3 scenes have no filename
    field; date clusters are the best proxy) -> do subjects recur across sessions?
  - wifi<->imu pairing coverage at MAX_GAP seconds
  - PROBE>0: open that many wifi pickles per node -> data shape, native packet
    rate, timestamp monotonicity, duration (feeds the native-rate prep design)

Output: manifest.csv next to this script (override with MANIFEST_OUT) + census
to stdout.

  ROOT=~/zerdani/buffer/octonet/OctoNet-upload PROBE=6 python3 00_manifest.py
"""
import os, glob, pickle, re
import numpy as np
import pandas as pd

ROOT = os.path.expanduser(os.environ.get("ROOT", "~/zerdani/buffer/octonet/OctoNet-upload"))
OUT  = os.path.expanduser(os.environ.get("MANIFEST_OUT", os.path.join(os.path.dirname(__file__) or ".", "manifest.csv")))
PROBE = int(os.environ.get("PROBE", "6"))            # wifi pickles to open per node
MAX_GAP = float(os.environ.get("MAX_GAP", "10"))     # s, wifi<->imu stamp pairing
WPAT = re.compile(r"exp-(\d{14})_node_(\d)_modality_wifi_subject_(\d+)_activity_(.+?)_trial_(\d+)")
IPAT = re.compile(r"(\d{14})_node_\d_modality_imu")

imu_keys = np.array(sorted(int(m.group(1)) for p in glob.glob(f"{ROOT}/imu/*.pickle")
                           if (m := IPAT.match(os.path.basename(p)))))
rows = []
for f in sorted(glob.glob(f"{ROOT}/node_*/wifi/*/*.pickle")):
    d = os.path.basename(os.path.dirname(f))
    m = WPAT.match(d)
    if not m: rows.append((f, d, *[None] * 7)); continue
    stamp = int(m.group(1))
    gap = float(np.abs(imu_keys - stamp).min()) if len(imu_keys) else np.inf
    rows.append((os.path.relpath(f, ROOT), d, stamp, str(stamp)[:8], int(m.group(2)),
                 int(m.group(3)), m.group(4), int(m.group(5)), gap))
df = pd.DataFrame(rows, columns=["file", "recdir", "stamp", "date", "node",
                                 "subject", "act", "trial", "imu_gap"])
bad = df[df.stamp.isna()]
df = df.dropna(subset=["stamp"]).reset_index(drop=True)
df.to_csv(OUT, index=False)

print(f"{len(df)} wifi pickles in {df.recdir.nunique()} recording dirs | "
      f"{len(bad)} unparseable | {len(imu_keys)} imu files | -> {OUT}\n")
ch = df.groupby("recdir").size()
print(f"chunks per recording dir: min {ch.min()}  median {int(ch.median())}  max {ch.max()}")
print(f"nodes {sorted(df.node.unique())}  subjects {df.subject.nunique()}  "
      f"activities {df.act.nunique()}  trials/act median "
      f"{int(df.groupby(['subject','act']).trial.nunique().median())}")
print(f"imu pairing at {MAX_GAP:.0f}s: {(df.groupby('recdir').imu_gap.first() <= MAX_GAP).mean()*100:.1f}% of dirs\n")

print("recordings per node:")
print(df.groupby("node").recdir.nunique().to_string())
print("\nsessions (dates) and whether subjects recur across them:")
dates = sorted(df.date.unique())
print(f"  {len(dates)} distinct dates: {dates[:12]}{' ...' if len(dates) > 12 else ''}")
subj_dates = df.groupby("subject").date.nunique()
print(f"  subjects on >1 date: {(subj_dates > 1).sum()} / {len(subj_dates)}"
      f"   (max dates for one subject: {subj_dates.max()})")
pivot = df.groupby(["date"]).subject.nunique()
print("  subjects per date:"); print(pivot.to_string())

if PROBE > 0:
    print(f"\nprobe ({PROBE}/node): shape | rate Hz | dur s | monotonic-ts frac")
    rng = np.random.default_rng(0)
    for n, g in df.groupby("node"):
        picks = g.iloc[rng.choice(len(g), min(PROBE, len(g)), replace=False)]
        for r in picks.itertuples():
            try:
                d = pickle.load(open(os.path.join(ROOT, r.file), "rb"))
                x = np.asarray(d["data"])
                ts = d.get("timestamp", d.get("timestamps"))
                t = np.array([(v - ts[0]).total_seconds() for v in ts])
                mono = float((np.diff(t) > 0).mean()) if len(t) > 1 else 0.0
                dur = float(t[-1]) if len(t) else 0.0
                rate = len(t) / dur if dur > 0 else 0.0
                print(f"  node{n} s{r.subject:>2} {r.act[:18]:18s} {str(x.shape):>16s} "
                      f"{rate:7.2f} {dur:7.1f} {mono:6.3f}")
            except Exception as e:
                print(f"  node{n} s{r.subject:>2} {r.act[:18]:18s}  UNREADABLE {type(e).__name__}")
print("""
READ: 'subjects on >1 date' answers the cross-environment identity question IF
dates map to scenes (verify against the paper's office/lab/living-room count:
expect ~3 date clusters). Chunks>1 per dir means prep must concatenate or treat
chunks as separate streams -- decide before writing prep_v3. Rate far from 75
or monotonic<0.99 flags timestamp repair work.
""")
