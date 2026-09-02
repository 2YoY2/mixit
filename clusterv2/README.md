# clusterv2 — the clusterer with PURITY as a first-class objective

Goal: same identity-free set-transformer clusterer (v1 = limbtok/persontok),
upgraded so slots are trained, measured, and self-certified for PURITY:
"this slot contains only this actor's information." Tested on BOTH
PerceptAlign (limbs) and WiMANS (persons).

## Planned components (training code awaits green light)
- `bench/`  superposition bench: sum real solo recordings' raw CSI
            (physically valid, cross-terms included) -> exact per-token
            ownership GT. Purity ground truth + free training labels.
            Seed: legacy/superposition_seed.py (probe-50 construction).
- `train/`  v2 trainer = v1 losses (PIT-envelope / PIT-activity +
            MixIT-origin) PLUS:
            * per-token OWNERSHIP loss on bench mixes (purity supervised)
            * exclusivity-MARGIN loss (matched minus wrong-perm, trained
              not just measured)
            * M sweep (8 -> 12/16) with usage regularization
- `eval/`   purity metrics: slot purity (energy fraction of owning actor),
            slot-SIR (dB form), ARI vs ownership; reported WITH slot-count
            context (purity inflates with M). Plus the untouchable v1
            gates: probe-42 clusterer gate (PA), persontok testroom gate
            (WiMANS), presence-head calibration.
- purity HEAD: calibrated per-slot purity prediction at inference
            (pattern: legacy/presence_head.py).

## Acceptance
1. Bench purity up vs v1 at SAME M (no slot-count inflation trick).
2. v1 gates do not regress (limb gate 71%/+0.679; persontok matched
   0.89/0.79/0.70/0.68/0.63).
3. Downstream sanity: slot-stream pose TC does not drop; ideally rises
   (purer streams -> better decoding; the purity->TC probe quantifies).

## Data (existing, untouched)
- PA:      cluster/tok/pa-v1 (+limbenv GT; limbenv2/ankles = separate queued fix)
- WiMANS:  cluster/tok/wimans-v1 (+manifest activities; empty prints)
- runs ->  cluster/runs/clusterer/*.v2.*

## legacy/ (copied v1 reference, do not edit in place)
train_limbtok.py eval_limbtok.py train_persontok.py eval_persontok.py
presence_head.py gate_tokens.py superposition_seed.py

STATUS: scaffold only — actual v2 training code NOT yet written
(user green light pending).

## PRE-FLIGHT CHECK (run before trusting any WiMANS purity number)
The WiMANS bench was first built with a broken split (val reserved at pair
index 3000+, but only ~1782 pairs exist -> empty val). Fixed by clamping
`ntr = min(NPAIR, len(pairs) - NVAL)` plus a train/val contamination guard
in `load_bench` (drops files not in the current train meta, or whose (a,b)
pair appears in the val meta). The rebuild was still running at handoff and
its bookkeeping is NOT yet verified — meta row counts did not obviously
reconcile with the printed pair count. Verify:

```bash
# on rosebyte
python3 - <<'PY'
import pandas as pd, glob, os
B=os.path.expanduser("~/zerdani/buffer/clusterv2/bench")
for ds in ("pa","wimans"):
    tr=pd.read_csv(f"{B}/{ds}_train/meta.csv"); va=pd.read_csv(f"{B}/{ds}_val/meta.csv")
    tp={(r.a,r.b) for r in tr.itertuples()}; vp={(r.a,r.b) for r in va.itertuples()}
    print(ds,"train rows",len(tr),"val rows",len(va),
          "npz",len(glob.glob(f"{B}/{ds}_train/*.npz")),len(glob.glob(f"{B}/{ds}_val/*.npz")),
          "OVERLAP",len(tp&vp))
PY
```
Requirements: val rows > 0, OVERLAP == 0. (Stale npz in `*_train` beyond the
meta are expected and are filtered by the guard — do not delete them without
asking the user.)

## Purity sanity gate
PA training shows PURITY ~0.99 within 14k steps. **If the referee eval
(`eval/eval_purity.py`) shows v1 scoring ~the same, the bench is too easy
and the gain is meaningless.** Harden it before believing anything:
tighten ownership thresholds (0.8/0.2 -> 0.65/0.35), pair energy-matched
recordings, or prefer pairs with higher TF overlap. Report purity together
with slot-SIR (dB) and the excluded-energy fraction, at fixed M.
