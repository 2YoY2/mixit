---
name: clusterv2-purity
description: "CURRENT WORK — clusterer v2 with purity as a first-class objective (superposition bench, ownership loss), running on PA + WiMANS"
metadata: 
  node_type: memory
  type: project
  originSessionId: a9cc9e6b-eaff-4bc0-b6af-9293cd89d6f3
  modified: 2026-09-02T07:34:53.841Z
---

**THE CURRENT THREAD (started 2026-09-01 night).** Everything else in
[[pa-limb-pipeline]] is history/context; this is what is running now.

**Why**: v1's losses never charge a slot for carrying a second actor's
energy (PIT rewards only the best pair's correlation; MixIT-origin only
demands *a* split exists; the other slots are unsupervised each step).
Purity was emergent, never optimized. User's framing: the product claim
is absolute — "this slot contains ONLY this actor's information, with
calibrated confidence" — not the relative "matches actor k best".
Evidence purity is THE performance axis: slot-stream decoder hits TC 0.97
on train clips vs 0.445 heldout (decoder isn't the bottleneck, its input
is); probe 52 gets R²0.82 only with per-recording fitted unmixing.

**WORKSPACE (user ruling: sibling of cluster/, v1 untouched):**
`~/zerdani/buffer/clusterv2/{code,bench,runs,logs}` — `code/` is its own
git clone of the mixit repo; v2 code lives in repo dir `clusterv2/`.
v1 clusterer + runs stay in `~/zerdani/buffer/cluster/`, read-only here.

**THE THREE COMPONENTS (written, pushed):**
- `clusterv2/bench/make_bench.py` — superposition bench = exact per-token
  ownership GT. PA: sum two real solo recordings' RAW complex CSI (same
  scene+node) → products → PRODUCTION tokenizer on the mix; owner from
  the two solos' CMN'd STFT energy maps (r>0.8 → own=0, r<0.2 → own=1,
  else −1 = collision, excluded). WiMANS: same via empty-print pipeline
  (CAVEAT: product-domain sum is first-order, cross terms are the residue).
- `clusterv2/train/train_clusterv2.py` — v1 base losses (PIT-envelope PA /
  PIT-activity WiMANS) + **MARGIN loss** (exclusivity trained not just
  measured) + **OWNERSHIP loss = the purity objective** (slot origin by
  energy-majority (detached) → per-token BCE for sitting in a wrong-origin
  slot + slot-origin entropy weighted by slot energy). Env: DATASET,
  M, HOURS, OWNW/PURW/MARGW.
- `clusterv2/eval/eval_purity.py` — referee: v1 vs v2 on the same bench
  val, SAME M → purity, slot-SIR (dB), token-ownership acc, excluded
  fraction.

**RUNNING (launched 09-01 ~23:50, chain `clusterv2/logs/launch_v2_chain.sh`):**
benches (DONE: pa_train 2966 / pa_val 393) → **PA v2 6h** (running,
`runs/clusterv2.pa.r1`, log_train_pa.txt) → **WiMANS v2 4h** → purity
evals both datasets → marker `logs/v2chain.marker`.
PA at 14k steps: matched 0.619-0.660, wrong −0.104..−0.113 (gap 0.764 >
v1's 0.723 — margin loss working), PURITY 0.992.
**WATCH: PURITY 0.99 may mean the bench is TOO EASY (high excluded
fraction / clearly-separated solos). The v1-vs-v2 eval is the test: if v1
also ~0.99 there is no headroom → make the bench harder (tighter own
thresholds, energy-matched pairs, more overlap) before believing gains.**

**BUG FIXED at handoff**: WiMANS bench had NO val split (only 1782 pairs
exist, builder reserved val at index 3000+). Fixed by clamping
ntr=min(NPAIR, len(pairs)−NVAL) + a train/val contamination guard in
load_bench (drops stale files not in the current train meta or whose
(a,b) pair is in val meta). WiMANS val rebuild launched
(log_bench_wimans_fix.txt, marker benchfix.marker) — must finish before
the chain reaches WiMANS training (~6h away).

**SOFT-PURITY (user's addition to eval_purity.py):** bench now also saves
a per-token `ratio` (continuous eA/(eA+eB)); eval computes SOFT-purity
over ALL tokens instead of only the hard-thresholded ones — the honest
answer to "is the bench too easy" (hard purity ignores collisions by
construction). **Benches built before 09-02 have no `ratio` field, so
SOFT-purity is silently absent until a bench rebuild.**

**ACCEPTANCE (in clusterv2/README.md):** (1) bench purity up vs v1 at the
SAME M — purity inflates with slot count, classic caveat; (2) v1 gates
must not regress (PA limb gate 71%/+0.679; WiMANS matched
0.89/0.79/0.70/0.68/0.63); (3) downstream slot-stream pose TC holds or
rises.

**QUEUED NEXT (agreed, not started):** ankle GT (limbenv2: replace hips
with ankles in `phase2/limbgt_tokens.py` JB=[7,4,12,9,0] → feet are
currently an UNSUPERVISED actor = the biggest known impurity hole) +
clusterer retrain; purity HEAD (calibrated per-slot purity at inference,
pattern = presence_head.py); M sweep 8→12/16 with usage reg; recursive
product architecture (room→persons→limbs, presence-gated recursion,
confidence propagation) — design note NOT yet written;
`notes/tokenizer2_design.md` = PARKED (learned atom emitter + cross-rate
distillation, do not start).

**LITERATURE (checked)**: purity is a standard clustering metric; SIR
(BSS-Eval) is its energy-domain twin in audio separation; slot-attention
/ object-centric work evaluates slots with FG-ARI/mBO/Purity. Ours to
claim: the physical superposition bench in RF, purity as a training
objective for identity-free clustering on real mixtures, and calibrated
per-slot purity self-certification at inference. Report purity WITH
slot-SIR and slot-count context, citing that lineage.
