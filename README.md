# mixit — session handoff (2026-09-02)

**Read this first, then `clusterv2/README.md` (current work), then
`memory/pa-limb-pipeline.md` (accumulated doctrine/history).**

## Access / workflow
- Jump host: `SSH_AUTH_SOCK= ssh -i ~/.ssh/id_ed25519 -p 22427
  zerdani@csinfra.eurecom.fr` then `ssh -i ~/y netsoft@rosebyte`.
  **csinfra goes down intermittently — fallback host, same port/keys:
  `zerdani@vulcanix.netsoft.cs.eurecom.fr`.**
- git is the transport (repo clones on the server, no scp). Long jobs:
  write a launcher `.sh` under `logs/`, run it with `setsid nohup`, touch
  a marker file at the end, poll the marker. **Do not inline long python
  heredocs through the double ssh hop** — quoting breaks silently.
- Python: `~/kimodo-env/bin/python`.
- **USER RULE: the assistant never deletes anything on the server.**
  Propose `rm`; the user runs it.
- **USER RULE: no subagents / no Workflow fan-outs. Work in-session.**

## Server layout (renamed 09-01; old paths still resolve via symlinks)
```
~/zerdani/buffer/
  cluster/            v1 workspace (was octonet/)
    tok/pa-v1  pa-v2-fine  pa-v1-absgt  pa-v1-relmove  wimans-v1  wimans-v2-rx
        GT dirs: limbenv/ (MOVING limb-speed envelopes from PA keypoints —
        NOT IMU data), pose_rel/ (fixed-pelvis), pose_abs/, pose_relmove/
    runs/clusterer/   limbtok.pa-v1.r12 (THE v1 limb clusterer),
                      persontok.wimans-v1.r1 (THE v1 person clusterer)
    runs/downstream/  pose heads (posemh.*, slotstream.*, poseprobe.*)
    code/ logs/ archive/ ref/ attic/   MANIFEST.md documents all of it
  clusterv2/          v2 workspace (CURRENT WORK) — code/ bench/ runs/ logs/
  PerceptAlign/ (867G raw)   wimans/ (65G raw)
```

## The system in one paragraph
Tokenize CSI (clean → normalize by a static reference → Slepian STFT →
per-bin readout) into a cloud of tokens `[w, f, φ, ψ, logE]`; an
identity-free set transformer assigns every token to one of M anonymous
slots (permutation-invariant losses, no slot ever named); downstream heads
consume the slots. Instantiated twice with the same recipe: **limbs**
(PerceptAlign) and **people** (WiMANS). Verification is always
matched vs **wrong-permutation** vs rolled null.

## What is PROVEN (v1)
- **Limbs**, rooms 4/5 never trained: gate matched +0.679 / wrong-perm
  −0.053 / win 71% / beats k-means control on 99% of recordings paired.
  All-limb Hungarian: matched +0.61…+0.69 at k=2..5, slot→limb argmax
  accuracy 0.985/0.933/0.847/0.823.
- **People** (WiMANS, unseen room): matched activity per slot
  0.89/0.79/0.70/0.68/0.63 for n=1..5 vs mis-assignment 0.40→0.19
  (chance 0.111). Presence head AUC **0.998**, calibrated; occupancy
  (incl. motionless) 97–100%. Per-person activity from their own slot
  **0.695**.
- **Slots explain kinematics**: CV-R² of the 5 limb envelopes from 8 slot
  curves = **0.84 / 0.81** (scene 1 / unseen scene 4) vs raw CSI
  0.75/0.80 vs cleaned Doppler bands 0.68/0.69.
- **Room-map law**: a room's effect on dynamic tokens ≈ one phase
  rotation per receiver, recoverable label-free from ~200 unlabeled
  statics after DC removal → free site calibration, zero retraining.
- **Beats-raw HAR cross-room**: frozen slot features 21.0% vs raw 16.4%
  (17 classes, chance 5.9). Movement recognition itself is best
  tokens-direct (49.3% in-domain / 22.8% cross-room).

## Pose: how the trajectory wall fell (09-01, the day's arc)
1. GT convention ladder: frame-pinned root-relative = transfers but
   **frozen** (motion ratio 0.06); room-absolute = **doesn't train**
   multi-scene (scenes' keypoint frames differ); **clip-centered**
   (subtract the clip-mean pelvis ONCE — keeps the whole body path, drops
   room position) = correct target → `tok/pa-v1-relmove`.
2. `train/train_slotstream.py` (**the architecture that works**): 24 slot
   streams (3 rx × 8 slots) × [logE, 3 bands, bearing sin/cos, purity,
   ψ sin/cos, oracle-role one-hot, rx one-hot] → shared stream GRU →
   16 queries (1 root + 15 joints) cross-attending the streams per window
   (content-based correspondence: permutation-safe) → temporal GRU →
   **factored anchor + offsets**, pelvis supervised. NB the mh trainer
   masks joint 8 — a bug on moving targets (pelvis got no gradient).
3. **Motion-weighted loss** (`MOTW`, weight = 0.2 + speed/mean-speed):
   trajectory correlation **+0.157 → +0.445** in-domain and **+0.102 →
   +0.316** cross-room at unchanged MPJPE. The wall was
   objective-alignment, not information: the loss never asked for motion.
   Runs: `slotstream.r1` (plain), `slotstream.r2-motw` (best).

## Metrics (use these; MPJPE alone misranks)
- **motion ratio (MR)** = predicted / true temporal std. **trajectory
  correlation (TC)** = normalized inner product of mean-removed
  trajectories. Per-clip MPJPE *inverts* quality: a 62 mm clip was frozen
  (MR 0.14) while a 466 mm squat rendered MR 0.76 / TC 0.89. This table
  is a paper contribution.
- **Eval hygiene (user caught three real flaws — obey these):**
  1. Qualitative GIFs only from **split-verified heldout** clips
     (`build(with_act=True)`, counts must match the training log).
  2. Caption clips with MR/TC, not MPJPE.
  3. Beware memorization gaps (slot-stream: TC 0.97 on train clips).

## CURRENT WORK — clusterer v2 with purity (see `clusterv2/README.md`)
The product claim is absolute: *this slot contains only this actor's
information, with calibrated confidence*. v1 never optimized purity.
v2 adds, on a **superposition bench** (sum two real solo recordings →
exact per-token ownership GT): an **ownership loss**, an **exclusivity
margin loss**, and purity reported every eval. Two separate models
(PA 6h, WiMANS 4h), workspace `~/zerdani/buffer/clusterv2/`, v1 untouched.
Chain: `clusterv2/logs/launch_v2_chain.sh` → benches → PA → WiMANS →
v1-vs-v2 purity table → marker `logs/v2chain.marker`.
**CONFIRMED 09-02 morning: the bench IS too easy as first built** — v1
baseline scores purity 0.993 (same as v2-in-training) because hard
ownership (>0.8 dominance) excluded **73% of token energy**; both models
were graded on the trivial remainder. Fix landed: **soft ownership
ratios** (`ratio` per token; SOFT-purity in the referee charges contested
energy). Val splits rebuilt as `{ds}_val2` (`SPLITS=val SUFF=2`) —
**rerun the referee with `SUFF=2`; the chain's built-in referee uses the
saturated val.** Also fixed: WiMANS val was empty (NPAIR=3000 > its
~1,780 total solo pairs; corrected 1450/300). Owed: train-split soft
regeneration (~1.5h with `OMP_NUM_THREADS=1` — forgetting that pin is
why the benches took all night) → r2 with full-coverage ownership loss
if the r1 referee underwhelms.
**Ops 09-02**: server↔github outage — `clusterv2/code` received the
bench/eval fixes via ssh and is DIRTY vs git; reconcile on next
successful pull. The bench doubles as a reusable asset: ~5,200 labeled
physical 2-actor mixes, incl. the first 2-person-PA corpus.

## Deliverables + honest boundaries
| deliverable | state |
|---|---|
| attribution (which limb / which person) | strong 2–3 actors, fair 4–5 |
| presence / occupancy incl. still bodies | 97–100%, calibrated |
| per-person activity | 0.695 unseen room |
| pose with real motion | MR 0.51 / TC +0.445 in-domain, TC +0.316 x-room |
| people counting | poor cross-room (~40%); occupancy transfers, count doesn't |
| persistent actor naming | impossible without an anchor (slot↔actor is a fresh permutation per recording — probe 49e) |
| absolute position | room-coded; needs geometry or site calibration |

## Queued (agreed, not started)
1. **Ankle GT** — feet are an unsupervised actor (`limbgt_tokens.py`
   JB=[7,4,12,9,0] uses hips); replace hips with ankles → `limbenv2` →
   clusterer retrain. Biggest known impurity hole; explains jumpjack/squat
   failures.
2. **Purity head** — calibrated per-slot purity at inference (pattern:
   `presence_head.py`), plus M sweep 8→12/16 with usage regularization.
3. **Recursive product architecture** — room → persons → limbs, recursion
   gated by presence/purity confidence, confidence propagated to leaves.
   Design note not yet written.
4. `notes/tokenizer2_design.md` — **PARKED** (learned atom emitter +
   cross-rate distillation). Do not start until the above concludes.
5. Paper `~/Desktop/final_version/actorsep_ieee.tex` (4 pp, compiles):
   needs the slot-stream/motion results + the MPJPE-inversion table folded
   in; prose TODOs marked in red.
