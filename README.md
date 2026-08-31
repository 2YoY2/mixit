# mixit — session handoff (2026-09-01)

**Read this first in any new session.** Then `memory/pa-limb-pipeline.md` (a
copy of the assistant's persistent memory, kept here so any model can cold
start), then `phase2/README.md`.

## Access / workflow
- Jump host: `SSH_AUTH_SOCK= ssh -i ~/.ssh/id_ed25519 -p 22427 zerdani@csinfra.eurecom.fr`
  then `ssh -i ~/y netsoft@rosebyte`. (vulcanix was used for one day, 08-31;
  csinfra is current.) BatchMode; git = code transport (repo clone at
  `~/zerdani/buffer/octonet/mixit`), no scp. Long jobs: `setsid nohup` + marker
  files in `archive2/` + local pollers. `OMP_NUM_THREADS=1` for multiproc preps.
  Python: `~/kimodo-env/bin/python` (h5py, torch cu130; asteroid via
  `PYTHONPATH=~/zerdani/buffer/octonet/ref_asteroid`).
- **USER RULE: the assistant never deletes anything on the server.** Propose
  `rm` commands; the user runs them.
- LANDMINES:
  - Piping a script through the double hop (`cat f | ssh 'ssh ... cat > f'`)
    writes **0 bytes**. Embed base64 in the command string instead.
  - `pkill -f <pattern>` also kills the ssh wrapper whose cmdline contains the
    pattern (exit 255). Use `[b]racket` patterns; never put script names inside
    waiting-chain command strings.
  - `CKPT` env collides: `eval_v2big.py` / `act_from_mh.py` want the *pose*
    checkpoint, but importing `train_posetok3.py` consumes `CKPT` at module
    scope as `RUNS/CKPT` (the *separator*). Current scripts pop it; new ones
    should use a distinct name (`PCKPT`, `CK`, `POSECKPT`).
  - Disk 1.9T: OctoNet + XRF deleted 09-01 → 496G free. PerceptAlign raw 867G,
    WiMANS 49G (+ `wimans.zip` still present, deletable).

## The three deliverables (keep separate — user ruling)
1. **Movement recognition** — best route is **tokens-direct**, never through
   pose: **49.3% in-domain / 22.8% scene-4** (17-class, chance 5.9) on coarse
   tokens. The fine grid does not beat it (42.2 / 19.9). Frozen classifier
   saved: `act_cls_fine2.pt` (fine grid).
2. **Pose** — root-relative 15-joint, train scenes 1-2-3, test scene 4/5. Best
   cross-room operating point = **mh3** (fine grid): scene-4 **133 mm /
   PCK@20 5.5 / PCK@50 21.3** selected; oracle 82 / 13.3 / 50.4.
3. **Separation** — phase-2 gate passed (below). Multi-person separation is the
   WiMANS thread: onboarded and tokenized, not yet modeled.

## Paper target (corrected 08-31)
Our protocol (train scenes 1-3, test scene 4/5) ⇒ compare to PerceptAlign
**Table 5**, not Table 3:
- scene 4: **222.4 mm / PCK@20 39.7 / PCK@50 64.1**
- scene 5: **317.1 / 32.2 / 56.1**

Table 3's 181.5/44.2/79.5 is leave-one-out *among scenes 1-3* — a different
experiment. Caveat both directions: their protocol is absolute-frame and
geometry-conditioned, ours root-relative and geometry-free. We already beat
their scene-4 MPJPE; the PCK gap is the open problem.

Their method (arXiv 2601.12252, read 08-31): CSI-ratio denoise → ~27-sample
chunks per 30 fps video frame → mag+phase+DFS as 3×224×224 → shared ResNet-34
→ tokens + learnable temporal embeddings + geometry spatial embedding →
6-layer transformer → per-frame MLP → **plain MSE at 30 Hz**. No velocity,
trajectory, or multi-hypothesis loss. Their cross-domain power is geometry
conditioning alone (ablation without it: 181.5 → 729.5 mm). Worth stealing:
dense supervision grid, learnable temporal embeddings. Not worth stealing: the
224×224 resize front end.

## What is PROVEN
- **Tokenizer** `phase2/tokenize_pa.py`: raw → clean (conj products) →
  subtract static / ÷|static| → Slepian STFT → per-bin MUSIC → tokens
  `[w, f, φ, ψ, logE]`. `WINF`/`HOPF` are env vars now.
  - Coarse grid (WINF=256 HOPF=128): `pa_tokens/` — 51,927 recs, manifest.csv,
    `imu/` limb GT, `pose/` root-rel 15-joint GT, `statics/`, `statics_add/`,
    `tenv/`, `static_peaks.npz`.
  - **Overlap grid (WINF=256 HOPF=32 → 12.5 Hz): `pa_tokens_fine2/`** — same
    per-window physics, 4× denser in time. **Clusterer-verified best**
    (+0.213 matched / 73% win, vs coarse +0.192 / 65%). All 5 scenes.
- **Separator gate PASSED**: `archive/limbtok12_best_step43k_gate71.pt`
  (8 slots, identity-free PIT+MixIT): rooms-4/5 matched +0.679, wrong-perm gap
  +0.723, win 71%, beats dopp-k-means on 99% paired. Emergent room slot = s3
  (probe 30); pose carriers s1/s2 (probe 31).
- **Limb IDENTIFICATION closed** (aperture physics, probes 23-26); identity-free
  clustering works (27/28). The clusterer is the **verification instrument**:
  `diagnostics/42_fine_gate.py` scores any cached tokenization. Nothing is
  trained in it (per-recording k-means), so it can never be "retrained".
- **TWO-TREATMENT DOCTRINE** (probes 33-39): dynamics = product clean +
  STFT/MUSIC tokens; statics = ADDITIVE clean (`bench/static_additive.py`) +
  ensemble axis. Person is first-class in the static (receiver-dependent
  0-99%); human-free room print exists (split-half 0.84-0.97, probe 37); DC
  identified by invariance (ψ≈0, rx+room-shared); **walls unresolvable at
  20 MHz** (ψ ≤ 0.065 < Rayleigh 0.11); never peak-parameterize statics.

## The 2026-08-31 → 09-01 campaign (this session)

### 1. v2-BIG battery → its real role, and probe 40
- Battery part 1 had crashed on the `CKPT` landmine; relaunched. Heldout
  114 / 9.6 / 33.8, scene-4 124 / 4.6 / 23.6; **pred motion 0.6 cm vs GT
  2.8 cm** — not mean-collapsed (dist-to-mean 9-11 cm, beats baseline 66-74%)
  but temporally *frozen*: the optimal-shrinkage signature. Skeleton semantics
  14.1% scene-4 vs 91.9% GT ceiling.
- `bench/act_from_pose.py` gained `INDOM` (heldout fraction of the train
  scenes) + per-scene breakdown: in-domain PRED **38.5%** vs ceiling 93.7
  (s1 39.8 / s2 38.5 / s3 36.1) ⇒ **the semantic destruction is
  cross-room-specific**, not a generic head defect.
- **Probe 40** (`diagnostics/40_static_shuffle.py`): true / room-mean /
  shuffled-in-scene / same-act-shuffled / zeros / wrong-room statics — **all
  identical to 0.1 PCK**. v2-BIG *ignores its static input entirely*. Its gain
  was capacity + schedule; "no poison at scale" means no fusion (SDROP/SWAP
  taught it to distrust the channel). v2-BIG ≡ tokens-only baseline.

### 2. Lever 2 — multi-hypothesis decoding (`phase3/train_posetok_mh.py`)
K=8 full trajectory hypotheses, clip-level relaxed winner-takes-all (losers get
`EPSW`) + selector cross-entropy. Every eval prints **selected / oracle-K /
mean-hyp** (mean-hyp reproduces the old shrunk decoder = built-in control).
- **mh1** (L1, coarse): heldout selected 114 / 13.9 / **39.7** vs mean control
  114 / 3.6 / 25.3 — PCK@20 ~4× the mean decoder at 1/11 of v2-BIG's size.
  **Oracle-K 60 / 25.4 / 67.9 in-domain, 79 / 13.5 / 51.7 cross-room at step
  10k** ⇒ **the 116 mm wall and the PCK collapse were decoding artifacts, not
  information limits.** Long training room-specializes the hypotheses (oracle
  decays 79 → 93 by 38k).
- **Selection is the sole bottleneck**, and is not recoverable from physics
  (`phase3/rerank_mh.py`): envelope correlation 9-13%, slot-resolved
  limb-envelope matching under the phase-2 gate protocol 11-12%, action-majority
  20-22%, vs the learned selector 37.9 in-domain / 18.2 cross-room (chance
  12.5). Hypotheses share timing and differ in spatial pose — envelopes are
  blind to that.
- **PCK ⊥ semantics** (`phase3/act_from_mh.py`): oracle-picked skeletons give
  17.9% cross-room / 40.4% in-domain. The oracle picks by MPJPE, which is
  stance-dominated ⇒ the deliverables must stay separate.
- **mh2** — user hypothesis: "MPJPE punishes bold motion, so freezing wins".
  Added `SOFTPCK` (soft-PCK@20/50 for both the loss and the winner choice):
  heldout **116 / 16.9 / 41.2**, oracle 62 / 31.6 / 73.1 — PCK up, but
  std-ratio *fell* 0.29 → 0.08. The threshold objective taught **smarter
  freezing** (park inside the band); semantics fell with it (9.9% x-room).
  ⇒ any per-timestep loss favours freezing while trajectory-phase information
  is this weak.
- **mh3** — fine (overlap) grid + soft-PCK: in-domain unchanged (122 / 14.5 /
  36.7, oracle 64 / 30.1 / 71.0) ⇒ **temporal supervision density is NOT the
  in-domain wall**. But scene-4 **133 / 5.5 / 21.3** (+67% PCK@20 over mh2,
  ≈ the room-map's gain without a map) and skeleton semantics **room-flat**
  (~21% in-domain and ~21% cross-room — the first model with no semantic
  room-collapse).

### 3. THE ROOM-MAP LAW (probe 41 series) — validated
User hypothesis: the operator that maps room A's static to room B's should
also map A's *dynamics* to B's ⇒ statics become free site calibration.
Protocol (user-set): **zero retraining** — frozen models, transform only the
test-room **tokens**, evaluate.

| arm | what | scene-4 (frozen mh2) |
|---|---|---|
| 41b | per-channel scalar on raw CSI + retokenize | harmful (v2-BIG 143 / 3.1 / 14.9) |
| 41c | rigid (Δφ,Δψ) from **raw** static peak | harmful (164 / 2.9 / 12.3) |
| **41d** | rigid shift, **DC removed first** | **sel 133.6 / 6.1 / 23.7; oracle 82.5 / 13.9 / 51.3** |
| 41e | matrix CORAL in the 40-dim MUSIC snapshot space | beats native, loses to 41d |
| 41g | anchor = repeated movements instead of statics | loses to statics |

- Native baselines for that column: sel 150 / 3.3 / 16.8, oracle 87 / 9.1 / 45.0.
- Why 41c failed: the dominant peak of the raw static **is** the DC (ψ≈0.08 in
  every room, room-shared) — the wrong correspondence. Removing it exposes the
  room-specific path.
- **41d improves the ORACLE row too**, which no selector artifact can explain:
  the mapped data genuinely sits closer to the model's manifold.
- **41f replication** on a scene-1-**only** model (maximally room-dependent;
  in-domain sel 100.4 / 23.2 / 49.3 — best in-domain numbers we have): all four
  test cells improve (room-2 oracle 9.9 → 13.4, room-4 14.2 → 16.8; room-4
  selector pick-acc 17.7 → 25.7%). Direction replicated **6/6, zero failures**.
  Magnitude is thin there ⇒ map and multi-room training are **complements**
  (training strips the deep room dependence; the map calibrates the phase-like
  residual — which is why mh2 gained more).
- **LAW: at 20 MHz a room's effect on the dynamic tokens ≈ one phase rotation
  per receiver, recoverable label-free from ~200 unlabeled statics after DC
  removal. Site calibration is free — 3 numbers per site, no retraining.**
  ψ has no dynamic range (Rayleigh). v2-BIG is neutral in every arm, consistent
  with probe 40.
- Scripts: `phase3/tokmap41.py` (`DCRM`, `SRC`, `NEG` placebo),
  `tokmap41_mat.py`, `tokmap41_dyn.py`, `retok_map41.py`. Token dirs
  `pa_tokens_tokmap41*`, `pa_tokens_dynmap*` are symlink overlays — heldout
  rows reproduce native exactly (pairing sanity).

### 4. Estimator campaign — MUSIC is not the problem; spatial SR is dead
Bake-off (`phase2/tokenize_pa_bench.py`, 100 recs, same grid, gate-judged):

| estimator | matched / win |
|---|---|
| MUSIC grid-locked (incumbent) | +0.213 / 73% |
| MUSIC + off-grid refine | +0.213 / 72% |
| **Bartlett + refine (control)** | **+0.208 / 68%** |
| FB-MUSIC | +0.181 / 73% |
| Matching-pursuit atoms, K≤3, off-grid, complex amplitudes (`tokenize_pa_mp.py`) | +0.180 / 72% |
| Capon / MVDR | +0.165 / 73% |

**Bartlett ties MUSIC ⇒ spatial super-resolution never contributed anything on
2 baselines.** The estimator axis is closed. (MP is banked as the only variant
that adds amplitudes / invertibility / multi-atom bins at zero gate cost —
the candidate if reconstruction is ever needed.)

User's follow-up — "then you're using the STFT Doppler wrong" — was right in
direction: SR was spent on the dead spatial axis while the *signal* axis got
Fourier-limited. But the tonal fix failed: `phase2/tokenize_pa_lines.py`
(greedy off-grid complex exponentials + plain per-line spatial readout) gates
at +0.110 / 57% (K≤4) and +0.103 / 52% (K=8). **Micro-Doppler at 0.3 s is
chirps, not tones** ⇒ the right Doppler-SR atom is a **chirplet / ridge**.
That is the open frontier.

### 5. Tokenization grid
- WINF=128 / HOPF=32 (short window): **fails** the gate (+0.134 / 57%) — the
  Fourier trade destroys Doppler resolution.
- **WINF=256 / HOPF=32 (overlap): best** (+0.213 / 73%) → `pa_tokens_fine2`.
- `phase3/posegt_fine.py` rebuilds pose GT on any grid (window-centre interp of
  the root-relative BODY25[:15] track, root = MidHip j8). `VALIDATE=1`
  reproduces the cached coarse pose files exactly (0.0 mm, corr 1.0000).

### 6. Other closures
- **Within-room consistency** (`act_from_mh.py` + `MUSC` env, scene-1-only
  model): its predictions classify at **52.3% within scene 1**, **18.8% /
  26.4% within the unseen rooms** (GT ceiling 95-99%). Cross-room output is
  **not** class-noise (3-4× chance) but ⅔ of the class information is
  destroyed; a further layer is format shift (26.4% read in-room vs 18% by a
  train-room-trained classifier). The in-room-recoverable slice is the
  few-shot adaptation budget.
- **OctoNet heartbeat: CLOSED** (`diagnostics/43_octo_heartbeat.py`). Estimation
  arm at chance (hit 20% vs null 16%), detection arm marginal on n=12 only.
  Physics: OctoNet WiFi phase is dead (amplitude-only regime) and ~0.5 mm
  cardiac motion needs phase. Their vitals modality is the Vayyar UWB radar.
  OctoNet has since been deleted.
- **Literature** (08-31): WiFi-JEPA (masked-link latent prediction, −48%
  cross-env — strongest documented rescue, needs no geometry, maps onto our 3
  receivers), RePos (root-relative factorization — we already do it), AdaPose
  (few-shot target adaptation), physics-informed disentangling.

### 7. WiMANS onboarded (multi-person thread)
49.3 G from Kaggle (`kaggle datasets download shuokanghuang/wimans`) →
`~/zerdani/buffer/wimans`. Intel 5300 `trace` structs (3rx × 3tx × 30,
~945 pkt/s, 3 s samples), 11,286 samples, 3 environments × dual band, 0-5
users, plus synchronized video.
- **Probe 44** (`diagnostics/44_wimans_contact.py`): int8 quantization is a
  non-issue (dynamics ~8000× the floor); phase spread 1.35 is **ambiguous**
  (measured on *occupied* samples — must be redone on 0-user); **empty-room
  print is deterministic** (split-half ≈ 1.000) but the occupied-ensemble proxy
  matches it only 0.13-0.59. WiMANS users stand at fixed canonical locations
  a-e, so bodies never average out of the ensemble — a boundary condition on
  the statics doctrine (PA's varied positioning is why the proxy worked there).
- **Tokenizer** `phase2/tokenize_wimans.py`: PA-style clean (AGC out, cross-RX
  conj products per TX, 400 Hz binning) + **empty-room CMN** (user spec:
  normalize by the *true* empty print so standing bodies stay in the dynamics)
  + overlap grid → `wimans_tokens/` (11,279 ok, `empty_prints.npz`, per-sample
  `staticdev`).
- **Probe 45** (`diagnostics/45_wimans_count.py`): scalar-level count
  correlation is weak — logE Spearman +0.14…+0.33, `staticdev` ≈ 0 (3 s of
  moving users), 5-feature ridge ≈ majority baseline (27-42% vs 32%). This is
  the **floor**; the token-level readout has not been run.

## Next-step queue
1. **Token atoms — chirplet / ridge trajectory atoms.** Both stuck in-domain
   numbers (PCK@20, the ~50% classification ceiling) survived grid, loss,
   capacity and estimator changes. This is the frontier. Gate every design with
   `diagnostics/42_fine_gate.py` before training anything.
2. **mh3 + room-map compound** (~15 min, frozen eval): the two cross-room
   levers have never been stacked.
3. **WiMANS**: token-level people-count readout, then identity-free separation
   on real 2-5-person mixtures (per-user activity/location labels exist).
   Redo the phase-aliveness check on 0-user samples first.
4. **Selector work** — it is the entire remaining in-domain PCK gap (oracle
   60 mm / PCK@50 68 vs selected 114 / 40). Ideas: richer selector features
   (person-residual statics as a posture prior), soft targets, `EPSW`
   annealing, room-honest checkpoint choice.
5. **Deployment write-up**: frozen model + free static calibration (3 numbers
   per site) + optional few-shot local readout (the 26%-vs-18% within-room
   budget).

## Rules (standing)
No statistic INPUTS cross-room (parroting); CMN / fixed algebra OK. Controls
before believing numbers. Same-seed runs = cloned early curves (by design).
**The clusterer judges tokenizations, not us.** Checkpoints archived to
`~/zerdani/buffer/octonet/archive/`. Phase-1 code: `archive2/`; probes 23-45:
`diagnostics/`; actor-mixit + teacher-student (written, user-vetoed): `phase3/`.

## Key artifacts on the server
```
~/zerdani/buffer/octonet/
  pa_tokens/                 coarse tokens + all GT (statics, pose, imu)
  pa_tokens_fine2/           VERIFIED overlap grid (12.5 Hz) + fine pose GT
  pa_tokens_tokmap41dc/      scene-4 tokens with the DC-removed room map
  pa_tokens_mp_test/         matching-pursuit atoms (100-rec sample)
  wimans_tokens/             WiMANS tokens, empty-room CMN
  posetok_v2big_runs/        v2-BIG (7.3M; ignores its statics — probe 40)
  posetok_mh_runs/           mh1  (L1 MH-WTA)
  posetok_mh2_runs/          mh2  (soft-PCK, coarse)
  posetok_mh3_runs/          mh3  (soft-PCK, fine grid) ← best x-room pose
  posetok_mh_s1_runs/        scene-1-only model (room-map replication)
  act_cls_fine2.pt           frozen movement classifier (fine grid)
  archive2/log_*.txt         every log referenced above
```
