# mixit — Phase 2 handoff

WiFi CSI sensing on **PerceptAlign**. Written 2026-08-30 for whoever (human or
model) continues. Phase-1 code lives in `archive2/` here; phase-1 results and
checkpoints live on the server in `~/zerdani/buffer/octonet/archive2/`.

## The validated pipeline (the product of phase 1)

```
raw CSI (PA, 400 Hz, per receiver, 3 ant, 57→30 subc islands = (T,264))
  → CLEAN      per-packet AGC norm · Hampel · SFO/PDD island detrend · SNR damping
  → NORMALIZE  CMN: divide the recording's OWN static out. Room spatial pattern
               is removed from the signal (not appended — unparrotable). This is
               calibration, not cheating: single-recording, deployment-honest;
               but state honestly that STATICS are then assigned by the
               closed-form reference, not the network (b̄/still-person floor
               is the disclosed limitation).
  → STFT       0.64 s windows, ±2–150 Hz micro-Doppler of the modulation
  → recognize / separate in this basis
```

## Key results (rooms 4/5 gate = scenes 4+5, never trained on)

| system | pairSNR | leak | stress |
|---|---|---|---|
| **CMN separator** (best learned) | **35.4 dB** | 0.81 | **0.000** |
| trivial control (site template) | 20.1 | 0.09 | — |
| closed-form morph deflation | **44.7 dB** | 0.07 | — |

- CMN model = first learned system to beat a control. Checkpoint:
  `~/zerdani/buffer/octonet/archive2/pa400_cmn_runs/best.pt` (step 10k,
  cfg in ckpt: M=7, DEG=6, CMN=1, ARCH=conv; eval rebuilds from cfg).
- Composite architecture = deflation owns statics + CMN model owns dynamics.
- Limb extraction: closed on current data. Every instrument agrees
  (energy probes, micro-Doppler probes, per-action classification after
  motion-matching: wrist-vs-leg fell to ~chance 0.46/0.55; laterality with
  3-rx features 0.55 = marginal). Revival conditions: data with independent
  limb episodes, or a coherent 200 Hz aperture (XRF authors' raw — unemailed).
- Action recognition from these spectra works (0.85 held-out on a
  high-vs-low-motion pair) → Track 2 is feasible.

## Phase 2 tracks

1. **Separator v2, STFT-native**: masks over time-frequency bins of the CMN
   modulation (speech-style; time-domain masks cannot split overlapping
   oscillations). Target ≥ deflation. Gate: rooms 4/5 + controls, always.
2. **← START HERE. "Beats raw" demonstration**: action recognition, body
   channel vs raw CSI vs room channel, train scenes 1–3 / test rooms 4/5,
   clip-level. Needs NO training — the CMN checkpoint produces body channels
   today. Room-at-chance + body ≥ raw = the flagship claim.
3. **Deployment packaging**: single receiver, self-calibrating, causal
   composite; per-site unlabeled fine-tune as optional tier.

## Data & assets on the server (netsoft@rosebyte)

| path (~/zerdani/buffer/octonet/) | contents |
|---|---|
| `prep_pa_xrf400/` | PA scene-1 train streams (T,264)@400 Hz + limb GT in imu/ |
| `prep_pa_xrf400t/` | rooms 4/5 test streams + limb GT (1,200 clips) |
| `prep_pa_xrf/`, `prep_pa75/`, `prep_v75/`, `prep_xrf/` | 50/75 Hz variants, XRF islands |
| `archive2/` | all phase-1 logs, markers, run dirs incl. the CMN checkpoint |
| `archive/` | pre-fine-tune checkpoint copies (standing rule) |
| `mixit/` | this repo's clone (git is the code transport; no scp) |

Reference implementations to lift from `archive2/` here: `prep/prep_pa_xrf.py`
(pipeline front end), `eval/eval_xrf.py` (gate incl. CMN + M-slot + limb rows),
`train/train_xrf.py` (trainer: morph targets, route losses, INIT/GROUPBY/CMN
flags), `diagnostics/20–22` (STFT probes, per-action machinery).

## Landmines (each cost a run)

- GB10 unified memory: big file sweeps fill page cache → cudaMalloc fails.
  Claim the GPU **before** heavy reads; keep the retry loop (in train_xrf).
- Always `OMP_NUM_THREADS=1` for multiprocessing preps (load-110 thrash).
- Server pulls fail silently if the tree is dirty — never chmod on the
  server; exec bits belong in git.
- H5 layouts nest: `samples/<name>/{amp,pha}`, `samples/<name>/imu`.
- PA `.mat` = HDF5, `csi/csi` (3,57,T) compound + `csi/timestamp` (µs-ish;
  rate ~860–915 Hz from timestamps; never assume fps).
- Sensor GTs are only coarsely aligned (±0.5–1 s) — per-recording lag
  correction via total-motion cross-correlation, one offset per recording.
- No signal statistics as model INPUTS (hint and self-conditioning both
  failed by parroting); dividing statistics OUT (CMN) is allowed and works.
- Time-domain softmax masks can't separate overlapping oscillations (why
  Track 1 exists). Slot-norm penalty needed for unconstrained slots (Prop 4).
- Controls (trivial + deflation) before believing any number, always.

## Workflow

SSH: `ssh -i ~/.ssh/id_ed25519 -p 22427 zerdani@csinfra.eurecom.fr` then
`ssh -i ~/y netsoft@rosebyte` (key lives on jump host; pipe scripts via
`bash -s` stdin; BatchMode; scope = ~/zerdani/buffer/octonet/ + datasets
read-only; no deletes). Long jobs: nohup + marker files + local
run_in_background pollers. Logs stay outside the repo.
