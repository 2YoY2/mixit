# Phase 2 — per-limb clustering of WiFi CSI (the working pipeline)

Raw PerceptAlign CSI in → limb-coherent clusters out, validated on rooms the
model never saw. Identity-free by design: no slot is ever told which limb it
is; permutation is resolved per recording (deep-clustering principle).

## The pipeline

```
raw .mat (3 ant × 57 subc, ~810-915 Hz, per receiver)
  1. CLEAN       hardware-AGC removal (per-packet gain division);
                 CFO/SFO/CTO cancelled exactly by cross-antenna conjugate
                 products c_a·conj(c_1) (one LO) → (T, 2, 57) coherent rep
  2. NORMALIZE   spectral-domain rule: SUBTRACT the complex static
                 (multipath is additive in H) then divide by |static|
                 (gain is multiplicative; floor 5%·median) — phases carrying
                 delay/angle are never rotated
  3. STFT        Slepian multitaper (K=4, NW=2.5), 0.64 s / 0.32 s hop,
                 400 Hz grid, +2..150 Hz motion band
  4. TOKENIZE    per-TF-bin MUSIC over the antenna×subcarrier aperture
                 (subcarrier smoothing L=20; D=1 completeness identity —
                 exact, ~23× faster) → token [window, Doppler f, angle φ,
                 delay ψ, logE] per kept bin
  5. CLUSTER     set-transformer (5.5M params: D=256, 6 layers, M=8 slots,
                 softmax slot assignment per token), trained identity-free:
                   PIT-envelope: slot energy envelopes best-permutation
                     matched per recording to keypoint-GT limb envelopes
                   MixIT-origin: two recordings' token sets unioned; slots
                     must partition to reconstruct each origin
```

## Files (run in this order)

| file | what | typical run |
|---|---|---|
| `tokenize_pa.py` | raw .mats → tokens, all 5 scenes (scenes 1-3 train / 4-5 test) | `OMP_NUM_THREADS=1 NPROC=10 python3 tokenize_pa.py` (~30 min) |
| `limbgt_tokens.py` | keypoint GT → per-limb envelopes on the token grid (phase-1 recipe, uniform across scenes; the dataset's own preprocessing also start-syncs) | `CLIPS=0 NPROC=8 python3 limbgt_tokens.py` (~15 min) |
| `train_limbtok.py` | the clustering model | `DIM=256 LAYERS=6 M=8 BP=16 BM=16 STEPS=60000 HOURS=12 MIXIT_RUNS=<runs> python3 train_limbtok.py` (~8 h GPU) |
| `eval_limbtok.py` | rooms-4/5 gate vs doppler k-means control | `MIXIT_RUNS=<runs> python3 eval_limbtok.py` (~15 min) |

## Gate results (9,521 unseen-room recordings, two-limb criterion)

| | matched | wrong-perm | gap | null | win |
|---|---|---|---|---|---|
| model (12 h) | **+0.679** | −0.053 | **+0.723** | +0.593 | **71%** |
| doppler k-means control | +0.172 | −0.057 | +0.162 | +0.044 | 63% |

Model beats the control on **99%** of recordings (paired). Every limb pair
wins, including left-vs-right wrist (gap +0.729). Checkpoint:
`archive/limbtok12_best_step43k_gate71.pt` (server), runs `limbtok12_runs/`.

Boundary (measured five independent ways, diagnostics/23–28): single-limb
laterality identity is not recoverable at this 3-antenna aperture — grouping
works, naming which side does not; whole-body chirality (spin/lunge
direction) does resolve via multi-receiver temporal order.
