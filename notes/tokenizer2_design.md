# Tokenizer 2.0 — learned atom emitter with cross-rate distillation
*(design note 2026-09-01; PARKED until current pose/slot work concludes)*

## Motivation
The engineered tokenizer (STFT + per-bin readout) has two measured walls:
temporal atoms are Fourier-limited (micro-Doppler at 0.3 s is chirps, not
tones — line-spectral tokenizer failed the gate; chirplet/ridge atoms are
the known frontier), and tokens are rate-locked (probe 32: cross-hardware
transfer null; XRF iceboxed because its release is 50 Hz). Replace
hand-designing atoms with letting separation pressure choose them, and
make the token space sample-rate-invariant.

## Two design rulings (from discussion, do not relitigate)
1. **Not MixIT-shaped.** A separation model with M output streams is the
   wrong architecture for a tokenizer: a full CSI recording cannot be 8
   outputs. The tokenizer must emit a CLOUD — hundreds of atoms,
   cardinality scaling with content. MixIT survives only as a *pressure*
   (origin purity, below), never as the output structure. Clustering into
   M slots stays downstream (limbtok's job, unchanged).
2. **This is NOT the vetoed RadEar bootstrap.** That teacher's only edge
   was its own earlier opinion (self-reinforcing error). Here the teacher
   sees physically more signal (400–500 Hz vs the student's 50 Hz):
   privileged-information distillation, not pseudo-label circularity.

## Architecture
Learned matching pursuit (the MP estimator we benched is the invertible,
zero-gate-cost ancestor):
- **Encoder** (teacher & student, shared architecture): raw sanitized CSI
  -> variable-size set of atoms, each with parameters
  `(t, f, chirp-slope, bearing phi, amplitude, codebook id)`.
  Shared discrete codebook (VQ) or shared continuous parameter space so
  teacher/student tokens are directly comparable.
- **Decoder**: reconstruct CSI from the atom set (analysis-by-synthesis).

## Loss stack
| loss | role |
|---|---|
| reconstruction | completeness — the cloud must explain the whole CSI |
| sparsity | atomicity — few, meaningful atoms |
| origin purity on unions | separability — mix two recordings, every atom must be attributable to ONE origin (per-token, no fixed partition count) |
| cross-rate KD | invariance — student(50 Hz input) matches teacher(400–500 Hz) tokens |

## Honesty controls (build in from day one)
- **Gate acceptance**: student tokens must pass the probe-42 clusterer
  gate (untrainable instrument; standing rule: the clusterer judges
  tokenizations). Target: at/near teacher and current +0.213 / 73%.
- **Aliasing disclosure**: >25 Hz content from a 50 Hz student is
  inference, not measurement. Report teacher/student token disagreement
  per Doppler band on held-out clips; the folded band's tokens are
  labeled inferred. (Lesson from VELW: fabricated detail must be
  measurable, not hidden.)

## Pilot plan (PA-only, one GPU slot, ~days not hours)
1. Teacher at 400 Hz on PA raw (native rate is honest there);
   train with reconstruction + sparsity + origin purity.
2. Gate the teacher's tokens (probe 42). No pass -> stop, redesign atoms.
3. Student on 8x-decimated input, KD to teacher tokens. Gate again.
4. **The prize cell**: run the student on TRUE 50 Hz XRF recordings
   (real hardware, not simulated decimation) and gate. Pass = probe 32's
   cross-hardware null is fixed; the stack becomes rate/hardware-portable.

## What success buys
- Chirplet-capable atoms without hand-design (the token-atom frontier).
- XRF unlocked at its release rate; WiMANS/PA rate differences dissolve.
- Fine-timing priors distilled into coarse observations — a
  representation-level attack on the trajectory-phase wall.

## Status
PARKED. Do not start until the current slot-stream pose line and its
follow-ups conclude. When resumed: this note is the spec; the clusterer
gate is the judge; the XRF cell is the win condition.
