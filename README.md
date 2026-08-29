# mixit — Phase 2

WiFi CSI sensing on **PerceptAlign**, built on the pipeline validated in
Phase 1 (everything from that campaign — code, diagnostics, trainers — lives
in `archive2/`, results archived on the server under
`~/zerdani/buffer/octonet/archive2/`).

## The validated pipeline (Phase 2 foundation)

```
raw CSI (400 Hz, per receiver, 3 ant × 57→30 subc islands)
  → CLEAN      per-packet AGC norm · Hampel · SFO/PDD island detrend · SNR damping
  → NORMALIZE  CMN: divide the recording's own static out (room pattern
               removed from the signal — unparrotable, field-standard)
  → STFT       0.64 s windows, ±2–150 Hz micro-Doppler of the modulation
  → recognize / separate in this basis
```

Carried-over components: the CMN separator (35.4 dB on unseen rooms,
control-beating, zero room-memorization), closed-form morph deflation (44.7 dB
statics — the composite's static engine), per-action ground truth machinery,
the rooms-4/5 gate with controls.

## Phase 2 tracks

1. **Separator v2 — STFT-native**: masks over time-frequency bins of the CMN
   modulation (speech-separation style; time-domain masks cannot split
   overlapping oscillations). Target: close the gap to / pass deflation, with
   Doppler-clean body channels. Gate: rooms 4/5 vs the same controls.
2. **Demonstrated value — the "beats raw" claim**: action recognition on PA
   from the separated body channel vs raw CSI vs room channel, rooms 4/5 held
   out. The historically unwon claim, now in reach (spectral features already
   classify actions at 0.85).
3. **Deployment packaging**: single-recording, self-calibrating (CMN from own
   traffic), causal composite (deflate statics + learned dynamics).

Parked, with named revival conditions: limb extraction (needs data with
independent limb episodes or a coherent 200 Hz aperture); zero-shot beyond
CMN (simulation-scale room randomization).

## Workflow

Unchanged: develop locally, push, `git pull` on `netsoft@rosebyte`
(`~/zerdani/buffer/octonet/mixit`); data read-only; no deletes; archive
checkpoints before fine-tuning; controls before conclusions.
