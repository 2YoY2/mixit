# mixit — unsupervised room/body separation of WiFi CSI

Splits a receiver's CSI into `x(t) = room(t) + body(t)` with no ground truth.
Successor to the PerceptAlign-era codebase (`~/Desktop/roombody`, handoff in
`~/Desktop/final_version/README.md`). **Founding rule: the split is room vs
body, NOT static vs dynamic** — motion statistics (Doppler, IMU) may *name*
the body channel, never *define* the boundary.

## Method (current design, 2026-08-28)

Cross-recording prediction with morph tolerance, scored on the private side —
no MixIT mixtures, no pseudo-label bootstrapping, no exchange game:

```
s, p = model(x_A)                    # one recording in, no hint input
L_private = min_{m∈ℳ} SNR(p, x_A − m(x_B))      # body scored at body scale
L_shared  = min_{m∈ℳ} SNR(m(s), static(x_B))    # predict what recurs
L_route   = IMU/Doppler routing (naming only)    # losses/imu_loss.py
```

A, B are same-node recordings; ℳ is a tiny room-deformation family (complex
gain, delay ramp, smooth spectral filter). Rationale: the old objectives'
*optimum* provably leaves per-recording room deviations in the body channel;
the morph moves that content to the room side, and private-side SNR scoring
prices the error at body scale instead of 20 dB below room scale.

## Data

- **Primary: OctoNet** (WiFi + IMU + MoCap, 41 subjects, 62 activities) —
  raw at `netsoft@rosebyte:~/zerdani/buffer/octonet/OctoNet-upload/`,
  prepped v2 at `~/zerdani/buffer/octonet/mixit_data_v2/`.
- **Held-out external benchmark: PerceptAlign** (never trained on).

## Layout

| dir | contents |
|---|---|
| `prep/` | `00_manifest.py` (census — run first), `imu_stream_v2.py` (IMU envelopes, v2 grid) |
| `diagnostics/` | `08_native.py` (CSI↔IMU at native timing), `09_phase.py` (conj sanitization validity), `10_morph_pilot.py` (morph-family ceiling) |
| `losses/` | `imu_loss.py` — IMU routing losses (scale-blind, static-blind; shape/grad checked) |
| `legacy/` | `train_roombody_imu.py` — gen-1 Sem-MixIT + IMU port (superseded; sampler/model reusable) |

## Open gates (run on the server, in this order, before any training)

1. `diagnostics/09_phase.py` — do OctoNet's two antenna slices share a phase
   reference? (dyn_conj ≈ 0.90 observed vs 0.13 known-good → suspect prep.)
2. `diagnostics/08_native.py` — does CSI track the wearer's IMU at native
   packet timing? (ρ ≈ +0.05 on interpolated streams — near zero.)
3. `prep/00_manifest.py` — census; decides prep_v3 design (chunking, rates,
   sessions↔scenes, subjects-across-sessions for the identity claim).
4. `diagnostics/10_morph_pilot.py` — ceiling of the morph family; gates the
   objective itself.

`prep_v3` (native-rate binning, no complex interpolation, sanitization per the
09 verdict) gets written against the outputs of 1–3.

## Workflow

Heavy work runs on `netsoft@rosebyte` (jump: `zerdani@csinfra.eurecom.fr:22427`);
env `source ~/zerdani/phase1/.venv/bin/activate`. One shared GPU — check
`nvidia-smi` first. Scripts are developed here, shipped by scp, logs pasted
back. Archive checkpoints before fine-tuning anything.
