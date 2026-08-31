---
name: roombody-project-state
description: "Current state of the WiFi CSI room/body separation project — nepo abandoned, HINT model best but asymmetric"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3dc8a204-07e6-4a40-b969-a1caba422726
  modified: 2026-08-28T14:33:33.106Z
---

WiFi CSI room/body separation project (`~/Desktop/final_version`, working tree `~/Desktop/roombody`, compute on netsoft@rosebyte).

State as of 2026-08-28 (user update, supersedes README.md's "nepo unvalidated" status):
- **nepo (generation 3, exchange game) does NOT work** — user said to forget it.
- **Best working model: generation 1/Run 7 with HINT** (per-node statistical mean template as extra input) + `res=body`.
- **The open problem is asymmetry**: HINT removed *body from room* (clean room channel, HAR from room at chance) but did NOT remove *room from body* (body channel still contaminated by room).
- Next direction being explored: fixing the objective, not tuning it. Key derivation (2026-08-28, from critical read of exchange_game.pdf): with approximate room invariance x_i = ā + δa_i + b_i, the exchange objective + private-norm penalty has exact optimum shared = ā + b̄, private = (b_i − b̄) + δa_i — i.e. **the observed asymmetry (room deviations in body channel) is the objective's optimum**, and that optimum equals the trivial group mean (stat_separation.py). Candidate fixes discussed: per-recording room-equivariance training on model-free perturbations (multiplicative drift, template-morphs; generic static clutter unsafe per Lemma 1), recursive deflation at inference, and motion-anchored signature matching to attribute person statics (uses DC, outside Lemma 1's barred class). Rejected: RemixIT-style pseudo-label bootstrap from [[radear-paper]] — user correctly objected it self-reinforces the teacher's systematic error (RadEar's gains relied on supervised pre-training we lack).

The README.md in final_version is the authoritative handoff doc but predates the nepo verdict.

**Working repo (since 2026-08-28): https://github.com/2YoY2/mixit, local clone `~/Desktop/mixit`** (gh authed as 2YoY2). Scaffolded with prep/ (00_manifest.py census, imu_stream_v2.py), diagnostics/ (08_native, 09_phase, 10_morph_pilot), losses/ (imu_loss.py), legacy/ (train_roombody_imu.py), README with the cross-prediction method spec. Pushed to main (public repo). **Workflow: git is the transport** — develop locally, push, `git pull` on the server (clone at `~/zerdani/buffer/octonet/mixit`); no scp. prep_v3 to be written against the 08/09/manifest outputs.
