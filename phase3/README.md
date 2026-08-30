# Phase 3 — CSI actor decomposition in the ADDITIVE domain

Charter (from the parallel "actor decomposition" campaign, merged here):
split the CSI mixture into one clean CSI per ACTOR (room | body | limbs...),
exclusive and COMPLETE (streams sum back to the mixture). Completeness
requires the LINEAR domain: conjugate products cancel CFO but destroy
superposition, so phase-2's product representation cannot reconstruct.
This phase uses additivity-preserving sanitization instead:

  STO: fit + subtract the linear phase ramp across subcarriers (per snapshot)
  CFO: de-rotate each snapshot so the static component holds constant phase
  -> hardware phase gone, superposition intact, per-actor reconstruction possible

Key prior measurements motivating this (from the actor campaign):
  - CSI is ~93% static; room removed -> effective rank 2.26 (~3 actors)
  - the ant x subcarrier DELAY PROFILE separates actors (range), so the
    separator consumes full channel vectors, never single-peak parameters
  - MVDR isolation beat all arms cross-layout (131 vs 144 mean-pose)

`train_actor_mixit.py`: true MixIT (mixtures-of-mixtures, greedy 2-way
assignment + mixture-consistency) on sanitized additive CSI @400 Hz, TCN
masker with softmax slot partition (completeness by construction), plus a
small identity-free limb-envelope PIT aux (keypoint GT), gated on rooms 4/5
with the probe-27 battery. legacy/ = the retired pose-head experiments.
