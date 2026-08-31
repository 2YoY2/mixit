# multi-person — WiMANS campaign (separated from PA code)

Goal: person-per-slot clustering on WiMANS (0-5 users, 3 rooms, per-user
activity+location labels, empty-room statics GT).

- `tokenize_wimans.py` — raw 5300 traces -> tokens, PA clean + EMPTY-ROOM CMN
  (per-env 0-user print; standing bodies stay in dynamics), overlap grid.
  Output `~/zerdani/buffer/octonet/wimans_tokens/`.
- `train_persontok.py` — **the phase-2 clusterer recipe ported to people**:
  SetSep token clusterer (M=8, D=256, NL=6, same 7-dim feats as limbtok12)
  + co-trained activity head; PIT-ACTIVITY loss (Hungarian best slot<->user
  assignment on CE cost — identity-free, activities replace PA's limb
  envelopes as the per-person GT). Train rooms 1-2, room 3 never seen.
- `eval_persontok.py` — the gate: real multi-user samples, matched activity
  accuracy vs WRONG-PERM control (slot k tracks human k and not human j),
  top-energy-slot activity multiset vs shuffle null (probe-47/48 protocol).
- `train_wimanstok.py` — superseded first attempt (MixIT-origin on 0/1-user
  unions; flawed: 0-user origins are pure noise after empty-CMN).

Probes 44-48 (contact, scalar count, token count+location, k-means person
test, limbtok12 zero-shot) live in `../diagnostics/`.
