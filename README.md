# mixit — session handoff (2026-08-31)

**Read this first in any new session.** Then `phase2/README.md` and memory.

## Access / workflow
- Jump host (CHANGED 08-31): `SSH_AUTH_SOCK= ssh -i ~/.ssh/id_ed25519 -p 22427 zerdani@vulcanix.netsoft.cs.eurecom.fr` then `ssh -i ~/y netsoft@rosebyte`. Pipe scripts via `bash -s` (heredoc file in scratchpad); BatchMode; git = code transport (repo clone at `~/zerdani/buffer/octonet/mixit`), no scp. Long jobs: nohup + marker files in `archive2/` + local background pollers. `OMP_NUM_THREADS=1` for multiproc preps; claim GPU before heavy reads (retry loop). Python: `~/kimodo-env/bin/python` (has h5py, torch cu130; asteroid via `PYTHONPATH=~/zerdani/buffer/octonet/ref_asteroid`).
- LANDMINE: `pkill -f <pattern>` also kills wrapper bash chains whose cmdline contains the pattern — use `[b]racket` patterns AND never put script names inside waiting-chain command strings.

## What is PROVEN (phase 2, done)
- Tokenizer `phase2/tokenize_pa.py`: raw→clean(conj products)→subtract-static/÷|static|→Slepian STFT→per-bin MUSIC→tokens [w,f,φ,ψ,logE]. All 5 scenes cached: `pa_tokens/` (51,927 recs, manifest.csv; imu/=limb GT; pose/=root-rel 15-joint GT scenes1-4; statics/=product statics; statics_add/=ADDITIVE statics; tenv/, static_peaks.npz).
- Separator gate PASSED: `archive/limbtok12_best_step43k_gate71.pt` (8 slots, identity-free PIT+MixIT): rooms-4/5 matched +0.679, wrong-perm gap +0.723, win 71%, beats dopp-k-means on 99% paired; LW+RW included. Emergent room slot = s3 (probe 30); pose carriers s1/s2 (probe 31).
- Limb IDENTIFICATION closed (aperture physics, probes 23-26); identity-free clustering works (27/28).

## Downstream findings (bench/)
- Pose wall: ~116mm/PCK@50 24 cross-room is INFORMATIONAL (v1=v4=v4b=7.3M=lag-fix all flat). GT-skeleton action ceiling 94%; predicted skeletons only 16-23% (head lossy); tokens direct = best HAR (49.3% in-domain / 22.8% scene4).
- Statics: raw input = poison (v2 small: heldout 95 but scene4 210; DFR: in features); ComBat/person-tokens/swap all null. **v2-BIG (7.3M, 12h, tokens+raw statics): best heldout 109mm/PCK@20 10.8/PCK@50 36.5 @step100k, scene4 STABLE ~126mm (no poison at scale!), overfit after 100k, stopped.** Ckpt `posetok_v2big_runs/best.pt`. **BATTERY DONE (logs archive2/log_battery_v2big.txt + log_battery1b.txt): full heldout 114/9.6/33.8, scene4 124mm/4.6/23.6, beats-baseline 66% paired; DIAG: dist-to-mean 9-11cm (posture committed, no collapse) but pred-std 0.6 vs GT 2.9cm (motion-poor, std-ratio 0.2), traj-corr +0.10 scene4; skeleton→action semantics 14.1%/16.8% vs 91.9% GT ceiling (WORSE than small head). Verdict: v2-BIG = best cross-room POSTURE estimator (no poison), poorest motion renderer; role = posture half of the adaptation tier; movement products come from tokens directly.**

## The TWO-TREATMENT DOCTRINE (night probes 33-39, settled)
- Dynamics: product clean (differential delay GOOD when differencing mover-vs-room) + STFT/MUSIC tokens. Unchanged.
- Statics: ADDITIVE clean (`bench/static_additive.py`, all extracted) + ENSEMBLE axis. Facts: person is first-class in the static (receiver-dependent 0-99%); human-free room print exists (split-half 0.84-0.97, probe 37); DC identified by invariance (ψ≈0, rx+room-shared); WALLS unresolvable at 20MHz (structure ψ≤0.065 < Rayleigh 0.11 — closed); NEVER peak-parameterize statics (SR-spectra info-free). Recipe: subtract DC → ensemble room print (label-free site calib) → per-recording residual = person/stance (room-coded: within-site/adaptation only; cross-room only relationally).

## Next-step queue
1. Read the battery results (marker above) → decide v2-BIG's role (adaptation-tier base?).
2. Doctrine build: wire static objects (DC-subtract, room print, person residual) into pipeline; v2-as-per-site-adaptation-tier experiments (shuffle probe, few-shot k-curve).
3. Tokenizer atoms: K-peak matching pursuit w/ complex amplitudes (reconstruction inside token domain), ridge tokens, slow-band 0.05-2Hz branch.
4. Environments lever (untouched): PA has 7 device layouts (scene3=A/B/C; repo/ clone + user geometry configs) → 5-7 real static environments; revives GRL/IRM family.
5. Paper: vs PerceptAlign Table 3 needs absolute frame + geometry (scene2 official cfg frame mismatch w/ our fresh3d labels; scenes1/4 geometry missing). User's actor-campaign folder: ~/Desktop/'New Folder 1' (MVDR isolator best; additivity doctrine origin).

## Rules (standing)
No statistic INPUTS cross-room (parroting), CMN/fixed-algebra OK. Controls before believing numbers. Same-seed runs = cloned early curves (by design). Checkpoints archived to `~/zerdani/buffer/octonet/archive/`. Phase-1: `archive2/` here; probes 23-39: `diagnostics/`; actor-mixit + teacher-student (written, user-vetoed): `phase3/`.
