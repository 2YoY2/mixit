---
name: radear-paper
description: RadEar (arXiv 2603.12446) — self-supervised RF backscatter voice separation; remix/teacher-student method relevant to roombody
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3dc8a204-07e6-4a40-b969-a1caba422726
  modified: 2026-08-28T13:13:26.381Z
---

RadEar: A Self-Supervised RF Backscatter System for Voice Eavesdropping and Separation.
arXiv 2603.12446 (submitted 2026-03-12), Qijun Wang, Peihao Yan, Chunqi Qian, Huacheng Zeng.
https://arxiv.org/abs/2603.12446

Key method (relevant to [[roombody-project-state]]): MixIT-inspired **main/target (teacher-student) model pair** on Conv-TasNet. Teacher separates two real segments into pseudo-sources; one pseudo-source from each is recombined into a remix; student must re-separate the remix; permutation-invariant SI-SDR loss against the pseudo-sources. Student updates teacher via EMA (θ ← λθ + (1−λ)θ'). Pre-training on synthetic LibriMix was the biggest ablation gain (SI-SDR 7.56→10.87 dB). This is essentially RemixIT-style bootstrapping — training on the same chimera construction the roombody CYCLE eval uses.
