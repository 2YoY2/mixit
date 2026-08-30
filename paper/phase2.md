# Per-limb WiFi CSI separation without limb identity

**Phase-2 working paper — methods.** Results section intentionally left empty.

---

## 1. Problem statement

Phase 1 established a validated front end for WiFi CSI sensing on the
PerceptAlign dataset (clean → channel-mean normalisation → STFT) and a learned
body-level separator that beat its controls on unseen rooms. Phase-1 also
recorded a negative result: per-limb extraction was closed on this data, every
instrument (energy probes, micro-Doppler probes, per-action classification
after motion matching) agreeing that limb-class information was absent.

Phase 2 re-opens the per-limb question by splitting it into two questions that
the phase-1 instruments had conflated:

- **Identification** — does a room-stable signature exist that says *which*
  limb produced a given component ("this is the left wrist")?
- **Separation** — can the received signal be partitioned into limb-coherent
  components *without* naming them, resolving the correspondence only
  afterwards?

Sections 3.1–3.3 address the first question, Sections 3.4–3.7 the second. The
distinction is the one deep clustering draws in speech separation (Hershey et
al., 2016): a network trained only to group time–frequency bins by "belongs to
the same source" separates speakers it has never heard, because the grouping
criterion never requires speaker identity. The permutation is resolved after
the fact.

## 2. Data and conventions

### 2.1 Dataset

PerceptAlign: 5 scenes (rooms), 21 subjects, 17 actions used here, 3 receivers
per capture, each receiver 3 antennas × 57 subcarriers. Raw CSI is stored per
receiver as HDF5 `.mat` files with a compound `csi/csi` array of shape
(3, 57, T) and a `csi/timestamp` vector. Packet rate is nominally ~810 Hz
(Intel 5300) but varies per recording; it is always derived from timestamps,
never assumed (see §2.3). Synchronised multi-view 3D keypoints (BODY25,
reconstructed with EasyMocap) accompany each clip under `fresh3d/keypoints3d`.

**Split convention, fixed for all of Phase 2**: scenes 1–3 train, scenes 4–5
test. No model, classifier, or clustering rule in this work ever observes
scenes 4–5 during fitting; every reported test number is on rooms never
trained on.

### 2.2 Two representations

Phase 1 stored streams in an *island* representation — per antenna, products
of adjacent subcarriers, `z_{a,s} = c_{a,s} · conj(c_{a,s+1})` — chosen to match
a companion dataset's format and to cancel per-packet carrier and sampling
frequency offsets (CFO/SFO), which are common across subcarriers within one
antenna. That cancellation is exact and is what makes the representation
temporally coherent; but the same difference also cancels every *per-antenna*
constant, including the angle-of-arrival phase. The stored (T, 264) stream —
90 amplitudes plus real and imaginary parts of 87 island products — therefore
contains no quantity relating antenna *a* to antenna *b*, and its
subcarrier-axis phase ramp has additionally been flattened by the SFO/PDD
detrend. Cleaning and normalisation are not responsible for this; the island
construction is.

Phase 2 therefore introduces a second, **coherent representation** for every
experiment that needs spatial phase, formed from the same raw files by
conjugate-multiplying antennas 2 and 3 against antenna 1 at each packet:

```
y_{a,s}(t) = c_{a,s}(t) · conj(c_{1,s}(t)),   a ∈ {2,3},  s = 1…57
```

CFO and SFO are identical across antennas of one receiver (a single local
oscillator and sampling clock), so they cancel exactly here as well — but the
per-antenna offsets survive, so angle of arrival remains observable, and all
57 subcarriers are retained, so the delay axis keeps its full aperture. This
representation, (T, 2, 57) complex, is the input to Sections 3.2 onward.

### 2.3 Cleaning

Applied identically wherever the coherent representation is built:

1. **Rate recovery.** The median timestamp difference is tested against unit
   scales {1, 10⁻³, 10⁻⁶, 10⁻⁹ s}; the scale placing the rate in [100, 5000] Hz
   is adopted (fallback 810 Hz). Non-monotonic samples are dropped. Recordings
   shorter than 2 s are rejected.
2. **Hardware AGC removal.** Each packet is divided by its own measured gain,
   `g(t) = sqrt(mean_{a,s} |c_{a,s}(t)|²)` — a real positive scalar, so phase is
   untouched. This precedes the conjugate products, which would otherwise
   square any residual gain variation.
3. **CFO/SFO/CTO cancellation** via the conjugate products of §2.2.
4. **Resampling to a uniform 400 Hz grid** by binned averaging of the complex
   products; empty bins are nearest-neighbour filled, and a recording is
   rejected if more than 35 % of bins are empty.

### 2.4 Normalisation: subtract the static, divide by its magnitude

In the spectral domain the channel frequency response is a *sum* over
propagation paths, so the static room and the moving person superpose
additively:

```
H(t, e) = S(e) + D(t, e),     e = (antenna pair, subcarrier)
```

Removing the room is therefore a subtraction, not a division. Phase 1's
channel-mean normalisation (CMN) divided by the complex static — appropriate
where the downstream consumer reads magnitudes, since it also equalises gain,
but it rotates every element by `−arg S(e)`, which destroys precisely the
phase ramps that encode delay and angle. Phase 2 uses the **phase-safe**
variant everywhere the coherent representation is used:

```
S(e) = (1/T) Σ_t H(t, e)                                 (own-recording static)
g(e) = max( |S(e)|, 0.05 · median_e |S(e)| )             (floored magnitude)
D̃(t, e) = ( H(t, e) − S(e) ) / g(e)
```

Subtraction removes the room exactly and leaves `D`'s complex structure
intact; the division is by a real positive scalar per element, so it equalises
strong and weak elements (conditioning the covariance estimates of §3.2)
without rotating anything. The 5 %-of-median floor prevents dead elements from
dominating the quotient, matching phase 1's constant. As in phase 1, the
static is estimated from the recording itself — single-recording,
deployment-honest, and carrying the disclosed limitation that a person who
never moves is absorbed into the static.

Subtraction matters even though the analysis band starts at 2 Hz: the static
term is orders of magnitude stronger than `D`, and its leakage through window
sidelobes would otherwise flood the low-Doppler bins.

### 2.5 Time–frequency analysis

All spectra use 256-sample windows with 128-sample hop on the 400 Hz grid
(0.64 s windows, 0.32 s hop), retaining bins in ±2–150 Hz — the phase-1
micro-Doppler band. Sections 3.1–3.3 use a Hann window. From Section 3.4
onward we switch to **Thomson multitaper analysis with K = 4 discrete prolate
spheroidal (Slepian) sequences, NW = 2.5**. Slepian tapers maximise energy
concentration in the main lobe and are the standard remedy where the sidelobes
of a dominant component would obscure a weaker superimposed one — the
torso-versus-limb situation exactly — and, as a second benefit, the K tapers
supply K nearly independent spectral looks per window, which serve directly as
extra snapshots for the covariance estimates in §3.4.

### 2.6 Limb reference signals

PerceptAlign has no IMUs; the per-limb reference is derived from the dataset's
3D keypoints (stored in the pipeline under a legacy `imu/` directory name).
For five joints — left wrist, right wrist, left hip, right hip, head (BODY25
indices 7, 4, 12, 9, 0) — keypoints with confidence > 0.3 are converted to 3D
speeds, resampled onto the recording's CSI grid, smoothed with a 0.5 s moving
average, and normalised to unit standard deviation per limb.

Alignment follows phase 1's recipe unchanged, applied identically to all five
scenes so that every room shares one convention: keypoint frames and CSI are
assumed co-started and mapped proportionally (frame rate inferred from frame
count over clip duration), then interpolated to the CSI time base. This
matches the dataset authors' own released preprocessing, which likewise maps
each keypoint frame to a proportional CSI segment with no timestamp
synchronisation, interpolation, or offset estimation available in the release;
the 0.5 s smoothing is our tolerance for the residual error. This reference
was validated at routing grade in phase 1 (r ≈ 0.3 against a null of 0).

Where two limbs must be scored against two estimated components, the five
envelopes are first **residualised**: each limb's envelope is regressed on the
other four and the non-negative residual retained. Raw envelopes are heavily
collinear through whole-body motion, which leaves a limbs-by-components
assignment weakly identified; the residualisation is phase-1's measured fix.

## 3. Methods

### 3.1 Spatial-signature probe (identification, non-parametric)

**Question.** After normalisation removes the static room, does the *dynamic*
signature — which subcarriers and antennas light up when a given limb moves —
distinguish limbs, and is it stable across rooms?

**Features.** On the phase-1 island streams at 400 Hz: CMN, then STFT, then
the ±2–150 Hz band power of each of the 87 islands summed over windows,
log-transformed, and mean-removed per receiver. Mean removal makes the feature
a *pattern* rather than an intensity; the removed mean is retained separately
as an intensity scalar for use as a control. The three receivers' 87-vectors
are concatenated (261 dimensions per clip). Only clips complete in all three
receivers are used.

**Design.** Motion-matched action pairs, so that motion intensity cannot serve
as a shortcut: the two-hand pair, and a wrist-versus-leg pair. Rooms available
at 400 Hz at this stage: scene 1, scene 4, scene 5. Four tests:

- **(A)** cosine similarity between per-(action, room) mean signatures — a
  direct look at whether the signature tracks the limb or the room;
- **(B)** within-room classification, 70/30 split, ridge (λ = 100);
- **(C)** leave-one-room-out classification, all rotations;
- **(D)** two controls on every fit: a label-shuffle null (20 draws) and an
  intensity-only feature set, which must fail if the pattern is doing the work.

The probe is parameterised by action pair and, optionally, by a single clip
directory, allowing a fast single-clip variant.

### 3.2 Coherent super-resolution probe (identification, parametric)

**Question.** Does limb identity live in the angle/delay structure that the
island representation discards?

Built on the coherent representation with the cleaning and phase-safe
normalisation of §2.3–2.4. Per recording, the ±2–150 Hz *positive* STFT bins
(one-sided selection isolates the dynamic-times-conjugate-static term from its
mirror) are collected as snapshots. Spatial smoothing over subcarriers with
sub-aperture length L = 20 yields 38 sub-arrays per snapshot and a 40 × 40
covariance over the joint (2 antenna-pairs × 20 subcarriers) aperture. MUSIC
is applied on a grid of antenna phase φ (49 points) × subcarrier phase ψ (97
points), the steering vector being the outer product of the two axes, giving a
2-D pseudo-spectrum per recording. Two products are recorded: the log
pseudo-spectrum (3 receivers concatenated) as a signature, and the
circular-mean direction of its φ marginal as a scalar estimate of the mover's
direction, with the circular spread across clips as its repeatability. The
same A/B/C/D battery of §3.1 is applied, plus a per-(room, action, receiver)
direction table.

Note that this arrangement pools all Doppler bins into a single covariance.

### 3.3 Doppler-gated joint-space probe (identification, corrected)

The pooling in §3.2 is exactly what the multi-dimensional WiFi-sensing
literature identifies as the resolution-limiting choice: mD-Track (2019) and
Widar2.0 (2018) resolve paths that are inseparable in any single dimension by
estimating jointly in (angle, delay, Doppler) — two scatterers sharing an
angular lobe but differing in Doppler are separable in the product space and
not in the marginal. The corrected probe therefore estimates **per Doppler
bin**: for each in-band bin, the windows supply snapshots, subcarrier
smoothing supplies sub-arrays, and MUSIC with a one-dimensional signal
subspace returns that bin's angle/delay peak. Outputs are a Doppler-angle map
per receiver, and the scalar

```
Δφ = (energy-weighted circular-mean angle over 15–80 Hz)
   − (energy-weighted circular-mean angle over 2–8 Hz)
```

i.e. the fast-limb band's direction *relative to the torso band's* — a
self-referenced quantity, independent of how the array happens to be oriented
in a given room, and therefore the one candidate carrier of laterality that
could transfer across rooms. Battery as before, plus a Δφ-only classifier
(sin/cos of the three receivers' Δφ, 6 dimensions) and a per-pair breakdown.
Sampling was scaled to 60 clips per (room, action) with the estimation
parallelised over workers.

### 3.4 Tokenisation

The remaining sections stop asking which limb a component belongs to. The
signal is instead reduced to a set of **tokens**, one per retained
time–frequency bin, each carrying that bin's dominant scatterer's coordinates:

```
raw CSI
  → clean            (§2.3: AGC out, conjugate products, 400 Hz)
  → normalise        (§2.4: subtract static, divide by |static|)
  → Slepian STFT     (§2.5: K = 4 tapers, NW = 2.5)
  → per-bin MUSIC    (angle φ, delay ψ)
  → token [ window w, Doppler f, angle φ, delay ψ, log energy E ]
```

Bins below the median energy of the recording are discarded. For a retained
bin, the K taper snapshots and the 38 subcarrier sub-arrays form a 40 × 40
covariance; MUSIC with a one-dimensional signal subspace gives the peak over a
37 × 37 (φ, ψ) grid.

**Exact fast evaluation.** Rather than projecting each grid steering vector on
the 39-dimensional noise subspace, we use the completeness identity for a
one-dimensional signal subspace,

```
‖Eₙᴴ a‖² = 1 − |v₁ᴴ a|²    for unit-norm a,
```

where `v₁` is the principal eigenvector. This is algebraically identical — the
same peaks — and reduces the projection cost by the subspace-dimension factor.
Measured effect on the full tokeniser: 3.46 s → 0.15 s per file, which is what
made tokenising the entire dataset (52,199 files, 51,927 retained) practical
as a single ~30-minute pass rather than a multi-day job.

Tokens are cached one compressed file per recording (1.9 GB total) with a
manifest carrying scene, subject, action, take, receiver and the train/test
split of §2.1.

### 3.5 Feasibility of identity-free clustering, without learning

Before training anything, we test whether clustering these tokens groups them
by limb at all, using a procedure with no free parameters fitted to any label.

Per recording: k-means with K = 2 on the token set, weighted by √E, over
features (cos φ, sin φ, cos ψ, sin ψ, f/150) with 8 restarts. Each cluster's
energy is accumulated per window to give two envelopes. The reference is the
residualised keypoint envelopes (§2.6) of the two most-active limbs; a
recording enters the test only if those two limbs' envelopes are decorrelated
(|r| ≤ 0.7), which is what makes the two-component question meaningful.

Three quantities are recorded per recording:

- **matched** — the better of the two cluster↔limb assignments (mean of the
  two correlations);
- **wrong permutation** — the same pair scored with the assignment swapped;
- **null** — the matched score recomputed against the reference circularly
  shifted by half its length, scored by the identical best-permutation rule so
  that the null inherits any optimism in the procedure.

The matched-minus-wrong-permutation gap is the quantity that distinguishes
"two clusters that are different limbs" from "two copies of whole-body
motion": the latter correlates with both limbs and leaves no gap. Results are
broken down per limb pair, since a head-versus-wrist pair is an easier case
than wrist-versus-wrist.

**Ablations.** The same recordings are re-clustered with (i) Doppler alone,
isolating the contribution of the spatial token axes; and, in a separate probe,
(ii) tokens replaced by the bin's *raw* aperture vector — the dominant complex
114-dimensional (antenna-pair × subcarrier) vector obtained as the leading
right singular vector of the K × 114 taper matrix, phase-referenced to the
recording's strongest element — with sub-variants using amplitude and phase,
amplitude only, and Doppler only. This second probe removes the parametric
(MUSIC) step entirely, so that any spatial contribution is measured without
the steering model as an intermediary.

### 3.6 Learned separator over token sets

**Architecture.** A recording is a *set* of tokens. Each token is embedded from
seven features — sin φ, cos φ, sin ψ, cos ψ, f/150, w/(n_windows − 1), and the
per-recording z-scored log energy (circular quantities enter as sine/cosine
pairs so their geometry is preserved) — by a linear map to width D, followed by
NL pre-norm transformer encoder layers with 4 heads and feed-forward width 2D,
and a linear head to M slots with a softmax over slots. The softmax makes the
slots a partition of the recording's token energy. Padding is masked.

**Losses.** Both are permutation- and identity-free; this is the deliberate
inversion of the phase-1 design, which pinned slot *i* to limb *i* globally and
failed by learning motion-intensity shortcuts.

1. *Envelope PIT loss* (weak supervision). Slot energy envelopes are formed
   differentiably by scattering assignment-weighted token energies into their
   windows. Against the two most-active residualised keypoint envelopes, the
   loss is `1 − max over ordered slot pairs of the mean matched correlation`.
   The permutation is resolved per recording; no slot ever acquires a name.
2. *MixIT-style origin loss* (no supervision). Two recordings from the same
   receiver have their token sets unioned, with the energy feature re-z-scored
   over the union. The model must be able to split its M slots into two groups
   that reconstruct each recording's share of the energy: the loss is the
   minimum, over all 2^M − 2 non-trivial slot bipartitions, of the
   energy-weighted squared error against the origin indicator. This follows the
   mixture-invariant training family (MixIT, 2020; RemixIT, 2022), which learns
   separation from mixtures only — necessary here because isolated single-limb
   recordings do not exist, so permutation-invariant supervised targets are
   impossible in principle.

**Optimisation.** Adam with cosine decay, gradient-norm clipping at 5. Each
step draws a batch of recordings for the envelope loss and a batch of pairs for
the origin loss; the pairs are evaluated in a single padded forward pass. The
GPU is claimed before any bulk file reads, with a retry loop, following a
phase-1 landmine on this unified-memory machine. Checkpoints record the
configuration; training is resume-safe.

Two configurations were run: a small one (M = 6, D = 128, NL = 4, batches of 8
and 8, 20 k steps) and a scaled one (M = 8, D = 256, NL = 6, batches of 16 and
16, 60 k steps). During training, a held-out-room evaluation on 200 test
recordings is run periodically and the best checkpoint retained.

### 3.7 Gate

The gate applies the §3.5 battery to the learned model on **every** test-room
recording that has a reference and satisfies the two-decorrelated-limbs
criterion, and scores the zero-learning Doppler k-means clustering on the
*same* recordings as a paired control. Reported: matched, wrong-permutation,
gap, null, and win-rate against the null for both arms; the fraction of
recordings on which the model beats the control; and the per-limb-pair
breakdown. The controls-before-conclusions rule from phase 1 is unchanged — a
learned number is only credited if it beats the trivial procedure on the same
data.

### 3.8 Transfer probe: pose estimation on frozen outputs

**Question.** Is the separator's output a *universal* representation — does a
downstream task trained on its outputs in one set of rooms transfer to an
unseen room?

The separator is frozen. Per window, its outputs are summarised as, for each
slot, the total assigned energy plus the energy in three Doppler bands (2–10,
10–40, 40–150 Hz); the three receivers are concatenated. A control arm replaces
this with band energies computed *without* the separator (eight bands plus the
total, per receiver). Both are log-transformed and z-scored per recording, so
the arms differ only in whether the separator mediates the features.

The task is the one the dataset's own paper addresses: 3D pose. Targets are
BODY25 joints 0–14 expressed **root-relative** (referred to the mid-hip joint)
— absolute coordinates differ per room by construction, so pose *shape* is the
transferable quantity — interpolated to window centres, with per-joint gaps
linearly filled where at least half the frames are valid.

The head is deliberately small: a 2-layer GRU (width 256) and a linear output,
trained with masked L1 (root joint excluded), identical architecture and budget
in both arms. The metric is MPJPE in centimetres, median over clips. A third
reference is the **mean-pose baseline** — the training-set average skeleton —
which is the floor any claim of transfer must clear.

Two configurations were run: training on one room, and training on the three
training rooms pooled; both tested on an unseen room, with an in-domain
held-out set reported alongside.

## 4. Results

*(Left empty — to be written.)*

## 5. Reproduction

| Stage | Script |
|---|---|
| Spatial-signature probe (§3.1) | `diagnostics/23_limb_signature.py` |
| Coherent MUSIC probe (§3.2) | `diagnostics/25_superres_signature.py` |
| Doppler-gated probe (§3.3) | `diagnostics/26_dopplergated_music.py` |
| Feasibility clustering (§3.5) | `diagnostics/27_limb_clustering.py` |
| Raw-aperture ablation (§3.5) | `diagnostics/28_limb_clustering_raw.py` |
| Tokenisation (§3.4) | `prep/tokenize_pa.py` |
| Limb reference envelopes (§2.6) | `prep/limbgt_tokens.py` |
| Separator training (§3.6) | `train/train_limbtok.py` |
| Gate (§3.7) | `eval/eval_limbtok.py` |
| Pose targets (§3.8) | `prep/posegt_tokens.py` |
| Transfer probe (§3.8) | `train/train_pose_probe.py` |

All stages are configured by environment variables and are resume-safe; the
split of §2.1 is enforced inside every script rather than by convention.

## References

Yılmaz & Rickard (2004), blind separation by time–frequency masking (DUET) ·
Hershey et al. (2016), deep clustering · Zhang et al. (2018), Widar2.0 ·
Xie et al. (2019), mD-Track · Wisdom et al. (2020), MixIT ·
Zeghidour & Grangier (2021), Wavesplit · Tzinis et al. (2022), RemixIT ·
Wang et al. (2023), TF-GridNet · Thomson (1982), multitaper spectrum estimation.
