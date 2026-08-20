# Phase 1 Implementation Plan: State–Nuisance Geometry

**Status:** conditional go for preflight; no-go for hypothesis-bearing GPU work until verified state/identity/localization and motion-common-support gates pass  
**Specification audited:** `docs/project_methodology.md`, version 1.1, read in full  
**Dataset/repository snapshot re-audited:** 21 August 2026  
**Scope:** Phase 1 only

## 1. Executive verdict

The narrow research question is valid and falsifiable:

> Given a fixed observation construction, object-region pooling, and cosine readout, does a frozen predictive video encoder rank a verified semantic state change above a verified state-preserving nuisance change for the same physical object?

The downloaded data cannot yet answer that question automatically. The current snapshot contains synthetic EgoInteract videos, task annotations, and aligned COCO annotations for separately released static JPGs. It does **not** contain semantic before/after object-state labels. The COCO regions are aligned to 1280×720 JPGs, not to the downloaded 1408×1408 videos. TAS labels `action_0`, `action_1`, and `action_2` are action-segmentation labels, not object states. `solo_instance_id` is not a globally persistent physical-object identifier.

Accordingly:

- Phase 1 should proceed first as a data-audit and feature-pipeline preflight.
- A feature-blind, manually verified observation manifest is mandatory before constructing state–nuisance triplets.
- Static JPGs may be used immediately to validate mask/box pooling and same-sequence identity diagnostics, but they cannot by themselves validate the central state-change hypothesis.
- V-JEPA 2.1 ViT-B/16 at 384 px is the only local extraction target initially.
- The laptop has an RTX 4050 Laptop GPU with 6 GiB VRAM. Extraction must start
  at batch size 1. ViT-L is Runpod-only and only after the ViT-B gates pass.
- Raw SNS remains an operational nuisance stress test. Semantic attribution
  requires reliable within-input motion measurement, temporal **and** motion
  common support, motion-matched SNS, and a motion-only role-prediction control.
- The proposed “ordering loss” is the ordinary margin triplet/ranking loss. The diagnostic protocol and constrained residual adaptation may be useful; the loss itself is not novel.
- A successful generic MLP, PCA, or whitening control does not refute the diagnostic. It refutes or weakens the need for a special adapter.

No Phase 2 component, Pneuma integration, JEPA predictor training, long-video memory, LoRA, or full-backbone fine-tuning belongs in this plan.

## 2. Claims and hypothesis separation

The project must keep four hypotheses separate.

### H-info: state information is decodable

Within a shared and explicitly defined state ontology, a train-only fitted linear or small nonlinear probe predicts state above balanced and shuffled-label controls on held-out physical-object instances.

### H-geometry: the preregistered native readout is misaligned

For verified, temporal- and motion-matched triplets with adequate common support,
frozen V-JEPA 2.1 object-region cosine distance has materially weaker
state–nuisance ordering than the annotation/noise ceiling. Raw all-triplet SNS
is reported separately as a broader stress test. This claim is conditional on
the selected layer, pooling operator, temporal sampling, motion measurement and
triplet distribution. It is not a claim that the entire representation is
“bad.”

### H-method: a tiny residual correction is useful

A residual bottleneck trained with the ordinary ordering loss improves held-out object-disjoint geometry beyond identity, train-only PCA/whitening, and simpler same-supervision metric transforms under matched tuning budgets.

### H-preserve: correction is non-destructive

The adapter meets preregistered non-inferiority tolerances for identity/persistence retrieval and retains healthy output variance and effective rank.

H-geometry can be supported even when H-method fails. A generic transform solving the problem is still a valid diagnostic result, but it removes the basis for a special-method claim.

## 3. Research/theoretical and mathematical validation

### 3.1 What is theoretically sound

- Information content and a chosen metric readout are different. A representation can support a strong classifier while having poor raw cosine neighborhoods.
- A relative same-instance comparison is more defensible than forcing all objects with a coarse state word into a universal prototype.
- A symmetric distance cannot encode transition direction. Direction/type requires an ordered relation model; that model is outside Phase 1.
- A near-identity residual adapter is a reasonable low-capacity post-hoc correction when the backbone stays frozen.
- Grouped inference over independent objects/videos is required; frame-level bootstrap would be pseudo-replication.

The JEPA connection must remain narrow. V-JEPA was trained by latent prediction, but JEPA principles do not imply a theorem that pooled-token cosine distance must rank a human semantic state change above every nuisance. Failure of this readout is not a contradiction of JEPA or proof that a world model “lost” state. Phase 1 is supervised post-hoc metric analysis of a frozen predictive encoder, not world-model training.

### 3.2 Critical corrections to the specification

| Severity | Specification issue | Required correction for Phase 1 |
|---|---|---|
| **Critical** | The “ordering loss” and “standard triplet loss” are the same hinge loss. Treating them as distinct objectives creates a false baseline. | Name it the standard state–nuisance triplet/ranking loss. Compare architectures or metric parameterizations under the same triplets and loss. Do not claim a novel loss. |
| **Critical** | Same-physical-object positives do not prevent state-prototype collapse. Mapping every observation to a state prototype can achieve zero ordering loss while destroying identity. | Measure within-state cross-instance variance, same-instance retrieval, residual rank, and separate `d_n`/`d_s` distributions. Preservation is an empirical constraint, not guaranteed by sampling. |
| **Critical** | Triplets are underconstrained with respect to time, action, hands, motion, and context. | All observations must be stable-state windows; exclude transition/action frames; prohibit overlapping/near-duplicate clips; **hard-match anchor–nuisance and anchor–state temporal gaps** within a preregistered tolerance, or exclude the triplet from confirmatory SNS. Bidirectional anchors alone do not remove temporal-distance confounding. |
| **Critical** | The specification lists motion as a nuisance but has no measured-motion estimand. Low raw SNS could reflect role-correlated motion severity rather than semantic-state geometry. | Measure motion only on exact encoder input frames, subtract camera flow vectorially before taking magnitudes, require quality/common support, report `SNS_all` and motion-matched SNS, and run a grouped motion-only role classifier. |
| **Critical** | The pooling equation indexes a spatial mask against spatiotemporal tubelet tokens. | Use an explicit `[B,T/2,H/16,W/16,d]` token grid and transform masks/boxes jointly with RGB before area pooling to the tubelet grid. |
| **Critical** | Shuffling state-label metadata after triplets are built does not change the ordering gradients. | For probes, permute train labels at independent group level. For the adapter, rebuild roles from permuted labels or swap nuisance/state roles. Keep validation/test labels and roles untouched. |
| **Critical** | “Object-disjoint” is used interchangeably with “unseen category.” | Separate instance-disjoint from category-disjoint. The current data can support only held-out episode/object instances after manual identity curation. Category-disjoint evaluation is conditional on enough objects sharing a valid state ontology. |
| **Critical** | VICReg variance with the usual `gamma=1` is incompatible with unit-normalized high-dimensional features. | Do not use VICReg in the MVP. If a predeclared rescue is needed, apply variance/covariance terms to unnormalized adapter outputs or to `sqrt(d) u`, with dimension-aware scaling. |
| **High** | “Background-only pooling” of contextual transformer tokens is presented as object-free evidence. | Call it **context-token pooling**. A stronger control requires a second encoder forward on object-pixel-erased input, while explicitly testing fill/silhouette OOD artifacts; it is still not object-free. |
| **High** | Full-dimensional PCA is listed as if rotation alone changes cosine geometry. | Fit centering on train only. A full orthogonal rotation after centering is a no-op; only centering, truncation, scaling, or whitening can change the readout. Include center-only and random-projection controls. |
| **High** | The state probe is interpreted as a global state test even when labels are category-conditioned. | Probe within shared state families or use an explicitly shared ontology. Never use generic `before`/`after` as the state target. |
| **High** | The identity loss can oppose state ordering if the positive pair spans states. | In Phase 1, preservation is evaluation-only. Do not add identity or geometry-distillation loss unless a later, separately justified rescue is approved. |
| **High** | FP32 is reserved mainly for covariance/rank although the scientific signal may be a small distance difference. | BF16 is acceptable for the frozen forward. Convert pooled features to FP32 and compute normalization, PCA, whitening, distances, margins, probes, adapter training, and statistics in FP32. |
| **High** | A parameter-matched classifier and metric adapter are treated as an automatically fair comparison. | Match observations, triplets, supervision, representation dimension, search budget, stopping rule, and seeds. A classifier remains a readout control, not a same-objective metric baseline. |
| **High** | “A generic MLP solves it” is listed as failure of the central claim. | It is failure of H-method, not H-geometry. Record the simpler solution. |
| **Medium** | High SNS with a weak global probe is called benchmark pathology. | This can be legitimate when changes are object-conditioned but absolute states are not globally aligned. Inspect family-conditioned probes before declaring pathology. |
| **Medium** | DINO-style image features are treated as a JEPA causal ablation. | At most use the same sampled frames and regions as a descriptive non-JEPA comparison; architecture, temporal support, data, and objective all differ. It cannot isolate the effect of JEPA training. |
| **Medium** | TrackMAE is read as proof that motion is an independent factor or that separate prediction heads validate this project. | TrackMAE provides empirical motivation to measure motion; it proves neither independence nor a V-JEPA cosine failure. Do not add its motion loss/model to the MVP. A later frozen baseline would be descriptive only. |
| **Medium** | The optional extended loss originally defined `L_dir` but omitted it while including `L_trans`, and nested VC weights redundantly. | Methodology v1.1 now declares categorical and language relation losses as distinct optional terms and uses one variance/covariance weighting level. All remain outside Phase 1. |
| **Medium** | Methodology v1.0 assumed an RTX 5090/32 GiB and extraction micro-batch 4. | Version 1.1 and this plan use measured RTX 4050/6 GiB settings: batch 1, workers 0, BF16 forward, and an OOM smoke test. |

### 3.3 Equation-level audit, including terms not implemented in Phase 1

The complete methodology was checked, including objectives that remain out of
scope. This table records their status so exclusion is not mistaken for lack of
review.

| Equation/objective | Audit result |
|---|---|
| Desideratum `d(same state, nuisance) < d(changed state)` | Mathematically valid as an empirical, annotation-conditioned ranking desideratum; it is not implied by JEPA training and is not an identifiability theorem. |
| Object pooling and learned multi-layer sum | A weighted sum with projected, dimension-matched layers and softmax weights is valid, but the framewise mask equation is not valid for spatiotemporal tubelets. Use Section 4.1. A learned layer mixture is a separately supervised readout and cannot be counted as a frozen-layer result. |
| Cosine distance | Correct after explicit normalization; range `[0,2]` and equal to one-half squared Euclidean distance on the unit sphere. |
| SNS | Sign is correct. Define ties, macro-average independent groups, and use grouped inference; frame/triplet-level IID inference is invalid. |
| Normalized margin `(d_s-d_n)/(d_s+d_n+eps)` | Correct and bounded in magnitude by one for nonnegative distances. It must accompany, not replace, raw distances/margins. |
| Residual bottleneck shapes | Dimensionally correct. The stated zero-scale/zero-output initialization can suppress first-step gradients; use the nonzero small-scale initialization in Section 4.6 and test gradients. |
| Ordering objective | Correct hinge triplet loss; zero loss requires `d_s-d_n >= m`. It is not distinct from “standard triplet loss.” |
| VICReg-style variance/covariance | Covariance alone does not prevent collapse. With unit-norm `u`, average coordinate variance is at most `1/d`, so a per-coordinate standard-deviation target near one is infeasible. Methodology v1.1 removes the redundant outer VC scale. Omit VC terms in Phase 1; any later use must act on unnormalized/rescaled outputs and unique object-balanced observations. |
| Directional transition score/loss | The ordered formulation is algebraically valid and the need for asymmetry is correct. Methodology v1.1 now exposes categorical `L_dir` and language `L_trans` as distinct optional supervision; repeated transition descriptions can still be false negatives in a contrastive denominator. Not implemented here. |
| Identity triplet | Algebraically valid, but can directly conflict with state ordering when the positive pair spans states. Preservation is evaluation-only in Phase 1. |
| Pairwise geometry distillation | Algebraically valid, but “non-state-sensitive pairs” is undefined and may include pairs meant to change. Do not activate without a train-only pair protocol. |
| Optional predictor cosine loss | Algebraically valid with a stopped-gradient target and nonzero predicted/target norms. Jointly training both representation branches would reintroduce collapse/teacher-definition questions. Predictor training is explicitly excluded. |
| Memory confidence/redundancy score | Sigmoid confidence and `q_i - lambda max cosine` are valid heuristics, not representation objectives; calibration, the empty-memory case, and cosine normalization are unspecified. Deferred. |
| Cross-attention memory update | Attention is dimensionally valid only after specifying projection sizes and the softmax key axis. The gate equation is ambiguous: `[M; M_tilde]` must mean feature-wise concatenation per slot, not token-axis concatenation, for elementwise gating to work. Deferred. |
| Conditional metric `M_c=L_c^T L_c` | Positive semidefinite as written; it is a pseudometric if `L_c` is rank deficient. Not needed for Phase 1. |
| Pairwise transition verifier | Concatenation and softmax classifier are dimensionally valid, but it changes the question from metric geometry to supervised relation prediction. It is a later pivot, not evidence for H-method. |
| Adversarial state/nuisance saddle | The sign is consistent only if the nuisance head minimizes its CE while the encoder/adapter maximizes it via explicit alternating or gradient-reversal semantics. It is outside Phase 1. |

### 3.4 Identifiability and interpretation limits

“State” and “nuisance” are task-relative annotations, not identifiable latent factors. For example, lid pose can be the defining evidence for open/closed, while another pose change may be nuisance. Viewpoint can reveal or hide state. The experiment therefore supports at most:

> supervised, task-conditioned alignment of a specified readout on a specified verified triplet distribution.

It does not support causal sufficiency, identity/state disentanglement, a universal state algebra, or a universal claim about all predictive video models.

Motion control can also over-control. For stable configurational state families
such as open/closed, matching observable within-window motion helps isolate the
readout question. If motion is constitutive evidence, such as moving/stopped or
pouring/not-pouring, it must be a separately declared dynamic state family;
conditioning it away changes the estimand and is not used for the primary
configurational-state claim.

## 4. Mathematical objectives and diagnostics actually required

### 4.1 Frozen descriptor

For a 16-frame clip after deterministic preprocessing,

\[
X\in\mathbb{R}^{B\times 3\times16\times384\times384}.
\]

V-JEPA 2.1 ViT-B/16 uses patch size 16 and tubelet size 2. Its final token tensor is

\[
Z=E(X)\in\mathbb{R}^{B\times4608\times768},
\qquad
4608=\frac{16}{2}\frac{384}{16}\frac{384}{16}.
\]

Reshape it as

\[
Z^{grid}\in\mathbb{R}^{B\times8\times24\times24\times768}.
\]

Let `M` be the object mask after the exact RGB resize/crop. Area-average it over each two-frame, 16×16 tubelet cell:

\[
w_{b,\tau,i,j}
=
\frac{1}{2\cdot16\cdot16}
\sum_{t=2\tau}^{2\tau+1}
\sum_{y=16i}^{16i+15}
\sum_{x=16j}^{16j+15}
M_{b,t,y,x}.
\]

Use continuous occupancy weights and FP32 accumulation:

\[
z_b^{mask}
=
\frac{\sum_{\tau,i,j} w_{b,\tau,i,j}Z^{grid}_{b,\tau,i,j}}
{\sum_{\tau,i,j}w_{b,\tau,i,j}+\epsilon}.
\]

Box pooling uses a rasterized per-frame box in place of `M`. Full-frame pooling uses all-ones weights. Context-token pooling uses the complement of a dilated object region, but it is not object-free because all tokens are globally contextualized.

For the released static JPG pathway, `T=1`, so the image-modality output is expected to be `[B,576,768]`, reshaped as `[B,1,24,24,768]`. Static-image results must not be silently mixed with video-clip results.

### 4.2 Cosine geometry

Compute in FP32:

\[
\hat z=\frac{z}{\lVert z\rVert_2+\epsilon},
\qquad
d(a,b)=1-\hat z_a^\top\hat z_b.
\]

For unit vectors, `d` lies in `[0,2]` and equals one half of squared Euclidean distance. For triplet `i`:

\[
d_n^{(i)}=d(z_a,z_n),
\qquad
d_s^{(i)}=d(z_a,z_s),
\qquad
\delta_i=d_s^{(i)}-d_n^{(i)}.
\]

Let \(r\) index sampled triplet rows, \(a\) index an anchor within a transition,
\(t\) index a verified transition, and \(g\) index a connected
`dependency_group_id`. Define the tie-aware row score and explicit nested means

\[
s_r=\mathbf{1}[\delta_r>\tau]
+\tfrac12\mathbf{1}[|\delta_r|\le\tau],
\qquad
\bar s_a=\frac1{|R_a|}\sum_{r\in R_a}s_r,
\qquad
\bar s_t=\frac1{|A_t|}\sum_{a\in A_t}\bar s_a,
\qquad
\bar s_g=\frac1{|T_g|}\sum_{t\in T_g}\bar s_t,
\]

and use the equal-group macro estimator

\[
SNS=\frac1G\sum_{g=1}^{G}\bar s_g.
\]

Here `tau=1e-6` is the numerical tie tolerance. Report tie rate separately.
This nesting gives each transition equal weight inside its dependency group,
even when one transition has two valid anchor directions and another has one;
within a transition, each anchor direction has equal weight regardless of its
number of sampled row combinations. Apply the same row -> anchor -> transition
-> dependency-group nesting to
raw/normalized margins and paired adapter deltas. Do not count a Cartesian
product as independent evidence.

Also report

\[
\Delta_i
=
\frac{d_s^{(i)}-d_n^{(i)}}{d_s^{(i)}+d_n^{(i)}+\epsilon},
\]

the raw paired margin `delta`, and both distance distributions. SNS alone can improve by pushing state distances toward the cosine ceiling while leaving nuisance displacement poor.

### 4.3 Motion-controlled estimand and leakage diagnostic

Motion is measured per observation using only the 16 exact frames passed to
V-JEPA, never the unseen path between temporally separated stable clips. For
visible object point \(j\) between sampled frames \(k\) and \(k+1\), fit a
robust background warp \(H_k\) outside a dilated object/hand region and define

\[
g_k(p)=\pi(H_k\tilde p)-p,
\]

\[
r_{j,k}
=
\frac{
(p_{j,k+1}-p_{j,k})-g_k(p_{j,k})
}{
\Delta t_k\,s_k
},
\qquad
s_k=\max\!\left(\sqrt{w_k^2+h_k^2},\epsilon\right).
\]

Here \(\Delta t_k=(f_{k+1}-f_k)/fps\) is in **seconds**, derived from
authoritative frame timestamps (or frame indices plus the verified constant
frame rate); units may never be mixed across observations. The 16-of-30
sampling intervals are not all equal. The scale \(s_k\) is object-box diagonal.
Estimate points/warps in the recorded post-resize/crop encoder coordinate
system, or analytically map source coordinates through that exact transform;
reject background points outside the encoder field of view. Scalar
object-motion minus global-motion
magnitude is invalid: magnitudes cannot remove position-dependent camera
rotation or projective flow.

Predeclare one complete declared-schema measured-motion vector \(\phi_M(x)\) per
`motion_backend_id + motion_feature_schema_version`. The minimum
reviewed-stationary backend contains signed object-centroid velocity `(vx,vy)`,
signed log-area rate and signed aspect-ratio rate, plus robust magnitude
summaries. A quantitative background-registration bound is mandatory in
addition to double review; otherwise this backend is exploratory.

A global-compensated point-track schema contains signed confidence-weighted mean
residual-flow `(vx,vy)`, signed mean background-flow `(vx,vy)`, median and
90th-percentile residual magnitude, and moving-point fraction. Visibility,
confidence, valid-track count, inlier ratio, warp error, region coverage and
manual decisions are **quality fields**, not coordinates of \(\phi_M\). Missing
tracker fields are explicit nulls in the joined table and are never imputed as
zero or passed to a distance. All three members of a confirmatory triplet must
have the same backend/config/schema and complete required motion coordinates;
different backends are analyzed as separate estimands.

For the minimum stationary-camera schema, use transformed encoder-crop boxes.
Let \(c_k\) be box center, \(A_k=w_kh_k\),
\(R_k=w_k/h_k\), \(s_k=\sqrt{w_k^2+h_k^2}\), and let
\(\omega_k=\Delta t_k\min(v_k,v_{k+1})\), where \(v_k\in[0,1]\) is reviewed
visible box fraction. For each of the 15 intervals,

\[
v^c_k=\frac{c_{k+1}-c_k}{\Delta t_k\max(s_k,\epsilon)},\qquad
v^A_k=\frac{\log(A_{k+1}+\epsilon)-\log(A_k+\epsilon)}{\Delta t_k},
\qquad
v^R_k=\frac{\log(R_{k+1}+\epsilon)-\log(R_k+\epsilon)}{\Delta t_k}.
\]

With `wmean` and weighted quantiles using \(\omega_k\), freeze

\[
\phi_M^{stationary}(x)=
[
\operatorname{wmean}(v^c_x),
\operatorname{wmean}(v^c_y),
Q_{50}(\lVert v^c\rVert),
Q_{90}(\lVert v^c\rVert),
\operatorname{wmean}(v^A),
Q_{90}(|v^A|),
\operatorname{wmean}(v^R),
Q_{90}(|v^R|)
].
\]

Quantify the stationary assumption rather than relying on review alone. On each
transformed frame pair, estimate a deterministic FFT phase-correlation
translation \(\hat b_k\) over pixels outside the fixed dilated object/hand
exclusion region and record

\[
b_k=\frac{\lVert\hat b_k\rVert_2}
{\Delta t_k\sqrt{384^2+384^2}}.
\]

The observation passes only if weighted \(Q_{90}(b_k)\) is below a
feature-blind pilot-frozen bound and phase-correlation peak-to-sidelobe quality
is above its frozen bound. Reject hand-dominated pairs without a reviewed
exclusion region. These registration values remain quality diagnostics.

For a global-compensated point-track schema, compute \(r_{j,k}\) from the vector
residual equation above. Its declared vector is the confidence/visibility/
duration-weighted signed mean `(rx,ry)`, weighted median and 90th percentile of
the residual norm, weighted moving-point fraction above a train-frozen native
threshold, and the correspondingly weighted signed background-flow mean
`(gx,gy)`. Tracker visibility, cycle error, valid count, warp inliers and warp
residual are componentwise quality fields. Every aggregation has a
deterministic empty/insufficient-support failure rule.

Fit a robust scaler on unique training observations only:

\[
\widetilde\phi_{M,k}(x)
=
\frac{
\phi_{M,k}(x)-\operatorname{median}_{train,k}
}{
\max(\operatorname{IQR}_{train,k},s_{min,k})
}.
\]

Freeze native-unit floors \(s_{min,k}>0\) before test and drop dimensions that
are constant on train under a recorded rule. Record active dimensions and
scales; never let an arbitrarily small IQR amplify numerical jitter.

For ordered pair \((i,j)\), retain both signed change and severity:

\[
\Delta_M(x_i,x_j)=\widetilde\phi_M(x_j)-\widetilde\phi_M(x_i),
\qquad
c_M(x_i,x_j)=|\Delta_M(x_i,x_j)|,
\qquad
M(x_i,x_j)=\lVert c_M(x_i,x_j)\rVert_2.
\]

The triplet has separate signed-change and severity mismatches:

\[
q_M^{signed}(T)=
\left\lVert
\Delta_M(x_a,x_n)-\Delta_M(x_a,x_s)
\right\rVert_2,
\qquad
q_M^{sev}(T)=
\left\lVert
c_M(x_a,x_n)-c_M(x_a,x_s)
\right\rVert_2,
\qquad
\overline M(T)
=
\tfrac12\{M(x_a,x_n)+M(x_a,x_s)\}.
\]

Confirmatory motion-matched rows require
\(q_M^{signed}(T)\le\epsilon_M^{signed}\) **and**
\(q_M^{sev}(T)\le\epsilon_M^{sev}\), all three observations to pass frozen
motion-quality checks, and adequate common support. Choose both calipers, the
componentwise quality rules and low/medium/high bins of
\(\overline M\) from feature-blind train/validation feasibility only; never use
V-JEPA features or test labels. Report group-macro SNS-all,
SNS-motion-matched, matched coverage, independent-group counts,
both mismatch distributions, \(\overline M\), signed component balance and
sensitivity over predeclared calipers.
If the background model is invalid under parallax, rolling shutter,
insufficient static background or tracker drift, reject/flag the observation.
If common support collapses, H-geometry is not estimable.

The motion-only leakage control creates \((a,n,y=0)\) and \((a,s,y=1)\) pair
rows. Its preregistered logistic input is
`[phi_a, phi_b, Delta_M(a,b), abs(Delta_M(a,b))]`; a deliberately weak scalar
threshold uses only \(M(a,b)\). It fits preprocessing/model/threshold using
train/validation dependency groups only, and evaluates only inside the single
atomic locked test on
object/video-disjoint test. Report balanced accuracy, AUROC, permutation null
and grouped confidence intervals. Shared-anchor pairs are dependent and must
never be row-randomized or row-bootstrapped. Strong prediction is a sampling
failure for semantic attribution; weak prediction is not proof that all
confounding is absent. Any dyadic mixed-effects distance regression is
exploratory only and cannot replace matching.

The role-association randomization null is paired and dependency-preserving.
For each Monte Carlo replicate draw one Bernoulli swap for each complete
`dependency_group_id`; if selected, swap the nuisance/state role labels for
**every** `(a,n)/(a,s)` pair in that component, otherwise retain all of them.
Keep fitted predictions fixed and recompute the grouped statistic. Never flip
individual rows, anchors, or triplets independently. This conditional null is a
diagnostic of role association under paired-role exchangeability, not proof that
the motion descriptor captures every confound.

[TrackMAE](https://arxiv.org/html/2603.27268) motivates this audit empirically
through complementary trajectory and appearance/semantic reconstruction
targets; its [official implementation](https://github.com/rvandeghen/TrackMAE)
uses CoTracker3, but supplies no theorem for this geometry hypothesis. No
TrackMAE loss, predictor, encoder dependency or CoTracker-supervised adapter
objective is added to Phase 1.

### 4.4 Probe objectives

The linear and MLP probes use cross-entropy on unique observations, not
triplet-expanded rows:

\[
\mathcal L_{probe}
=
-\frac1B\sum_i w_i\log p_\phi(y_i\mid z_i).
\]

Use a two-stage sampler (sample a dependency group uniformly, then an eligible
observation within it) and inverse-class loss weights, or fit iterative
proportional/calibrated weights. Report the realized class and group mass; the
simple product of inverse class and inverse group size is not guaranteed to
balance both when group class compositions differ. They answer H-info only for
state labels shared across the evaluated state family.
Report dependency-group-macro balanced accuracy, macro-F1, per-class recall,
calibration, majority baseline, shuffled-training negative controls (and a
formal block-randomization null only if separately valid), and source-video-clustered
confidence intervals.

### 4.5 Train-only geometry controls

Fit all transforms on unique, object-balanced training observations only (or a
fixed, preregistered cap per object). With training mean `mu` and covariance
eigendecomposition `C=V Lambda V^T`:

\[
y_{PCA,k}=V_k^\top(z-\mu),
\]

\[
y_{white,k}
=(\Lambda_k+\rho I)^{-1/2}V_k^\top(z-\mu).
\]

L2-normalize after projection. Consider `k` from `{64,128,256}` only when
`k <= min(d, N_unique_train-1)` and the preregistered minimum effective-group
support per retained dimension passes; record skipped values. Select `k` and
whitening shrinkage `rho` on validation only. Express shrinkage and eigenvalue
floors relative to the mean positive train eigenvalue, not as an unscaled
absolute constant. Include:

- raw L2-normalized features;
- center-only features;
- truncated PCA;
- dimension-matched seeded random orthogonal projection;
- shrinkage whitening with an eigenvalue floor;
- full-dimensional PCA as an implementation invariant: after the same centering and normalization, its orthogonal rotation must reproduce center-only cosine distances within tolerance.

### 4.6 Tiny residual adapter

For `d=768` and bottleneck `b=256`:

\[
h=W_1\operatorname{LN}(z),
\qquad
v=z+\alpha W_2\operatorname{GELU}(h),
\qquad
u=\frac{v}{\lVert v\rVert_2+\epsilon}.
\]

This has approximately 395,777 trainable parameters including LayerNorm, biases, and scalar `alpha`. Initialize `W_1` and `W_2` randomly with a small residual scale `alpha=10^{-3}`. Do not initialize both the residual output and scale to zero; that can create a dead or nearly dead first step. A unit test must confirm nonzero first-step gradients for `W_1`, `W_2`, and `alpha`.

The only required adapter objective is the standard hinge ranking loss:

\[
\boxed{
\mathcal L_{ord}
=
\frac1B\sum_i
\max\left(0,m+d(u_a,u_n)-d(u_a,u_s)\right)
}.
\]

Zero loss requires `d_s-d_n >= m`. Select `m` from `{0.05,0.10,0.20}` using
training distributions and validation performance only. Train cached features
in FP32 with AdamW, starting at learning rate `3e-4`, weight decay `1e-4`,
triplet batch 512, and fixed seeds. Form each batch by sampling
dependency/transition groups uniformly and then at most one triplet per sampled
transition; early-stop on dependency-group-macro validation margin. No
transition head, identity loss, geometry-distillation loss, or VICReg term is
part of the Phase 1 MVP.

## 5. Exact dataset audit

### 5.1 Snapshot and provenance

Only `datasets/phase1/EgoInteract/` exists. No OSCaR, STATUS, HowToChange/VidOSC, TOC-Bench, or real-world state dataset is present.

All 6,855 Hugging Face `*.metadata` records identify dataset revision:

```text
313d1ef6586571d6ce1fe85581f690c507110fea
```

The current downloader does not pin this revision; the implementation must add a `--revision` argument before any future re-download. The project root is not a Git repository, so root source-file hashes must be recorded until version control is initialized by the user. A nested official V-JEPA repository now exists at `vjepa2/`; its independently audited state is recorded in Section 8.1.

The final 21 August 2026 repository re-audit found **no dataset delta**:
payload count/bytes, cache metadata, revision, annotations, frames and videos
are unchanged; there are still no aligned square-video regions or semantic
physical-state labels. A new zero-byte root file named `2]` is unrelated to
the experiment, is excluded from every manifest, and is preserved rather than
deleted without explicit cleanup authorization.

The `.cache/` tree has 6,857 files and 793,083 bytes in total: the 6,855
metadata records above plus `CACHEDIR.TAG` and `.gitignore`. It has no lock or
incomplete-download files. Payload inventory, excluding `.cache/`, at the
stable 2026-08-21 00:05 +06:00 snapshot (reconfirmed in the final pass) is:

| Item | Exact count/size |
|---|---:|
| All payload files | **68,073 files; 19,902,705,085 bytes** |
| Root metadata | 2 files; 6,354 bytes |
| Configs, including extracted ZIP | 54 files; 243,404 bytes |
| HOI annotations | 15 files; 2,900,918,617 bytes |
| Interaction-anticipation annotations | 1 file; 1,117,134 bytes |
| NAO annotations | 9 files; 51,524,595 bytes |
| TAS annotations | 3,390 files; 6,599,070 bytes |
| MP4 videos | 3,390 files; 10,646,382,394 bytes |
| `frames/frames/hoi` JPGs | 29,598 files; 1,677,098,830 bytes |
| `frames/frames/hoi_enigma` JPGs | 20,225 files; 987,216,111 bytes |
| `frames/frames/nao` JPGs | 11,388 files; 544,177,311 bytes |
| Redundant extracted-frame archive | `data/frames.tar.gz`; 1 file; 3,087,421,265 bytes |

The JPGs are already extracted. Do not unpack `frames.tar.gz` again. The archive
SHA-256 is
`9962d4c7dfe5ea2af05dcf8bf2c5c73f82a9ba1adc80f62a8066589273371c81`.
A streamed tar audit found 61,211 files and four directories, with exact
name/size agreement with the extracted tree. The downloaded config ZIP also
passes integrity testing; its six extracted files match the archive, although
the extracted path contains a redundant doubled directory component.

For migration, copy the extracted JPGs or the archive, not both. The current
`scripts/download_phase1_datasets.py` does **not** request `data/frames.tar.gz`,
so rerunning it alone cannot reproduce this snapshot. Before migration or a
clean re-download, add an explicit pinned frame-archive acquisition step and
verify the revision, byte count, and SHA-256 above; otherwise the dataset audit
must fail closed.

The 54 config files comprise 34 `.yaml`, two `.yml`, 13 `.txt`, four `.py`, and
one `.zip`. The local YAML configs reference absent base YAMLs and retain `PATH`
placeholders; they are not runnable Phase 1 configs. The 13 TAS split lists are
EgoExo/EK100 transfer splits, not EgoInteract physical-object-disjoint splits,
so none may substitute for the new split manifest.

### 5.2 Videos and TAS annotations

Use paths:

```text
data/videos/0/sequence_<id>.mp4
data/annotations/tas/sequence.<id>.txt
```

There are exactly 3,390 MP4s and 3,390 TAS files. Every MP4 has one `avc1`
video track, is 1408×1408 at exactly 30 fps, and contains 99–237 frames. The
video-ID and TAS-ID sets match exactly; MP4 sample count equals TAS line count
for every file, with zero structural parse errors. The complete TAS corpus has
659,907 frame labels with:

| TAS value | Frames |
|---|---:|
| `action_0` | 396,144 |
| `action_1` | 82,535 |
| `action_2` | 181,228 |

These labels may be used only to propose temporal regions and to exclude active
transition/action intervals after visual verification. They are not state
labels. Exactly 3,263/3,390 sequences have the run pattern
`action_0 -> action_1 -> action_0 -> action_2 -> action_0`, which creates a
severe procedural-phase shortcut if action labels are reinterpreted as state.

### 5.3 Interaction-anticipation labels

Use only as weak candidate metadata:

```text
data/annotations/interaction_anticipation/labels.json
```

It is a dictionary with 2,294 entries keyed by `sequence.<id>`. Every value has:

```text
inputs.video 1.id
question
choices
correct_idx
timestamp_hoi
```

`timestamp_hoi` ranges from 1.2 to 2.1 seconds. Of the records, 2,143 have five
choices and 151 have six. The selected answer is a noisy next-object noun
proposal, not a physical-object ID and not a state. There are 273 normalized
answer strings, including scene nouns such as `building` and `house`; 223 rows
contain duplicate normalized choices and 166 repeat the correct text. This
metadata requires manual verification before use.

### 5.4 Aligned static HOI masks and boxes

The following base pair is exactly aligned and is one of the two immediately
supported static sources for mask pooling:

```text
data/frames/frames/hoi/<sequence>_<frame>.jpg
data/annotations/hoi/coco_annotations_egointeract.json
```

Facts:

- 29,592 JSON image records, all present on disk;
- 29,598 JPGs on disk, with six unreferenced extras for sequence 5062;
- every JPG is 1280×720;
- 48,606 annotations: 28,966 hand and 19,640 object;
- 4,224 `processed_sequences` strings and 4,196 unique filename sequence prefixes;
- categories are `{1: hand, 2: object}`;
- object regions are generic `object`, not semantic object categories.

Image-level category patterns are 19,615 hand+object, 9,351 hand-only, and 25
object-only. Every image containing both categories is a contact sample; object
regions are therefore effectively contact-selected, a major shortcut.

All 61,211 extracted JPGs across the three frame directories verify as RGB
1280×720 JPEGs with no decode errors. This media integrity does not imply
semantic label suitability.

Image fields:

```text
id
file_name
width
height
```

Annotation fields:

```text
id
image_id
category_id
bbox                 # COCO [x, y, width, height]
area
iscrowd
exhaustive
solo_instance_id
handside
isincontact
offset
segmentation         # multipart polygon lists; components can be degenerate
```

For hand annotations, `handside` and `isincontact` are binary; for object
annotations they are `-1`. Local files do not define the numeric hand-side or
offset semantics, so do not infer them. The COCO `area` value equals bbox area,
not rasterized polygon area. HOI image and annotation IDs can exceed signed
64-bit range; store them as decimal strings in Parquet/NumPy indices.

Polygon validation is mandatory. In the base HOI file, 4,552 annotations have
at least one component with fewer than three points; 84 object and 39 hand
annotations have no valid component after filtering. Drop degenerate
components and exclude an annotation if the resulting mask is empty. Do not
assert that polygon-mask area equals the supplied `area` field.

The `processed_sequences` entries are contributor-machine absolute paths and
must not be used as portable data paths or model features.

`solo_instance_id` can identify repeated annotations within a sequence only
when combined with the source family, object category, and sequence. Its
numeric namespace also collides between hand and object categories. Local
documentation does not establish that a repeated numeric ID across episodes is
the same physical asset. Define a conservative local ID as:

```text
egointeract:<source_family>:<sequence_id>:<category_id>:<solo_instance_id>
```

The static HOI data are suitable for:

- unit-testing polygon rasterization and joint RGB/mask transforms;
- image-modality mask, box, full-frame, and context-token pooling;
- same-sequence object-preservation retrieval;
- candidate nuisance-pair curation after manual state-preservation review.

They are **not** sufficient for the central state diagnostic because no semantic object-state labels are supplied.

### 5.5 Other static annotations

The following aligned pair is also complete:

```text
data/frames/frames/hoi_enigma/
data/annotations/hoi/coco_annotations_hand_egointeract.json
```

It contains 20,225 1280×720 JPGs, 20,225 image records, and 32,359 polygon
annotations (19,432 hand and 12,927 object) with the same HOI field schema. Of these, 2,726 annotations contain
at least one degenerate polygon component; 50 object and 81 hand annotations
have no valid polygon after filtering. Its sequence identifiers are 7001–10998
and do not correspond to video partition 0.

Its image-level category patterns are 12,870 hand+object, 6,562 hand-only, and
57 object-only, with the same contact-selected object policy.

The base NAO annotation is:

```text
data/frames/frames/nao/
data/annotations/nao/coco_annotations_egointeract.json
```

It has 6,885 image records and 6,885 object annotations; every referenced JPG
is present. NAO annotations contain `bbox` and `solo_instance_id` but no
segmentation. The `nao/` directory has 11,388 JPGs; the remaining 4,503 are not
referenced by **any** of the nine local NAO JSONs and must be excluded.

Its image fields are `id`, `file_name`, `width`, and `height`; annotation fields
are `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`, `exhaustive`,
and `solo_instance_id`. The only category is `{1: object}`. It has no
segmentation, contact, state, or nuisance field.

Files whose names contain `+egohos`, `+visor`, `+enigma`, `+ego4d`, or `+meccano` combine external-domain annotations. They must not be used in the minimal Phase 1 run because the corresponding external datasets/splits are not present and the combined JSONs duplicate EgoInteract records.

There are exactly 377 byte-identical JPEG pairs, all shared between
`hoi_enigma/` and `nao/`; 371 pairs are referenced in both base tasks. Reject a
hash entirely if its two records receive conflicting proxy labels; canonicalize
same-label copies to one observation before splitting. Exclude the six
unreferenced `frames/hoi` files
`5062_{38,58,59,65,67,74}.jpg` and the 4,503 unreferenced `frames/nao` files.

### 5.6 Runnable static interaction-phase proxy

The strongest join available without new annotation is:

```text
data/annotations/nao/coco_annotations_egointeract.json
data/frames/frames/nao/<sequence>_<frame>.jpg
                joined by sequence + object solo_instance_id
data/annotations/hoi/coco_annotations_hand_egointeract.json
data/frames/frames/hoi_enigma/<sequence>_<frame>.jpg
```

Raw object-presence join facts before enforcing a contacting hand:

- 2,219 sequences occur in both base annotations, and 2,171 have object
  annotations on both sides;
- object `solo_instance_id` agrees in 2,171/2,171 joined sequences;
- all 6,785 joined NAO boxes occur 6–51 frames before the first HOI
  object-presence/contact-selected sample (mean 16.95, median 16 frames);
- 1,979 sequences have at least two pre-contact boxes and at least two contact
  boxes/masks, spanning 1,827 numeric object IDs;
- only 142 of those IDs repeat across sequences (294 sequences), so most
  within-sequence same-phase pairs are near-duplicates rather than strong
  cross-scene nuisance changes.

After requiring at least one hand with `isincontact=1`, the effective proxy has
2,170 joined sequence/object keys, 6,784 pre-contact NAO boxes, and 1,978
sequences with at least two observations per phase, spanning 1,826 numeric
object IDs. The gap remains 6–51 frames (mean 16.9587); the repeat-ID counts
remain 142 IDs/294 sequences. These filtered counts, not the raw counts above,
must appear in proxy artifacts.

This permits an **image-only interaction-phase smoke/shortcut benchmark** with
`phase in {pre_contact, contact}`. Use GT **box pooling** as the symmetric
primary localization because NAO has no masks. Contact-mask pooling is an
asymmetric diagnostic only; mask presence itself reveals the phase. A fair
mask comparison would require applying the same frozen segmenter and review
policy to both sides.

Define `contact` only when the HOI image has at least one hand annotation with
`isincontact=1`; exclude the 57 object-only hand-HOI frames and any object frame
without a positively contacting hand. Do not infer contact merely from the
presence of an object annotation.

This proxy must be reported in a separate table and never renamed object state.
It is dominated by temporal order, hand presence, occlusion, mask availability,
and procedural phase, so it cannot validate H-info or H-geometry for semantic
physical state. Its value is to exercise the loader/pooling/split/control code
and demonstrate whether the shortcut suite detects an intentionally confounded
target.

### 5.7 The video/static alignment boundary

Do **not** apply 1280×720 COCO coordinates to 1408×1408 MP4 frames. Numeric sequence-ID overlap does not establish a spatial transform; the releases use different image products/views. Until a documented and visually verified mapping exists:

```text
TAS ∩ interaction-anticipation sequence IDs = 2,294
TAS ∩ pure-HOI sequence IDs              = 2,579
interaction-anticipation ∩ pure-HOI       = 1,656
TAS ∩ hand-HOI                            = 0
TAS ∩ base NAO                            = 0
hand-HOI ∩ base NAO                       = 2,219
```

These are identifier intersections only, not evidence that different spatial
products are registered.

- static JPGs support region pooling in V-JEPA's image modality;
- square MP4s support video extraction only after new aligned boxes/masks are curated;
- static and video results remain separate tables;
- no cross-modality triplet may be used in the primary SNS estimate.

### 5.8 State and identity supervision actually available

| Required concept | Available now? | Defensible use |
|---|---|---|
| Semantic state before/after | **No** | Must be manually annotated; TAS cannot substitute. |
| Physical identity within one static sequence | Partial | Namespaced sequence/category plus local `solo_instance_id`; label as an asset proxy unless manually verified. |
| Physical identity in square video | **No annotation** | Must be manually tracked/verified. |
| Cross-video persistent physical identity | **No** | Do not claim; split videos conservatively. |
| Semantic object category | Weak/noisy | Interaction answer may propose a noun; manual normalization required. |
| Static HOI box | Yes | Base HOI and hand-HOI JPGs. |
| Static HOI mask | Partial | Validated polygon segmentation in HOI JSONs; degenerate/empty polygons must be filtered. |
| Static NAO box | Yes | Base NAO JSON. |
| Video box/mask | **No aligned region** | Must be newly annotated or generated and manually approved. |
| Nuisance label | **No** | Must be manually tagged and measured without using embeddings. |

## 6. Phase 1 data construction

### 6.1 Mandatory fail-closed manifest gate

The data contract is staged; a single table must never be required to contain
fields that are created only by later steps.

The feature-blind curation output `curated_observations.parquet` must contain:

```text
observation_id
dataset_revision
source_video_id
media_relpath
media_type                    # image or video
physical_object_id
verified_asset_group_id         # nullable; only independently verified cross-video asset
object_category_manual
state_family
state_label
stable_segment_id
transition_id
start_frame
end_frame
sampled_frame_indices
fps
state_observable
identity_verified
box_annotation_relpath         # required per sampled frame for the pilot primary
mask_annotation_relpath        # nullable; verified mask subset only
mask_available                 # frozen before extraction; mask subset only
nuisance_tags
hand_present
stationary_background_verified # manual review; not a substitute for measured motion
motion_review_status           # pending/pass/fail
curator_id
reviewer_id
review_status
media_sha256                    # required derived exact hash
perceptual_hash                 # required derived near-duplicate fingerprint
duplicate_group_id              # derived connected exact/near-duplicate group
```

The split builder writes an immutable `splits.parquet` mapping
`observation_id -> dependency_group_id, split`. Motion estimation writes a
separate `motion_observations.parquet` with:

```text
observation_id
motion_backend_id               # implementation + frozen config hash
motion_source_revision
motion_checkpoint_sha256        # null only for a non-learned backend
motion_feature_schema_version
motion_feature_vector_raw       # complete declared coordinates for its raw schema
motion_valid_track_count        # quality/diagnostic, not a distance coordinate
motion_visibility_fraction      # quality/diagnostic
motion_global_inlier_ratio      # quality/diagnostic
motion_global_fit_error         # quality/diagnostic
motion_background_registration_error
```

`fit_motion_scaler_quality.py` writes `motion_validated.parquet` with the raw
diagnostics plus `motion_feature_vector_scaled`, `motion_active_dimensions`,
`motion_scaler_hash`, `motion_quality_rule_hash`, and `motion_quality_pass`.
Only train observations fit centers/scales/active dimensions; componentwise
quality rules are frozen from feature-blind pilot/train/validation policy.

`assemble_phase1_analysis_manifest.py` then performs one-to-one joins and writes
`internal_analysis_observations.parquet`. It contains every curated field
plus split and validated motion provenance/quality, but no V-JEPA feature.
Build the exact temporal candidate-triplet universe from this table **before**
selecting signed/severity calipers. The caliper selector receives that candidate
manifest, uses train/validation candidates only, and emits a hash-locked rule;
the finalizer applies it without refitting.

Before any model-selection command, a release-view builder separates
`analysis_trainval.parquet`, label-redacted `feature_inputs.parquet`, and a
hash-locked `sealed_test_targets.parquet`. It likewise separates
`triplets_trainval.parquet` from sealed test triplet roles. This is a procedural
research seal rather than cryptographic secrecy, but no pre-test CLI accepts the
sealed target/role path or a full labeled analysis table. The builder may apply
the already frozen rules to test rows, but it must not emit or display test
class balance, matched coverage, exclusion counts, role distributions, or any
other outcome-bearing test summary before the locked evaluator.

The redacted feature-input view has an exact allowlist, hash-locked in
`configs/phase1/feature_input_columns.txt`:

```text
observation_id
dataset_revision
media_relpath
media_type
media_sha256
start_frame
end_frame
sampled_frame_indices
fps
box_annotation_relpath
mask_annotation_relpath
mask_available
coordinate_space
motion_backend_id
motion_feature_schema_version
motion_quality_pass
```

No extra column is permitted. In particular it must exclude `state_family`,
`state_label`, all triplet roles/targets, `physical_object_id`, asset/category,
source-video/transition/stable-segment IDs, nuisance/hand/time-phase labels,
`dependency_group_id`, `split`, and confirmatory selectors. Extraction logs and
feature indices repeat only `observation_id` plus feature provenance; they do not
copy target-bearing columns.

All media paths stored in Parquet/JSON must be dataset-relative POSIX strings. Never store `P:\...` or `/workspace/...` in a manifest.

COCO identifiers are stored as strings, and static records additionally carry
`source_family`, `sequence_id`, `frame_id`, `category_id`, `solo_instance_id`,
`media_sha256`, `duplicate_group_id`, and `coordinate_space`. Generated candidates must never copy
`processed_sequences` paths into learning tables.

The **curation-stage** validator must reject:

- missing or generic `before`/`after` state labels;
- state labels not defined in a state-family ontology;
- observations that include action/transition frames;
- invisible or ambiguous defining evidence;
- uncertain physical identity;
- masks/boxes whose declared coordinate system does not match the media;
- missing exact/perceptual hashes or unresolved duplicate groups;
- reused raw frames across splits;
- a state/nuisance pair with zero verified nuisance change;
- tracker points, thresholds, calipers or quality decisions selected using
  state labels, V-JEPA embeddings or test outcomes;
- any triplet selected by looking at embeddings.

The **analysis-stage** validator additionally rejects missing split/dependency
mapping, non-one-to-one joins, missing physical-motion coordinates, failed
quality, mixed backend/config/schema within an analysis stratum, and incomplete
motion provenance. A stationary-background review can waive global
compensation only after a quantitative registration bound passes; it never
waives motion measurement.

### 6.2 Minimal curation protocol

Start with a feature-blind screen of 50–100 square videos. For every candidate:

1. inspect the full video;
2. identify one physical object and a genuine, visually observable state change;
3. define the state family and its valid labels, such as `open/closed`, only when the semantic relation is actually present;
4. mark stable pre-state and post-state intervals, excluding hands/actions/transitions where possible;
5. select 16-frame observations within fixed-duration stable windows;
6. annotate a box on every sampled video frame; masks are optional for the first pilot but must be generated/reviewed for the mask-pooling subset;
7. select at least two non-overlapping same-state observations with a verified
   nonzero nuisance change, including a repeated-cycle or long-gap observation
   capable of matching the anchor–state temporal gap within the frozen temporal
   tolerance;
8. tag nuisance type, observability, hand motion and whether the
   background/camera is stationary; do not use embeddings;
9. initialize any motion tracks or background points by a fixed grid/region rule
   independent of state labels; CoTracker-style continuity can support motion
   measurement but cannot replace physical-identity review;
10. have a second reviewer independently verify identity, state, interval
    stability, observability and the stationary-background assessment;
11. resolve disagreement without seeing model features.

Simple grasp, translation, camera motion, or pose change is not automatically a semantic state change. Exclude episodes where the only difference is procedural phase, hand presence, or object location unless a preregistered positional-state family explicitly treats that relation as state.

The first pilot continues only if at least 30 independent, high-confidence
object transitions spanning at least three recurring state families can be
curated. It is **protocol/power work only** and makes no confirmatory test
claim. Use pilot cluster variance to perform a power analysis, then curate a
separate powered cohort before creating the final frozen split. A 15% test split
with a target of at least 50 independent test groups requires roughly 334 total
independent groups even before family stratification (and potentially more after
connected-component grouping); also target at least 10 test groups per reported
family. If that expansion is infeasible, report an exploratory pilot and do not
open a nominal confirmatory test set.

Double-annotate at least 20% of candidates (and every test candidate). Report
Cohen's kappa or Krippendorff's alpha for categorical state/identity/visibility
decisions and temporal IoU for stable intervals; retain raw disagreements and
derive the adjudication/noise ceiling without consulting features.

### 6.3 Observation sampling

Use a fixed observation rule, frozen before feature extraction:

- source rate: 30 fps;
- primary observation: for verified half-open stable interval `[a,b)` with
  `b-a>=30`, choose `s=floor((a+b-30)/2)` and window `[s,s+30)`, then sample
  `i_k=s+floor(k*29/15+0.5)` for `k=0,...,15`; this freezes 16-of-30 rounding
  and includes both endpoints;
- no shared decoded frame between two observations in one triplet;
- no sampled transition/action frame;
- equal temporal span for anchor, nuisance, and state observations;
- retain exact integer frame indices;
- record exact consecutive sampled-frame/time deltas, relative timestamp and
  temporal gaps for motion and shortcut controls;
- use the same sampled frames for motion measurement and encoder input; never
  measure an unobserved transition path and call it input motion.

If a stable interval is shorter than one second, reject it rather than silently changing temporal support. A separately labeled sensitivity ablation may use another duration after the primary result.

### 6.4 Triplet construction

Every triplet must satisfy:

\[
o_a=o_n=o_s,\qquad y_a=y_n,\qquad y_a\ne y_s,
\]

plus the non-mathematical requirements:

- same verified physical track;
- all three observations are stable and state-observable;
- `a` and `n` have a recorded, nonzero state-preserving nuisance difference;
- no overlapping raw frames or near-duplicate observation hashes;
- no active transition/action cue;
- role assignment uses annotations only;
- nuisance severity is recorded and stratified;
- temporal gaps satisfy the hard confirmatory tolerance; hand/action context is
  matched, with any residual mismatch reported as a covariate and sensitivity
  stratum;
- all three observations pass frozen motion-quality checks and the componentwise
  pair-motion mismatch satisfies the train/validation-frozen caliper.

Where both states contain two stable observations, create paired directions:

```text
(pre_anchor, pre_nuisance, post_state)
(post_anchor, post_nuisance, pre_state)
```

This reduces one-direction procedural bias but does not match temporal distance.
For confirmatory triplets require

\[
\left|\,|t_a-t_n|-|t_a-t_s|\,\right|\leq\eta_t,
\]

where times are source-frame midpoints and `eta_t` is frozen from feature-blind
pilot feasibility, initially at most two source frames. Use repeated cycles or
same-state observations spanning comparable time when available. If the source
episode cannot supply such observations, label its triplets
`exploratory_temporally_confounded=true` and exclude them from H-geometry,
H-method, and confirmatory confidence intervals; recording the gap as a
covariate is not enough. Limit triplets per transition and aggregate within
transition before inference; do not enumerate all combinations.

Temporal matching alone is insufficient. After split construction, fit the
robust motion scaler and choose the motion caliper using train/validation
feasibility without embeddings. Mark a triplet confirmatory only when both
temporal and motion criteria pass. Report all-triplet results descriptively, but
do not use them to rescue H-geometry if the motion-matched subset lacks adequate
independent groups or common support.

Triplet manifest fields:

```text
triplet_id
anchor_observation_id
nuisance_observation_id
state_observation_id
physical_object_id
transition_id
dependency_group_id
anchor_state
state_target
nuisance_type
nuisance_severity
temporal_gap_an
temporal_gap_as
temporal_gap_mismatch_frames
temporal_gap_tolerance_frames
temporal_matched
motion_pair_an
motion_pair_as
motion_signed_pair_an
motion_signed_pair_as
motion_signed_mismatch
motion_signed_tolerance
motion_severity_mismatch
motion_severity_tolerance
motion_common_level
motion_quality_min
motion_backend_id
motion_feature_schema_version
motion_matched
confirmatory_eligible
direction
split
verification_status
```

`motion_signed_pair_*` stores signed scaled changes and `motion_pair_*`
stores their componentwise absolute severities. Both mismatch norms must pass
their frozen calipers. `motion_common_level` is the scalar average severity
norm. The confirmatory flag is the conjunction of verification, homogeneous
complete motion schema, temporal match, motion quality/match and leakage checks,
never a manually editable label.

## 7. Object-disjoint splitting and leakage prevention

Split groups **before** enumerating triplets. For current EgoInteract video curation, define the physical object as a manually verified local track and require all tracks from one source video to stay in one split. If the same synthetic asset is later shown to recur across videos, all occurrences of that asset must also stay in one split.

Construct a leakage/dependence graph whose nodes are observations and whose
edges connect a shared source video, verified physical object/asset, transition,
or exact/near-duplicate media hash. Its connected-component ID is
`dependency_group_id`. A component is indivisible across splits and is the
default bootstrap unit; this is stricter than grouping only by
`physical_object_id` when several objects share one video.

For every nullable grouping key, null means **no edge**. Never stringify null or
connect all missing values into one component. Unit-test this specifically for
`verified_asset_group_id` and any future nullable asset/hash field.

Primary split:

```text
train / validation / test = 70% / 15% / 15%
seed = 20260820
group constraints = dependency_group_id (connected object + video + asset + hash groups)
stratification = state_family + state_label + multilabel nuisance_tags
```

Required assertions:

\[
\mathcal O_{train}\cap\mathcal O_{val}
=
\mathcal O_{train}\cap\mathcal O_{test}
=
\mathcal O_{val}\cap\mathcal O_{test}
=\varnothing.
\]

Also assert zero source-video overlap, zero observation/frame overlap, and zero exact or near-duplicate media hash overlap. Category-disjoint evaluation is a secondary split only when several categories share the same semantic state family with enough independent groups. Do not call an instance-disjoint split “unseen category.”

Fit PCA, whitening, normalization statistics, probe hyperparameters, class weights, layer selection, margin, early stopping, and adapter hyperparameters on train/validation only. The test split is opened once after the protocol and code paths are frozen. Use a separate exploratory pilot rather than repeatedly consulting test results.

## 8. Frozen V-JEPA 2.1 extraction plan

### 8.1 Reproducible model loading

Use the existing nested [official V-JEPA 2 repository at the audited
commit](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)
at:

```text
https://github.com/facebookresearch/vjepa2
commit 204698b45b3712590f06245fbfba32d3be539812
```

The re-audit verified:

- local path `vjepa2/`;
- origin `https://github.com/facebookresearch/vjepa2.git`;
- HEAD `204698b45b3712590f06245fbfba32d3be539812`;
- imported paths `src/`, `app/`, and `hubconf.py` have no tracked, staged, or
  untracked change that can shadow imported code;
- nine unrelated evaluation/training YAMLs are user-modified and
  `checkpoints/` is untracked.

Preserved modified files:

```text
configs/eval_2_1/vitG-384/coin.yaml
configs/eval_2_1/vitG-384/diving48.yaml
configs/eval_2_1/vitG-384/ek100.yaml
configs/eval_2_1/vitG-384/in1k.yaml
configs/eval_2_1/vitG-384/jester.yaml
configs/eval_2_1/vitG-384/k400.yaml
configs/eval_2_1/vitG-384/ssv2.yaml
configs/train_2_1/vitG16/cooldown-256px-64f.yaml
configs/train_2_1/vitG16/pretrain-256px-16f.yaml
```

Preserve those changes. Do not reset, checkout over, move, or reclone this tree.
Record full status, but gate encoder reproducibility on `git diff HEAD` plus an
explicit untracked-file check over imported paths and the commit, rather than
falsely claiming the entire nested repository is clean.

Required primary ViT-B checkpoint, still absent locally:

```text
https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
Content-Length: 1,664,223,428 bytes
ETag: "be0dc26f052ae6a7476714cd53176836-199"
```

An existing checkpoint was inspected but is **not** the primary asset:

```text
vjepa2/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt
bytes: 5,151,198,524
SHA-256: 7EA9B7CB4A75D10644A8A8D42CFF9E177B10DCA8F02173F0EAF2B0BED82838C6
top-level keys: batch_size, ema_encoder, encoder, epoch, loss, lr, opt,
                predictor, scaler, world_size
ema_encoder: 302 tensors; 304,680,960 elements; width 1024
```

The SHA is a local integrity fingerprint, not proof of official authenticity in
the absence of an independently published checksum. Its filename matches the
official model map and its checkpoint structure/key/width are internally
consistent with ViT-L. It has not passed the planned strict wrapper load because
the report-only dependency check still finds required encoder imports missing.
It must remain Runpod-only/post-gate and must never be auto-selected merely
because it is present: all primary tensor shapes and adapter counts assume
ViT-B width 768.

Compute and record SHA-256 for the missing ViT-B after download. The audited official source commit
points its Hub base URL to `http://localhost:8300`; the production-URL repair is
tracked in [upstream pull request
#161](https://github.com/facebookresearch/vjepa2/pull/161). Therefore do not use
unpinned `torch.hub.load(..., pretrained=True)`. The architecture/context claims
in this plan should be read alongside the [official V-JEPA 2.1
paper](https://arxiv.org/abs/2603.14482), not as a theorem about cosine geometry.

The wrapper must:

1. import the encoder implementation from the pinned local source;
2. instantiate ViT-B/16, 384 px, tubelet 2, 16 frames, RoPE, SDPA, and the official V-JEPA 2.1 arguments;
3. load checkpoint key `ema_encoder`;
4. strip only documented `module.` and `backbone.` prefixes;
5. load with `strict=True`;
6. instantiate no predictor;
7. freeze every parameter, call `eval()`, and run under `torch.inference_mode()`;
8. record source commit, checkpoint SHA, parameter count, preprocessing config, torch/CUDA versions, and device.

### 8.2 Deterministic preprocessing

Reproduce the pinned official evaluation **geometry** and record the pixel
implementation exactly. At commit `204698b...`, the video evaluation path uses
`int(crop_size * 256 / 224)` and bilinear resize (`cv2.INTER_LINEAR`); the image
evaluation path also defaults to bilinear. It does not specify bicubic resize.
For cross-platform Phase 1, use torchvision bilinear with `antialias=False` as a
preregistered, version-pinned deviation from the ndarray/OpenCV video path; do
not call it bitwise official equivalence.

- resize the RGB short side to `int(384 * 256 / 224) = 438` with
  torchvision bilinear interpolation and `antialias=False`, recording the
  library-computed integer output size and this deviation;
- take the deterministic 384×384 center crop using torchvision's integer crop
  convention and record its top/left offsets;
- convert to `[C,T,H,W]` float;
- normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`;
- no random crop, flip, color jitter, or temporal augmentation.

Interpret COCO `[x,y,w,h]` as the continuous half-open rectangle
`[x,x+w) × [y,y+h)` and rasterize box pixels by pixel-center inclusion.
Rasterize valid COCO polygon components with
`pycocotools.mask.frPyObjects`/`merge`/`decode`; record the pycocotools version.
Then resize binary region masks with `InterpolationMode.NEAREST_EXACT` to the
same realized geometry and apply the same crop offsets as RGB. This mask rule is
ours, not an upstream V-JEPA guarantee. Never bilinearly interpolate categorical masks.
Area-average the resulting binary mask only when reducing to tubelet occupancy;
this is where fractional weights are introduced. Store interpolation modes,
antialias flag, realized resize size, crop offsets, box convention, and
torchvision version, and render contact sheets with overlays for a fixed QA
subset.

### 8.3 Expected tensor flow

| Stage | Dtype | Shape |
|---|---|---|
| Decoded source | uint8 | `[B, F, 1408, 1408, 3]` |
| Sampled observation | uint8 | `[B, 16, 1408, 1408, 3]` |
| Per-frame boxes | FP32 | `[B, 16, 4]` in source pixels |
| Per-frame masks | bool/uint8 | `[B, 16, 1408, 1408]` |
| Exact sampled time deltas | FP32 seconds | `[B, 15]` |
| Raw/scaled motion descriptor | FP32 | `[N_obs, K_M]` within one complete backend/schema, plus separate quality/provenance table |
| Signed + absolute pair changes | FP32 | two `[N_triplet, 2, K_M]` arrays for anchor–nuisance/state |
| Signed-change mismatch, severity mismatch, common level | FP32 | three `[N_triplet]` arrays |
| Preprocessed RGB | BF16 forward | `[B, 3, 16, 384, 384]` |
| Transformed mask | FP32 | `[B, 16, 384, 384]` |
| Encoder tokens | BF16 | `[B, 4608, 768]` |
| Token grid | BF16 | `[B, 8, 24, 24, 768]` |
| Occupancy grid | FP32 | `[B, 8, 24, 24]` |
| Pooled descriptor | FP32 | `[B, 768]` per layer/pool |
| Feature cache | FP32 | `[N_obs, L, P, 768]` plus Parquet index |
| Triplet gather | FP32 | `[B_triplet, 3, 768]` |
| Adapted features | FP32 | `[3 B_triplet, 768]` |

For static 1280×720 JPGs, replace the video-specific source shapes with
`[B,1,720,1280,3]` in `[B,T,H,W,C]` order and a `[B,576,768]` image-token
output after the 384×384 crop (`24 × 24 = 576` tokens).

### 8.4 Pooling variants

For every curated observation, extract from one encoder forward:

- `box`: rasterized verified boxes;
- `full`: all tokens;
- `context_tokens`: complement of a dilated object region.

Create and freeze a `mask_available` observation subset before feature
extraction. On that subset only, extract `mask` occupancy features into a
distinct immutable run/index. Compare mask versus box only on their common rows;
never silently drop box-only observations or merge mask and box populations.

Run a second forward for:

- `object_pixel_erased_mean`: replace a dilated object region with a fixed,
  label-independent RGB mean before preprocessing;
- `object_pixel_erased_blur`: optional sensitivity control using a fixed blur
  rule.

Object-pixel erasure can create out-of-distribution boundaries and preserves a
mask-shaped silhouette. Report mean-fill plus blur/inpainting sensitivity and
context-token results; do not call any of them object-free.

Every immutable feature run has its own index. A read-only catalog binds each
human-readable `feature_key` (for example
`vjepa21b/layer11/box/original/all`) to the full backbone source commit,
checkpoint SHA-256, preprocessing hash, layer, pool, input control, subset, run
ID and artifact hashes. A key may never be rebound or overwritten. Every
downstream CLI must name one frozen key; hidden/default pool or layer selection
is forbidden.

Reject pooled observations with non-finite features, zero region mass, too few occupied tokens, or a region clipped out by preprocessing. Set the minimum occupancy rule on the curation/pilot split and freeze it before test.

### 8.5 Layer policy

Preregister the final ViT-B layer as primary. Cache pooled outputs from layers `[2,5,8,11]` in one forward only if the 6 GiB smoke test passes. Intermediate-layer selection is validation-only and is invoked when final-layer probes are weak. Do not select the best test layer.

## 9. Diagnostics, probes, and controls

### 9.1 Raw geometry report

For every pool/layer variant, write:

- group-macro SNS-all and motion-matched SNS with tie rates;
- motion-matched coverage, independent-group count, signed/severity motion
  mismatches, \(\overline M\), role-conditioned component distributions/overlap
  and caliper sensitivity;
- raw paired margin and normalized margin;
- `d_n` and `d_s` distributions;
- 95% cluster-bootstrap confidence intervals;
- breakdown by state family, nuisance type/severity, hand presence, visibility, and direction;
- per-object/transition rows for error audit;
- split-specific tables: pre-test reports contain train/validation only; the
  locked final bundle adds test without pooling it with either selection split.

Use the coarsest dependency unit for resampling, normally source video/physical object. If a cluster contributes several transitions, resample the cluster once with all its observations.

### 9.2 Linear and MLP probes

Primary linear probe:

- multinomial logistic regression;
- two-stage dependency-group sampling plus inverse-class weighting (or
  calibrated/IPF weights), with realized group/class mass reported;
- validation-selected `C` from a small fixed grid;
- one model per shared state family or a carefully defined shared ontology;
- no object/category identifiers as inputs.

Primary MLP probe:

```text
768 -> 256 -> number_of_states
LayerNorm -> GELU -> dropout 0.1
AdamW, FP32, validation early stopping
```

Use a group-balanced sampler for the MLP. Report dependency-group-macro balanced
accuracy and macro-F1 with source-video-clustered intervals against majority,
stratified-random, and shuffled-label controls. A probe trained on `pre` versus
`post`, absolute timestamp, or TAS action phase is invalid for H-info.

### 9.3 Shortcut baselines

Run the following using the same split:

| Control | What it tests |
|---|---|
| Full-frame pooled feature | Scene, hands, action, and procedural context. |
| Box-pooled feature | Coarse object localization with background leakage. |
| Mask-pooled feature | Best available object-region readout; still globally contextual. |
| Context-token feature | Whether non-object tokens alone carry state-correlated evidence. |
| Object-pixel-erased re-encoding | Stronger but OOD-prone test of scene/hand/procedure shortcuts; mask silhouette remains. |
| Mask silhouette only | State leakage through mask geometry. |
| Box coordinates/area/aspect only | State leakage through annotation geometry. |
| Motion-only role models | Frozen motion-control artifact only: signed endpoint/change/severity features plus scalar-threshold and regularized-logistic baselines. Quality fields are reported separately, not silently used as motion. Fit scaler/model/threshold on train/validation groups and test once with grouped permutation/CI. Pixel difference alone is rejected because it mixes state appearance with motion. |
| Category-prior only | Train-split state frequency by manually normalized object category, with global backoff for unseen categories. |
| Metadata only | Timestamp, phase, temporal gap, nuisance tags, hand presence, object/category prior, and sequence/scene/video statistics where meaningful. |
| Temporal-order only | Whether `pre/post` chronology predicts labels. |

The desired result is not merely `object > background`; it is that the
object-region result remains materially above all label-independent shortcut
baselines and survives matched temporal, motion and context sampling. If the
motion-only role model is strong on the exact matched confirmatory subset, or
that subset lacks motion common support, resample before interpreting
H-geometry. Strong prediction on all eligible/unmatched rows is a secondary
stress warning, not by itself a kill condition. A weak model does not prove the
absence of all motion confounding; the matched estimand remains primary.

### 9.4 Shuffled-label and role controls

Two distinct controls must not be conflated. The required **shuffled-label
negative control** is:

- create at least 99 deterministic shuffled training assignments;
- treat each stable segment as a label block; within **train only** and
  `(state_family, object_category)` stratum, permute the vector of segment
  labels among blocks once and assign that value to every observation in the
  block. This preserves the segment-level class multiset while breaking its
  link to features; strata without at least two represented classes are
  excluded from negative-control training and scoring rather than left with
  true labels or globally shuffled;
- train on permuted train labels;
- reuse the corresponding true-label probe's already frozen architecture,
  regularization and epoch/stopping choice (or use an inner-train selection);
  never select a shuffled control on the same true validation score it reports;
- evaluate once against untouched true-label validation/test targets as a
  leakage check;
- report it as a negative-control distribution, not an exchangeability-valid
  permutation p-value or the null quantile for H-info.

If a formal randomization null/p-value is reported, define a separate
group/block exchangeability scheme, apply the same permutation consistently to
all held-out targets used by the frozen training/selection procedure, and run
the full statistic for every permutation. Otherwise H-info uses a preregistered
chance/effect-size criterion plus the negative control, not a formal
permutation-null threshold.

Run this procedure for both linear and MLP probes. Never shuffle individual
frames inside a segment or across splits.

For the adapter:

- shuffling a metadata column after triplets exist is invalid;
- either rebuild train triplets from group-permuted state labels or swap nuisance/state roles in 50% of train triplets;
- retain true validation/test triplets;
- repeat across seeds.

Meaningful true-label generalization from a role-shuffled adapter is evidence of leakage or an invalid evaluation path.

## 10. Residual-adapter experiment and fair baselines

Run the adapter only if H-info has evidence and H-geometry has validation
headroom. Train and evaluate every adapter/baseline on the identical rows with
`temporal_matched=true` and `motion_matched=true`; otherwise an adapter can
appear to improve merely by learning role-correlated motion suppression. Use
five seeds if the pilot size supports them.

Minimum comparison set evaluated on the same triplets:

1. identity/no adaptation;
2. centered/PCA/whitened readouts;
3. learned positive diagonal feature scaling;
4. linear bottleneck residual;
5. non-residual bottleneck MLP with the same output dimension and comparable parameter count;
6. residual bottleneck adapter.

Identity/centering/PCA/whitening are train-fit or fixed readout controls and do
not use a margin loss. Learned positive-diagonal, linear-residual,
non-residual-MLP, and residual-bottleneck metric transforms share the ordinary
margin ranking loss, train rows, validation rule, hyperparameter search budget,
maximum epochs, stopping rule and seeds. Classification and
supervised-contrastive adapters may be secondary different-objective controls,
not same-objective baselines.

Do not interpret a training-SNS gain as evidence. The required evidence is improvement on held-out object/video groups, recurring state families, nuisance strata, and a manually verified test set.

## 11. Identity, preservation, and collapse diagnostics

Compute all collapse statistics on unique, object-balanced observations, never on duplicated triplet positions.

### Identity/persistence retrieval

- query and gallery must not contain overlapping frames;
- positive: same verified physical-object track in another observation;
- within-video positives are labeled **track persistence**, not physical
  identity generalization;
- hard negatives: different objects from the same source video/context and
  matched procedural phase, temporal gap, category, and state family where
  possible, plus same-category same-state negatives;
- compare region, context-token, and object-pixel-erased retrieval so globally
  contextual transformer tokens cannot pass only by recognizing the episode;
- claim cross-video identity only for independently verified recurring assets,
  with source videos disjoint between query and gallery;
- report Recall@1, Recall@5, MRR, and per-family results;
- report same-state nuisance retrieval separately from cross-state same-identity retrieval.

For static COCO preflight, use the namespaced
`source_family + sequence_id + category_id + solo_instance_id` key and report
it as same-sequence identity only. Do not call it cross-video persistence.

### Geometry preservation

On a fixed, train-independent stratified set of pairs, report:

- raw versus adapted pairwise-cosine Spearman correlation;
- median and 95th-percentile residual ratio `||v-z||/(||z||+eps)`;
- identity retrieval delta;
- state ordering delta.

### Collapse and state-prototype collapse

For unnormalized `v` and normalized `u`, report:

- feature norm and per-dimension standard-deviation distributions;
- covariance eigenvalue spectrum;
- effective rank

\[
r_{eff}=\exp\left(-\sum_j p_j\log p_j\right),
\qquad
p_j=\frac{\lambda_j}{\sum_k\lambda_k};
\]

- participation ratio `(sum lambda)^2 / sum lambda^2`;
- mean off-diagonal cosine;
- within-state cross-instance variance;
- within-instance versus between-instance retrieval.

An adapter can maintain global rank while collapsing each state to a prototype, so both spectral and identity-conditioned diagnostics are required.

## 12. Statistical protocol and decision rules

- Freeze the primary pool, layer, state families, sampling rule, motion backend,
  componentwise quality rules, robust scaler, signed/severity calipers, split and metrics before opening
  test.
- Use paired comparisons because raw and adapted results share triplets.
- Cluster-bootstrap `dependency_group_id` connected components (source video,
  object/asset, transition, and duplicate-hash links) for 95% confidence
  intervals.
- Report point estimates, intervals, number of clusters, number of observations, and number of derived triplets.
- Report effect sizes and raw distributions, not only p-values.
- Correct formal multiplicity when several layers/pools are promoted to confirmatory hypotheses; otherwise label them exploratory.
- Use a human/adjudication ceiling based on double-reviewed triplets. Geometry is not expected to reach 1.0 when labels or observability are uncertain.
- Matching is the primary motion control. A regression of pair distance on
  state/motion/viewpoint/occlusion is exploratory because pairs share endpoints,
  covariates are noisy and the state coefficient is not causally identified;
  use crossed/grouped inference only when enough clusters exist.

Pre-register practical decision tolerances after the pilot and before test. Recommended initial engineering values are:

| Gate | Continue when |
|---|---|
| Data | At least 30 valid pilot transitions across at least 3 recurring families; confirmatory sample is power-justified. |
| Motion validity | The exact quality-passing temporal-and-motion confirmatory subset retains a power-justified number of dependency groups; signed and severity balance pass sensitivity checks; matched-subset validation `abs(AUROC-0.5)` is not materially above its dependency-group permutation null. All-eligible AUROC is secondary. |
| H-info | Probe lower CI exceeds balanced chance by a preregistered practical margin and shuffled-training negative controls do not generalize. If a formal exchangeability-valid block randomization test is implemented, its frozen statistic also clears that null. |
| H-geometry headroom | Motion-matched SNS upper CI is below `min(0.90, adjudication_ceiling - 0.05)`, with adequate coverage and failures in at least two families; SNS-all is reported separately. |
| H-method | Residual adapter paired SNS gain has lower CI above 0 and point gain at least 0.03; simpler matched transforms do not fall within a 0.02 equivalence band. |
| H-preserve | In paired dependency-cluster inference, the lower 95% CI for `R@1_adapted-R@1_raw` is at least `-0.02`, and the lower 95% CI for the effective-rank ratio is at least `0.90`, with no state-prototype collapse. |
| Shortcut | Object-region metrics exceed every metadata/object-pixel-erased control by a preregistered practical margin. |

These are internal gates, not universal scientific constants. Revise them only on pilot data and document the revision before test.

## 13. Local GPU plan

Measured `llm` environment:

```text
Python 3.12.12
torch 2.10.0+cu130
torchvision 0.25.0+cu130
GPU NVIDIA GeForce RTX 4050 Laptop GPU
VRAM 6,141 MiB
compute capability 8.9
CUDA available; BF16 supported
```

Local extraction settings:

```text
model: V-JEPA 2.1 ViT-B/16 only
frames: 16
resolution: 384
encoder batch: 1
workers on Windows: 0 initially
forward precision: BF16 autocast
pool/cache/statistics: FP32
mode: eval + inference_mode
```

The smoke test must measure allocated/reserved peak VRAM for one full observation and every requested output layer. If the primary 16×384 protocol OOMs after batch 1, inference mode, BF16, SDPA, and one-layer output are verified, migrate extraction to Runpod. Do not silently lower resolution or frame count and then report it as the primary experiment.

Cached-feature probes and the adapter run locally in FP32; their memory footprint is small. Gradient checkpointing is unnecessary. ViT-L, non-JEPA confirmation, and large sweeps are forbidden until all ViT-B gates pass.

## 14. Proposed code and project structure

```text
configs/
  phase1/
    vjepa21_vitb.yaml
    motion_controls.yaml
    probe_linear.yaml
    probe_mlp.yaml
    adapter_margin_triplet.yaml
    feature_input_columns.txt

src/
  state_geometry/
    __init__.py
    config.py
    provenance.py
    data/
      schema.py
      egointeract.py
      inventory.py
      curation.py
      splits.py
      triplets.py
    backbones/
      vjepa21.py
    features/
      transforms.py
      pooling.py
      cache.py
      catalog.py
    controls/
      geometry.py
      motion.py
      shortcuts.py
      permutations.py
    evaluation/
      geometry.py
      probes.py
      bootstrap.py
      preservation.py
      collapse.py
    models/
      residual_adapter.py
    training/
      probes.py
      adapter.py
    utils/
      hashing.py
      seeding.py

scripts/
  check_phase1_dependencies.py        # created in this audit
  download_phase1_datasets.py         # existing; add revision + explicit frame-archive options
  audit_phase1_dataset.py
  build_interaction_phase_proxy.py    # confounded static pipeline smoke test only
  build_phase1_manifest.py
  validate_phase1_manifest.py
  build_phase1_splits.py
  estimate_motion_controls.py
  validate_motion_controls.py
  fit_motion_scaler_quality.py
  assemble_phase1_analysis_manifest.py
  build_phase1_candidate_triplets.py
  select_motion_calipers.py
  finalize_phase1_triplets.py
  build_phase1_release_views.py
  fetch_vjepa21_checkpoint.py
  smoke_test_vjepa21.py
  extract_features.py
  fit_geometry_controls.py
  eval_state_nuisance.py
  train_probe.py
  eval_motion_role_control.py
  run_shortcut_controls.py
  train_state_adapter.py
  eval_state_adapter.py
  record_phase1_skip.py
  freeze_phase1_selection.py
  run_locked_phase1_test.py
  build_phase1_report.py

tests/
  test_dataset_schema.py
  test_manifest_stages.py
  test_static_proxy_join.py
  test_polygon_validation.py
  test_joint_transforms.py
  test_pooling.py
  test_split_leakage.py
  test_dependency_groups.py
  test_triplets.py
  test_temporal_matching.py
  test_motion_vector_residual.py
  test_motion_signed_change_matching.py
  test_motion_schema_completeness.py
  test_motion_nonuniform_dt.py
  test_motion_train_only_scaler.py
  test_motion_scaler_floor.py
  test_motion_encoder_crop.py
  test_motion_matching.py
  test_motion_quality_failures.py
  test_motion_group_role_permutation.py
  test_block_permutations.py
  test_pca_train_only.py
  test_ordering_loss.py
  test_adapter_gradients.py
  test_cache_alignment.py
  test_nested_group_metrics.py
  test_release_view_sealing.py
  test_optional_component_resolution.py
  test_final_test_lock.py

vjepa2/                              # existing nested official source
  src/ + app/                        # imported paths match audited commit
  checkpoints/
    vjepa2_1_vitl_dist_vitG_384.pt   # existing; Runpod-only/post-gate

checkpoints/
  vjepa2_1_vitb_dist_vitG_384.pt     # required primary; currently absent

artifacts/
  phase1/
```

Use memory-mappable FP32 `.npy` arrays plus a Parquet index rather than one opaque pickle. Every artifact directory must contain resolved configuration, input hashes, source hashes/revisions, seed, command line, environment summary, and completion status.

## 15. Implementation order

1. Keep the current research scope locked to Phase 1.
2. Run and satisfy the dependency checker in `llm`; do not create an environment.
3. Implement dataset inventory and staged schema validation.
4. Pin the dataset revision, make the frame-archive acquisition explicit, and hash the payload manifest.
5. Implement static JPG/COCO alignment, 128-bit ID handling, polygon validation,
   duplicate grouping, and transform/pooling unit tests.
6. Build the separately labeled NAO↔hand-HOI interaction-phase **data join** as
   a schema/shortcut-covariate preflight only; defer its feature/pooling run until
   the primary checkpoint has strictly loaded.
7. Build feature-blind video curation candidates from TAS intervals and weak IA metadata.
8. Curate and double-review the 30-transition protocol/power pilot; do not form
   or inspect a nominal confirmatory test split from this pilot.
9. Use pilot variance to set the power target, expand curation to the powered
   cohort, then freeze `curated_observations.parquet`.
10. Validate the curation manifest; stop on any missing state, identity,
    observability, hash, or aligned-box field.
11. Split object/video/dependency groups before constructing triplets.
12. Estimate and QA label/embedding-blind motion measurements on the exact
    encoder frames and spatial crop, with homogeneous schema and provenance.
13. Fit robust motion scaling and componentwise quality rules, assemble the
    joined analysis manifest, and build the exact feature-blind temporal
    candidate-triplet universe.
14. Select signed/severity calipers on train/validation candidates only; stop if
    common support is inadequate, finalize triplets, then emit train/validation,
    label-redacted feature-input, and sealed test-target/role views.
15. Run the validation-only motion-role gate before significant V-JEPA GPU work.
16. Verify the existing V-JEPA source (including staged/untracked imported
    paths), acquire/hash the missing primary ViT-B checkpoint, and strictly load
    the encoder only.
17. Run a 1-clip and fixed 32-observation extraction smoke test, then run the
    static interaction-phase feature/pooling/shortcut preflight.
18. Extract immutable ViT-B final-layer `box/full/context` features for all
    curated rows, mask features for the frozen mask subset, and object-pixel-
    erased variants into distinct run IDs; build the read-only feature catalog.
19. On train/validation only, run SNS-all, motion-matched SNS/margins,
    PCA/whitening controls, probes, shortcuts, and shuffled-label controls.
20. Freeze the primary feature key, transforms, probe checkpoints, thresholds,
    and all H-info/H-geometry selections without test access; resolve the mask
    component as selected or evidence-backed skipped.
21. Only if validation gates pass, train/select the tiny residual ordering
    adapter and same-loss learned metric baselines on the same confirmatory rows.
22. Run validation-only preservation/collapse checks, or record the failed gate
    that skips adapters, then freeze every selected/skipped component and hash.
23. Invoke one atomic locked evaluator that creates its permanent access marker
    immediately before opening test once and computes raw,
    control, probe, shortcut, motion, adapter, preservation, and collapse results.
24. Build a single immutable Phase 1 results report from that locked bundle.
25. Consider Runpod ViT-L/DINO/TrackMAE descriptive confirmation only after the
    local ViT-B conclusion is stable.

## 16. Exact CLI execution sequence

All planned scripts accept forward-slash relative paths and `pathlib` paths.
The argument sequences are shell-agnostic; the launcher is intentionally
different: local Windows/Linux Conda uses `conda run -n llm python`, while a
Runpod base image uses its already-active `python`. Commands below show the
required local launcher and are single-line intentionally.

### 16.1 Commands valid now

```bash
conda run -n llm python scripts/check_phase1_dependencies.py
```

At this snapshot the checker reports only these necessary installs:

```bash
conda run -n llm python -m pip install "scikit-learn>=1.5,<2" "scipy>=1.13,<2" "av>=12,<17" "timm>=1.0,<2" "einops>=0.8,<1" "pycocotools>=2.0.8,<3" "pytest>=8,<10"
```

`pycocotools` supplies canonical COCO polygon-to-mask semantics; do not replace
it with an unstated Pillow/OpenCV rasterizer. Re-run the checker after
installation. It imports every required module as well as checking distribution
versions, and performs no install itself.

The current payload already contains the frame archive and extracted JPGs. On a
clean machine, the existing downloader must first be extended with the planned
arguments below; until then, this command is a specification, not an executable
claim:

```bash
conda run -n llm python scripts/download_phase1_datasets.py --revision 313d1ef6586571d6ce1fe85581f690c507110fea --include-frame-archive --verify-frame-archive-sha256 9962d4c7dfe5ea2af05dcf8bf2c5c73f82a9ba1adc80f62a8066589273371c81
```

### 16.2 Source/checkpoint setup after the corresponding scripts exist

```bash
git -C vjepa2 remote get-url origin
git -C vjepa2 rev-parse HEAD
git -C vjepa2 diff HEAD --exit-code -- src app hubconf.py
git -C vjepa2 status --porcelain --untracked-files=all -- src app hubconf.py
conda run -n llm python scripts/fetch_vjepa21_checkpoint.py --url https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt --expected-bytes 1664223428 --output checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

The first two outputs must equal the official origin and audited commit above;
the imported-path diff command must exit zero and the path-scoped porcelain
command must print nothing. Full `git status --short` is
still recorded so the nine preserved YAML edits and untracked ViT-L checkpoint
are visible. Do not require a globally clean nested repository. If Git reports a
safe-directory ownership warning, resolve the absolute repository path and use
a per-command `-c safe.directory=<resolved-path>` override; do not mutate a
global Git allowlist.

### 16.3 Data gate

```bash
conda run -n llm python scripts/audit_phase1_dataset.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --output-root artifacts/phase1/00_audit
conda run -n llm python scripts/build_phase1_manifest.py --dataset-root datasets/phase1/EgoInteract --output artifacts/phase1/01_manifests/candidates.parquet
conda run -n llm python scripts/validate_phase1_manifest.py --manifest artifacts/phase1/01_manifests/candidates.parquet --require-state-labels --require-physical-object-ids --require-observability --require-aligned-regions --output-root artifacts/phase1/01_manifests/validation
```

The last command should fail on automatically generated current metadata.
Manual curation and the powered-cohort expansion produce
`curated_observations.parquet`; only then continue:

```bash
conda run -n llm python scripts/validate_phase1_manifest.py --stage curated --manifest artifacts/phase1/01_manifests/curated_observations.parquet --require-state-labels --require-physical-object-ids --require-observability --require-aligned-boxes --output-root artifacts/phase1/01_manifests/curation_validation
conda run -n llm python scripts/build_phase1_splits.py --observations artifacts/phase1/01_manifests/curated_observations.parquet --build-connected-groups physical_object_id,source_video_id,transition_id,verified_asset_group_id,duplicate_group_id --null-group-values-no-edge --group dependency_group_id --ratios 0.70,0.15,0.15 --stratify state_family,state_label --stratify-multilabel nuisance_tags --stratification-objective deterministic_iterative_group_balance --report-realized-stratum-mass --seed 20260820 --output artifacts/phase1/01_manifests/splits.parquet
conda run -n llm python scripts/estimate_motion_controls.py --config configs/phase1/motion_controls.yaml --observations artifacts/phase1/01_manifests/curated_observations.parquet --splits artifacts/phase1/01_manifests/splits.parquet --backend reviewed_stationary_region_kinematics --encoder-preprocess-config configs/phase1/vjepa21_vitb.yaml --use-exact-sampled-frames --time-unit seconds --require-background-registration-bound --reject-nonstationary-background --output artifacts/phase1/02_motion/motion_observations.parquet
conda run -n llm python scripts/validate_motion_controls.py --stage raw --config configs/phase1/motion_controls.yaml --motion artifacts/phase1/02_motion/motion_observations.parquet --observations artifacts/phase1/01_manifests/curated_observations.parquet --splits artifacts/phase1/01_manifests/splits.parquet --require-complete-homogeneous-schema --no-fit --output-root artifacts/phase1/02_motion/raw_validation
conda run -n llm python scripts/fit_motion_scaler_quality.py --config configs/phase1/motion_controls.yaml --motion-raw artifacts/phase1/02_motion/motion_observations.parquet --observations artifacts/phase1/01_manifests/curated_observations.parquet --splits artifacts/phase1/01_manifests/splits.parquet --fit-split train --quality-select-split validation --scaler-denominator max_iqr_native_floor --drop-train-constant-features --no-test-access --output artifacts/phase1/02_motion/motion_validated.parquet
conda run -n llm python scripts/assemble_phase1_analysis_manifest.py --curated-observations artifacts/phase1/01_manifests/curated_observations.parquet --splits artifacts/phase1/01_manifests/splits.parquet --motion-validated artifacts/phase1/02_motion/motion_validated.parquet --output artifacts/phase1/01_manifests/internal_analysis_observations.parquet
conda run -n llm python scripts/validate_phase1_manifest.py --stage analysis --manifest artifacts/phase1/01_manifests/internal_analysis_observations.parquet --require-state-labels --require-physical-object-ids --require-observability --require-aligned-boxes --require-split-and-dependency-groups --require-complete-motion-provenance --output-root artifacts/phase1/01_manifests/analysis_validation
conda run -n llm python scripts/build_phase1_candidate_triplets.py --observations artifacts/phase1/01_manifests/internal_analysis_observations.parquet --base-eligibility verified_stable_state_nuisance --max-temporal-gap-mismatch-frames 2 --no-motion-caliper --output artifacts/phase1/01_manifests/internal_candidate_triplets.parquet --seed 20260820
conda run -n llm python scripts/select_motion_calipers.py --candidates artifacts/phase1/01_manifests/internal_candidate_triplets.parquet --motion artifacts/phase1/02_motion/motion_validated.parquet --fit-split train --select-split validation --signed-caliper-grid 0.25,0.50,1.00,1.50 --severity-caliper-grid 0.25,0.50,1.00,1.50 --select-smallest-calipers-with-min-coverage 0.50 --no-test-access --output artifacts/phase1/02_motion/caliper_selection.json
conda run -n llm python scripts/finalize_phase1_triplets.py --candidates artifacts/phase1/01_manifests/internal_candidate_triplets.parquet --motion artifacts/phase1/02_motion/motion_validated.parquet --calipers artifacts/phase1/02_motion/caliper_selection.json --require-homogeneous-motion-schema --apply-without-refit --all-selector eligible_base --matched-selector confirmatory_eligible --output artifacts/phase1/01_manifests/internal_triplets.parquet
conda run -n llm python scripts/build_phase1_release_views.py --internal-observations artifacts/phase1/01_manifests/internal_analysis_observations.parquet --internal-triplets artifacts/phase1/01_manifests/internal_triplets.parquet --output-trainval-observations artifacts/phase1/01_manifests/analysis_trainval.parquet --output-feature-inputs artifacts/phase1/01_manifests/feature_inputs.parquet --feature-input-column-allowlist configs/phase1/feature_input_columns.txt --reject-extra-feature-input-columns --output-trainval-triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --output-sealed-test-targets artifacts/phase1/01_manifests/sealed_test_targets.parquet --output-sealed-test-triplets artifacts/phase1/01_manifests/sealed_test_triplets.parquet --label-redact-feature-inputs --suppress-test-summaries-until-locked --hash-lock-outputs --output-manifest artifacts/phase1/01_manifests/release_views.json
conda run -n llm python scripts/eval_motion_role_control.py --motion-controls artifacts/phase1/02_motion/motion_validated.parquet --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --models scalar_threshold,logistic --logistic-features endpoints,signed_change,absolute_change --primary-subset confirmatory_eligible --secondary-subset eligible_base --fit-split train --select-split validation --evaluate-split validation --permutations 999 --permutation-group dependency_group_id --permutation-action swap_all_paired_roles_in_group --bootstrap-group dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/02_motion/role_control_validation
```

The minimum local backend deliberately accepts only double-reviewed stationary
backgrounds and derives kinematics from verified per-sampled-frame boxes/masks.
It still requires a quantitative background-registration bound; setting
`g_k=0` is an audited design restriction, not an estimated camera correction.
Any moving-camera observation is excluded until a separately pinned,
hashed and quality-tested global-compensated tracking backend exists. CoTracker3
is an optional measurement backend after such pinning; it is not an identity
oracle or a Phase-1 training target. The held-out test motion-control command is
run only inside the final locked evaluation, not during this validation gate.

The available static data may exercise the code path before manual state
curation, but only under an explicit proxy namespace:

```bash
conda run -n llm python scripts/build_interaction_phase_proxy.py --nao-json datasets/phase1/EgoInteract/data/annotations/nao/coco_annotations_egointeract.json --nao-frame-root datasets/phase1/EgoInteract/data/frames/frames/nao --hoi-json datasets/phase1/EgoInteract/data/annotations/hoi/coco_annotations_hand_egointeract.json --hoi-frame-root datasets/phase1/EgoInteract/data/frames/frames/hoi_enigma --phase-labels pre_contact,contact --min-observations-per-phase 2 --require-contacting-hand --deduplicate-by-sha256 --reject-cross-label-hash-conflicts --output-root artifacts/phase1/01_proxy
conda run -n llm python scripts/build_phase1_splits.py --observations artifacts/phase1/01_proxy/observations.parquet --build-connected-groups asset_proxy_id,sequence_id,duplicate_group_id --null-group-values-no-edge --group dependency_group_id --ratios 0.70,0.15,0.15 --stratify interaction_phase --stratification-objective deterministic_iterative_group_balance --report-realized-stratum-mass --seed 20260820 --output artifacts/phase1/01_proxy/splits.parquet
```

Here `asset_proxy_id` conservatively groups the object-category
`solo_instance_id` across sequences; sequence is an additional indivisible
group. The script must label the output `interaction_phase`, never `state`, and
must report the 377 duplicate pairs, invalid polygons, hand/occlusion cues, and
time index as shortcut variables. It should output proxy triplets only for
pipeline testing, with no contribution to H-info/H-geometry decisions.

### 16.4 Frozen extraction

The minimal powered experiment preregisters `layer11/box/original/all` as its
primary feature key. Mask pooling is a supported-subset analysis, never an
implicit replacement population. The `all` cache covers every base-eligible
observation, including rows that later fail the confirmatory motion-quality
predicate, so SNS-all remains a genuine broader stress estimand. Extraction
requires the frozen motion provenance columns but does not filter on their pass
flag; only the matched selector applies that flag.

```bash
conda run -n llm python scripts/smoke_test_vjepa21.py --vjepa-source-root vjepa2 --checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --observations artifacts/phase1/01_manifests/feature_inputs.parquet --require-label-redacted-input --require-motion-provenance --subset-query motion_quality_pass=true --frames 16 --resolution 384 --batch-size 1 --precision bf16 --output-root artifacts/phase1/02_smoke/vjepa21b_16x384
conda run -n llm python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_manifests/feature_inputs.parquet --require-label-redacted-input --require-motion-provenance --no-motion-quality-filter --layers 11 --pools box,full,context_tokens --input-control original --subset-name all --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id vjepa21b_l11_all_original --output-root artifacts/phase1/03_features/vjepa21b_l11_all_original --catalog artifacts/phase1/03_features/catalog.parquet
conda run -n llm python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_manifests/feature_inputs.parquet --require-label-redacted-input --require-motion-provenance --no-motion-quality-filter --subset-query mask_available=true --layers 11 --pools mask,box --input-control original --subset-name mask_common --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id vjepa21b_l11_mask_common --output-root artifacts/phase1/03_features/vjepa21b_l11_mask_common --catalog artifacts/phase1/03_features/catalog.parquet --component-resolution artifacts/phase1/03_features/mask_component_resolution.json
conda run -n llm python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_manifests/feature_inputs.parquet --require-label-redacted-input --require-motion-provenance --no-motion-quality-filter --layers 11 --pools box --input-control object_pixel_erased_mean --subset-name all --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id vjepa21b_l11_all_pixel_erased_mean --output-root artifacts/phase1/03_features/vjepa21b_l11_all_pixel_erased_mean --catalog artifacts/phase1/03_features/catalog.parquet
```

The mask extraction command is conditional. If no powered, frozen mask/common
subset exists, do not fabricate or silently drop rows; replace that command with:

```bash
conda run -n llm python scripts/record_phase1_skip.py --component mask_common --reason no_powered_frozen_mask_common_subset --evidence artifacts/phase1/01_manifests/curation_validation/power_report.json --output artifacts/phase1/03_features/mask_component_resolution.json
```

Exactly one mask resolution is valid: a successful extraction writes
`status=selected` and the two common-row feature keys to that path, while the
alternative writes `status=skipped` with a hashed reason and evidence.

For the static proxy preflight, run the exact extraction below; do not request
symmetric mask pooling because the NAO side has no masks.

```bash
conda run -n llm python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_proxy/observations.parquet --layers 11 --pools box,full,context_tokens --input-control original --subset-name proxy --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id interaction_phase_proxy_l11 --output-root artifacts/phase1/01_proxy/features/interaction_phase_proxy_l11 --catalog artifacts/phase1/01_proxy/features/catalog.parquet
conda run -n llm python scripts/run_shortcut_controls.py --feature-catalog artifacts/phase1/01_proxy/features/catalog.parquet --feature-key vjepa21b/layer11/box/original/proxy --observations artifacts/phase1/01_proxy/observations.parquet --triplets artifacts/phase1/01_proxy/proxy_triplets.parquet --splits artifacts/phase1/01_proxy/splits.parquet --controls context_tokens,box_geometry,metadata,temporal,hand_presence --evaluate-split validation --no-test-access --bootstrap-group dependency_group_id --seed 20260820 --output-root artifacts/phase1/01_proxy/shortcut_report/validation
```

### 16.5 Diagnostics and probes

```bash
conda run -n llm python scripts/fit_geometry_controls.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --variants center,pca64,pca128,pca256,random64,random128,random256,whiten64,whiten128,whiten256,full_pca_invariant --max-k-from-train-rank --min-unique-train-per-dimension 2 --record-skipped-k --whitening-shrinkage-grid 0.0001,0.001,0.01 --shrinkage-units mean_positive_train_eigenvalue --eigenvalue-floor-relative 0.000001 --fit-split train --select-split validation --fit-balance dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/04_controls/vjepa21b_l11_box
conda run -n llm python scripts/eval_state_nuisance.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --controls artifacts/phase1/04_controls/vjepa21b_l11_box --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --all-selector eligible_base --matched-selector confirmatory_eligible --validate-match-provenance --report sns_all,sns_motion_matched,raw_margin,normalized_margin,motion_strata --select-control-on validation --evaluate-split validation --no-test-access --bootstrap-group dependency_group_id --seed 20260820 --output-root artifacts/phase1/05_geometry/vjepa21b_l11_box_validation
conda run -n llm python scripts/eval_state_nuisance.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/mask/original/mask_common --comparison-feature-key vjepa21b/layer11/box/original/mask_common --common-rows-only --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --all-selector eligible_base --matched-selector confirmatory_eligible --validate-match-provenance --report sns_all,sns_motion_matched,raw_margin,normalized_margin --evaluate-split validation --no-test-access --bootstrap-group dependency_group_id --seed 20260820 --output-root artifacts/phase1/05_geometry/vjepa21b_l11_mask_common_validation
conda run -n llm python scripts/train_probe.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --probe linear --fit-split train --select-split validation --evaluate-split validation --weighting ipf_class_and_dependency_group --report-realized-weight-mass --metric-aggregation dependency_group_id --bootstrap-group dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/06_probes/linear/vjepa21b_l11_box
conda run -n llm python scripts/train_probe.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --probe mlp --fit-split train --select-split validation --evaluate-split validation --sampler dependency_group_then_observation --class-weight inverse_train_frequency --report-realized-sampling-mass --metric-aggregation dependency_group_id --bootstrap-group dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/06_probes/mlp/vjepa21b_l11_box
conda run -n llm python scripts/train_probe.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --probe linear --fit-split train --evaluate-split validation --label-control shuffled_train_stable_segment_blocks --permutation-strata state_family,object_category_manual --permutations 99 --locked-hyperparameters-from artifacts/phase1/06_probes/linear/vjepa21b_l11_box/selection.json --no-hyperparameter-selection --exclude-unpermutable-strata-from-train-and-score --score-against true_validation_labels --control-interpretation negative_control_not_formal_null --weighting ipf_class_and_dependency_group --metric-aggregation dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/06_probes/shuffled_linear/vjepa21b_l11_box
conda run -n llm python scripts/train_probe.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --probe mlp --fit-split train --evaluate-split validation --label-control shuffled_train_stable_segment_blocks --permutation-strata state_family,object_category_manual --permutations 99 --locked-hyperparameters-from artifacts/phase1/06_probes/mlp/vjepa21b_l11_box/selection.json --no-hyperparameter-selection --exclude-unpermutable-strata-from-train-and-score --score-against true_validation_labels --control-interpretation negative_control_not_formal_null --sampler dependency_group_then_observation --metric-aggregation dependency_group_id --no-test-access --seed 20260820 --output-root artifacts/phase1/06_probes/shuffled_mlp/vjepa21b_l11_box
conda run -n llm python scripts/run_shortcut_controls.py --feature-catalog artifacts/phase1/03_features/catalog.parquet --primary-feature-key vjepa21b/layer11/box/original/all --control-feature-keys vjepa21b/layer11/full/original/all,vjepa21b/layer11/context_tokens/original/all,vjepa21b/layer11/box/object_pixel_erased_mean/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --controls full,context_tokens,object_pixel_erased_mean,box_geometry,category_prior,metadata,temporal,hand_presence --optional-control-from-resolution mask_geometry=artifacts/phase1/03_features/mask_component_resolution.json --mask-common-row-only --evaluate-split validation --no-test-access --bootstrap-group dependency_group_id --seed 20260820 --output-root artifacts/phase1/07_shortcuts/vjepa21b_l11_box_validation
```

The mask/common-row geometry command and `mask_geometry` shortcut are executed
only when `mask_component_resolution.json` has `status=selected`. When it has
`status=skipped`, omit those optional requests; the required box, full-frame,
context-token, object-pixel-erased, box-geometry, metadata and temporal controls
still run and the skip manifest is carried into the locked test.

### 16.6 Conditional adapter and final report

In every command below, `--confirmatory-only` must fail unless each selected
row has both `temporal_matched=true` and `motion_matched=true` plus valid
motion provenance; it is not a loose alias for any verified triplet.
The five training commands and validation evaluator are one conditional branch:
run them only if the frozen validation H-info/H-geometry gates authorize an
adapter comparison. They must not be run merely to make the final table complete.

```bash
conda run -n llm python scripts/train_state_adapter.py --architecture positive_diagonal --positive-parameterization softplus --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --fit-split train --select-split validation --no-test-access --sampler dependency_transition_balanced --loss margin_triplet --margin-grid 0.05,0.10,0.20 --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 --seeds 20260820,20260821,20260822,20260823,20260824 --output-root artifacts/phase1/08_adapters/positive_diagonal/vjepa21b_l11_box
conda run -n llm python scripts/train_state_adapter.py --architecture linear_residual --bottleneck 256 --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --fit-split train --select-split validation --no-test-access --sampler dependency_transition_balanced --loss margin_triplet --margin-grid 0.05,0.10,0.20 --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 --seeds 20260820,20260821,20260822,20260823,20260824 --output-root artifacts/phase1/08_adapters/linear_residual/vjepa21b_l11_box
conda run -n llm python scripts/train_state_adapter.py --architecture mlp_nonresidual --bottleneck 256 --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --fit-split train --select-split validation --no-test-access --sampler dependency_transition_balanced --loss margin_triplet --margin-grid 0.05,0.10,0.20 --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 --seeds 20260820,20260821,20260822,20260823,20260824 --output-root artifacts/phase1/08_adapters/mlp_nonresidual/vjepa21b_l11_box
conda run -n llm python scripts/train_state_adapter.py --architecture residual_bottleneck --bottleneck 256 --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --fit-split train --select-split validation --no-test-access --sampler dependency_transition_balanced --loss margin_triplet --margin-grid 0.05,0.10,0.20 --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 --seeds 20260820,20260821,20260822,20260823,20260824 --output-root artifacts/phase1/08_adapters/residual/vjepa21b_l11_box
conda run -n llm python scripts/train_state_adapter.py --architecture residual_bottleneck --bottleneck 256 --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --fit-split train --select-split validation --no-test-access --sampler dependency_transition_balanced --loss margin_triplet --role-control random_nuisance_state_swap --swap-probability 0.5 --margin-grid 0.05,0.10,0.20 --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 --seeds 20260820,20260821,20260822,20260823,20260824 --output-root artifacts/phase1/08_adapters/role_shuffled/vjepa21b_l11_box
conda run -n llm python scripts/eval_state_adapter.py --adapter-roots artifacts/phase1/08_adapters/positive_diagonal/vjepa21b_l11_box,artifacts/phase1/08_adapters/linear_residual/vjepa21b_l11_box,artifacts/phase1/08_adapters/mlp_nonresidual/vjepa21b_l11_box,artifacts/phase1/08_adapters/residual/vjepa21b_l11_box,artifacts/phase1/08_adapters/role_shuffled/vjepa21b_l11_box --feature-catalog artifacts/phase1/03_features/catalog.parquet --feature-key vjepa21b/layer11/box/original/all --observations artifacts/phase1/01_manifests/analysis_trainval.parquet --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet --confirmatory-only --evaluate-split validation --no-test-access --evaluate-preservation --evaluate-collapse --bootstrap-group dependency_group_id --paired-noninferiority-r1 -0.02 --paired-noninferiority-effective-rank-ratio 0.90 --component-resolution artifacts/phase1/09_adapter_eval/adapter_component_resolution.json --output-root artifacts/phase1/09_adapter_eval/vjepa21b_l11_box_validation
```

If the adapter gate is not passed, replace the entire adapter branch above with
this command. It is a valid scientific outcome, not an execution failure:

```bash
conda run -n llm python scripts/record_phase1_skip.py --component adapters --reason validation_h_info_or_h_geometry_gate_not_passed --evidence artifacts/phase1/05_geometry/vjepa21b_l11_box_validation/metrics.json,artifacts/phase1/06_probes/linear/vjepa21b_l11_box/metrics.json,artifacts/phase1/06_probes/mlp/vjepa21b_l11_box/metrics.json --output artifacts/phase1/09_adapter_eval/adapter_component_resolution.json
```

Exactly one adapter resolution is permitted: successful validation writes
`status=selected` and the selected checkpoint/config hashes; the alternative
writes `status=skipped` and a hashed gate reason. Once both optional component
resolutions exist, freeze every validation-selected input. This command also
requires the cached full-frame shortcut, so it cannot disappear from the final
control bundle:

```bash
conda run -n llm python scripts/freeze_phase1_selection.py --primary-feature-key vjepa21b/layer11/box/original/all --feature-catalog artifacts/phase1/03_features/catalog.parquet --geometry-controls artifacts/phase1/04_controls/vjepa21b_l11_box --geometry-selection artifacts/phase1/05_geometry/vjepa21b_l11_box_validation --probe-selections artifacts/phase1/06_probes/linear/vjepa21b_l11_box,artifacts/phase1/06_probes/mlp/vjepa21b_l11_box,artifacts/phase1/06_probes/shuffled_linear/vjepa21b_l11_box,artifacts/phase1/06_probes/shuffled_mlp/vjepa21b_l11_box --shortcut-selection artifacts/phase1/07_shortcuts/vjepa21b_l11_box_validation --require-control-feature-key vjepa21b/layer11/full/original/all --motion-selection artifacts/phase1/02_motion/role_control_validation --adapter-resolution artifacts/phase1/09_adapter_eval/adapter_component_resolution.json --mask-resolution artifacts/phase1/03_features/mask_component_resolution.json --require-each-optional-component-selected-or-skipped --release-views-manifest artifacts/phase1/01_manifests/release_views.json --no-test-access --output artifacts/phase1/10_report/frozen_selection.json
conda run -n llm python scripts/run_locked_phase1_test.py --sealed-test-targets artifacts/phase1/01_manifests/sealed_test_targets.parquet --sealed-test-triplets artifacts/phase1/01_manifests/sealed_test_triplets.parquet --release-views-manifest artifacts/phase1/01_manifests/release_views.json --feature-catalog artifacts/phase1/03_features/catalog.parquet --selection-manifest artifacts/phase1/10_report/frozen_selection.json --all-selector eligible_base --matched-selector confirmatory_eligible --validate-match-provenance --evaluate-split test --bootstrap-group dependency_group_id --require-frozen-hashes --access-start-marker artifacts/phase1/10_report/test_access_started.lock --completion-marker artifacts/phase1/10_report/test_completed.lock --output-root artifacts/phase1/10_report/locked_test
conda run -n llm python scripts/build_phase1_report.py --artifact-root artifacts/phase1 --locked-test artifacts/phase1/10_report/locked_test --selection-manifest artifacts/phase1/10_report/frozen_selection.json --require-completion-marker artifacts/phase1/10_report/test_completed.lock --output artifacts/phase1/10_report/phase1_results.md
```

`run_locked_phase1_test.py` verifies all non-test hashes and the absence of an
access marker **without reading sealed labels or roles**. Immediately before its
first sealed-target, sealed-role, or test-prediction read, it atomically creates
`test_access_started.lock` with exclusive-create semantics. The marker is retained
on success, crash, or metric failure, so an interrupted run cannot silently be
repeated. Results are written into a temporary sibling directory and atomically
renamed only when the whole raw/control/probe/shortcut/motion/optional-component
bundle succeeds; only then is `test_completed.lock` created. The report builder
requires that completion marker. Any exceptional rerun requires a documented
protocol deviation and new preregistration; deleting the access marker is not a
normal recovery path. No earlier command may read a sealed target/role artifact
or test prediction.

## 17. Windows/Linux and Runpod migration

Code must never depend on drive letters, backslashes, current working directory, or a user home path. Every CLI path is resolved from an explicit argument; manifests store paths relative to their declared dataset root.

Local Windows execution:

- use `conda run -n llm python ...`;
- start PyAV/DataLoader workers at 0;
- use forward slashes in configs and commands;
- keep checkpoint and artifacts below the project root;
- record NVIDIA driver, torch CUDA build, and peak VRAM.

Runpod Linux execution:

1. place the project at an arbitrary root such as `/workspace/state_geometry_video`;
2. use the base image's already-active Python; do not create another environment;
3. run `python scripts/check_phase1_dependencies.py --skip-env-name-check` and
   install only what it reports into that active container environment;
4. exclude the local dirty `vjepa2/` tree from transfer and clone a clean source
   at the exact audited commit (commands below); do not copy the unused 5.15 GB
   ViT-L checkpoint for the ViT-B run;
5. re-download the dataset at pinned revision or copy payload files while
   excluding `.cache/` and one of the duplicate frame representations;
6. store data/checkpoints/artifacts on a persistent volume;
7. verify dataset, manifest, motion-control, V-JEPA source and primary ViT-B
   checkpoint hashes before extraction;
8. load checkpoint containers on CPU with `weights_only=True` and mmap where
   supported, extract only `ema_encoder`, and never instantiate the predictor;
9. run the identical 32-observation smoke subset and compare shapes,
   finite-value checks and FP32 pooled-feature cosine values;
10. only then increase decoder workers or encoder batch size;
11. never refit motion scaling/calipers, PCA/whitening or alter splits after
    migration.

```bash
git clone https://github.com/facebookresearch/vjepa2.git vjepa2
git -C vjepa2 checkout --detach 204698b45b3712590f06245fbfba32d3be539812
git -C vjepa2 diff HEAD --exit-code -- src app hubconf.py
git -C vjepa2 status --porcelain --untracked-files=all -- src app hubconf.py
```

For every Section 16 command on Runpod, replace only the literal launcher
`conda run -n llm python` with `python`; all following script names and
arguments remain identical. For example:

```bash
python scripts/smoke_test_vjepa21.py --vjepa-source-root vjepa2 --checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --observations artifacts/phase1/01_manifests/feature_inputs.parquet --require-label-redacted-input --require-motion-provenance --subset-query motion_quality_pass=true --frames 16 --resolution 384 --batch-size 1 --precision bf16 --output-root artifacts/phase1/02_smoke/vjepa21b_16x384
```

Recommended copy exclusions when extracted JPGs are retained:

```text
datasets/phase1/EgoInteract/.cache/
datasets/phase1/EgoInteract/data/frames.tar.gz
vjepa2/                              # clone the exact commit on Runpod instead
artifacts/phase1/**/temporary/
```

The dataset inputs excluding the redundant archive and cache occupy
16,815,283,820 bytes. The local ViT-L checkpoint alone adds 5,151,198,524 bytes
if intentionally transferred; omit it for the primary ViT-B run. Feature caches
and the primary checkpoint require additional persistent storage.

## 18. Expected artifacts by stage

| Stage | Required artifacts | Completion condition |
|---|---|---|
| Dependency check | console report; later `environment.json` | Only necessary installs reported; CUDA/ABI test passes. |
| Dataset audit | `inventory.json`, `codec_report.parquet`, `annotation_alignment.json`, file hashes | Counts, dimensions, frame alignment, revision, missing/extras recorded. |
| Static proxy preflight | `01_proxy/observations.parquet`, proxy triplets/splits, immutable feature run/catalog, shortcut report | Data join and, after strict model load, loader/pooling work; confounding is exposed; artifact is labeled interaction phase and excluded from state-hypothesis gates. |
| Curation | `candidates.parquet`, `curated_observations.parquet`, `exclusions.parquet`, agreement/power report | Every retained observation passes state/identity/visibility/box/hash checks; pilot and powered cohort are distinguished. |
| Splits | `splits.parquet`, leakage report | Zero object/video/frame/hash overlap. |
| Motion controls | `motion_observations.parquet`, `motion_validated.parquet`, active-dimension/scaler/backend/config/source hashes, overlay/track QA, signed-change/severity calipers, common-support and role-control reports | No test fit; complete homogeneous schema; quality valid; validation role prediction near null; enough matched independent groups. |
| Joined/internal manifests | `internal_analysis_observations.parquet`, `internal_candidate_triplets.parquet`, `internal_triplets.parquet`, join/validation and caliper reports | One-to-one curated/split/motion join; no embedding fields; exact candidate universe precedes caliper selection; confirmatory prerequisites complete. |
| Release/sealed views | `analysis_trainval.parquet`, label-redacted `feature_inputs.parquet`, `triplets_trainval.parquet`, `sealed_test_targets.parquet`, `sealed_test_triplets.parquet`, `release_views.json` | Pre-test CLIs receive only train/validation labels or redacted model inputs; every view is hash locked. |
| Triplets | internal and released triplet tables, temporal/motion match and exclusion reports, sampling summary | Confirmatory rows pass both frozen calipers; dependency-cluster counts, coverage and nuisance strata reported. |
| Model assets | source revision/full-status/imported-path verification, local ViT-L inventory hash, primary ViT-B SHA-256, strict-load report | Imported source matches commit; exact ViT-B encoder loads with no missing/unexpected keys. |
| Smoke test | VRAM/latency report, token-shape JSON, 32-observation QA features, overlay contact sheets | Expected shapes, finite outputs, deterministic repeat. |
| Feature extraction | immutable per-run `features.npy`, `index.parquet`, `metadata.json`, extraction log, read-only `catalog.parquet` | One-to-one alignment within each declared subset; unique feature keys; no silent skips/overwrites; mask/common-row population explicit. |
| Geometry | per-triplet/cluster Parquet, metrics JSON, bootstrap samples, motion-balance figures | SNS-all and motion-matched raw/PCA/whitening results with CIs, coverage and strata. |
| Probes | configs, checkpoints, predictions, metrics, shuffled-training controls and optional valid block-randomization null | Object-disjoint performance and chance/leakage controls complete. |
| Shortcuts | input/control manifests, metrics, erasure QA images | Object result is interpretable against metadata/context controls. |
| Adapter | seed checkpoints, training logs, selected hyperparameters | Validation-only selection and role-shuffle control complete. |
| Preservation | retrieval predictions, spectra, rank/variance JSON, pairwise distortion | Non-inferiority decision possible. |
| Final locked test/report | `frozen_selection.json`, transactional `locked_test/`, `test_access_started.lock`, `test_completed.lock`, `phase1_results.md`, tables, figures, run manifest | Access marker is created before the first sealed/test read and retained on failure; completion marker exists only after one atomic test bundle; skipped optional components carry reasons. |

Recommended artifact tree:

```text
artifacts/phase1/
  00_audit/
  01_proxy/                           # interaction-phase smoke test, never state evidence
  01_manifests/
    curated_observations.parquet
    splits.parquet
    internal_analysis_observations.parquet
    internal_candidate_triplets.parquet
    internal_triplets.parquet
    analysis_trainval.parquet
    feature_inputs.parquet
    triplets_trainval.parquet
    sealed_test_targets.parquet
    sealed_test_triplets.parquet
    release_views.json
  02_motion/
  02_smoke/
  03_features/<run_id>/
    features.npy
    index.parquet
    metadata.json
    extraction.jsonl
  03_features/catalog.parquet
  04_controls/<run_id>/
  05_geometry/<run_id>/
  06_probes/<probe>/<run_id>/
  07_shortcuts/<run_id>/
  08_adapters/<method>/<run_id>/
  09_adapter_eval/<run_id>/
  10_report/
    frozen_selection.json
    test_access_started.lock
    test_completed.lock
    locked_test/
```

## 19. Sanity checks and leakage tests

### Data and media

- Verify every MP4 frame/sample count against TAS length.
- Verify every COCO image size and image filename against disk.
- Filter polygon components with fewer than three points, exclude empty masks,
  and assert rasterized masks are nonempty and lie inside their boxes. Do not
  compare mask area to COCO `area`, which is bbox area in these files.
- Render source and transformed overlays for at least 32 fixed observations.
- Reject masks/boxes transformed with different resize/crop parameters than RGB.
- Verify the recorded `int(crop*256/224)` bilinear geometry against the pinned
  source and label the torchvision/OpenCV pixel-kernel difference explicitly.
- Assert state clips contain no annotated/reviewed transition frames.
- Assert every confirmatory triplet meets the frozen temporal-gap tolerance;
  merely storing the gap is not sufficient.
- Assert motion is computed on exactly the sampled encoder frames, in the same
  spatial field of view, using exact nonuniform **seconds** deltas and the same
  transformed regions; reject background points outside the crop.
- For the stationary backend, require double-reviewed stationary background and
  a quantitative registration bound, and reject any camera-motion flag; for a future global backend, test synthetic
  translation/rotation cases and fail on inadequate inliers/high warp residual.
- Assert vector camera displacement is removed before norms; include a toy case
  where scalar magnitude subtraction gives the wrong answer.
- Render fixed motion/region/background QA overlays without state labels or
  embeddings. Keep motion coordinates separate from quality fields; report null
  tracker fields as null, never zero, and never feed them to a distance.
- Compute exact frame hashes and perceptual hashes for near-duplicate detection.
- Assert 128-bit COCO identifiers survive a JSON→Parquet→JSON round trip as
  strings without truncation.
- Assert the 377 exact cross-directory duplicate pairs cannot cross splits.
- Reject proxy hashes assigned to both phases and require a positively
  contacting hand for every retained contact record.

### Splits and fitting

- Split before triplet construction.
- Assert zero physical-object, video, asset-proxy, frame, and hash overlap.
- Assert every nullable dependency-graph field creates no edge when null; null
  values must never merge otherwise unrelated observations into one component.
- Validate curated, split, motion, and joined-analysis stages separately; assert
  one-to-one joins and prohibit derived-field requirements at the curation gate.
- Assert PCA/whitening/scalers expose a `fit_split=train` provenance field.
- Assert the motion robust scaler is train-only, caliper selection uses
  validation feasibility only, and test rows never influence points, quality,
  thresholds, bins or calipers.
- Assert one complete backend/config/schema per confirmatory triplet, frozen
  active dimensions, `max(IQR,native_floor)` denominators, and no null imputation.
- Unit-test that test rows never enter fit calls or early stopping.
- Assert no observation is repeated as both positive and negative across splits.

### Features

- Strict checkpoint load with exact key report.
- Frozen parameter assertion and zero gradient allocation.
- Expected token count and dimension assertions.
- Assert imported V-JEPA paths have no tracked, staged, or untracked shadowing
  file at the pinned commit.
- Finite-value, norm, variance, and deterministic-repeat checks.
- Full-PCA distance invariant against center-only features.
- Pooling on an all-ones mask must equal full-frame pooling.
- Zero/empty masks must fail, not produce a zero vector.
- Assert unique immutable feature keys/run roots, box/full/context coverage for
  all curated rows, mask coverage only for the frozen mask subset, and
  common-row-only mask/box comparisons.
- Assert each requested PCA/whitening dimension is below train covariance rank
  and meets support; relative shrinkage/floor provenance must be recorded.

### Learning and metrics

- Closed-form toy tests for SNS sign, ties, margin, and hinge loss.
- Closed-form tests for signed-change and severity matching (including
  `anchor=0, nuisance=+v, state=-v`), matched coverage and motion strata.
- Motion-only pairs sharing an anchor must stay in one dependency group; no
  row-level split, permutation, bootstrap or confidence interval is permitted.
- Unit-test that one randomization draw swaps every paired role in a selected
  dependency component together and never swaps a row/anchor independently.
- Every confirmatory evaluator and adapter must assert both
  `temporal_matched=true` and `motion_matched=true`.
- Adapter first-step gradient test.
- Identity adapter must reproduce frozen metrics.
- Shuffled-training probes and role-shuffled adapters must fail to generalize;
  call them negative controls unless a separate exchangeability-valid
  randomization null is implemented.
- Metrics must macro-average clusters rather than prolific triplets.
- Fit/batch weights must give each dependency/transition group its declared
  influence; unit-test against a deliberately prolific synthetic track.
- Bootstrap must resample whole `dependency_group_id` connected components.
- Assert pre-test commands cannot accept sealed target/role paths or a full
  labeled test table, and feature extraction accepts only the redacted view.
- Test labels/roles/predictions may be opened only by the atomic final evaluator.
  Verify that it creates the exclusive access-start marker immediately before
  the first such read, retains it on failure, commits results transactionally,
  creates the completion marker only on success, and refuses to rerun whenever
  the access-start marker exists.
- Assert each optional mask/adapter component resolves to exactly one of
  `selected` or `skipped`, with hashes and evidence, before selection freeze.

## 20. Kill criteria and pivots within Phase 1

Stop data/model work when any of the following holds:

1. no recurring semantic state families can be curated from EgoInteract;
2. fewer than 30 independent high-confidence pilot transitions survive review
   **and** the preregistered temporal-plus-motion match; unmatched triplets
   remain exploratory and cannot rescue this gate;
3. reliable motion measurement/common support cannot be established, or the
   validation motion-only role model has materially non-null
   `abs(AUROC-0.5)` after the frozen signed-change+severity matching protocol—resample before any
   semantic H-geometry claim;
4. region alignment, physical identity, or observability cannot be verified;
5. reviewer agreement is below 0.80 for state/identity/observability or ambiguity remains high after adjudication;
6. in the curated semantic-state experiment (not the intentionally confounded
   proxy), metadata, temporal order, hands, mask geometry, or object-pixel-erased
   inputs explain the apparent state signal; motion is evaluated by criterion 3;
7. linear and MLP probes remain at balanced chance and shuffled-training
   controls across reasonable frozen layers;
8. the preregistered motion-matched SNS is already near the adjudication ceiling
   with a tight cluster CI, leaving no practical H-geometry headroom;
9. a train-fit PCA/whitening readout control, or a simpler learned metric
   transform trained with the same loss, matches the residual adapter within the
   preregistered equivalence band—retain the diagnostic, drop the special-method claim;
10. adapter gain exists only on train, one narrow state family, synthetic shortcuts, one sampling direction, or the unmatched rather than temporal-and-motion confirmatory subset;
11. the lower paired cluster-CI bound for identity R@1 change is below `-0.02`,
    the lower bound for the effective-rank ratio is below `0.90`, or
    state-prototype collapse is detected;
12. the 16×384 primary extraction OOMs locally after safe batch-1 settings and cannot be moved to Runpod without changing protocol;
13. no real-data Phase 1 validation set is later added—results must remain a synthetic pilot and cannot support a general natural-video claim.

Allowed Phase 1 responses are limited and explicit:

- improve/manual-review the dataset;
- resample to restore temporal/motion common support;
- probe the preregistered intermediate layers;
- report a simpler PCA/whitening/metric solution;
- move identical frozen extraction to Runpod;
- terminate the adapter claim.

Do not respond by adding a JEPA predictor, Pneuma, memory modules, transition-language heads, LoRA, or full backbone fine-tuning.

## 21. Final Phase 1 deliverable

The smallest valid result is a reproducible diagnostic table for a verified, object/video-disjoint test set:

| Readout/method | Pool | SNS all | SNS motion-matched | Matched coverage | Motion-only `abs(AUROC-0.5)` ↓ | Raw margin | Normalized margin | Linear probe | MLP probe | Identity R@1 | Effective rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen V-JEPA 2.1 ViT-B primary | box (all curated rows) | | | | | | | | | | |
| Frozen V-JEPA 2.1 ViT-B supported subset | mask (mask/box common rows only) | | | | | | | | | | |
| Centered/PCA | same | | | | | | | | | | |
| Shrinkage whitened | same | | | | | | | | | | |
| Simple matched metric | same | | | | | | | | | | |
| Residual margin-triplet adapter | same | | | | | | | | | | |

Motion-only AUROC and matched coverage are protocol-level diagnostics, not
method scores; report them once (or repeat them only for table readability).
If the powered mask subset or adapter gate is absent, keep the corresponding row
and write `not run` plus the frozen skip reason/evidence hash; never delete the
row or fill it from a different population.

That table is scientifically useful whether it yields:

- no geometry mismatch;
- a mismatch with no decodable state information;
- a mismatch repaired by a simple transform;
- a mismatch repaired by a residual adapter without preservation loss; or
- an invalid dataset/shortcut result.

The project should spend significant GPU time only after the curated-manifest,
motion/common-support, information, headroom, shortcut and preservation gates
make the result interpretable.
