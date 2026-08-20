# State–Nuisance Geometry for Object-State Understanding with Predictive Video Representations

**Definitive Project Methodology and Manuscript Blueprint**  
**Working name:** StateOrder  
**Version:** 1.1 — 21 August 2026  
**Target:** rigorous MSc thesis with a credible path to CVPR/ICLR/NeurIPS-level submission if the diagnostic and generalization results support the claim.

---

## 0. Scope Lock

This document deliberately separates **the publishable research core** from **the downstream long-video integration**.

### Core research contribution

The core project studies whether modern frozen video foundation representations organize object features so that **semantic object-state changes are more salient in representation geometry than state-preserving nuisance variation**, and whether a small object-local residual adapter can correct systematic failures without destroying object persistence or general representation quality.

### Downstream integration

The existing Pneuma stack—chunk analysis, relational SQL storage, multi-aspect FAISS retrieval, temporal graph logic, and answer synthesis—is retained as an **application testbed**, not as the source of novelty.

The state-aware representation may later be used to:

- validate symbolic object-state transitions;
- rerank state-transition memories;
- provide a visual state key;
- improve evidence localization for state-related long-video questions.

### Explicitly out of scope for the primary paper

- a new full long-video architecture;
- a new JEPA foundation model;
- a literal identity/state disentanglement claim;
- a universal additive “state delta” algebra;
- human-brain anatomical memory claims;
- Kimi KDA/MoE as required components;
- a general physics engine;
- causal discovery from passive video;
- cross-attention memory consolidation as a required module.

The research should not be expanded beyond this scope unless the core diagnostics first establish a real representation-level failure.

---

# 1. Executive Summary & Audit Verification

## 1.1 Executive summary

Let an object observation be represented by a frozen dense video encoder

\[
z_{o,t}=\operatorname{Pool}(E(V_t),M_{o,t}),
\]

where \(E\) is initially V-JEPA 2.1 and \(M_{o,t}\) is a ground-truth or frozen-model object region/mask.

For a triplet

\[
T=(x_a,x_n,x_s),
\]

where:

- \(x_a\): anchor observation of an object;
- \(x_n\): the **same physical object in the same semantic state** under a state-preserving nuisance change;
- \(x_s\): the **same physical object after a genuine semantic state change**;

we measure whether the native representation satisfies

\[
d(f(x_a),f(x_s))
>
d(f(x_a),f(x_n)).
\]

The primary diagnostic is the tolerance-aware nested group-macro score

\[
\boxed{
SNS_\tau
=
\operatorname{Macro}_{g,t,a}
\left\{
\mathbf 1[d_s-d_n>\tau]
+\tfrac12\mathbf 1[|d_s-d_n|\le\tau]
\right\},
\qquad \tau\ge0,
}
\]

where SNS is our internal shorthand for **State–Nuisance Separation** and is **not** presented as an established community benchmark.
Freeze the numerical tolerance without test access. Report \(SNS_{strict}\)
and the tie rate separately; Section 9 defines the exact nesting.

The project proceeds only if the following regime is observed:

\[
\boxed{
\text{state information is decodable}
\quad\land\quad
\text{native similarity geometry is materially misaligned}
}
\]

If this regime exists, we train a lightweight residual adapter \(A_\theta\) on frozen features using a relative state–nuisance ordering objective, with optional transition-semantic supervision and anti-collapse regularization.

The final representation is tested on:

- unseen physical-object instances, with category-disjoint testing only when a
  shared state ontology and enough independent categories make it valid;
- state-transition retrieval;
- rigorous STATUS consistency;
- identity/persistence preservation;
- real-vs-synthetic transfer.

A later Pneuma integration tests whether the learned representation improves retrieval or verification of symbolic state-transition memories under a matched evidence budget.

---

## 1.2 Final critical audit

| Earlier claim / assumption | Audit result | Definitive resolution |
|---|---|---|
| A full renovated Pneuma architecture should be the thesis novelty. | **Rejected.** Modern long-video memory is extremely crowded. | Pneuma is infrastructure/downstream evaluation only. |
| V-JEPA necessarily “loses” object-state information. | **Unsupported.** STATUS shows that subtle visual state information may survive even when final decisions fail. | Diagnose **information content** and **metric geometry** separately. |
| Identity and state can be uniquely disentangled into two learned heads. | **Rejected as a strong theoretical claim.** Such decomposition is generally non-identifiable without much stronger assumptions. | Use a **state-sensitive residual adapter**, not claimed disentanglement. |
| Raw \(z_b-z_a\) constitutes a universal state-transition operator. | **Rejected.** There is no justified universal additive transition algebra. | Raw delta is a baseline; transition semantics use a learned ordered-pair head. |
| Viewpoint/background/pose should always be invariant. | **Rejected.** Some viewpoints change state observability. | Positive pairs require **verified state preservation and observability**. |
| Semantic state changes should be farther apart than nuisance changes. | **Plausible but empirical.** | Establish the failure before training. |
| Low raw SNS uniquely diagnoses a semantic-state geometry failure. | **Unsupported when nuisance/state roles have systematically different within-input motion.** | Report raw SNS as an operational stress test, then require a motion-quality-controlled, motion-matched estimand and a motion-only role-prediction control for semantic attribution. |
| A symmetric metric can represent transition direction. | **False.** \(d(a,b)=d(b,a)\). | State-change separation uses geometry; direction/type uses \(\psi(u_a,u_b)\). |
| Frozen backbone means collapse cannot occur. | **False.** The adapter itself can collapse. | Monitor covariance, variance and effective rank. |
| Strong state classification proves correct state geometry. | **False.** Decodability and nearest-neighbor geometry are different properties. | Report both probes and geometry. |
| State-related gains prove object-state understanding. | **Unsafe.** Hands, backgrounds, motion or procedural phase may create shortcuts. | Box, supported-subset mask, full-frame, context-token, object-pixel-erasure sensitivity, motion and metadata controls are mandatory. |
| A JEPA predictor is necessary for the initial paper. | **Rejected.** | Predictor is an optional post-validation ablation. |
| Cross-attention memory updates are necessary. | **Rejected.** | Deterministic SQL + FAISS first; learned memory slots are deferred. |
| Raw V-JEPA features can be queried directly from text. | **False unless explicitly aligned.** | Text retrieves candidates first; visual state representations rerank/verify. |
| Persistent object memory is novel. | **Rejected.** ObjectStream and R4DSG directly occupy this direction. | Persistence is a prerequisite/control, not novelty. |
| Surprise/state change as the memory-write mechanism would itself be novel. | **Rejected.** | Memory selection is downstream only. |

---

## 1.3 Defensible core claim

The strongest defensible claim is:

> **Modern frozen video foundation representations may preserve object-state information while still exhibiting a metric-geometry mismatch: state-preserving nuisance variation can induce latent displacement comparable to or greater than genuine semantic state changes. After auditing observable motion and other role-correlated shortcuts, we diagnose this explicitly and, where the failure exists, learn a lightweight object-local residual adapter that improves state–nuisance ordering and transition retrieval—particularly on unseen physical-object instances—while preserving object identity/persistence and the underlying frozen representation.**

This claim must be weakened or abandoned if the diagnostic does not support it.
It is not representation-intrinsic: “state” and “nuisance” are task-relative
annotations, and SNS changes with the sampled state-change and nuisance-severity
distribution. Preregister that distribution, record nuisance type/severity, and
report severity-stratified results. The defensible result is supervised
readout alignment on that declared triplet population, not identifiable factor
separation or a universal ordering law.

---

## 1.4 Precise novelty positioning

The project does **not** claim novelty for:

- state-change-aware representation learning;
- appearance/dynamics latent factorization;
- latent transition residuals;
- physical-state identifiability in JEPA;
- persistent object memory;
- hierarchical long-video memory;
- multi-key episodic retrieval.

The candidate novelty is instead the combination of:

1. an explicit **state-vs-nuisance metric diagnostic** for dense predictive video representations at object level;
2. a **post-hoc frozen-backbone state-sensitive residual adapter** trained directly on local state/nuisance ordering;
3. explicit preservation and shortcut controls;
4. unseen-object generalization as a primary target;
5. optional evidence that corrected geometry improves symbolic state-memory retrieval.

This novelty boundary must be rechecked immediately before submission.

---

# 2. Theoretical Foundations

## 2.1 Information content and representation geometry are different

A representation \(z\) may encode sufficient state information for

\[
P(y_{\text{state}}\mid z)
\]

to be highly accurate while simultaneously having poor cosine or Euclidean neighborhoods.

For example,

\[
d(z_{\text{open},1},z_{\text{open},2})
>
d(z_{\text{open}},z_{\text{closed}})
\]

can coexist with a strong nonlinear classifier.

Therefore this project independently measures:

### Decodability

Can a simple probe recover object state?

### Metric geometry

Do semantic state changes dominate state-preserving nuisance variation?

### Relational information

Can the ordered pair distinguish transition type/direction?

### Preservation

Does adaptation retain identity and useful general features?

This distinction is essential because STATUS Bench reports evidence that subtle object-state information can remain available even when a model's ultimate decision is inconsistent.

---

## 2.2 Relative metric learning

The objective does **not** force every “open” object into a universal cluster.

Instead it imposes a local relation:

\[
d(u_a,u_n)+m
<
d(u_a,u_s).
\]

The preferred positive is:

\[
\text{same object}
+
\text{same state}
+
\text{different nuisance}.
\]

This reduces some category-level shortcut opportunities but does **not** prevent
state-prototype collapse. A counterexample maps every observation to one vector
per coarse state with prototypes separated by at least the margin: the loss can
be zero while physical identity is destroyed. Same-instance retrieval,
within-state cross-instance variance, separate nuisance/state distances, output
rank and residual rank are therefore mandatory preservation diagnostics.

---

## 2.3 Why universal transition vectors are rejected

“Open” may mean:

- a drawer translated outward;
- a laptop screen rotated;
- a book unfolded;
- a bottle cap removed;
- a door rotated around hinges.

There is no theoretical reason these transformations should correspond to one shared Euclidean vector.

Therefore we make no assumption that

\[
\Delta_{\text{open}}
\]

is globally transferable.

The primary geometry is **object-conditioned/local**.

---

## 2.4 Why residual adaptation is preferred

Recent Video-JEPA auxiliary-objective experiments show that improving one representational capability can degrade another, so preserving the pretrained representation must be treated as part of the problem rather than assumed automatically.

The trainable adapter therefore starts near the identity map.

---

# 3. Comprehensive Literature Map

## 3.1 JEPA and predictive video representation learning

### V-JEPA 2 / V-JEPA 2.1

V-JEPA 2.1 improves dense predictive video features through a denser predictive objective, deep self-supervision, improved multimodal tokenization and scaling. The released family includes smaller distilled ViT-B and ViT-L models suitable for frozen evaluation.

**Relevance:** primary frozen video backbone.

**Gap relative to us:** it does not directly test whether semantic object-state changes are better separated than state-preserving nuisance changes.

---

### TrackMAE

TrackMAE empirically augments masked video modeling with CoTracker3 point-track
displacement prediction and motion-aware masking. Its reported ablations show
that trajectory targets can complement pixel or semantic feature targets on its
downstream benchmarks. It contains no theorem that motion is an independent or
identifiable latent factor, and it does not establish that frozen V-JEPA cosine
distance is motion-confounded.

**Adopted insight:** measure within-input object and camera motion explicitly and
audit whether triplet roles are motion-separable.

**Not adopted in Phase 1:** TrackMAE pretraining, its motion loss, or a claim that
its separate-decoder ablation justifies our architecture. A frozen TrackMAE
encoder would be a later descriptive baseline, not a causal JEPA ablation.

---

### Factorized Latent Dynamics for Video JEPA

This work explicitly investigates appearance/dynamics factorization and reports capability trade-offs across auxiliary objectives.

**Adopted insight:** preservation must be measured.

**Difference:** we do not claim latent appearance/state disentanglement.

---

### PhyLatent

PhyLatent targets closely related issues of invariance, physical-state identifiability and counterfactual separation in JEPA world-model latents.

**Collision:** high conceptual overlap.

**Difference:** our target is post-hoc adaptation of frozen foundation representations for natural semantic object states, not training a controlled physical world model.

---

## 3.2 State-aware and object-state video representations

### Learning State-Aware Visual Representations from Audible Interactions

Mittal et al. explicitly argue that successful temporal invariance objectives can conflict with the need to remain sensitive to interaction-induced changes in environmental state.

This is foundational motivation for the project.

---

### Look for the Change

This work jointly learns initial object state, state-modifying action and end state using temporal ordering in untrimmed web videos.

It means we cannot claim novelty merely for modeling before/action/after state.

---

### OSCaR

OSCaR provides 14,084 annotated egocentric video segments involving nearly 1,000 objects and focuses explicitly on object states and state changes.

**Role:** principal real-world state-change dataset.

---

### VidOSC / HowToChange

VidOSC formulates open-world object-state-change localization and explicitly evaluates known vs novel object/state-change combinations. Its HowToChange protocol reports frame-level F1, precision and state Precision@1.

**Role:** strongest open-world generalization benchmark in the project.

---

### State-Change Counterfactuals

This work uses state-change descriptions and counterfactual descriptions as representation supervision for procedure-aware video tasks.

**Implication:** transition-semantic supervision itself is not novel.

---

### STATUS Bench

STATUS contains 404 curated quintuplets and evaluates object-state identification, image retrieval and state-change identification simultaneously. It introduces rigorous per-task metrics and ROA, which requires consistency across the complete quintuple.

The benchmark's analysis also provides the strongest warning against assuming state information disappears at the encoder: subtle visual distinctions often remain recoverable deep into the model.

---

### DEHOI

Recent hand/object cue-isolation work shows that egocentric models can rely on superficial cues and that stronger object-dynamics modeling improves STATUS state-change identification.

**Implication:** shortcut controls are mandatory.

---

### TOC-Bench

TOC-Bench explicitly grounds queries in object tracks and structured temporal event timelines and finds significant weaknesses in identity/state continuity in current Video-LLMs.

**Role:** optional downstream temporal consistency test.

---

# 4. Long-Video Memory & Search Literature

## 4.1 HAVEN

HAVEN builds a structured global/scene/segment/entity hierarchy with audiovisual entity cohesion and agentic search.

Therefore hierarchical entity memory itself is not a plausible Pneuma novelty claim.

---

## 4.2 SimpleStream

SimpleStream finds that a four-frame recent-window baseline can match or outperform substantially more complex streaming-memory systems and identifies a perception-memory trade-off.

This is a mandatory downstream simplicity baseline.

---

## 4.3 StreamMem

StreamMem maintains bounded, query-agnostic KV-cache memory during streaming video processing.

Query-agnostic bounded memory itself is therefore not novel.

---

## 4.4 SelectStream

SelectStream formulates streaming memory as budgeted online latent evidence allocation with mechanisms governing writing, preservation and retrieval.

State/surprise-based memory allocation cannot be our novelty claim.

---

## 4.5 ObjectStream

ObjectStream makes persistent latent objects the primary memory anchors, explicitly retaining object histories, transient changes and recent visual evidence under bounded memory.

Persistent object memory is therefore already occupied.

---

## 4.6 WorldMM / VideoARM

WorldMM constructs episodic, semantic and visual memory at multiple temporal scales, while VideoARM combines hierarchical multimodal memory with adaptive agentic reasoning.

These works strengthen the case against turning the thesis into another broad multimemory agent.

---

## 4.7 MERIT

MERIT deliberately prioritizes high-recall multi-key episodic memories and performs temporal expansion after retrieval rather than aggressively precomputing every high-level relation.

This strongly influences our downstream design:

\[
\text{retrieve}
\rightarrow
\text{expand temporally}
\rightarrow
\text{verify},
\]

rather than constructing an increasingly elaborate graph before the query is known.

---

## 4.8 R4DSG

R4DSG stores persistent object identities and anchor-relative object-state changes in long egocentric video.

Thus even object-state memory construction is not sufficient novelty.

Our contribution must occur **before or below memory construction**.

---

## 4.9 StreamFlow

StreamFlow combines dynamics-aware mid-term filtering with latent long-term memory and attention-guided retrieval.

Again, predictive/dynamics-aware streaming memory is already an active research area.

---

# 5. Long-Context Vector Search

FAISS remains an appropriate implementation substrate for exact and approximate nearest-neighbor search, including inverted indexing and product quantization.

However:

> vector-search infrastructure is not a research contribution.

The most important design point is modality compatibility.

Raw V-JEPA features are not text-aligned, so a language query cannot legitimately be projected into raw V-JEPA space unless a separately trained cross-modal mapping has been validated.

Therefore our downstream search is two-stage:

\[
\boxed{
\text{text/SQL retrieval}
\rightarrow
\text{visual state verification/reranking}
}
\]

rather than:

\[
\text{text query}
\rightarrow
\text{raw V-JEPA FAISS}.
\]

---

# 6. Comparative Literature Table

| Work | What it established | Limitation relative to our question | What we adopt | Direct difference |
|---|---|---|---|---|
| V-JEPA 2.1 | Dense predictive spatial-temporal frozen features. | No explicit semantic state-vs-nuisance metric study. | Frozen dense backbone and layer probing. | Diagnose/correct object-local geometry. |
| Factorized Latent Dynamics | Appearance/dynamics factorization; capacity trade-offs. | Broad factorization rather than semantic object-state diagnosis. | Preservation warning. | No claimed disentanglement. |
| CAST | Learned state-conditioned latent residuals for consistent retrieval. | Transition residual model rather than nuisance ordering. | Ordered relation baseline. | Raw delta is not our claimed mechanism. |
| PhyLatent | Physical invariance/identifiability/counterfactual separation. | Controlled physical world models. | Invariance-vs-identifiability insight. | Natural semantic object states and frozen adaptation. |
| State-Aware Audible Interactions | State-sensitive SSL from egocentric interactions. | Audio/event-centric objective. | State sensitivity motivation. | Modern predictive feature diagnostic. |
| Look for the Change | Initial/action/end state learning. | Procedure/state localization task. | Temporal state concepts. | Representation geometry rather than temporal-order novelty. |
| OSCaR | Large egocentric state-change benchmark. | Not a JEPA geometry method. | Real training/evaluation data. | Adapter + diagnostic. |
| VidOSC | Open-world state-change localization. | Localization architecture. | Known/novel evaluation. | Feature geometry is primary. |
| State-Change Counterfactuals | Transition semantic/counterfactual supervision. | Procedure-aware representation. | Optional semantic auxiliary loss. | State/nuisance ordering is central. |
| STATUS | Rigorous state consistency benchmark and readout analysis. | Primarily VLM evaluation. | Mandatory information-vs-geometry analysis. | We alter frozen visual geometry. |
| DEHOI | Hand/object cue-specific dynamics. | HOI model. | Shortcut warning. | Object-state metric adaptation. |
| EgoInteract | Controlled synthetic egocentric interaction generation. | Synthetic domain gap. | Controlled nuisance/state triples. | Never sole evidence. |
| TOC-Bench | Object-track-grounded temporal consistency. | Video-LLM QA. | Optional application benchmark. | Not core training. |
| HAVEN | Hierarchical entity-aware long-video search. | Broad architecture. | Entity coherence insight. | We do not build another hierarchy. |
| SimpleStream | Extremely strong recent-window baseline. | No representation mechanism. | Complexity sanity control. | Downstream only. |
| StreamMem | Bounded query-agnostic KV memory. | Cache-level memory. | Causality/budget discipline. | Not core. |
| SelectStream | Selective bounded latent evidence allocation. | Memory-selection problem. | Prevents surprise-memory novelty claims. | Downstream only. |
| ObjectStream | Persistent latent object memory. | Occupies object-memory novelty. | Strong downstream comparator. | Representation reliability is upstream. |
| WorldMM | Episodic/semantic/visual memory agent. | Broad multimemory system. | Separate symbolic and visual evidence. | No multimemory novelty claim. |
| MERIT | Multi-key episodic retrieval + temporal expansion. | Retrieval-centric. | Candidate-first expansion. | State feature is a verification/reranking signal. |
| R4DSG | Persistent object-state memory using relative anchors. | Direct object-state memory solution. | Downstream comparator. | Our focus is representation quality before storage. |
| StreamFlow | Dynamics-aware streaming memory. | Broad streaming architecture. | Efficiency comparator. | Not core. |
| FAISS | Scalable vector similarity indexing. | Infrastructure. | Implementation. | No novelty claim. |
| VICReg | Variance/covariance anti-collapse objective. | Generic SSL. | Optional regularization. | Activated only when diagnostic warrants it. |
| DINOv2 | Strong generic dense visual representation. | Not temporally predictive. | Non-JEPA control. | Tests whether failure is JEPA-specific. |

---

# 7. Core Architectural Methodology

## 7.1 Core pipeline

```text
video / before-after observations
        │
        ▼
object grounding
(GT masks/boxes preferred; frozen detector otherwise)
        │
        ▼
frozen V-JEPA 2.1
        │
        ▼
dense object-local feature pooling
        │
        ├── frozen geometry diagnostic
        ├── linear probe
        ├── MLP probe
        │
        ▼
residual state adapter Aθ
        │
        ├── ordering objective
        ├── optional transition objective
        ├── optional preservation objective
        └── optional variance/covariance regularization
        │
        ▼
state-sensitive feature u
        │
        ├── SNS
        ├── retrieval
        ├── STATUS
        ├── HowToChange
        └── identity-preservation evaluation
```

---

## 7.2 Frame/clip encoder

Let

\[
V_t\in\mathbb R^{T\times H\times W\times3}.
\]

The primary configuration uses V-JEPA 2.1 frozen at 384-pixel input resolution.

The official V-JEPA 2.1 training/evaluation recipes use short video clips and released 384-resolution checkpoints; the model family includes 80M ViT-B and 300M ViT-L distilled versions.

For our first experiments:

\[
T=16
\]

frames sampled over the observation window.

For the primary EgoInteract protocol, use one fixed 30-source-frame stable
window and the deterministic 16-of-30 index rule in the Phase 1 implementation
plan. Do not vary sampling rate by observed transition speed.

---

## 7.3 Object-local pooling

For 16 frames, 384-pixel input, patch size 16 and tubelet size 2, the ViT-B
dense-token grid is

\[
Z=E(V)\in\mathbb R^{B\times8\times24\times24\times768}.
\]

Apply exactly the same resize/crop to RGB and every framewise mask/box. For each
tubelet/patch cell \(q\), area-average the two transformed frame masks into an
occupancy weight \(W_q\in[0,1]\), producing
\(W\in\mathbb R^{B\times8\times24\times24}\). Pool in FP32:

\[
z_o
=
\frac{\sum_q W_q Z_q}
{\sum_q W_q+\epsilon}.
\]

Require a predeclared minimum token coverage and reject empty regions. This is
region-pooled rather than strictly object-isolated: transformer tokens remain
globally contextualized.

Phase-1 pooling contract:

1. bounding-box pooling is the pilot primary on every curated observation;
2. mask pooling is required where supported, on a frozen mask-available subset,
   and is compared with boxes only on common rows;
3. full-frame pooling;
4. context-token pooling as a localization diagnostic;
5. object-pixel-erased input re-encoding as a stronger shortcut control, with
   mean-fill plus blur/inpainting sensitivity because it preserves a mask
   silhouette and can be out of distribution.

Transformer tokens are globally contextualized, so complement-region token
pooling is not object-free and must not be labeled background-only evidence.
Object-pixel erasure is also not literally object-free.

---

## 7.4 Multi-layer probing

Let

\[
z^{(\ell)}_{o,t}
\]

denote features from encoder layer \(\ell\).

If final-layer probes are weak, test

\[
z^{multi}_{o,t}
=
\sum_{\ell\in\mathcal L}
\alpha_\ell
P_\ell(z^{(\ell)}_{o,t}),
\]

where

\[
\alpha=\operatorname{softmax}(a).
\]

This is cheaper and more interpretable than immediately fine-tuning the backbone.

---

# 8. State–Nuisance Triplet Construction

Every triple must satisfy

\[
\operatorname{object}(x_a)
=
\operatorname{object}(x_n)
=
\operatorname{object}(x_s),
\]

\[
\operatorname{state}(x_a)
=
\operatorname{state}(x_n),
\]

\[
\operatorname{state}(x_a)
\ne
\operatorname{state}(x_s).
\]

### Candidate nuisance factors

- camera viewpoint;
- ego-motion;
- translation/scale;
- illumination;
- background;
- hand presence;
- partial occlusion;
- state-preserving object pose.

### Critical observability constraint

A nuisance example is invalid if the changed viewpoint hides or reveals the defining state evidence.

For example, a top-down viewpoint that reveals cup fill level cannot automatically be treated as a nuisance relative to a side view where fill level is invisible.

### Temporal, action and motion constraints

Primary confirmatory observations must be stable-state windows. They exclude the
intervening manipulation/transition and are non-overlapping, near-duplicate
filtered, and matched in anchor–nuisance versus anchor–state temporal gap. Motion
must be measured only on the exact frames supplied to the frozen encoder. A low
raw SNS remains a useful stress-test result, but it is not specific to semantic
state geometry if nuisance and state roles have systematically different motion.
Section 9.3 therefore defines the confirmatory motion-matched estimand.

If motion is constitutive of the state definition, such as `moving/stopped` or
`pouring/not-pouring`, it must be declared as a separate dynamic state family;
matching or regressing that motion away would change the scientific question.

Point tracking may support localization and motion measurement, but it cannot by
itself verify physical identity through occlusion, cuts, drift, or visually
similar objects. Ground-truth or feature-blind manual identity approval remains
mandatory.

---

# 9. Frozen Geometry Diagnostic

Normalize:

\[
\bar z
=
\frac{z}{\|z\|_2}.
\]

Use cosine distance:

\[
d(a,b)
=
1-a^\top b.
\]

For each triple:

\[
d_n=d(\bar z_a,\bar z_n),
\]

\[
d_s=d(\bar z_a,\bar z_s).
\]

## 9.1 State–Nuisance Separation

For triplet row (r), anchor (a) within verified transition (t), and connected
dependency group (g), use

\[
s_r=\mathbf1[d_s^{(r)}-d_n^{(r)}>\tau]
+\tfrac12\mathbf1[|d_s^{(r)}-d_n^{(r)}|\le\tau],
\]

\[
\bar s_a=|R_a|^{-1}\sum_{r\in R_a}s_r,
\qquad
\bar s_t=|A_t|^{-1}\sum_{a\in A_t}\bar s_a,
\qquad
\bar s_g=|T_g|^{-1}\sum_{t\in T_g}\bar s_t,
\qquad
\boxed{SNS_\tau=G^{-1}\sum_{g=1}^{G}\bar s_g}.
\]

## 9.2 Normalized margin

\[
\Delta_i
=
\frac{
d_s^{(i)}-d_n^{(i)}
}{
d_s^{(i)}+d_n^{(i)}+\epsilon
}.
\]

Aggregate \(\Delta_i\), raw margins, tie rate, and paired adapter deltas with the
same row -> anchor -> transition -> equal-weight dependency-group nesting. Each
transition has equal weight inside its group even when it has a different number
of valid anchor directions; each anchor has equal weight inside its transition.
Confidence intervals must be bootstrapped by
connected object/video groups rather than by individual correlated frames or
Cartesian-product triplets.

## 9.3 Motion-controlled SNS

Raw SNS is an operational test of all declared nuisance variation, including
ego-motion. However, low raw SNS is non-specific when the state and nuisance
roles differ systematically in motion within the clips seen by the encoder. The
primary semantic attribution therefore also uses a motion-quality-controlled,
motion-matched subset.

For object point \(p_{j,k}\) at sampled frame \(k\), let \(H_k\) be a robust
background warp fitted outside a dilated object/hand region, let \(\pi\) project
homogeneous image coordinates, and define the background-induced displacement

\[
g_k(p)=\pi(H_k\tilde p)-p.
\]

The object-relative residual displacement is

\[
r_{j,k}
=
\frac{
(p_{j,k+1}-p_{j,k})-g_k(p_{j,k})
}{
\Delta t_k\,s_k
},
\qquad
s_k=\max\!\left(\sqrt{w_k^2+h_k^2},\epsilon\right),
\]

where \(\Delta t_k=(f_{k+1}-f_k)/fps\) is in seconds from authoritative
timestamps (or verified constant-FPS frame indices); units may not be mixed,
and \(s_k\) is object-box diagonal. Fit tracks/warps in the recorded encoder
resize/crop coordinate system or map source coordinates through it exactly;
reject points outside the encoder field of view. **Do not** compute
\(M_{object}-M_{global}\) from scalar magnitudes: magnitude subtraction can be
negative and cannot account for position-dependent rotation or projective flow.

For each exact encoder input \(x\), predeclare a complete **declared-schema**
measured-motion descriptor \(\phi_M(x)\) under one fixed
`motion_backend_id + config + schema_version`. A minimum backend may use signed
box/mask centroid velocity `(vx,vy)`, signed log-area/aspect rates and robust
magnitudes only when two reviewers certify a stationary background, a
quantitative background-registration bound passes, and the window is stable.
Moving-camera clips require a separately pinned global-compensated point-track
backend. That schema includes visibility/confidence-weighted signed mean
residual-flow `(vx,vy)`, signed mean
background-flow `(vx,vy)`, residual-magnitude quantiles and moving-point
fraction. Tracker visibility/confidence, valid-track count, inlier ratio, fit
error, coverage and manual decisions are quality fields, not coordinates in
\(\phi_M\). Missing track fields are null, not zero, and are never passed to a
distance. Confirmatory triples require the same complete backend/config/schema
for all three observations; analyze different backends separately. Fit a robust
scaler on unique training observations only:

\[
\tilde\phi_{M,k}(x)
=
\frac{
\phi_{M,k}(x)-\operatorname{median}_{train,k}
}{
\max(\operatorname{IQR}_{train,k},s_{min,k})
}.
\]

Freeze positive native-unit floors \(s_{min,k}\), the active dimensions and a
train-constant-feature drop rule before test. Do not allow near-zero IQRs to
amplify numerical jitter.

For a pair define the componentwise change vector

\[
\Delta_M(x_i,x_j)=\tilde\phi_M(x_j)-\tilde\phi_M(x_i),
\qquad
c_M(x_i,x_j)=|\Delta_M(x_i,x_j)|,
\qquad
M(x_i,x_j)=\lVert c_M(x_i,x_j)\rVert_2.
\]

For triplet \(T=(x_a,x_n,x_s)\), define signed-change mismatch, severity mismatch,
and common motion level

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

Define frozen componentwise motion-quality acceptance

\[
\operatorname{pass}_Q(x)
=
\bigwedge_{\ell}
\left[Q_\ell(x)\in\mathcal A_\ell\right],
\]

where each \(\mathcal A_\ell\) is an acceptable set for visibility, valid-track
count, coverage, inlier ratio, fit/registration error, or review status. These
heterogeneous fields are not collapsed into an undefined scalar. The matched
set is

\[
\mathcal T_M(\epsilon_M^{signed},\epsilon_M^{sev})
=
\{T:q_M^{signed}(T)\le\epsilon_M^{signed},
q_M^{sev}(T)\le\epsilon_M^{sev},
\operatorname{pass}_Q(x_a)\land
\operatorname{pass}_Q(x_n)\land
\operatorname{pass}_Q(x_s)\},
\]

and \(SNS_{motion\text{-}matched}\) is the same nested group-macro estimator as
SNS, restricted to \(\mathcal T_M\). Choose both calipers, all componentwise
quality rules, robust-scale
floors, and low/medium/high bins of \(\overline M\) using feature-blind
train/validation feasibility only, then freeze them before test. Always report
raw SNS, matched SNS, common-support coverage, group counts, role-conditioned
signed/severity distributions, and sensitivity over the predeclared calipers.
If too few
independent groups have reliable motion and common support, matched SNS is not
estimable and the semantic geometry claim is a no-go.

A motion-only leakage model creates the two dependent pairs
\((x_a,x_n,y=0)\) and \((x_a,x_s,y=1)\). Its regularized model uses
`[phi_a, phi_b, Delta_M(a,b), abs(Delta_M(a,b))]`; a deliberately weak
scalar-threshold baseline uses only \(M(a,b)\). It fits preprocessing/model/threshold on
train/validation groups, and evaluates only inside the single atomic locked test
on object/video-disjoint test.
Report balanced accuracy, AUROC, label-permutation nulls and grouped confidence
intervals. High role prediction requires resampling or tighter matching; low
prediction does not prove that all motion confounding is absent.
The null uses one paired-role swap per connected `dependency_group_id`: within a
replicate, either retain all `(a,n)/(a,s)` labels in the component or swap every
pair in that component, then recompute the statistic with predictions fixed.
Never permute dependent rows, anchors, or triplets independently. This is a
conditional paired-role association null, not evidence that the measured-motion
schema is complete.
Report the model on (a) all base-eligible triplets as a stress diagnostic and
(b) the exact motion-matched confirmatory subset. Gate 3b uses (b), with the
same connected dependency groups as matched SNS; unmatched performance cannot
by itself kill a successfully matched design.

Background compensation must report inlier count/ratio and warp residual.
Insufficient static background, severe parallax, rolling shutter, occlusion or
tracker drift invalidates the corrected measurement; fail or flag the clip
rather than silently accepting it. Any dyadic mixed-effects distance regression
is exploratory sensitivity analysis, not a substitute for matching or a causal
identification argument.

---

# 10. State-Sensitive Residual Adapter

Use a bottleneck residual adapter:

\[
h
=
W_1\operatorname{LN}(z),
\]

\[
r
=
W_2\operatorname{GELU}(h),
\]

\[
\tilde z
=
z+\alpha r,
\]

\[
u
=
\frac{\tilde z}{\|\tilde z\|_2}.
\]

Recommended starting configuration:

- bottleneck width: 256;
- trainable \(\alpha\) initialized to a small nonzero value such as \(10^{-3}\);
- random nonzero residual-branch weights, with a first-step gradient test for
  both projections and \(\alpha\);
- frozen backbone;
- total trainable parameters preferably below roughly 5M.

The near-identity initialization limits destructive changes to the pretrained representation.

---

# 11. Primary Loss

For

\[
u_a,u_n,u_s,
\]

use

\[
\boxed{
\mathcal L_{\mathrm{ord}}
=
\max
\left(
0,
m
+
d(u_a,u_n)
-
d(u_a,u_s)
\right)
}
\]

with an initial margin sweep

\[
m\in\{0.05,0.10,0.20\}.
\]

Select the margin from frozen **training** distance distributions and validation
performance only; never inspect test distances or labels.

This is the ordinary margin triplet/ranking loss and is the **MVP loss**. The
name “ordering loss” is descriptive; it is not a distinct or novel objective.

No additional objective is added unless an observed failure requires it.

---

# 12. Transition Direction & Type

Since

\[
d(a,b)=d(b,a),
\]

metric distance cannot represent direction.

Construct

\[
r_{ab}
=
\psi_\phi
[
u_a,
u_b,
u_b-u_a,
u_a\odot u_b
].
\]

For categorical transition labels:

\[
\mathcal L_{\mathrm{dir}}
=
-\sum_c
y_c\log p_\phi(c\mid r_{ab}).
\]

If transition-language descriptions are available, optionally align

\[
Wr_{ab}
\]

with frozen text representation \(t_c\):

\[
\mathcal L_{\mathrm{trans}}
=
-
\log
\frac{
\exp(\operatorname{sim}(Wr_{ab},t_c)/\tau)
}{
\sum_j
\exp(\operatorname{sim}(Wr_{ab},t_j)/\tau)
}.
\]

Hard negatives should include inverse transitions where possible.

---

# 13. Preservation Objectives

## 13.1 Identity preservation

For same-object observations \(u_i,u_j\) and different-object \(u_k\):

\[
\mathcal L_{\mathrm{id}}
=
\max
\left(
0,
m_{\mathrm{id}}
+
d(u_i,u_j)
-
d(u_i,u_k)
\right).
\]

Phase 1 uses identity as evaluation-only preservation evidence and does not add
this loss. If a later rescue activates it, identity positives must be
state/category matched; a cross-state positive can directly oppose the ordering
objective.

The Phase-1 retrieval diagnostic uses non-overlapping query/gallery frames,
verified same-track positives, and state/category/context-matched hard negatives
where feasible. Within-video results are called **track persistence**. Claim
cross-video physical identity only when recurring assets are independently
verified and their source videos are disjoint; a tracker or repeated numeric
annotation ID is not sufficient evidence.

---

## 13.2 Geometry-preservation loss

For non-state-sensitive pairs \(\mathcal P\):

\[
\mathcal L_{\mathrm{geom}}
=
\frac1{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\left(
u_i^\top u_j
-
\bar z_i^\top\bar z_j
\right)^2.
\]

Do not enable in Phase 1. The phrase “non-state-sensitive pairs” is otherwise
undefined and may include pairs the adapter is intended to change. Any later
rescue requires a predeclared train-only pair protocol and demonstrated damage.

---

# 14. Anti-Collapse Regularization

The frozen encoder cannot collapse, but \(A_\theta\) can.

Monitor:

- per-dimension variance;
- singular-value spectrum;
- effective rank;
- mean pairwise cosine similarity.

Phase 1 uses these as diagnostics only. Do not add VICReg-style terms to the MVP.
For unit-normalized \(u_i\), using population variance (`correction=0`),
\(\sum_j\operatorname{Var}(u_j)\le 1\), so the usual per-coordinate target
\(\gamma=1\) is infeasible in 768/1024 dimensions.
Covariance loss alone is also minimized by collapse.

Only if a separately approved rescue is required may the following terms act on
unnormalized adapter outputs (or on \(\sqrt d\,u\)) with a dimension-aware
\(\gamma\). Compute them on unique, dependency-balanced observations rather than
duplicated anchor/nuisance/state triplet positions.

## 14.1 Variance

\[
\mathcal L_{\mathrm{var}}
=
\frac1d
\sum_j
\max
\left(
0,
\gamma
-
\sqrt{
\operatorname{Var}(U_{:,j})+\epsilon
}
\right).
\]

## 14.2 Covariance

For centered covariance matrix \(C(U)\):

\[
\mathcal L_{\mathrm{cov}}
=
\frac1d
\sum_{i\neq j}
C(U)_{ij}^2.
\]

Then

\[
\mathcal L_{\mathrm{VC}}
=
\lambda_v\mathcal L_{\mathrm{var}}
+
\lambda_c\mathcal L_{\mathrm{cov}}.
\]

---

# 15. Final Core Objective

## MVP

\[
\boxed{
\mathcal L_{\mathrm{MVP}}
=
\mathcal L_{\mathrm{ord}}
}
\]

## Extended model only if justified

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{ord}}
+
\lambda_{dir}\mathcal L_{\mathrm{dir}}
+
\lambda_{trans}\mathcal L_{\mathrm{trans}}
+
\lambda_i\mathcal L_{\mathrm{id}}
+
\lambda_v\mathcal L_{\mathrm{var}}
+
\lambda_c\mathcal L_{\mathrm{cov}}
+
\lambda_g\mathcal L_{\mathrm{geom}}
}
\]

Use \(\mathcal L_{dir}\) when categorical direction/type labels exist and
\(\mathcal L_{trans}\) when validated language targets exist; do not enable both
without distinct supervision and an explicit ablation. The variance/covariance
terms have only one weighting level, avoiding the redundant outer
\(\lambda_{VC}\) scale. A loss term is included only when its corresponding
failure has been demonstrated.

The project must resist the temptation to improve metrics by uncontrolled multi-loss accumulation.

---

# 16. JEPA Predictor — Optional Post-Validation Module

V-JEPA 2.1 itself is trained by latent prediction, with masked-token prediction and weighted context-token losses plus deep self-supervision.

We do **not** retrain the foundation predictor initially.

After the state representation succeeds, test a lightweight object-level predictor.

For

\[
U_{t-k:t}
=
[u_{t-k},...,u_t],
\]

predict

\[
\hat u_{t+h}
=
P_\phi(U_{t-k:t},h).
\]

Use

\[
\boxed{
\mathcal L_{\mathrm{pred}}
=
\sum_{h=1}^{H}
w_h
\left(
1-
\cos(
\hat u_{t+h},
\operatorname{sg}(u_{t+h})
)
\right)
}
\]

and optionally define predictive residual

\[
e_t
=
1-\cos(\hat u_t,u_t).
\]

This is only an ablation of whether state-aware geometry improves prediction.

It is **not** evidence of causal or physical understanding.

---

# 17. Pneuma Downstream Integration

The current Pneuma system already has:

- chunk-based ingestion;
- structured VLM analysis;
- SQL chunk/object/state storage;
- object-instance tracking;
- state-change records;
- five text-derived aspect embeddings;
- FAISS retrieval;
- temporal/entity graph traversal;
- query decomposition/planning;
- evidence verification;
- final synthesis. 
Its active graph, however, is mainly temporal containment, `NEXT`, and `HAS_ENTITY`; learned relation/node scoring is not active. 
This remains **application infrastructure**, not novelty.

---

# 18. Downstream Chunking Protocol

Initial integration configuration:

- chunk duration: **4 s**;
- overlap: **1 s**;
- visual state extraction: **16 frames**;
- resolution: **384 px**;
- timestamps retained exactly.

The configuration is tunable.

It is chosen as a compact state observation window rather than presented as theoretically optimal.

Candidate transitions spanning boundaries receive dedicated before/after observation windows.

---

# 19. Relational SQL State Store

Introduce a first-class transition table:

```sql
state_transitions(
    transition_id PRIMARY KEY,
    video_id,
    object_instance_id,
    object_name,
    start_ts,
    end_ts,
    state_before,
    state_after,
    location_before,
    location_after,
    symbolic_confidence,
    visual_validity,
    before_chunk_id,
    after_chunk_id,
    transition_type,
    feature_version
)
```

And visual-key metadata:

```sql
visual_state_keys(
    key_id PRIMARY KEY,
    transition_id,
    object_instance_id,
    faiss_index_name,
    faiss_vector_id,
    key_type,
    backbone,
    adapter_version,
    created_at
)
```

This is schema engineering, not a research contribution.

---

# 20. FAISS Indexing

Maintain the existing text indexes.

Add:

- `state_visual`;
- `transition_visual`.

For normalized features, begin with exact inner-product search:

```text
IndexFlatIP
```

Only test IVF or product quantization when scale makes exact search meaningfully costly.

This avoids contaminating representation experiments with ANN approximation error.

---

# 21. Query-Modality Constraint

Do **not** perform:

\[
\text{text embedding}
\rightarrow
\text{raw V-JEPA nearest neighbor}
\]

without an explicitly trained and evaluated text-to-state alignment.

Instead:

\[
\boxed{
\text{text / SQL candidate retrieval}
\rightarrow
\text{visual state verification}
\rightarrow
\text{reranking}
}
\]

The existing Pneuma symbolic retrieval remains the high-recall first stage.

---

# 22. Memory Write

For transition \(\tau_i\), optionally learn visual-validity confidence

\[
q_i
=
\sigma(w^\top r_i+b).
\]

Initially store **all** candidate transitions with their confidence.

Do not hard-filter until calibration is measured.

Under a fixed storage budget:

\[
s_i
=
q_i
-
\lambda_{\mathrm{red}}
\max_{j\in M}
\cos(k_i,k_j).
\]

Keep the highest-value nonredundant entries.

This is a downstream heuristic, not the core contribution.

---

# 23. Cross-Attention Memory Update — Deferred Variant

A learned slot memory is explicitly excluded from the MVP.

If later justified, define memory slots

\[
M_{t-1}\in\mathbb R^{S\times d}
\]

and new observations \(U_t\):

\[
Q=M_{t-1}W_Q,
\]

\[
K=[M_{t-1};U_t]W_K,
\]

\[
V=[M_{t-1};U_t]W_V.
\]

Then

\[
\tilde M_t
=
M_{t-1}
+
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V.
\]

With gate

\[
g_t
=
\sigma(
W_g[M_{t-1};\tilde M_t]
),
\]

\[
M_t
=
(1-g_t)\odot M_{t-1}
+
g_t\odot\tilde M_t.
\]

This experiment is not approved unless deterministic external memory becomes an established bottleneck.

---

# 24. Prompt-Guided Query Mechanics

The downstream decomposer returns structured data:

```json
{
  "intent": "state_transition | state_lookup | temporal_relation | generic",
  "object": "canonical object or null",
  "state_before": "value or null",
  "state_after": "value or null",
  "location": "value or null",
  "temporal_relation": "before | after | between | current | none",
  "anchor_event": "text or null",
  "required_evidence": [
    "symbolic",
    "visual_state",
    "audio"
  ]
}
```

Retrieval then selects:

1. SQL constraints;
2. symbolic/text FAISS keys;
3. visual state verification;
4. temporal expansion;
5. raw evidence replay if allowed.

---

# 25. Evidence Aggregation

For candidate \(i\):

\[
S_i
=
\alpha s_i^{\mathrm{text}}
+
\beta s_i^{\mathrm{symbolic}}
+
\gamma q_i^{\mathrm{visual}}
+
\delta s_i^{\mathrm{temporal}}.
\]

Use transparent fixed weights first.

A learned reranker is added only if fixed weighting becomes the identified bottleneck.

For top candidates, expand

\[
[t_i-\Delta_t,t_i+\Delta_t]
\]

before final answer synthesis.

---

# 26. Dataset Plan

| Dataset | Role |
|---|---|
| **EgoInteract** | Controlled state/nuisance mechanism tests |
| **OSCaR** | Principal real egocentric state-change training/evaluation |
| **STATUS Bench** | Rigorous object-state consistency evaluation |
| **HowToChange / VidOSC** | Open-world known-vs-novel state-change evaluation |
| **TOC-Bench** | Optional temporal-object downstream test |
| **EgoSchema** | Legacy Pneuma continuity test only |

EgoInteract provides controllable egocentric interaction generation and is therefore valuable for isolating nuisance/state mechanisms, but synthetic performance is not accepted as final evidence.

---

# 27. Required Splits

## Object-disjoint

\[
\boxed{
\mathcal O_{\mathrm{train}}
\cap
\mathcal O_{\mathrm{val}}
=
\mathcal O_{\mathrm{train}}
\cap
\mathcal O_{\mathrm{test}}
=
\mathcal O_{\mathrm{val}}
\cap
\mathcal O_{\mathrm{test}}
=
\varnothing
}
\]

The same pairwise-disjoint condition applies to verified asset IDs, exact/near-
duplicate groups, and connected dependency components.

## Video-disjoint

No source video may contribute clips to more than one of train, validation, and
test.

## State-category reporting

Report positional, functional and other state categories where applicable.

## Nuisance reporting

Break down by:

- viewpoint;
- occlusion;
- illumination;
- background;
- hand presence;
- state-change subtlety.

---

# 28. Data Curation Pipeline

1. identify persistent object instance;
2. obtain stable-state observations;
3. identify true state transition;
4. verify state observability, stable windows and aligned per-frame regions;
5. attach object/state/nuisance/hand/stationary-background metadata;
6. reject ambiguous identity transitions;
7. create object/video/dependency-disjoint partitions before pair/triplet enumeration;
8. estimate and quality-audit motion on exact encoder input frames;
9. fit motion scaling/calipers on train/validation only and construct both raw
   and temporal-plus-motion-matched triplets;
10. retain dependency-group IDs for grouped bootstrap analysis.

---

# 29. Measured Laptop-GPU Feasibility

The released V-JEPA 2.1 family includes approximately 80M ViT-B and 300M ViT-L variants in addition to much larger models.

Use:

### Iteration model

\[
\boxed{\text{V-JEPA 2.1 ViT-B}}
\]

### Confirmation model

\[
\boxed{\text{V-JEPA 2.1 ViT-L}}
\]

Do not make the 1B/2B variants necessary.

The audited local device is an RTX 4050 Laptop GPU with 6 GiB VRAM. Local
V-JEPA 2.1 ViT-B extraction therefore starts with batch 1, workers 0,
inference mode, BF16 forward, one requested output layer and an explicit OOM
smoke test. ViT-L is Runpod-only after the ViT-B/data gates. The primary
16-frame/384-pixel protocol must not be silently reduced after an OOM; migrate
the same frozen extraction instead.

---

# 30. Precision

Default:

- BF16 feature extraction;
- FP32 pooled features, normalization, distances, margins, probes, adapter
  training, PCA/whitening, covariance and effective-rank statistics.

Do not begin with FP8.

The scientific signal involves potentially small metric differences, so numerical approximation should not be introduced until BF16 behavior is stable.

---

# 31. Conservative Batch Starting Points

## V-JEPA 2.1 ViT-B

16 frames, 384 px:

```text
micro-batch = 1
```

Increase only after the fixed OOM/peak-VRAM smoke test; do not assume 4 or 8
fits the measured 6 GiB laptop.

## ViT-L

Runpod/post-gate only:

```text
micro-batch = 1
```

Increase to 2–4 only if memory allows.

Target sustained VRAM below roughly 90% to leave room for decoding and temporary tensors.

---

# 32. Cached Adapter Training

After object-pooled features are cached:

```text
batch size:        512 triplets
optimizer:         AdamW
initial LR:        3e-4
weight decay:      1e-4
bottleneck:        256
margin sweep:      0.05 / 0.10 / 0.20
precision:         FP32
```

The feature adapter is small enough that its training is not the computational bottleneck.

---

# 33. Gradient Checkpointing

Not needed for the main frozen-feature experiments.

If limited backbone adaptation later becomes necessary:

1. LoRA/final-block adapters;
2. BF16;
3. micro-batch 1–2;
4. gradient accumulation;
5. gradient checkpointing;
6. optional 8-bit optimizer.

Full V-JEPA pretraining remains out of scope.

---

# 34. Feature Cache

Cache:

```text
video_id
object_id
state_label
timestamp
layer_id
pooled_feature
mask/box metadata
nuisance metadata
motion feature-table reference
motion backend/config/source/checkpoint hashes
motion visibility/global-fit/quality fields
exact sampled frame indices and frame/time deltas
dependency_group_id and split
```

Do not cache all dense patch tokens for the entire dataset unless an experiment specifically requires repooling.

---

# 35. First-48-Hour Micro-Benchmark

The goal is **not training a final model**.

It is deciding whether the thesis mechanism exists.

## Block A — controlled data

The currently downloaded data do not yet contain semantic physical-state
labels, aligned video regions, or verified physical identities. Therefore the
first 48 hours are an inventory plus a 30-transition feature-blind
protocol/power pilot, not an automatic triple-generation run. Only after that
pilot may curation expand to a power-justified cohort. A 15% test allocation
with at least 50 independent test groups needs roughly 334 total independent
groups before family stratification; if that is infeasible, label the result
exploratory and do not claim a confirmatory test.

Triplet row counts are never power calculations. Report the much smaller number
of independent object/video dependency groups; no Cartesian product of frames
or shared-anchor pairs is additional evidence.

Verify:

- physical object identity;
- state labels;
- state visibility;
- per-sampled-frame boxes for all primary observations and masks for a frozen
  supported subset;
- absence of metadata leakage;
- temporal-gap and motion common support;
- motion-only triplet-role prediction near its grouped permutation null.

---

## Block B — extraction

The short commands in Blocks B–E are schematic stage summaries only. The sole
exact CLI, artifact, split/sealing, feature-key, and optional-skip contract is
`docs/phase1_implementation_plan.md`, Section 16. In particular, implementations
must use cataloged memory-mappable arrays plus Parquet indices,
`dependency_group_id`, and the `margin_triplet` loss name; do not revive an
opaque `.pt` cache or an `object_video` pseudo-group.

Schematic call:

```bash
python scripts/extract_features.py \
  --backbone vjepa2_1_vitb \
  --frames 16 \
  --resolution 384 \
  --precision bf16 \
  --pool box \
  --output-root artifacts/phase1/03_features/vjepa21b_l11_all_original \
  --catalog artifacts/phase1/03_features/catalog.parquet
```

On the frozen mask-available common-row subset, repeat for mask. Also run:

- full frame;
- context-token pooling;
- object-pixel-erased input re-encoding with mean-fill and blur/inpainting
  sensitivity.

Do not run the ViT-L subset until the ViT-B data, motion and information gates
pass.

---

## Block C — raw geometry

```bash
python scripts/eval_state_nuisance.py \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --metric cosine \
  --bootstrap-group dependency_group_id \
  --report_by nuisance,state_type
```

Report:

- SNS-all and motion-matched SNS;
- matched coverage and independent dependency-group counts;
- role-conditioned motion distributions, overlap and motion-only AUROC;
- normalized margin;
- distance distributions;
- nuisance breakdown;
- state-category breakdown;
- seen/unseen objects.

Also test:

- L2 normalization;
- PCA;
- whitening.

DINOv2 is a useful **post-gate** descriptive dense self-supervised visual
comparison, not part of the minimum 48-hour run. Because its data, temporal
support, architecture and objective differ, it cannot isolate JEPA training as
a causal factor.

---

## Block D — information probes

```bash
python scripts/train_probe.py \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --probe linear \
  --split object_disjoint
```

and

```bash
python scripts/train_probe.py \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --probe mlp \
  --split object_disjoint
```

Interpretation:

| SNS | Probe | Meaning |
|---|---|---|
| high | high | representation already adequate |
| low/moderate | high | **ideal geometry mismatch** |
| low | low | information problem; frozen adapter may be insufficient |
| high | low | may be valid object-conditioned relational ordering with weak global labels; inspect state-family/category-conditioned probes before calling pathology |

---

# 36. Shortcut Controls

Evaluate:

1. full frame;
2. bounding box;
3. object mask;
4. context-token pooling from the original forward;
5. object-pixel-erased input re-encoding with fill/silhouette sensitivity;
6. mask/box geometry, hands, time/procedural metadata and measured motion.

Desired: on identical eligible rows, the preregistered region-pooled readout
materially exceeds each full-frame, context-token, geometry-only, and
object-pixel-erased shortcut baseline by a practical margin. Do not denote this
as “object versus background”: globally contextualized tokens are not isolated,
and pixel erasure has fill/boundary artifacts.

If context, object-pixel-erased input, or temporal-position information predicts
triplet role/state strongly, the dataset or sampling procedure must be
audited and usually corrected. Motion gates the semantic claim only when its
role model remains materially non-null on the **exact motion-matched
confirmatory subset**; strong prediction on the deliberately broader unmatched
population is a reported stress diagnostic, not by itself a kill condition.
Context-token pooling alone is not proof of a background shortcut
because self-attention has already mixed object information globally.
Object-pixel erasure is stronger but is not object-free: it preserves a
mask-shaped boundary and may be out of distribution.

---

# 37. Shuffled-Label Negative Control

For probes, permute training state labels within predeclared category/state-family
and dependency-preserving blocks, then retrain while keeping validation/test
labels untouched. For the adapter, merely shuffling metadata after triplets are
built changes no gradients; instead rebuild training roles from permuted labels
or randomly swap nuisance/state roles, and evaluate against untouched true-role
validation/test triplets. Use several seeds.

This is a shuffled-training leakage **negative control**, not automatically an
exchangeability-valid permutation p-value/null for the true statistic. A formal
randomization test would require a separately defined group/block exchangeability
scheme and the full frozen train/selection/evaluation statistic under each
consistent permutation of all relevant targets.

If meaningful generalization remains, there is leakage.

Do not proceed until the shortcut is identified.

---

# 38. Tiny Adapter Test

```bash
python scripts/train_state_adapter.py \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --loss margin_triplet \
  --bottleneck 256 \
  --margin 0.1 \
  --lr 3e-4 \
  --batch 512
```

Primary learned comparators are positive-diagonal, linear-residual and
non-residual bottleneck transforms trained on identical confirmatory triplets
with the same ordinary margin loss, search budget, stopping rule and seeds.
Classification or supervised-contrastive MLPs are secondary different-objective
readout controls, not automatically fair metric baselines.

---

# 39. Go Gates

Proceed to larger training only if all core gates pass.

### Gate 1 — headroom

Frozen model shows reproducible state/nuisance ordering failures.

### Gate 2 — information

Simple probes show that state information exists.

### Gate 3 — method-specific gain

All methods are evaluated on identical triplets. Among learned metric
transforms, margin loss, tuning budget, stopping rule and seeds are identical,
and the residual adapter outperforms simpler same-supervision
parameterizations with uncertainty estimates supporting the difference.
Identity, centering, PCA and whitening are fixed/train-fit readout controls and
do not share the loss. A classification MLP is a different-objective readout
control.

### Gate 3b — motion specificity

Confirmatory triplets have adequate common support, complete homogeneous motion
schemas, signed-change and severity balance, a motion-only `abs(AUROC-0.5)` near
its dependency-group permutation null, and stable conclusions across
predeclared calipers/backend-quality sensitivities. Raw and motion-matched SNS
need not agree: a material difference is evidence of role-correlated motion and
makes the raw stress test non-specific, but does not by itself invalidate a
well-supported matched estimand.

### Gate 4 — preservation

Identity/persistence retrieval does not materially degrade.

### Gate 5 — anti-collapse

Effective rank and feature variance remain healthy.

### Gate 6 — real-data direction

The controlled-data improvement also appears directionally on a real subset.

---

# 40. Internal Kill Conditions

If

\[
SNS_{\mathrm{motion\text{-}matched}}\gtrsim 0.90
\]

and state probes are already strong, there is probably insufficient headroom for
the confirmatory formulation. Use a cluster confidence interval and the
adjudication/noise ceiling rather than interpreting \(0.90\) as a universal
cutoff. A high unmatched/raw SNS alone does not trigger this kill condition.

If probes remain near chance across layers, do not perform long adapter sweeps.

Move to intermediate layers or limited backbone adaptation.

If a simpler same-loss metric transform performs approximately the same as the
residual adapter, the special method has insufficient evidence; this does not
erase a valid frozen-geometry diagnostic.

If motion roles are strongly separable or too few independent groups survive
motion-quality and common-support matching, the semantic interpretation of low
SNS is blocked even when the raw stress test remains reportable.

These thresholds are internal engineering gates, not universal laws.

---

# 41. Mandatory Core Baselines

## Frozen foundation models

1. V-JEPA 2.1 ViT-B is the Phase 1 primary.
2. V-JEPA 2.1 ViT-L is a post-gate Runpod confirmation.
3. V-JEPA 2 and DINOv2 comparable-scale encoders are later descriptive
   comparisons, not JEPA causal ablations.
4. TrackMAE is optional/post-gate and descriptive only.

## Metric/readout

5. raw and centered cosine;
6. truncated PCA plus dimension-matched random projection;
7. shrinkage-whitened features;
8. linear and MLP state probes;
9. identity, positive-diagonal metric and linear residual transforms;
10. non-residual MLP and residual bottleneck transforms trained on the same
    triplets with the same standard margin triplet/ranking loss;
11. role-shuffled adapter and grouped label-permutation controls.

Supervised contrastive/classification adapters may be reported as different-
objective readout controls but are not automatically capacity- or
supervision-matched metric baselines. There is no separate “standard triplet”
versus “proposed ordering” loss baseline; the objectives are algebraically
identical.

A paper comparing only against frozen V-JEPA is not sufficient.

---

# 42. Literature Baselines

Where protocols and code permit:

- State-Aware Audible Interactions;
- Look for the Change;
- VidOSC;
- state-change counterfactual learning;
- DEHOI;
- CAST for transition retrieval.

Factorized Latent Dynamics and PhyLatent must be discussed even when task mismatch prevents direct numeric reproduction because they are the closest conceptual objective-space competitors.

---

# 43. Downstream Memory Baselines

Only if the Pneuma integration becomes a paper experiment:

1. Pneuma symbolic baseline;
2. Pneuma + raw V-JEPA reranking;
3. Pneuma + state-adapted reranking;
4. SimpleStream-style recent-window baseline;
5. MERIT-style multi-key retrieval where compatible;
6. ObjectStream/R4DSG where compatible;
7. StreamMem/SelectStream/StreamFlow only for genuine streaming protocols.

---

# 44. Primary Metrics

## State–Nuisance Separation

\[
SNS_{\mathrm{strict}}=P(d_s>d_n),
\qquad
SNS_\tau=
\operatorname{Macro}_{g,u}
\{\mathbf1[d_s-d_n>\tau]+\tfrac12\mathbf1[|d_s-d_n|\le\tau]\}.
\]

Use frozen \(SNS_\tau\) as primary. Report strict SNS and tie rate separately.
Report the primary score as SNS-all and as the preregistered
SNS-motion-matched estimand from Section 9.3, with matched coverage,
independent-group count and grouped confidence intervals.

## Motion leakage and common support

- motion-only role balanced accuracy and AUROC;
- grouped label-permutation null;
- \(q_M^{signed}\), \(q_M^{sev}\), and \(\overline M\) distributions;
- role-conditioned signed/severity balance and caliper sensitivity;
- motion backend quality/failure rate.

## Normalized margin

\[
\overline\Delta
=
\mathbb E
\left[
\frac{d_s-d_n}{d_s+d_n+\epsilon}
\right].
\]

## Probe performance

- accuracy;
- macro-F1;
- per-state category;
- unseen-object accuracy.

## State retrieval

- Recall@1;
- Recall@5;
- MRR.

## Identity/persistence

- same-instance Recall@1;
- same-instance Recall@5;
- same-state nuisance retrieval.

## Collapse

- covariance;
- variance;
- singular-value spectrum;
- effective rank.

---

# 45. STATUS Metrics

Use official:

- Acc\(_{OSI}\);
- Acc\(_{IR}\);
- Acc\(_{SCI}\);
- RAcc\(_{OSI}\);
- RAcc\(_{IR}\);
- RAcc\(_{SCI}\);
- **ROA**.

ROA explicitly requires simultaneous correctness across all five components of the STATUS quintuple and is substantially stricter than ordinary average accuracy.

---

# 46. HowToChange Metrics

Use official known/novel results:

- F1;
- precision;
- state Precision@1.

VidOSC's protocol computes these across initial, transitioning and end-state localization.

---

# 47. Downstream Retrieval Metrics

Where timestamps are annotated:

- Recall@K;
- MRR;
- temporal IoU;
- evidence precision;
- evidence coverage.

QA is secondary.

Retrieval quality and answer generation must be reported separately.

---

# 48. Efficiency Metrics

Report:

- encoder peak VRAM;
- trainable parameters;
- clip feature-extraction latency;
- adapter latency;
- FAISS index size;
- FAISS query latency;
- number of replayed evidence frames;
- downstream end-to-end latency.

FLOPs should be calculated using one consistent profiler.

---

# 49. Core Ablation Matrix

| ID | Object-local | Ordering | Transition head | VC regularization | Preservation | Purpose |
|---|---:|---:|---:|---:|---:|---|
| A0 | ✗ | ✗ | ✗ | ✗ | ✗ | full-frame frozen |
| A1 | ✓ | ✗ | ✗ | ✗ | ✗ | localization effect |
| A2 | ✓ | ✓ | ✗ | ✗ | ✗ | core mechanism |
| A3 | ✓ | ✗ | ✓ | ✗ | ✗ | relation-only |
| A4 | ✓ | ✓ | ✓ | ✗ | ✗ | geometry + transition |
| A5 | ✓ | ✓ | ✓ | ✓ | ✗ | collapse regularization |
| A6 | ✓ | ✓ | ✓ | optional | ✓ | final preserved model |

Also test:

- box vs mask;
- final vs intermediate layer;
- V-JEPA 2 vs 2.1;
- ViT-B vs ViT-L;
- raw vs whitened;
- raw/all-triplet vs temporal-plus-motion-matched triplets;
- motion-caliper sensitivity with matched coverage/group counts;
- minimum reviewed-stationary region kinematics vs a separately pinned,
  quality-passing global-compensated tracker where available;
- raw delta vs pair MLP;
- same-instance vs same-category positives;
- shuffled labels.

---

# 50. Predictor Ablation

Only after core success:

| ID | State adapter | Predictor | Target |
|---|---:|---:|---|
| P0 | ✗ | ✓ | frozen state feature |
| P1 | ✓ | ✓ | adapted state feature |
| P2 | ✓ | ✓ | adapted feature + transition semantics |

The predictor does not become part of the novelty claim unless it introduces a genuinely new mechanism.

---

# 51. Memory Ablation

| ID | Symbolic | Raw visual | Adapted visual | Visual validity | SQL | Temporal expansion |
|---|---:|---:|---:|---:|---:|---:|
| M0 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| M1 | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| M2 | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |
| M3 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| M4 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |

FAISS:

- Flat exact;
- IVF-Flat;
- IVF-PQ only at justified scale.

ANN approximation quality must be separated from representation quality.

---

# 52. Adaptive Pivot Strategy

The scientific goal remains unchanged:

\[
\boxed{
\text{distinguish meaningful state variation from state-preserving nuisance variation}
}
\]

## Pivot A — conditional metric

If one global space is too restrictive:

\[
d_c(z_i,z_j)
=
(z_i-z_j)^\top
M_c
(z_i-z_j),
\]

with

\[
M_c=L_c^\top L_c.
\]

Possible low-rank form:

\[
L_c
=
L_0
+
\sum_k
\alpha_k(c)L_k.
\]

This models object/state-dependent geometry.

---

## Pivot B — pairwise transition verifier

If metric distance cannot capture the relation:

\[
h
=
[
z_a,
z_b,
|z_b-z_a|,
z_a\odot z_b
].
\]

Then

\[
p(y\mid a,b)
=
\operatorname{softmax}(Wg(h)).
\]

This directly models the relation rather than attempting to encode it entirely through distance.

CAST provides an important nearby precedent for learned latent state transitions in frozen embedding spaces.

---

## Pivot C — explicit nuisance suppression

Where nuisance labels are trustworthy:

\[
\min_{A,C_s}
\max_{C_n}
\mathcal L_{\mathrm{state}}
-
\lambda
\mathcal L_{\mathrm{nuisance}}.
\]

Only use nuisance attributes verified to be state preserving.

Otherwise an adversary can remove genuinely useful visual evidence.

---

## Pivot D — intermediate layers / LoRA

If state information is absent from the final layer:

1. probe intermediate V-JEPA features;
2. learn scalar layer fusion;
3. LoRA final blocks if necessary;
4. do not jump to full foundation pretraining.

---

# 53. Statistical Protocol

- grouped bootstrap over videos/objects;
- paired comparisons;
- effect sizes and 95% CIs;
- no frame-level pseudo-replication;
- fixed test split;
- all motion point-selection rules, robust scalers, quality thresholds, bins and
  matching calipers fixed without test access;
- matching is the primary motion control; any distance regression is explicitly
  exploratory and uses dependency-respecting or crossed grouping;
- internal preregistration of primary metrics;
- report negative ablations;
- correct multiple formal hypothesis tests when necessary.

---

# 54. Failure Criteria

The central claim is unsupported if:

1. frozen V-JEPA already has near-ceiling state–nuisance ordering;
2. state labels are not decodable above balanced chance while shuffled-training
   controls fail to generalize (and, if implemented, a valid group/block
   randomization null);
3. reliable motion measurement/common support cannot be established **or** the
   motion-only model predicts roles materially on the exact matched
   confirmatory subset;
4. region-pooled gains are no stronger than full-frame, context-token, or
   object-pixel-erased controls under their stated OOD limitations;
5. background/procedure shortcuts explain gains;
6. synthetic gains fail to transfer;
7. unseen-object results do not improve;
8. identity/persistence collapses;
9. adapter effective rank collapses;
10. object correspondence is unreliable;
11. downstream gain merely comes from more context/replay;
12. results exist only for one very narrow state category.

These criteria apply to different claims. A generic MLP, PCA/whitening, or
simpler same-loss metric transform solving held-out ordering does **not** refute
the frozen-geometry diagnostic; it refutes the need or novelty of the proposed
residual method. Likewise, any material raw-versus-matched SNS divergence is
evidence about role-correlated motion/sampling; the raw result is then not a
clean semantic geometry estimand, while the matched result remains interpretable
only if its common-support and leakage gates pass.

---

# 55. Manuscript Blueprint

## Working title

**State–Nuisance Geometry in Predictive Video Representations for Object-State Understanding**

If results justify a method name:

**StateOrder: Correcting State–Nuisance Geometry in Frozen Video Foundation Representations**

---

## Manuscript story

### 1. Problem

Feature similarity is routinely used for retrieval, tracking and memory, but native visual distance need not distinguish meaningful state changes from irrelevant appearance variation.

### 2. Diagnostic

Construct controlled state/nuisance triplets and characterize V-JEPA 2.1 and other frozen encoders.

### 3. Analysis

Separate:

\[
\text{information content}
\]

from

\[
\text{metric geometry}.
\]

### 4. Method

Introduce the smallest justified residual state adapter.

### 5. Generalization

Evaluate:

- unseen objects;
- subtle state changes;
- controlled nuisances;
- real egocentric video.

### 6. Preservation

Verify that improved state sensitivity does not destroy identity/persistence.

### 7. Downstream application

Optionally demonstrate improved state-transition evidence retrieval in Pneuma.

---

# 56. Claims That Must Not Appear

Without substantial new evidence, do **not** write:

> “Our model understands physics.”

> “We disentangle identity and state.”

> “Our representation is causally sufficient.”

> “State transformations form a linear latent algebra.”

> “Our model solves long-video understanding.”

> “Our brain-inspired memory mimics human cognition.”

> “Pneuma is novel because it combines SQL, FAISS and a graph.”

---

# 57. Execution Order

1. freeze dataset/state/motion definitions;
2. curate a protocol/power pilot, expand to a powered cohort, and freeze the
   curated observation manifest with aligned boxes plus a mask subset;
3. split dependency groups before triplet enumeration;
4. estimate and quality-audit homogeneous-schema motion on the exact spatial and
   temporal encoder inputs;
5. fit train/validation-only motion controls, assemble the joined analysis
   manifest, and build raw plus
   temporal-and-motion-matched triplets;
6. run the validation motion-only/common-support gate;
7. extract ViT-B features;
8. report SNS-all and motion-matched raw geometry;
9. run PCA/whitening controls;
10. run remaining shortcut controls;
11. run linear and MLP probes;
12. determine whether information and geometry claims exist;
13. only if the validation gates authorize it, train the ordinary margin-triplet
    residual adapter and same-loss baselines; otherwise freeze an evidence-backed
    adapter skip;
14. run validation-only preservation/collapse checks where applicable, resolve
    each optional mask/adapter component as selected or skipped, and freeze all
    selections;
15. create a permanent exclusive access marker immediately before the first
    sealed-test read, then perform one atomic locked evaluation covering
    raw/control/probe/shortcut/motion and applicable adapter/preservation metrics;
16. confirm with ViT-L only if justified;
17. confirm descriptively with a non-JEPA/TrackMAE backbone only if justified;
18. add transition relation head if justified;
19. add predictor only if justified;
20. integrate into Pneuma only after core representation results exist.

---

# 58. Definitive Go/No-Go Statement

## GO

The **diagnostic project is scientifically grounded and computationally feasible**.

The primary research question is falsifiable:

\[
\boxed{
\text{Does a state/nuisance metric-geometry mismatch exist?}
}
\]

The ideal regime is

\[
\boxed{
\text{state probe strong}
\quad\land\quad
\text{state–nuisance ordering weak/moderate}.
}
\]

This would demonstrate that useful state information is already present but native similarity geometry is not organized appropriately for state-aware retrieval/comparison.

## NO-GO for expensive runs until the diagnostic passes

Do not spend significant compute on:

- long adapter sweeps;
- predictor training;
- LoRA;
- long-video integration;
- memory-system reconstruction

before the diagnostic gates pass.

If the global ordering formulation fails, use the predefined pivot ladder:

\[
\boxed{
\text{global ordering}
\rightarrow
\text{conditional metric}
\rightarrow
\text{pair relation}
\rightarrow
\text{explicit nuisance suppression}
\rightarrow
\text{limited backbone adaptation}
}
\]

rather than uncontrolled architecture growth.

---

# 59. Primary References

1. Mur-Labadia et al. **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.** arXiv:2603.14482, 2026.
2. Premi. **Factorized Latent Dynamics for Video JEPA: An Empirical Study of Auxiliary Objectives.** arXiv:2605.17165, 2026.
3. Liu et al. **CAST: Modeling Visual State Transitions for Consistent Video Retrieval.** arXiv:2603.08648, 2026.
4. Zeng et al. **PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models.** arXiv:2608.05720, 2026.
5. Mittal et al. **Learning State-Aware Visual Representations from Audible Interactions.** NeurIPS 2022.
6. Souček et al. **Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos.** CVPR 2022.
7. Nguyen et al. **OSCaR: Object State Captioning and State Change Representation.** 2024.
8. Xue et al. **Learning Object State Changes in Videos: An Open-World Perspective.** 2024.
9. Kung et al. **What Changed and What Could Have Changed? State-Change Counterfactuals for Procedure-Aware Video Representation Learning.** 2025.
10. Ukai et al. **STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models.** 2025.
11. Tateno et al. **Do Egocentric Video-Language Models Capture Both Hand- and Object-Centric Cues?** 2026.
12. Leonardi et al. **EgoInteract: Synthetic Egocentric Videos Generation for Interaction Understanding and Anticipation.** 2026.
13. Chen et al. **TOC-Bench: A Temporal Object Consistency Benchmark for Video Large Language Models.** 2026.
14. Yin et al. **Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search (HAVEN).** 2026.
15. Shen et al. **A Simple Baseline for Streaming Video Understanding.** 2026.
16. Yang et al. **StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding.** 2025.
17. Ge et al. **What Should a Streaming Video Model Remember?** 2026.
18. Dong et al. **ObjectStream: Latent Objects as Memory Anchors for Streaming Video Understanding.** 2026.
19. Yeo et al. **WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning.** 2025/2026.
20. Yin et al. **VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding.** 2025/2026.
21. Choi et al. **Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding.** 2026.
22. Ma et al. **R4DSG: Relative 4D Scene Graph Memory for Object-Centric Question Answering in Long Egocentric Video.** 2026.
23. Fu et al. **StreamFlow: Dynamic Memory Flows for Streaming Video Understanding.** 2026.
24. Johnson, Douze & Jégou. **Billion-scale similarity search with GPUs.** 2017.
25. Bardes, Ponce & LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** 2021.
26. Oquab et al. **DINOv2: Learning Robust Visual Features without Supervision.** 2023.
27. Assran et al. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.** 2025.
28. Vandeghen et al. **TrackMAE: Video Representation Learning via Track Mask and Predict.** CVPR 2026.

---

# Appendix A — Pneuma Compatibility

The existing Pneuma work remains useful because it already provides:

- chunk ingestion;
- structured video analysis;
- relational SQL memory;
- object-instance tracking;
- state-change records;
- multi-aspect vector search;
- temporal retrieval;
- query planning;
- evidence verification.

But some tables/transition structures are inactive in the benchmark path and the graph itself is simpler than the original conceptual framing.

Those are engineering issues to clean before downstream comparison, not research contributions.

---

# Appendix B — Manuscript Diagnostic Table

| Backbone | Pooling | SNS raw ↑ | SNS motion-matched ↑ | Matched coverage | Motion-only `abs(AUROC-0.5)` ↓ | Margin ↑ | Linear state ↑ | MLP state ↑ | Identity R@1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2.1 B | mask | | | | | | | | |
| V-JEPA 2.1 L | mask | | | | | | | | |
| V-JEPA 2 | mask | | | | | | | | |
| DINOv2 | mask | | | | | | | | |

---

# Appendix C — Adapter Result Table

| Method | SNS motion-matched ↑ | Unseen-instance SNS ↑ | State R@1 ↑ | STATUS SCI ↑ | Identity R@1 ↑ | Effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | | | | | | |
| PCA/whiten | | | | | | |
| Positive diagonal, margin triplet | | | | | | |
| Linear residual, margin triplet | | | | | | |
| Non-residual MLP, margin triplet | | | | | | |
| Residual bottleneck, margin triplet | | | | | | |
| Role-shuffled residual control | | | | | | |

---

# Appendix D — Downstream Memory Table

| Variant | Evidence R@5 ↑ | MRR ↑ | tIoU ↑ | QA ↑ | Query latency ↓ | Index size ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Pneuma symbolic | | | | | | |
| + raw V-JEPA | | | | | | |
| + state-adapted feature | | | | | | |
| + visual validity | | | | | | |
| + temporal expansion | | | | | | |

---

# Final Research Discipline

The project succeeds if it produces a clear and reproducible representation result.

It does **not** need to solve the entire long-video-memory problem.

The correct progression is

\[
\boxed{
\text{diagnose}
\rightarrow
\text{minimally correct}
\rightarrow
\text{verify generalization}
\rightarrow
\text{optionally demonstrate memory utility}
}
\]

and not

\[
\text{JEPA}
+
\text{world model}
+
\text{memory hierarchy}
+
\text{graph}
+
\text{MoE}
+
\text{LLM agent}.
\]

That scope discipline is necessary for the work to remain scientifically interpretable.
