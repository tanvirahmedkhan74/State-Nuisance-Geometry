# PHASE 2 PLAN — StateOrder (v2, audited)

**Project:** State–Nuisance Geometry in Predictive Video Representations for Object-State Understanding
**Document role:** Technical implementation plan for Phase 2 (post-Phase-1-extraction)
**Revision:** v2 — theoretical, mathematical, and literature audit applied (see §0 for the correction log)
**Snapshot date:** 21 August 2026
**Governing docs (locked):** `docs/project_methodology.md` (v1.1), `docs/phase1_implementation_plan.md` (§4 math, §16 CLI contract, §18 artifact tree, §19 sanity checks, §20 kill criteria)
**Hardware target:** NVIDIA RTX 5090 (32 GB class) primary; RTX 4050 Laptop (6 GB) retained; Runpod for overflow/confirmation
**Phase 1 status:** pipeline preflight extraction complete and verified; **no scientific state claims made or unlocked**

---

## 0. Audit Changelog (v1 → v2)

Mathematical audit against `phase1_implementation_plan.md` §4 and `project_methodology.md` §§9–16, plus new numerical analysis of the Phase 1 covariance spectrum. Corrections applied:

| # | v1 defect | v2 correction |
|---|---|---|
| 1 | Adapter param counts wrong (`mlp_nonresidual`/`residual_bottleneck` listed ~0.46M) | Exact: **395,776 / 395,777** (~0.40M), matching plan §4.6's audited 395,777 (incl. LN, biases, α) |
| 2 | SNS tie tolerance τ unspecified | Preregistered **τ = 1e-6** (plan §4.2); tie rate reported separately; strict SNS and tie-aware SNS both reported |
| 3 | VICReg-style regularization mentioned without the feasibility math | Full dimensional analysis added (§3.4): on unit-normalized outputs Σⱼ Var(uⱼ) ≤ 1 − ‖E[u]‖²; with our measured ‖E[u]‖ ≈ 0.918, avg per-dim std ≤ 0.014 → γ=1 (or any γ ≳ 0.036 = √(1/768)) **infeasible**; rescue terms must act on unnormalized outputs with train-frozen dimension-aware γ ∈ {0.10, 0.20} (measured avg per-dim std: box 0.378, full 0.233) |
| 4 | Barlow Twins not analyzed | BT requires per-column z-standardization; the diagonal-1 target then constrains a standardized space, decoupling the loss from the cosine metric geometry we adapt. **Discouraged**; VICReg-style on unnormalized outputs is the sanctioned rescue form (methodology §14) |
| 5 | Whitening control lacked numeric guards | Measured covariance condition number ≈ 10¹²–10¹³ (λ_min ~ 1e-15): unguarded whitening divides by numerical zero. Guards made mandatory: shrinkage ρ ∈ {1e-4, 1e-3, 1e-2} × mean-positive-eigenvalue, eigenvalue floor 1e-6 × same, post-whiten finiteness + condition-number bound checks (§3.2) |
| 6 | GRU predictor params wrong (~2.7M) | Correct: **~3.94M** (2-layer GRU-512 + 512→768 readout) |
| 7 | Linear-AR predictor underspecified | Exact form: Δ̂ₜ = A(uₜ − uₜ₋₁), A ∈ ℝ⁷⁶⁸ˣ⁷⁶⁸ (590,593 params), ûₜ₊ₕ = normalize(uₜ + h·Δ̂ₜ) |
| 8 | Predictor lacked degenerate-solution controls | Added: **copy-last is the mandatory reporting floor** (slowly-varying low-rank features make it strong at short horizons); prediction-norm + effective-rank monitors for predictor collapse; minimum segment length k+H_max+1 = 17 |
| 9 | No pipeline rehearsal track | Added Step 4a: full diagnostics/adapter **code-path rehearsal** on the Phase 1 proxy features, namespaced and selection-free (no hyperparameter from rehearsal may transfer to the real run) |
| 10 | `context_tokens` near-degeneracy noted but no decision rule | Preregistered rule (§1.2): cos(context, full) > 0.999 on a subset ⇒ context control marked non-discriminative there; object-pixel-erasure carries the complement-control burden |
| 11 | Batch composition rule unstated | Exact sampler rule from plan §4.6: sample dependency/transition groups uniformly, at most one triplet per sampled transition per batch; early-stop on dependency-group-macro validation margin |
| 12 | Literature grounding implicit | New §6 alignment audit (JEPA/world-model/probing/anti-collapse/statistics, 2024–2026) with explicit novelty-boundary recheck |
| 13 | Full-PCA invariance test absent from validation list | Added as a hard check (§3.2): full-dim PCA rotation must reproduce center-only cosine distances to tolerance |
| 14 | Measured spectrum not exploited | §1.1 now records λ_max, λ_min, condition numbers, and ‖μ‖/‖z‖ ≈ 0.92 mean-direction dominance — justifying center-only as a first-class control and grounding all whitening/γ numbers |

---

## 1. Phase 1 Verification Report (re-verified for v2)

### 1.1 Verified state + measured spectrum (proxy features, dim 768, N = 13,361)

| Check | Result |
|---|---|
| Catalog integrity | 3 rows (`box`, `full`, `context_tokens`), 13,361 × 768 FP32, `status=complete`; all `features_sha256`/`index_sha256` re-computed and matched |
| Row/order alignment | `index.parquet.observation_id` order == `observations.parquet` order |
| Provenance chain | observations SHA `bdd83f02…` consistent across metadata + splits report; checkpoint/config/source-commit hashes consistent |
| Determinism | Full run vs 8-row smoke re-extraction: **max abs diff = 0.0** (all 3 pools) — bitwise reproducible |
| Split integrity | `duplicate_group_id`, `(asset_proxy_id, sequence_id)` never cross splits; phase strata exactly 70/15/15 |
| Triplet smoke | Box pool P(d_s>d_n) = 0.984, mean(d_s−d_n) = 0.121 — **pipeline smoke only, confounded proxy** |
| Fail-closed gates | Curation validator exits 2 as designed; `physical_state_claim_allowed: false`; triplets tagged `interaction_phase_proxy_not_physical_state` |

**Measured covariance spectrum (new for v2):**

| Pool | ‖z‖ | ‖μ‖/‖z‖ | avg per-dim std | ΣⱼVar | λ_max | λ_min | cond | eff. rank (PR) |
|---|---|---|---|---|---|---|---|---|
| box | 26.30 | 0.918 | 0.378 | 109.6 | 21.16 | 2.3e−15 | 2.1e13 | 20.2 |
| full | 25.39 | 0.968 | 0.233 | 41.6 | 4.91 | 1.7e−15 | 4.9e12 | 24.6 |
| context_tokens | 25.53 | 0.967 | 0.240 | 44.2 | 5.27 | 1.8e−15 | 5.3e12 | 24.2 |

Three consequences now locked into the plan: (i) **mean-direction dominance** (‖μ‖/‖z‖ ≈ 0.92–0.97) makes center-only a mandatory first-class control, not a formality; (ii) **numerically null tail** (λ_min ~ 1e−15, cond ~ 10¹³) makes unguarded whitening undefined — §3.2 guards are mandatory; (iii) low effective rank (participation ratio (Σλ)²/Σλ² ≈ 20–25) sets the anti-collapse baselines for the adapter non-inferiority rule (ratio ≥ 0.90).

### 1.2 Representation gaps, leakage surfaces, and decision rules

1. **`context_tokens` ≈ `full`** (mean cos 0.9996; box occupancy ≈ 7.9/576 tokens). Decision rule: per subset, if cos(context, full) > 0.999, mark the context control non-discriminative and shift the complement-control burden to object-pixel-erasure (reported with its OOD caveat). Never interpret either as "background-only" — tokens are globally contextualized.
2. **Box pooling is localized** (cos(box, full) = 0.852) — the expected localization signal.
3. **Mask pool, object-pixel-erasure, video decode are code-complete but production-untested** — each requires an 8-observation smoke (hash-locked) before claim-level use; first video extraction additionally verifies per-frame decode indices.
4. **Leakage surfaces:** real extractions must use label-redacted `feature_inputs.parquet` (`--require-label-redacted-input`); `normalized_time_proxy` and `encoder_box_token_mass` enter shortcut-control reports only; proxy labels (confounded with hand presence and time) never support state claims.
5. **Missing sanity scripts:** `validate_feature_catalog.py`, `feature_sanity_report.py`, per-run `extraction.jsonl` + `metadata_sha256` catalog column; OMP Error #15 workaround documented.

---

## 2. Phase 2 Scope and Gate Ladder (preregistered, do not skip)

```
G1 data curation & split → G2 motion/common-support → G3 frozen extraction & raw geometry
→ G4 H-info (probes) & H-geometry (SNS) → G5 shortcut controls clean
→ G6 [conditional] adapter branch → G7 [conditional] predictor/world-model ablation
→ one atomic locked test → report
```

**Hard locks (methodology §39, §58):** no adapter training without validation evidence of decodable state ∧ misaligned geometry with headroom; no predictor training before the state representation validates (predictor is a post-validation ablation, never causal/physical evidence, never in the novelty claim); no ViT-L/1B/2B, no LoRA/fine-tuning, no FP8, no loss-term accumulation without a demonstrated failure.

---

## 3. Mathematical Specification (audited)

All distances in FP32. Notation: ẑ = z/(‖z‖₂+ε); d(a,b) = 1 − ẑₐᵀẑ_b ∈ [0,2] (= ½‖ẑₐ−ẑ_b‖² for unit vectors).

### 3.1 SNS estimator and margins (plan §4.2, verbatim contract)

For triplet row r with δ_r = d_s − d_n, tie-aware row score s_r = 1[δ_r > τ] + ½·1[|δ_r| ≤ τ] with preregistered **τ = 1e-6**; nested means row → anchor → transition → dependency group; **SNS = (1/G) Σ_g s̄_g** (equal-group macro). Report strict SNS = P(d_s > d_n) and the tie rate separately. Normalized margin Δᵢ = (d_s−d_n)/(d_s+d_n+ε), same nesting. Bootstrap resamples whole `dependency_group_id` components; a Cartesian product of frames is never independent evidence.

### 3.2 Geometry controls (train-fit only; selection on validation only)

With train mean μ and eigendecomposition C = VΛVᵀ (fit on unique, object-balanced training observations):
- center-only: ẑ after (z − μ);
- PCA-k: y = V_kᵀ(z−μ), L2-normalize;
- whitening-k: y = (Λ_k + ρI)^(−1/2) V_kᵀ(z−μ), with **ρ ∈ {1e-4, 1e-3, 1e-2} × mean-positive-train-eigenvalue** and **eigenvalue floor 1e-6 × same** (measured λ_min ~ 1e−15 makes these mandatory, not optional);
- dimension-matched seeded random orthogonal projection;
- k ∈ {64, 128, 256} only when k ≤ min(768, N_unique_train − 1) and each retained dimension has ≥ 2 unique train values (record skipped k);
- **invariance check:** full-dimensional PCA rotation must reproduce center-only cosine distances to tolerance (implementation invariant);
- **post-whiten guards:** all projected values finite; projected condition number ≤ 1/(floor-relative) bound recorded; variance-amplification ratio reported.

### 3.3 Residual adapter and ordering loss (plan §4.6, verbatim contract)

h = W₁ LN(z), v = z + α·W₂ GELU(h), u = v/(‖v‖₂+ε); W₁ ∈ ℝ²⁵⁶ˣ⁷⁶⁸, W₂ ∈ ℝ⁷⁶⁸ˣ²⁵⁶, α₀ = 1e−3 (nonzero; both-α-and-residual-zero initialization is a dead step — forbidden). Params: LN 1,536 + W₁ 196,864 + W₂ 197,376 + α 1 = **395,777**. First-step gradient test for W₁, W₂, α; identity adapter (u = ẑ) must reproduce frozen metrics exactly.

L_ord = (1/B) Σᵢ max(0, m + d(uₐ,uₙ) − d(uₐ,uₛ)), m ∈ {0.05, 0.10, 0.20} selected from frozen training distance distributions + validation performance only. Batch composition: sample dependency/transition groups uniformly, at most one triplet per sampled transition; early stop on dependency-group-macro validation margin. AdamW lr 3e-4, wd 1e-4, batch 512 triplets, FP32, seeds 20260820–20260824.

Same-loss comparators (identical supervision/budget/stopping/seeds): `positive_diagonal` (768 params, softplus scale, renormalize), `linear_residual` (592,129: z + α·W·LN(z), W ∈ ℝ⁷⁶⁸ˣ⁷⁶⁸), `mlp_nonresidual` (395,776), `role_shuffled` control (nuisance/state swap, p = 0.5). Cosine-distance margin domain [0,2] enforced in the loss.

### 3.4 Rescue losses — exact forms and activation conditions (methodology §§11–15)

Activated **only** on the corresponding demonstrated failure; one weighting level each; no redundant outer scales:

- **L_dir** (categorical transition labels exist): r_ab = ψ_φ[uₐ, u_b, u_b−uₐ, uₐ⊙u_b]; L_dir = −Σ_c y_c log p_φ(c | r_ab). Direction is not representable by a symmetric metric — this head is the sanctioned fix.
- **L_trans** (validated language targets): InfoNCE aligning W·r_ab with frozen text embeddings t_c, temperature τ.
- **L_id** (identity degradation): triplet with **state/category-matched positives only** (a cross-state positive directly opposes L_ord).
- **L_geom** (geometry damage): (1/|P|) Σ_{(i,j)∈P} (uᵢᵀu_j − ẑᵢᵀẑ_j)² over a predeclared train-only pair protocol.
- **L_VC (VICReg-style, corrected):** covariance-only terms are minimized by collapse, so the variance hinge must be active. **On unit-normalized outputs the standard γ = 1 target is infeasible:** Σⱼ Var(uⱼ) = 1 − ‖E[u]‖² ≤ 1, and with our measured ‖E[u]‖ ≈ 0.918 the total variance budget is ≈ 0.157, i.e. avg per-dim std ≤ 0.014 (even the mean-zero bound √(1/768) = 0.036 applies). Therefore compute on **unnormalized adapter outputs v** (or √d·u): L_var = (1/d) Σⱼ max(0, γ − √(Var(vⱼ)+ε)), L_cov = (1/d) Σ_{i≠j} C(v)ᵢⱼ², with **γ train-frozen and dimension-aware**: γ ∈ {0.10, 0.20} (≈ 0.25–0.5 × measured avg per-dim std of the box pool, 0.378; recompute per feature key and record). Fit on unique dependency-balanced observations, not duplicated triplet positions.
- **Barlow Twins (not adopted):** BT's identity-diagonal target presupposes per-column z-standardization, which constrains a standardized projection space rather than the adapter's cosine geometry; the off-diagonal reduction term alone is collapse-compatible. VICReg-style on unnormalized outputs is the sanctioned rescue form.
- **Collapse monitors (always on, not losses):** per-dim variance, singular spectrum, effective rank (participation ratio, PR = (Σλ)²/Σλ²), mean pairwise cosine. Non-inferiority: ER ratio = PR(adapted)/PR(frozen) ≥ 0.90 vs §1.1 baselines.

### 3.5 Predictor objective (methodology §16, with degeneracy guards)

L_pred = Σ_{h∈H} w_h (1 − cos(ûₜ₊ₕ, sg(uₜ₊ₕ))), H = {1,2,4,8}, w_h = 1/|H| uniform (uniform avoids over-weighting the easy h=1 horizon); targets stop-gradient; inputs and targets from the same feature key (P1: adapted u for both). Predictive residual e_t = 1 − cos(ûₜ, uₜ).

Guards added in v2: **copy-last (ûₜ₊ₕ = uₜ) is the mandatory reporting floor** — per-horizon improvements are reported as Δ over copy-last, and copy-last is expected to be strong at h ∈ {1,2} on slowly-varying low-rank features (PR ≈ 20); monitor ‖û‖ distribution and PR({û}) for predictor collapse; minimum usable segment length k + max(H) + 1 = **17** steps; fixed stride, homogeneous fps per segment, nonuniform-Δt segments excluded with hashed reasons.

---

## 4. Implementation Sequence

### Step 0 — Repo & validation hardening (blocking, ~2 h)
`git init` + `.gitignore` (datasets/, checkpoints/, `*.egg-info`, `__pycache__`, `.cache`, `*/temporary/`); remove junk `2]`; initial commit. Build + run `validate_feature_catalog.py` and `feature_sanity_report.py` against `01_proxy` (green required). Fix plan §16.4 run-id drift; canonicalize the runbook.

### Step 1 — Curation (the real scientific gate, human-in-the-loop)
30-transition feature-blind pilot → powered cohort → freeze `curated_observations.parquet` (identity, state family/label, observability, stable windows, per-sampled-frame boxes, mask subset, hashes, nuisance tags, duplicate/asset groups). Validate fail-closed (`validate_phase1_manifest.py --stage curated --require-state-labels --require-physical-object-ids --require-observability --require-aligned-boxes`). **Exit:** clean; reviewer agreement ≥ 0.80; ≥ 50 independent test groups targeted (≥ 334 total pre-stratification) or the result is labeled exploratory.

### Step 2 — Object/video-disjoint splits (before triplets)
`build_phase1_splits.py` on `physical_object_id,source_video_id,transition_id,verified_asset_group_id,duplicate_group_id`; 70/15/15; stratify `state_family,state_label` + multilabel `nuisance_tags`; seed 20260820. **Exit:** zero overlap on object/video/frame/hash; leakage report green.

### Step 3 — Motion controls (minimum backend: reviewed-stationary region kinematics)
`estimate_motion_controls.py` → `validate_motion_controls.py` (complete homogeneous schema, no fit) → `fit_motion_scaler_quality.py` (train-fit; `max_iqr_native_floor` denominator; drop train-constant dims) → `assemble_phase1_analysis_manifest.py` → analysis-stage validation → `build_phase1_candidate_triplets.py` (temporal-gap tolerance ≤ 2 frames, no motion caliper) → `select_motion_calipers.py` (signed/severity grids {0.25, 0.50, 1.00, 1.50}; smallest calipers with ≥ 50% validation coverage; no test access) → `finalize_phase1_triplets.py` (homogeneous schema enforced; `eligible_base` + `confirmatory_eligible` selectors) → `build_phase1_release_views.py` (label-redacted `feature_inputs.parquet`; hash-locked sealed test views; `release_views.json`) → `eval_motion_role_control.py` (scalar-threshold + logistic leakage models; 999 permutations, `swap_all_paired_roles_in_group`). **Exit (Gate 3b):** `abs(AUROC−0.5)` near its grouped permutation null on the confirmatory subset; adequate matched coverage and independent groups; else resample or no-go.

### Step 4 — Real frozen extraction (`03_features/`)
Label-redacted + motion-provenance inputs; primary key `vjepa21b/layer11/box/original/all`; mask run on the frozen common-row subset (else `record_phase1_skip.py --component mask_common`); `object_pixel_erased_mean` variant; append-only catalog; `extraction.jsonl` + `metadata_sha256` now mandatory.

```bash
conda run -n llm python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml \
  --observations artifacts/phase1/01_manifests/feature_inputs.parquet \
  --require-label-redacted-input --require-motion-provenance --no-motion-quality-filter \
  --layers 11 --pools box,full,context_tokens --input-control original --subset-name all \
  --feature-key-prefix vjepa21b --batch-size 1 --workers 0 \
  --run-id vjepa21b_l11_all_original --output-root artifacts/phase1/03_features/vjepa21b_l11_all_original \
  --catalog artifacts/phase1/03_features/catalog.parquet
```
Video-decode smoke first if any curated rows are video. **Exit:** catalog validator green; 8-row determinism re-run 0.0 max diff; feature-sanity report recorded.

### Step 4a — Pipeline rehearsal on Phase 1 proxy features (parallel track; new in v2)
While curation proceeds, exercise the *entire* downstream chain (geometry controls, SNS, probes, shortcut controls, adapter training, predictor harness) on the existing `01_proxy` features under `artifacts/phase2/rehearsal_01_proxy/` with `rehearsal` feature-key prefixes. **Constraint (leakage rule):** rehearsal is code-path validation only — no hyperparameter, margin, caliper, or architecture selection made on rehearsal may transfer to the real run; all real-run selections are made afresh on real train/validation; rehearsal artifacts carry `not_scientific_evidence` in every metadata file. This is sanctioned by plan §16.3 ("available static data may exercise the code path … under an explicit proxy namespace").

### Step 5 — Frozen geometry diagnostics (train/validation only)
`fit_geometry_controls.py` per §3.2 (with all guards), then `eval_state_nuisance.py` per §3.1:

```bash
conda run -n llm python scripts/eval_state_nuisance.py \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --controls artifacts/phase1/04_controls/vjepa21b_l11_box \
  --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet \
  --observations artifacts/phase1/01_manifests/analysis_trainval.parquet \
  --all-selector eligible_base --matched-selector confirmatory_eligible --validate-match-provenance \
  --report sns_all,sns_motion_matched,raw_margin,normalized_margin,motion_strata \
  --select-control-on validation --evaluate-split validation --no-test-access \
  --bootstrap-group dependency_group_id --seed 20260820 \
  --output-root artifacts/phase1/05_geometry/vjepa21b_l11_box_validation
```
**Exit (Gates 1, 3b):** SNS-all + SNS-motion-matched with grouped CIs, matched coverage, group counts; motion role control near null.

### Step 6 — Information probes + negative controls
Linear probe (multinomial logistic, IPF class×dependency-group weighting, realized weight mass reported, group-aggregated metrics) and MLP probe (1 hidden layer 512, `dependency_group_then_observation` sampler, inverse-frequency class weights); shuffled-training negative controls (permute within `state_family × object_category_manual` blocks; 99 permutations; locked hyperparameters; scored against true validation labels; labeled negative controls, not formal nulls). **Exit (Gate 2):** state decodable above chance; shuffled controls fail to generalize. Regime table: high probe + low SNS = target regime; low/low = information problem (pivot: intermediate layers); high/high = no headroom.

### Step 7 — Shortcut controls
`run_shortcut_controls.py`: full-frame, context-tokens (with §1.2 degeneracy rule), object-pixel-erasure (OOD caveat reported), box/mask geometry, metadata, temporal, hand-presence — all on identical eligible rows; region-pooled readout must materially exceed each control.

### Step 8 — Conditional adapter branch (Gate 3)
Per §3.3: five same-loss transforms, margins {0.05, 0.10, 0.20}, seeds 20260820–20260824, FP32, batch 512, group-balanced sampler; first-step gradient test; identity-adapter reproduction test.

```bash
conda run -n llm python scripts/train_state_adapter.py \
  --architecture residual_bottleneck --bottleneck 256 \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --triplets artifacts/phase1/01_manifests/triplets_trainval.parquet \
  --confirmatory-only --fit-split train --select-split validation --no-test-access \
  --sampler dependency_transition_balanced --loss margin_triplet --margin-grid 0.05,0.10,0.20 \
  --lr 3e-4 --weight-decay 1e-4 --batch-size 512 --precision fp32 \
  --seeds 20260820,20260821,20260822,20260823,20260824 \
  --output-root artifacts/phase1/08_adapters/residual/vjepa21b_l11_box
```
Rescue losses per §3.4 only on demonstrated failure. **Exit (Gates 3–5):** residual adapter beats same-loss simpler transforms within CI; identity-R@1 change lower grouped-CI bound ≥ −0.02; ER ratio ≥ 0.90; else `adapter_component_resolution.json = skipped` with hashed evidence (a valid outcome).

### Step 9 — Preservation & collapse evaluation
Identity/persistence retrieval (non-overlapping query/gallery frames, verified same-track positives, state/category/context-matched hard negatives; within-video results called track persistence only); covariance/variance/spectrum/ER monitors vs §1.1 baselines; state-prototype collapse check.

### Step 10 — One atomic locked test + report
`freeze_phase1_selection.py` → `run_locked_phase1_test.py` (exclusive access-start marker before the first sealed read, retained on failure; transactional commit; completion marker only on full success) → `build_phase1_report.py` producing the plan §21 diagnostic table (rows preserved with `not run` + skip reasons; motion-only AUROC and coverage reported once as protocol diagnostics).

---

## 5. JEPA Predictor / World-Model Pipeline (conditional, post-validation ablation)

**Activation gate:** only after the adapter step validates (methodology §50). Purpose: test whether state-aware geometry improves **object-level temporal prediction**. It is an ablation — explicitly not evidence of causal or physical understanding, and never part of the core novelty claim.

### 5.1 Data assembly
`scripts/build_predictor_sequences.py` → `artifacts/phase2/11_predictor/sequences.parquet`: per `physical_object_id` within stable segments, fixed stride, homogeneous fps, **minimum length 17** (k=8 context + max horizon 8 + 1 target); dependency-group-disjoint splits reused verbatim; null motion/track fields stay null; exclusions hashed. Feature rows resolved through the catalog (P0: frozen key; P1: adapted key + adapter checkpoint hash; P2: + transition-type conditioning where labels exist).

### 5.2 Architectures (cached 768-d inputs, FP32)

| Variant | Spec | Params |
|---|---|---|
| **P-A causal transformer (primary)** | input proj 768→256; 4 layers, d=256, 8 heads, MLP 1024, causal mask, RoPE over step index; per-horizon linear readouts 256→768 (×4) | **4,145,920** |
| P-B GRU | 2-layer GRU-512 + 512→768 readout | **3,939,072** |
| P-C linear AR | Δ̂ₜ = A(uₜ − uₜ₋₁), A ∈ ℝ⁷⁶⁸ˣ⁷⁶⁸ (+bias); ûₜ₊ₕ = normalize(uₜ + h·Δ̂ₜ) | **590,593** |
| P-D copy-last | ûₜ₊ₕ = uₜ (reporting floor, no params) | 0 |

Training: AdamW lr 3e-4, wd 1e-4, batch 256 segments, cosine schedule, early stop on group-macro validation loss; seeds 20260820–20260824; grouped bootstrap CIs.

```bash
conda run -n llm python scripts/train_state_predictor.py \
  --sequences artifacts/phase2/11_predictor/sequences.parquet \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --feature-key vjepa21b/layer11/box/original/all \
  --architecture causal_transformer --context-steps 8 --horizons 1,2,4,8 \
  --loss cosine_multihorizon --lr 3e-4 --weight-decay 1e-4 --batch-size 256 --precision fp32 \
  --fit-split train --select-split validation --no-test-access \
  --seeds 20260820,20260821,20260822,20260823,20260824 \
  --output-root artifacts/phase2/11_predictor/causal_transformer/vjepa21b_l11_box

conda run -n llm python scripts/eval_state_predictor.py \
  --predictor-roots artifacts/phase2/11_predictor/{copy_last,linear_ar,gru,causal_transformer}/vjepa21b_l11_box \
  --sequences artifacts/phase2/11_predictor/sequences.parquet \
  --feature-catalog artifacts/phase1/03_features/catalog.parquet \
  --evaluate-split validation --no-test-access --bootstrap-group dependency_group_id \
  --report-floor copy_last --output-root artifacts/phase2/11_predictor/eval/vjepa21b_l11_box_validation
```

### 5.3 Ablations and reporting

| ID | State adapter | Predictor | Question |
|---|---|---|---|
| P0 | ✗ | ✓ | raw predictive ability of frozen features |
| P1 | ✓ | ✓ | does corrected geometry improve prediction? |
| P2 | ✓ | ✓ (+transition semantics) | does transition-type signal help? |

Report per-horizon `1−cos` **and Δ vs copy-last floor**, predictive-residual distributions, prediction-norm and prediction effective-rank (collapse monitor), grouped CIs, paired P0→P1→P2 comparisons. Descriptive ablation only.

---

## 6. Literature Alignment Audit (2024–2026)

| Paradigm in this plan | Literature alignment | Verdict |
|---|---|---|
| Frozen-backbone probing (linear/MLP, layer-wise) | Standard protocol from the SSL linear-probe lineage (SimCLR/DINOv2 evaluation practice); our additions — IPF class×group weighting, dependency-group-macro aggregation — exceed the common rigor bar | Aligned; defensible |
| Post-hoc metric readout on frozen features (margin triplet) | Ordinary ranking loss (FaceNet/Hermans lineage); methodology §11 explicitly disclaims objective novelty — the claim is the diagnostic + minimal correction, not the loss | Aligned; novelty honest |
| Same-loss comparator battery (positive-diagonal, linear-residual, non-residual MLP, role-shuffled) | Satisfies Gate 3 fairness (identical supervision/budget/seeds); prevents the "only vs frozen V-JEPA" weakness (methodology §41) | Aligned; publication-grade |
| Latent-space prediction on frozen features (§5) | V-JEPA 2 (Assran et al. 2025, ref 27) builds an action-conditioned predictor over frozen V-JEPA latents for planning (the audited `vjepa2` repo ships `ac_predictor`); DINO-WM (Bar et al. 2024/2025, "DINO World Models") validates dynamics-on-frozen-features as an active paradigm. Our §5 is the object-level, multi-horizon, stop-gradient analog with a copy-last floor | Aligned; conservative and current |
| Anti-collapse regularization | VICReg (Bardes et al., ref 25) and Barlow Twins (Zbontar et al. 2021) — with our §3.4 dimensional-feasibility correction (unit-sphere γ infeasibility) and effective-rank monitoring (Roy & Vetterli 2007 notion, PR implementation) | Aligned + a genuine rigor improvement over naive porting |
| Motion-matched confirmatory estimand + leakage nulls | Consistent with the confound-aware evaluation trend (TrackMAE's motion-target insight, ref 28, absorbed as measurement not training; STATUS/DEHOI shortcut warnings, refs 10–11) | Aligned; this is the project's differentiator |
| Statistics | Cluster/grouped bootstrap + dependency-group paired-role permutation = cluster-randomized inference; avoids frame-level pseudo-replication | Aligned |
| Benchmarks (STATUS, OSCaR, HowToChange/VidOSC, TOC-Bench) | Not run in Phase 2 — mechanism-level result first; real-data direction gate (Gate 6) can use a curated real subset; full benchmark transfer is Phase 3 | Aligned; scope-locked |
| DINOv2 / V-JEPA 2 / TrackMAE backbones | Post-gate descriptive comparisons only (refs 26–28); never JEPA causal ablations | Aligned |

**Novelty boundary recheck (unchanged, methodology §1.4):** the candidate novelty remains (1) explicit state-vs-nuisance metric diagnostic for dense predictive video features at object level; (2) post-hoc frozen-backbone residual correction on local ordering; (3) preservation + shortcut controls; (4) unseen-object generalization as primary target; (5) optional downstream memory-retrieval evidence. Phase 2 adds no new claims; the predictor ablation is explicitly excluded from the novelty claim.

---

## 7. Evaluation Suite & Script Inventory

| Script | Status | Purpose |
|---|---|---|
| `validate_feature_catalog.py`, `feature_sanity_report.py` | **build (Step 0)** | catalog/provenance/determinism audit; spectrum & degeneracy report |
| `validate_phase1_manifest.py`, `build_phase1_splits.py`, `build_phase1_release_views.py`, `extract_features.py` | exists | data gate, splits, sealed views, extraction |
| `estimate_motion_controls.py`, `validate_motion_controls.py`, `fit_motion_scaler_quality.py`, `assemble_phase1_analysis_manifest.py`, `build_phase1_candidate_triplets.py`, `select_motion_calipers.py`, `finalize_phase1_triplets.py`, `eval_motion_role_control.py` | build | motion + triplet chain (Step 3) |
| `fit_geometry_controls.py`, `eval_state_nuisance.py` | build | §3.1–3.2 geometry (Step 5) |
| `train_probe.py`, `run_shortcut_controls.py` | build | probes, shuffled controls, shortcuts (Steps 6–7) |
| `train_state_adapter.py`, `eval_state_adapter.py`, `record_phase1_skip.py`, `freeze_phase1_selection.py` | build | adapter branch (Step 8) |
| `eval_identity_retrieval.py`, `eval_collapse.py` | build | preservation/collapse (Step 9) |
| `run_locked_phase1_test.py`, `build_phase1_report.py` | build | atomic locked test + report (Step 10) |
| `build_predictor_sequences.py`, `train_state_predictor.py`, `eval_state_predictor.py` | build (gated) | §5 predictor ablation |

---

## 8. Artifact Directory Layout

```
artifacts/phase1/                       (locked Phase 1 tree, append-only)
  01_manifests/  curated_observations, splits, internal_*, feature_inputs (redacted),
                 triplets_trainval, sealed_test_targets/triplets, release_views.json
  02_motion/     motion_observations, motion_validated, caliper_selection.json,
                 role_control_validation/
  03_features/   <run_id>/{features_{pool}.npy, index.parquet, metadata.json,
                 complete.json, extraction.jsonl}, catalog.parquet,
                 mask_component_resolution.json
  04_controls/ 05_geometry/ 06_probes/ 07_shortcuts/ 08_adapters/ 09_adapter_eval/ 10_report/
artifacts/phase2/
  rehearsal_01_proxy/        (Step 4a; every file marked not_scientific_evidence)
  11_predictor/  sequences.parquet, {copy_last,linear_ar,gru,causal_transformer}/,
                 eval/, predictor_component_resolution.json
```

Every artifact directory records resolved config, input/source hashes, seeds, command line, environment summary, completion status (plan §14).

---

## 9. Hardware & Memory Configuration

### 9.1 RTX 5090 (32 GB class) — Phase 2 primary

| Workload | Config | Precision | Est. peak VRAM | Policy |
|---|---|---|---|---|
| ViT-B 16×384 extraction | batch 1 (pinned) → 2–4 after peak-VRAM smoke re-audit | BF16 fwd / FP32 pool | ~3–5 GiB (b1) · ~6–12 GiB (b4) | never reduce frames/resolution; parity smoke vs b1 (cosine agreement) before adopting b>1; extraction remains deterministic at b1 (reproducibility anchor) |
| ViT-L 16×384 extraction | batch 1–2, post-gate only | BF16 | ~10–14 GiB (b1) · ~18–26 GiB (b2) | never before ViT-B gates pass |
| Object-pixel-erasure re-encode | same as extraction | BF16 | same | separate run-id |
| Probes / adapters (cached) | batch 512 | FP32 | < 1 GiB | trivial |
| Predictor training (cached) | batch 256 | FP32 | < 2 GiB | trivial |
| Motion estimation / PCA-eigh | — | FP64 intermediates | < 4 GiB | CPU-acceptable |

Storage: ~41 MB per pool per 13.4k rows; full Phase 2 feature set (3 pools + mask + erasure + indices) ≈ 250 MB per 13.4k rows. Dataset payload (19.9 GB) local.

### 9.2 RTX 4050 Laptop (6 GiB) — retained
Protocol anchor (b1, workers 0, one layer; the full 13,361-image proxy run completed on it). If the locked 16×384 protocol OOMs anywhere: migrate the identical frozen extraction to Runpod (plan §17 checklist) — never silently reduce the protocol.

### 9.3 Runpod
ViT-L confirmation and large sweeps, post-gate. Launcher-only difference; arguments identical.

---

## 10. 48-Hour Validation Plan (checkpoints with exit criteria)

**H0 (+2h)** — 5090 env verified (driver, CUDA build, BF16, peak-VRAM smoke); repo hardening committed; catalog validator + sanity report green on `01_proxy`; unit suite green; **rehearsal track started (Step 4a)**.
**H6 (data gate)** — curation pilot reviewed (30 transitions; agreement ≥ 0.80) or explicit blocker memo; splits leakage-green. *Fail → stop; the bottleneck is data, not compute.*
**H12 (motion gate)** — motion chain complete for the pilot cohort; homogeneous schema; calipers selected on train/validation; role control near its grouped permutation null; matched coverage ≥ 50% or documented resample plan. Rehearsal chain (SNS→probes→shortcuts→adapter harness) executes end-to-end on proxy features by this point.
**H24 (extraction + raw geometry)** — real `03_features` extraction complete (+ mask if powered subset); determinism smoke 0.0; SNS-all + SNS-motion-matched + margins with grouped CIs; PCA/whitening controls with all §3.2 guards.
**H36 (information + shortcuts)** — linear/MLP probes + shuffled controls; shortcut controls (with §1.2 degeneracy rule); regime identified.
**H48 (go/no-go + freeze)** — Gates 1–3b decision recorded; adapter branch authorized or skip recorded with hashed evidence; predictor activation decision recorded; `frozen_selection.json` prepared; locked-test checklist reviewed. Deliverable: `artifacts/phase1/10_report/phase2_48h_checkpoint.md` + plan §21 table rows executed so far.

Each checkpoint writes a hashed manifest; test labels/roles remain sealed until the atomic locked test.

---

## 11. Risk Register & Kill Criteria

(Highest-probability Phase 2 risks; full lists in plan §20, methodology §40/§54.)

1. **Curation infeasibility** → exploratory-only diagnostic, terminate method claim (kill #1).
2. **Motion matching destroys common support** → matched SNS not estimable; resample before no-go (kill #3).
3. **Shortcut leakage** (hands/temporal/metadata/geometry) → audit and correct before re-running gates (kill #6).
4. **No geometry headroom** (motion-matched SNS near adjudication ceiling with tight cluster CI) → diagnostic without adapter; predictor still runnable as descriptive ablation (kill #8).
5. **Simpler same-loss transform ties the residual adapter** → keep the diagnostic, drop the special-method claim (kill #9).
6. **Collapse / preservation failure** (identity R@1 CI bound < −0.02 or ER ratio < 0.90) → predeclared §3.4 rescues only (kill #11).
7. **Whitening instability** (new, from §1.1 spectrum) → mandatory shrinkage + floor guards; if whitened controls are numerically unstable at all grid points, record skipped-with-reason rather than loosen guards.
8. **Predictor degeneracy** (new) → copy-last floor unbeatable at short horizons ⇒ report as-is; never tune horizons post hoc to flatter the learned models.
9. **5090 OOM at locked protocol** → Runpod migration, protocol unchanged (kill #12).

---

## 12. Compliance Checklist (against locked methodology)

- [ ] No test access before the atomic locked test (sealed views + access markers enforced in code)
- [ ] Margin/PCA/whitening/scalers/calipers fit on train, selected on validation only; rehearsal makes no selections
- [ ] `margin_triplet` name, `dependency_group_id`, memory-mappable arrays + Parquet indices (no opaque `.pt` cache)
- [ ] SNS nesting with τ = 1e-6; strict SNS + tie rate reported; grouped bootstrap everywhere
- [ ] Whitening shrinkage/floor guards active (cond ≈ 10¹³ makes them mandatory)
- [ ] Adapter: first-step gradient test + identity-adapter reproduction test; ER ratio ≥ 0.90 vs recorded baselines
- [ ] VC-style terms only on unnormalized outputs with train-frozen dimension-aware γ; BT not adopted
- [ ] Shuffled/role controls labeled negative controls, not formal nulls
- [ ] Predictor is post-validation, descriptive-only, copy-last-floored, never in the novelty claim
- [ ] No FP8; BF16 forward / FP32 pooled, always
- [ ] Proxy never cited as state evidence; `physical_state_claim_allowed: false` respected in every downstream artifact
