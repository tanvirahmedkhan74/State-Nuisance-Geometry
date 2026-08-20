# PROJECT STATE — StateOrder

**Working title:** State–Nuisance Geometry in Predictive Video Representations for Object-State Understanding
**Phase:** Phase 1 extraction **complete** (pipeline preflight); central scientific gates pending
**Snapshot date:** 21 August 2026
**Authoritative sources:** `docs/project_methodology.md` (v1.1), `docs/phase1_implementation_plan.md`, `docs/phase1_cli_runbook.md`

---

## 1. Core Goal & Research Hypothesis

### What we are building

A **frozen-backbone diagnostic and correction pipeline** that asks whether modern predictive video foundation representations (V-JEPA 2.1) organize object features so that **semantic object-state changes are more salient in representation geometry than state-preserving nuisance variation** — and, where they do not, whether a tiny object-local residual adapter can correct the geometry without destroying object persistence.

For a triplet `T = (x_a, x_n, x_s)` — anchor, **same object/same state under nuisance change**, and **same object after a genuine semantic state change** — the primary diagnostic is the tolerance-aware nested group-macro score

```
SNS_tau = Macro_{g,t,a} { 1[d_s - d_n > tau] + 1/2 * 1[|d_s - d_n| <= tau] }
```

over connected dependency groups `g`, transitions `t`, anchors `a`, with cosine distance on L2-normalized pooled features. SNS is an internal diagnostic, **not** a community benchmark, and is reported together with the strict `P(d_s > d_n)` and the tie rate.

### The two-sided hypothesis regime

The project proceeds as a *method* only if both hold:

```
H-info:     state information is decodable by simple probes      (state probe strong)
H-geometry: native similarity geometry is materially misaligned  (SNS weak/moderate)
```

**High probe + low SNS = ideal geometry-mismatch regime.** High SNS + high probe = no headroom (kill condition). Low probe + low SNS = information problem, not fixable by a frozen adapter.

### Claim discipline (locked)

- We **do not** claim: latent disentanglement, a universal additive state-delta algebra, causal/physics understanding, a novel JEPA, or long-video-memory novelty (ObjectStream, R4DSG, MERIT et al. occupy that space).
- We **do** claim (candidate): (1) an explicit state-vs-nuisance metric diagnostic for dense predictive video features at object level; (2) a post-hoc frozen-backbone residual correction trained on local ordering; (3) preservation + shortcut controls; (4) unseen-object generalization as primary target; (5) optional downstream state-memory retrieval evidence.
- Pneuma is **downstream application infrastructure only**, never the novelty.

---

## 2. Theoretical Foundations

### 2.1 Frozen representation & object-local pooling

```
z_{o,t} = Pool(E(V_t), M_{o,t}),     E = V-JEPA 2.1 ViT-B/16 (frozen, 384 px)
```

- Encoder: V-JEPA 2.1 ViT-B, `patch_size=16`, `tubelet_size=2`, 16 frames → dense token grid `Z ∈ R^{B×8×24×24×768}`; images (1 frame) → `R^{B×1×24×24×768}`.
- Pooling: resize/crop geometry is **shared exactly** between RGB and per-frame boxes/masks (`short_side = int(384·256/224)`, bilinear, center crop). Boxes/masks are rasterized to `W ∈ [0,1]` per tubelet-patch cell by area-averaging; pooling is FP32 weighted:

```
z_o = sum_q W_q Z_q / (sum_q W_q + eps),   with a predeclared minimum token mass (1.0)
```

- **Pooling contract (Phase 1):** `box` (pilot primary) · `full` (whole-frame control) · `context_tokens` (complement region of box occupancy — localization diagnostic) · `mask` (frozen supported subset, compared with box only on common rows) · `object_pixel_erased_mean` (re-encoded shortcut control).
- **Caveat locked in metadata:** transformer tokens are *globally contextualized*; `context_tokens` and erasure are **not** "object-free" and must never be labeled background-only evidence.

### 2.2 Layer policy

- Primary key frozen to **final layer 11** (0-indexed of 12) at dim **768**.
- Multi-layer readout `z^multi = sum_ℓ α_ℓ P_ℓ(z^(ℓ))` with softmax weights is a cheaper pivot than backbone fine-tuning, permitted only if final-layer probes are weak.

### 2.3 Geometry, margins, and the ordering objective

Cosine distance `d(a,b) = 1 − ā·b̄`; normalized margin `Δ = (d_s − d_n)/(d_s + d_n + ε)`; aggregated with the same row→anchor→transition→group nesting; CIs bootstrapped by `dependency_group_id` (never by correlated frames or Cartesian-product triplets).

Residual adapter (near-identity initialization):

```
h = W1·LN(z),  r = W2·GELU(h),  z̃ = z + α·r,  u = z̃/||z̃||₂     (α₀ = 1e−3, bottleneck 256, <5M params)
```

MVP loss is the ordinary margin triplet/ranking loss (algebraically standard, not novel):

```
L_ord = max(0, m + d(u_a,u_n) − d(u_a,u_s)),   m ∈ {0.05, 0.10, 0.20}
```

Transition direction/type (a metric cannot represent direction) uses an ordered-pair head `ψ[u_a, u_b, u_b−u_a, u_a⊙u_b]`; anti-collapse (variance/covariance, effective rank) and preservation are **diagnostics in Phase 1**, activated as losses only on demonstrated failure.

### 2.4 Motion-controlled estimand

Low raw SNS is non-specific if state vs. nuisance roles differ in encoder-visible motion. The confirmatory estimand `SNS_motion-matched` restricts to triplets with frozen signed-change/severity calipers and componentwise quality gates on a homogeneous motion schema (`reviewed_stationary_region_kinematics` minimum backend), plus a motion-only role-prediction control whose grouped permutation null must be near zero. Motion-matched SNS, not raw SNS, carries the semantic claim.

### 2.5 Precision

BF16 forward (autocast, CUDA), **FP32 pooling/normalization/distances/probes/adapter training**. No FP8; numerical approximation is introduced only after BF16 behavior is stable.

---

## 3. Current Milestone — Phase 1: Extraction Complete

### 3.1 What Phase 1 actually delivered

Phase 1 exercised the **full data→split→extract→cache pipeline end-to-end on the static interaction-phase proxy** (13,361 observations; 3 feature pools at dim 768), while the **central physical-state experiment correctly stops at the fail-closed curation gate**: the downloaded EgoInteract metadata contains no verified physical-state, identity, observability, or aligned-region annotations (validator exit code 2, as designed).

| Component | Verified state |
|---|---|
| Environment | Python 3.12, torch 2.10.0+cu130, RTX 4050 Laptop (6,141 MiB, cc 8.9), BF16 supported |
| Test suite | **28/28 pass** (`python -m unittest discover -s tests`) — needs `KMP_DUPLICATE_LIB_OK=TRUE` on this Windows box (see §4) |
| Pinned V-JEPA source | `vjepa2` at commit `204698b45b3712590f06245fbfba32d3be539812`; `src/ app/ hubconf.py` clean; only 9 unrelated YAML edits + untracked ViT-L remain |
| ViT-B checkpoint | `checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`, 1,664,223,428 bytes, SHA-256 `848a77c3…df4d`, key `ema_encoder`, strict load |
| Dataset audit | `datasets/phase1/EgoInteract` at revision `313d1ef…107fea`: 68,073 payload files / 19,902,705,085 bytes; 3,390 videos TAS-aligned, zero alignment errors |
| Curation gate | Fail-closed as expected on smoke candidates (9 rows, 12 error classes, exit 2); full candidates queue: 540,610 bytes `candidates.parquet` |
| Proxy observations | **13,361 rows** — contact 7,943 / pre_contact 5,418; all `media_type=image` (1280×720); 1,574 asset proxies, 1,681 sequence-asset keys; 377 cross-directory exact-duplicate pairs reconciled via `duplicate_group_id` |
| Proxy exclusions | 6,451 rows (insufficient_phase_support 3,082; box_outside_encoder_crop 2,545; box_below_minimum_token_mass 767; no_positive_contacting_hand 57) |
| Proxy triplets | **3,362** — every row tagged `interaction_phase_proxy_not_physical_state`; `physical_state_claim_allowed: false` in `summary.json` |
| Splits | 70/15/15 group-disjoint by `asset_proxy_id`+`sequence_id`+`duplicate_group_id`, seed 20260820: train 9,357/1,102 groups · val 2,002/236 · test 2,002/236 (2,002+9,357+2,002 = 13,361) |

### 3.2 Feature catalog — integrity verified (re-verified this audit)

`artifacts/phase1/01_proxy/features/catalog.parquet` — 3 rows, all `status=complete`, FP32, dim 768, 13,361 rows each, run `interaction_phase_proxy_crop_v1`, layer 11, `input_control=original`, subset `proxy_crop_v1`:

| Feature key | File | Rows × Dim | SHA-256 (recomputed) | Match |
|---|---|---|---|---|
| `vjepa21b/layer11/box/original/proxy_crop_v1` | `features_box.npy` (41,045,120 B) | 13,361 × 768 | `7350d995…d302c0` | ✓ catalog |
| `vjepa21b/layer11/full/original/proxy_crop_v1` | `features_full.npy` | 13,361 × 768 | `15a4fa41…5ae05e93` | ✓ catalog |
| `vjepa21b/layer11/context_tokens/original/proxy_crop_v1` | `features_context_tokens.npy` | 13,361 × 768 | `608bcbbb…b96b2720d` | ✓ catalog |
| — shared index | `index.parquet` | 13,361 | `52a6c040…a822b54dd2` | ✓ catalog |

- Array shapes/dtypes on disk match catalog rows; all values finite FP32.
- `observations.parquet` SHA-256 `bdd83f02…4544ad17` matches both `metadata.json.observations_sha256` and `splits.report.json.source_sha256`; `splits.parquet` `78465ac3…` matches its report's `output_sha256`.
- `metadata.json` provenance chain: config SHA `4fc6166d…`, checkpoint SHA `848a77c3…`, source commit `204698b4…`, `context_tokens_are_globally_contextualized: true`, `pooled_precision: fp32`, `forward_precision: bf16`, device `cuda:0`.
- Smoke runs confirm reproducibility: 8-row runs (`proxy_smoke_001`, `proxy_smoke_crop_v1`) and a 256-row run (`proxy_smoke_256_crop_v1`) each with 3-pool catalogs.
- Immutability enforced in code: run roots are append-only (`FileExistsError`), catalogs reject duplicate feature keys/run IDs, and `load_cached_feature` re-verifies file hashes + shape at load time.

### 3.3 Honest status label

The extracted features are a **pipeline preflight on a deliberately confounded interaction-phase proxy**. They answer nothing about physical-state geometry by design. All three pools exist to exercise pooling code paths and the frozen-encoder pathway on the audited 6 GiB GPU. Phase 1 exit criteria for the *scientific* pipeline (curated observations, motion controls, sealed test views, locked evaluation) remain open.

---

## 4. Audit Findings — Debt & Risks Before Phase 2

### Critical

1. **No version control at repository root.** Root is not a git repo (only the nested `vjepa2` is). No history, no rollback, no remote for `src/`, `scripts/`, `docs/`, `configs/`, `tests/`. → `git init` + `.gitignore` + initial commit **before** any Phase 2 work.
2. **Windows-absolute paths embedded in artifacts.** `catalog.parquet` and `metadata.json` store `P:/research/state_geometry_video/…` absolute paths. Plan §17 requires relocation-safe paths; Runpod migration will break path resolution (hashes stay valid, paths won't). → Store workspace-relative paths + an explicit root anchor field; add a path-rebase utility.
3. **No standalone artifact-validation CLI.** Catalog integrity is checked only at load time (`load_cached_feature`) and by ad-hoc runbook assertions. → Add `scripts/validate_feature_catalog.py` (verify all hashes, shapes, row counts, append-only invariants, provenance chain).

### Moderate

4. **Doc–artifact drift in the implementation plan.** Plan §16.4 specifies proxy run-id `interaction_phase_proxy_l11` / subset `proxy`; the executed run (and both runbooks) use `interaction_phase_proxy_crop_v1` / `proxy_crop_v1`. Update §16.4 to the realized keys.
5. **Duplicate runbooks.** `phase1_cli_runbook.md` (PowerShell) and `phase1_cli_runbook_cmd.md` (cmd.exe) are near-identical twins — guaranteed drift. Keep one canonical runbook (or generate both from one source).
6. **Feature-run artifact gaps vs. plan §18:** no per-run `extraction.jsonl`; `metadata.json` has no hash column in the catalog (only features/index are hashed); catalog row has no per-run `extraction log`/environment record.
7. **OMP environment fragility.** `unittest discover` aborts with `OMP Error #15` (torch + sklearn both linking libiomp5md) unless `KMP_DUPLICATE_LIB_OK=TRUE` is set. The runbook omits this workaround. Document it or fix the library load order.
8. **Video decode path is production-untested.** `_decode_video` (PyAV, 16 sorted unique frames) is implemented and unit-tested, but **all 13,361 production extractions were images**. Real Phase 2 curation is video-based — the first video extraction must be a dedicated smoke with frame-index verification.
9. **Mask-pool path is production-untested.** `mask` pooling code exists and is unit-tested, but no mask features were ever extracted (NAO side has no masks; the proxy extraction deliberately used only box/full/context_tokens).

### Low

10. Junk file `2]` (0 bytes) at repo root (botched redirect). Delete.
11. `src/state_geometry_video.egg-info/` and scattered `__pycache__/` should be excluded from any future VCS/transfer.
12. Three separate smoke catalogs with inconsistent naming (`proxy_smoke_001`, `proxy_smoke_crop_v1`, `proxy_smoke_256_crop_v1`) — either one append-only smoke catalog or a documented naming scheme.
13. `01_proxy/` conflates several artifact categories (observations/triplets/splits/features). Acceptable for the proxy namespace; the real experiment must follow the plan §18 tree (`01_manifests/`, `02_motion/`, `03_features/`, …, `10_report/`).
14. `observations.parquet` co-locates the `interaction_phase` label with extraction inputs; the proxy run legitimately skipped `--require-label-redacted-input`, but real extractions must use the redacted `feature_inputs.parquet` view (flag exists, enforced by `validate_feature_inputs`).

---

## 5. Phase 2 Transition Plan

### 5.1 Immediate next steps (ordered)

1. **Repo hardening (blocking):** `git init` + `.gitignore` (datasets, checkpoints, `*.egg-info`, `__pycache__`, `.cache`), remove `2]`, commit a clean Phase 1 snapshot; add `scripts/validate_feature_catalog.py`; fix plan §16.4 drift; document the OMP workaround.
2. **Curation (the real gate):** from `candidates.parquet`, manually curate the **30-transition feature-blind protocol/power pilot** (physical identity, state family/label, observability, stable windows, per-sampled-frame aligned boxes + mask subset, media/perceptual hashes, nuisance tags, duplicate groups). Gate: reviewer agreement ≥ 0.80, no generic labels.
3. **Power + freeze:** pilot variance → powered cohort → freeze `curated_observations.parquet` → `validate_phase1_manifest.py --stage curated` must pass clean.
4. **Splits before triplets:** `build_phase1_splits.py` on `physical_object_id, source_video_id, transition_id, verified_asset_group_id, duplicate_group_id` (object/video/dependency-disjoint, 70/15/15, seed 20260820).
5. **Motion controls:** implement + run the §16.3 chain (`estimate_motion_controls.py`, `validate_motion_controls.py`, `fit_motion_scaler_quality.py`, `assemble_phase1_analysis_manifest.py`) with the stationary-region minimum backend; **validation motion-role gate must be near its grouped permutation null before any semantic work** (999 permutations, `swap_all_paired_roles_in_group`).
6. **Triplets + sealed views:** candidate universe → frozen signed/severity calipers on train/validation only → `finalize_phase1_triplets.py` → `build_phase1_release_views.py` (label-redacted `feature_inputs.parquet`, hash-locked `sealed_test_targets/triplets`, `release_views.json`).
7. **Real extraction (`03_features/`):** `vjepa21b/layer11/box/original/all` primary key; mask run on the frozen common-row subset (or evidence-backed `record_phase1_skip.py`); `object_pixel_erased_mean` variant; label-redacted + motion-provenance inputs; append-only catalog. Video decode smoke first.
8. **Diagnostics (train/validation only):** `eval_state_nuisance.py` (SNS-all, motion-matched SNS, margins, strata), `fit_geometry_controls.py` (center/PCA64-256/random-proj/whitening, train-fit), `train_probe.py` (linear + MLP, dependency-group IPF weighting, shuffled-training negative controls), `run_shortcut_controls.py` (full-frame, context-tokens, erasure, geometry, metadata, temporal, hand-presence).
9. **Conditional adapter branch:** only if H-info/H-geometry gates pass — 4 same-loss metric transforms + role-shuffled control (`margin_triplet`, 5 seeds, margin grid 0.05/0.10/0.20, lr 3e-4, wd 1e-4, batch 512, FP32); preservation (identity R@1 non-inferiority −0.02) and collapse (effective-rank ratio ≥ 0.90) checks on validation.
10. **Freeze + one atomic locked test:** `freeze_phase1_selection.py` → `run_locked_phase1_test.py` (access-start marker before first sealed read, transactional results, completion marker) → `build_phase1_report.py`.

### 5.2 Downstream training/probing pipeline (artifacts)

```
01_manifests/  curated_observations, splits, internal_analysis_observations,
               internal_candidate/finalized triplets, feature_inputs (label-redacted),
               triplets_trainval, sealed_test_targets/triplets, release_views.json
02_motion/     motion_observations, motion_validated, caliper_selection.json,
               role_control_validation/
03_features/   <run_id>/{features_{pool}.npy, index.parquet, metadata.json, complete.json},
               catalog.parquet (append-only), mask_component_resolution.json
04_controls/   PCA/whitening/random-projection geometry controls (train-fit)
05_geometry/   SNS-all + motion-matched reports, margins, bootstrap CIs
06_probes/     linear, mlp, shuffled-linear, shuffled-mlp (metrics + selection.json)
07_shortcuts/  full/context/erasure/geometry/metadata/temporal/hand controls
08_adapters/   positive_diagonal, linear_residual, mlp_nonresidual, residual,
               role_shuffled (checkpoints, logs, seeds)
09_adapter_eval/  preservation + collapse + adapter_component_resolution.json
10_report/     frozen_selection.json, test_access_started.lock, test_completed.lock,
               locked_test/, phase1_results.md
```

Configs already staged: `configs/phase1/{motion_controls,probe_linear,probe_mlp,adapter_margin_triplet}.yaml` + `feature_input_columns.txt`. Code already staged: `models/residual_adapter.py` (all 5 metric architectures + margin loss), `controls/geometry.py` (PCA/whitening/random projections), `controls/motion.py`, `evaluation/geometry.py`, `data/release_views.py`, `utils/hashing.py`, `utils/locking.py` — each with unit-test coverage.

### 5.3 Hardware execution path

- **Local (primary):** RTX 4050 Laptop, 6,141 MiB VRAM → V-JEPA 2.1 **ViT-B only**; 16×384, `batch-size=1`, `workers=0`, BF16 autocast, `inference_mode`, one output layer. Proven on-laptop: full 13,361-observation proxy run completed; smoke runs at 8 and 256 rows. Adapter/probe/control training is cached-feature FP32 and runs locally without the encoder.
- **OOM rule:** never silently lower resolution/frames. If the locked 16×384 protocol OOMs after batch-1/BF16/SDPA settings, migrate the *identical* frozen extraction to **Runpod** (plan §17: exclude `vjepa2/`, `.cache/`, `frames.tar.gz`, the 5.15 GB ViT-L; re-clone vjepa2 at the pinned commit; re-verify all hashes; reproduce the 32-observation smoke and compare FP32 pooled cosines).
- **ViT-L (300M) confirmation:** Runpod-only and **post-gate** — never before the ViT-B data/motion/information gates pass. DINOv2/TrackMAE/V-JEPA 2 are descriptive, post-gate, non-causal comparisons.
- **No-go discipline:** no predictor, LoRA, Pneuma integration, memory modules, or long sweeps until the diagnostic regime (state decodable ∧ geometry misaligned) is established on validation.
