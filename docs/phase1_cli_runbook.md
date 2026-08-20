# Phase 1 implementation runbook

This runbook contains only commands backed by code that currently exists in the
repository. Run it from PowerShell in the repository root. The central
physical-state experiment intentionally stops at the curation gate until valid
state, identity, observability, and aligned-region annotations exist.

## 1. Clone and prepare the environment

```powershell
Set-Location 'P:\research\state_geometry_video'

conda create --name llm-phase1 --clone llm --yes
conda activate llm-phase1

python -V
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

python -m pip install "scikit-learn>=1.5,<2" "scipy>=1.13,<2" "av>=12,<17" "timm>=1.0,<2" "einops>=0.8,<1" "pycocotools>=2.0.8,<3" "pytest>=8,<10"
python -m pip install --no-deps --editable .

python scripts/check_phase1_dependencies.py --skip-env-name-check
if ($LASTEXITCODE -ne 0) { throw 'Dependency/runtime check failed.' }
```

Do not reinstall or upgrade `torch`/`torchvision`; the clone preserves the
working CUDA-enabled pair.

## 2. Run implementation tests

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }

python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { throw 'Python compilation check failed.' }
```

The current expected result is 28 passing tests, including the CUDA-token/CPU-region
pooling regression test.

## 3. Verify the pinned V-JEPA source without modifying Git configuration

```powershell
$VjepaPath = ((Resolve-Path '.\vjepa2').Path -replace '\\','/')
$GitBase = @('-c', "safe.directory=$VjepaPath", '-C', 'vjepa2')
$ExpectedCommit = '204698b45b3712590f06245fbfba32d3be539812'

$Origin = (git @GitBase remote get-url origin).Trim()
$Commit = (git @GitBase rev-parse HEAD).Trim()
if ($Origin -ne 'https://github.com/facebookresearch/vjepa2.git') { throw "Unexpected origin: $Origin" }
if ($Commit -ne $ExpectedCommit) { throw "Unexpected V-JEPA commit: $Commit" }

git @GitBase diff HEAD --exit-code -- src app hubconf.py
if ($LASTEXITCODE -ne 0) { throw 'Imported V-JEPA source differs from the pinned commit.' }
$ImportedStatus = git @GitBase status --porcelain --untracked-files=all -- src app hubconf.py
if ($ImportedStatus) { throw "Imported V-JEPA paths contain staged/untracked changes: $ImportedStatus" }
```

Unrelated modified YAML files and the existing untracked ViT-L checkpoint are
outside this imported-source gate and must not be reset.

## 4. Re-run the real-data audit

```powershell
python scripts/audit_phase1_dataset.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --output-root artifacts/phase1/00_audit
if ($LASTEXITCODE -ne 0) { throw 'Dataset audit failed.' }

$Audit = Get-Content 'artifacts/phase1/00_audit/inventory.json' -Raw | ConvertFrom-Json
if ($Audit.filesystem.payload_files -ne 68073) { throw 'Unexpected payload file count.' }
if ($Audit.filesystem.payload_bytes -ne 19902705085) { throw 'Unexpected payload byte count.' }
if ($Audit.video_tas_alignment.videos -ne 3390) { throw 'Unexpected video count.' }
if ($Audit.video_tas_alignment.alignment_errors.Count -ne 0) { throw 'Video/TAS alignment errors found.' }
if (-not $Audit.revision_match) { throw 'Dataset revision mismatch.' }
```

## 5. Exercise the central curation gate

First run a bounded smoke candidate build. The validator must exit with code 2;
that is the expected scientific result for the downloaded metadata.

```powershell
python scripts/build_phase1_manifest.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --window-frames 30 --limit-videos 3 --output artifacts/phase1/01_manifests/candidates_smoke.parquet
if ($LASTEXITCODE -ne 0) { throw 'Candidate smoke build failed.' }

python scripts/validate_phase1_manifest.py --stage curated --manifest artifacts/phase1/01_manifests/candidates_smoke.parquet --require-state-labels --require-physical-object-ids --require-observability --require-aligned-regions --output-root artifacts/phase1/01_manifests/validation_smoke
$GateExit = $LASTEXITCODE
if ($GateExit -ne 2) { throw "Expected fail-closed exit 2, got $GateExit" }
```

Generate the full feature-blind curation queue separately:

```powershell
python scripts/build_phase1_manifest.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --window-frames 30 --output artifacts/phase1/01_manifests/candidates.parquet
if ($LASTEXITCODE -ne 0) { throw 'Full candidate build failed.' }
```

Do not relabel TAS `action_0/1/2` as physical states. The next central command is
the same validator applied to a genuinely curated
`curated_observations.parquet`; it is not runnable until that artifact exists.

## 6. Rebuild and verify the static interaction-phase proxy

This can test data joins, splits, image loading, box pooling, and the frozen
encoder pathway. It is not a physical-state experiment.

```powershell
python scripts/build_interaction_phase_proxy.py --nao-json datasets/phase1/EgoInteract/data/annotations/nao/coco_annotations_egointeract.json --nao-frame-root datasets/phase1/EgoInteract/data/frames/frames/nao --hoi-json datasets/phase1/EgoInteract/data/annotations/hoi/coco_annotations_hand_egointeract.json --hoi-frame-root datasets/phase1/EgoInteract/data/frames/frames/hoi_enigma --phase-labels pre_contact,contact --min-observations-per-phase 2 --require-contacting-hand --deduplicate-by-sha256 --reject-cross-label-hash-conflicts --encoder-crop-size 384 --minimum-box-token-mass 1.0 --output-root artifacts/phase1/01_proxy
if ($LASTEXITCODE -ne 0) { throw 'Proxy build failed.' }

python scripts/build_phase1_splits.py --observations artifacts/phase1/01_proxy/observations.parquet --build-connected-groups asset_proxy_id,sequence_id,duplicate_group_id --null-group-values-no-edge --group dependency_group_id --ratios 0.70,0.15,0.15 --stratify interaction_phase --stratification-objective deterministic_iterative_group_balance --report-realized-stratum-mass --seed 20260820 --output artifacts/phase1/01_proxy/splits.parquet
if ($LASTEXITCODE -ne 0) { throw 'Proxy split failed.' }

$Proxy = Get-Content 'artifacts/phase1/01_proxy/summary.json' -Raw | ConvertFrom-Json
if ($Proxy.physical_state_claim_allowed -ne $false) { throw 'Proxy was mislabeled as physical state.' }
if ($Proxy.observations -ne 13361) { throw 'Unexpected crop-poolable proxy observation count.' }
if ($Proxy.triplets -ne 3362) { throw 'Unexpected crop-poolable proxy triplet count.' }
if ($Proxy.eligible_sequence_asset_keys -ne 1681) { throw 'Unexpected eligible proxy-group count.' }
if ($Proxy.cross_directory_exact_duplicate_pairs -ne 377) { throw 'Unexpected duplicate-pair count.' }
```

The proxy build hashes many JPGs and can take several minutes.

## 7. Acquire and fingerprint the primary ViT-B checkpoint

The existing ViT-L checkpoint is not a substitute. Download the primary ViT-B
asset directly from the official Meta host:

```powershell
New-Item -ItemType Directory -Force 'checkpoints' | Out-Null
$Checkpoint = 'checkpoints/vjepa2_1_vitb_dist_vitG_384.pt'
$CheckpointUrl = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt'

curl.exe --fail --location --continue-at - --output $Checkpoint $CheckpointUrl
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint download failed.' }
if ((Get-Item $Checkpoint).Length -ne 1664223428) { throw 'Unexpected ViT-B checkpoint size.' }

$CheckpointHash = (Get-FileHash -Algorithm SHA256 $Checkpoint).Hash.ToLowerInvariant()
Write-Host "ViT-B SHA-256: $CheckpointHash"

$ConfigPath = (Resolve-Path 'configs/phase1/vjepa21_vitb.yaml').Path
$ConfigText = [System.IO.File]::ReadAllText($ConfigPath)
$ConfigText = [System.Text.RegularExpressions.Regex]::Replace(
    $ConfigText,
    '(?m)^  checkpoint_sha256:.*$',
    "  checkpoint_sha256: `"$CheckpointHash`""
)
[System.IO.File]::WriteAllText($ConfigPath, $ConfigText, [System.Text.UTF8Encoding]::new($false))
```

This records a reproducibility fingerprint for subsequent runs. The official
HTTPS origin plus the pinned loader/source provenance establishes the acquisition
path; a locally computed hash alone is not an independent authenticity proof.

## 8. Run the local GPU smoke extraction

This processes only eight feature-blind proxy observations and creates unique,
immutable output keys on every invocation.

```powershell
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), 'GiB'); print('bf16', torch.cuda.is_bf16_supported())"

$SmokeId = 'proxy_smoke_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$SmokeRoot = "artifacts/phase1/02_smoke/$SmokeId"

python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_proxy/observations.parquet --limit-observations 8 --layers 11 --pools box,full,context_tokens --input-control original --subset-name $SmokeId --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id $SmokeId --output-root "$SmokeRoot/run" --catalog "$SmokeRoot/catalog.parquet"
if ($LASTEXITCODE -ne 0) { throw 'Frozen V-JEPA smoke extraction failed.' }

python -c "import pandas as pd,sys; p=sys.argv[1]; c=pd.read_parquet(p); assert len(c)==3; assert set(c['pool'])=={'box','full','context_tokens'}; assert set(c['rows'])=={8}; assert set(c['dimension'])=={768}; print(c[['feature_key','rows','dimension','dtype']].to_string(index=False))" "$SmokeRoot/catalog.parquet"
```

If this OOMs, stop. Do not reduce resolution/frame count or switch to ViT-L;
record the failure and move extraction to Runpod.

## 9. Optional full static-proxy extraction

Run this only after the eight-image smoke succeeds. It may take hours on the
laptop and tests only the confounded interaction-phase proxy.

```powershell
python scripts/extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_proxy/observations.parquet --layers 11 --pools box,full,context_tokens --input-control original --subset-name proxy_crop_v1 --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id interaction_phase_proxy_crop_v1 --output-root artifacts/phase1/01_proxy/features/interaction_phase_proxy_crop_v1 --catalog artifacts/phase1/01_proxy/features/catalog.parquet
if ($LASTEXITCODE -ne 0) { throw 'Full proxy extraction failed.' }
```

Feature runs are append-only. Do not reuse a completed `run-id`, output root, or
feature key. The probe, motion-estimation, triplet-finalization, adapter-training,
and locked-test CLIs described as future work in the implementation plan are not
yet executable and are deliberately omitted here.
