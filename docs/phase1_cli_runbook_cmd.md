# Phase 1 CLI runbook — Windows Command Prompt

These commands are for an interactive `cmd.exe` terminal. Run each section from
the repository root and stop if an unexpected command returns a nonzero exit
code. Only commands backed by currently implemented code are included.

## 1. Clone and prepare the environment

```bat
cd /d P:\research\state_geometry_video

conda create --name llm-phase1 --clone llm --yes
call conda activate llm-phase1

python -V
python -c "import torch,torchvision; print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('cuda',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('bf16',torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

python -m pip install "scikit-learn>=1.5,<2" "scipy>=1.13,<2" "av>=12,<17" "timm>=1.0,<2" "einops>=0.8,<1" "pycocotools>=2.0.8,<3" "pytest>=8,<10"
python -m pip install --no-deps --editable .

python scripts\check_phase1_dependencies.py --skip-env-name-check
```

Do not reinstall or upgrade `torch` or `torchvision`; retain the CUDA-enabled
pair copied from `llm`.

## 2. Run the implementation tests

```bat
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q src scripts tests
```

The current expected result is 28 passing tests, including the CUDA-token/CPU-region
pooling regression test.

## 3. Verify the pinned V-JEPA source

The per-command safe-directory setting does not modify global Git
configuration.

```bat
git -c safe.directory=P:/research/state_geometry_video/vjepa2 -C vjepa2 remote get-url origin
git -c safe.directory=P:/research/state_geometry_video/vjepa2 -C vjepa2 rev-parse HEAD
git -c safe.directory=P:/research/state_geometry_video/vjepa2 -C vjepa2 diff HEAD --exit-code -- src app hubconf.py
git -c safe.directory=P:/research/state_geometry_video/vjepa2 -C vjepa2 status --porcelain --untracked-files=all -- src app hubconf.py
```

Expected origin and commit:

```text
https://github.com/facebookresearch/vjepa2.git
204698b45b3712590f06245fbfba32d3be539812
```

The `diff` command must return exit code zero and the path-scoped `status`
command must print nothing. Do not reset the unrelated modified YAMLs or the
existing untracked ViT-L checkpoint.

## 4. Re-run the dataset audit

```bat
python scripts\audit_phase1_dataset.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --output-root artifacts/phase1/00_audit

python -c "import json; a=json.load(open('artifacts/phase1/00_audit/inventory.json',encoding='utf-8')); assert a['filesystem']['payload_files']==68073; assert a['filesystem']['payload_bytes']==19902705085; assert a['video_tas_alignment']['videos']==3390; assert not a['video_tas_alignment']['alignment_errors']; assert a['revision_match']; print('dataset audit assertions passed')"
```

## 5. Exercise the fail-closed central-data gate

```bat
python scripts\build_phase1_manifest.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --window-frames 30 --limit-videos 3 --output artifacts/phase1/01_manifests/candidates_smoke.parquet

python scripts\validate_phase1_manifest.py --stage curated --manifest artifacts/phase1/01_manifests/candidates_smoke.parquet --require-state-labels --require-physical-object-ids --require-observability --require-aligned-regions --output-root artifacts/phase1/01_manifests/validation_smoke
echo Expected validator exit code 2; actual exit code %ERRORLEVEL%
```

Exit code `2` is expected because the downloaded video metadata does not contain
valid physical-state, identity, observability, and aligned-region annotations.
Any other result is an error.

Generate the full feature-blind manual-curation queue:

```bat
python scripts\build_phase1_manifest.py --dataset-root datasets/phase1/EgoInteract --dataset-revision 313d1ef6586571d6ce1fe85581f690c507110fea --window-frames 30 --output artifacts/phase1/01_manifests/candidates.parquet
```

Do not reinterpret TAS `action_0/1/2` as physical states.

## 6. Rebuild and split the static proxy

This is an interaction-phase pipeline test, not evidence about physical-state
geometry.

```bat
python scripts\build_interaction_phase_proxy.py --nao-json datasets/phase1/EgoInteract/data/annotations/nao/coco_annotations_egointeract.json --nao-frame-root datasets/phase1/EgoInteract/data/frames/frames/nao --hoi-json datasets/phase1/EgoInteract/data/annotations/hoi/coco_annotations_hand_egointeract.json --hoi-frame-root datasets/phase1/EgoInteract/data/frames/frames/hoi_enigma --phase-labels pre_contact,contact --min-observations-per-phase 2 --require-contacting-hand --deduplicate-by-sha256 --reject-cross-label-hash-conflicts --encoder-crop-size 384 --minimum-box-token-mass 1.0 --output-root artifacts/phase1/01_proxy

python scripts\build_phase1_splits.py --observations artifacts/phase1/01_proxy/observations.parquet --build-connected-groups asset_proxy_id,sequence_id,duplicate_group_id --null-group-values-no-edge --group dependency_group_id --ratios 0.70,0.15,0.15 --stratify interaction_phase --stratification-objective deterministic_iterative_group_balance --report-realized-stratum-mass --seed 20260820 --output artifacts/phase1/01_proxy/splits.parquet

python -c "import json; p=json.load(open('artifacts/phase1/01_proxy/summary.json',encoding='utf-8')); assert p['physical_state_claim_allowed'] is False; assert p['observations']==13361; assert p['triplets']==3362; assert p['eligible_sequence_asset_keys']==1681; assert p['cross_directory_exact_duplicate_pairs']==377; assert p['encoder_crop_size']==384; assert p['minimum_box_token_mass']==1.0; print('proxy assertions passed')"
```

The proxy build hashes many JPGs and can take several minutes.

## 7. Download and fingerprint the primary ViT-B checkpoint

The existing ViT-L checkpoint must not be substituted.

```bat
if not exist checkpoints mkdir checkpoints

curl.exe --fail --location --continue-at - --output checkpoints\vjepa2_1_vitb_dist_vitG_384.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt

python -c "from pathlib import Path; p=Path('checkpoints/vjepa2_1_vitb_dist_vitG_384.pt'); assert p.stat().st_size==1664223428, p.stat().st_size; print('checkpoint bytes',p.stat().st_size)"

python -c "from pathlib import Path; import hashlib,re; p=Path('checkpoints/vjepa2_1_vitb_dist_vitG_384.pt'); f=p.open('rb'); h=hashlib.file_digest(f,'sha256').hexdigest(); f.close(); c=Path('configs/phase1/vjepa21_vitb.yaml'); s=c.read_text(encoding='utf-8'); s=re.sub(r'(?m)^  checkpoint_sha256:.*$', '  checkpoint_sha256: \"'+h+'\"', s); c.write_text(s,encoding='utf-8'); print('ViT-B SHA-256',h)"
```

The hash becomes the immutable fingerprint checked on every extraction. The
official HTTPS acquisition path and source provenance establish where the file
came from; a locally computed hash alone is not an independent authenticity
proof.

## 8. Run the eight-image GPU smoke extraction

Use a new `SMOKE_ID` if you repeat this section because feature caches are
append-only.

```bat
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,2),'GiB'); print('bf16',torch.cuda.is_bf16_supported())"

set SMOKE_ID=proxy_smoke_002
set SMOKE_ROOT=artifacts\phase1\02_smoke\%SMOKE_ID%

python scripts\extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_proxy/observations.parquet --limit-observations 8 --layers 11 --pools box,full,context_tokens --input-control original --subset-name %SMOKE_ID% --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id %SMOKE_ID% --output-root %SMOKE_ROOT%\run --catalog %SMOKE_ROOT%\catalog.parquet

python -c "import pandas as pd; c=pd.read_parquet(r'%SMOKE_ROOT%\catalog.parquet'); assert len(c)==3; assert set(c['pool'])=={'box','full','context_tokens'}; assert set(c['rows'])=={8}; assert set(c['dimension'])=={768}; print(c[['feature_key','rows','dimension','dtype']].to_string(index=False))"
```

If this OOMs, stop. Do not change resolution/frame count or substitute ViT-L;
move feature extraction to Runpod.

## 9. Optional full static-proxy extraction

Run only after the eight-image smoke succeeds. It can take hours on the laptop
and remains a confounded pipeline test.

```bat
python scripts\extract_features.py --config configs/phase1/vjepa21_vitb.yaml --observations artifacts/phase1/01_proxy/observations.parquet --layers 11 --pools box,full,context_tokens --input-control original --subset-name proxy_crop_v1 --feature-key-prefix vjepa21b --batch-size 1 --workers 0 --run-id interaction_phase_proxy_crop_v1 --output-root artifacts/phase1/01_proxy/features/interaction_phase_proxy_crop_v1 --catalog artifacts/phase1/01_proxy/features/catalog.parquet
```

Do not reuse a completed run ID, output directory, or feature key. Motion,
triplet-finalization, probe-training, adapter-training, and locked-test CLIs that
remain future implementation work are deliberately omitted.
