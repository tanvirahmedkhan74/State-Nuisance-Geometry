from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from state_geometry.utils.hashing import sha256_file


PINNED_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_VIDEO_SHAPE = (3, 16, 384, 384)
EXPECTED_IMAGE_SHAPE = (3, 1, 384, 384)
EXPECTED_VIDEO_TOKEN_GRID = (8, 24, 24)
EXPECTED_IMAGE_TOKEN_GRID = (1, 24, 24)
EXPECTED_EMBEDDING_DIMENSION = 768


def _run_git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={source_root.as_posix()}", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def verify_pinned_source(
    source_root: str | Path,
    expected_commit: str = PINNED_SOURCE_COMMIT,
) -> dict[str, str]:
    """Verify the imported source paths without touching unrelated user changes."""
    root = Path(source_root).resolve()
    if not (root / "app" / "vjepa_2_1").is_dir():
        raise FileNotFoundError(f"V-JEPA 2.1 source is missing under {root}")
    commit = _run_git(root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"V-JEPA source commit {commit} != pinned {expected_commit}")
    _run_git(root, "diff", "HEAD", "--exit-code", "--", "src", "app", "hubconf.py")
    status = _run_git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "app",
        "hubconf.py",
    )
    if status:
        raise RuntimeError(f"imported V-JEPA source paths are dirty:\n{status}")
    return {"source_root": root.as_posix(), "source_commit": commit}


def clean_backbone_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    cleaned: dict[str, Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        if key in cleaned:
            raise ValueError(f"checkpoint key collision after prefix cleaning: {key}")
        cleaned[key] = value
    return cleaned


def build_vjepa21_vitb_encoder(source_root: str | Path) -> nn.Module:
    root = Path(source_root).resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise RuntimeError(f"V-JEPA import resolved outside pinned source: {module_path}")
    encoder = module.vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    if encoder.embed_dim != EXPECTED_EMBEDDING_DIMENSION or encoder.get_num_layers() != 12:
        raise RuntimeError("constructed encoder does not match the Phase 1 ViT-B contract")
    return encoder


def load_frozen_vjepa21_vitb(
    source_root: str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    device: str | torch.device = "cpu",
    expected_commit: str = PINNED_SOURCE_COMMIT,
) -> tuple[nn.Module, dict[str, str]]:
    source = verify_pinned_source(source_root, expected_commit)
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"primary ViT-B checkpoint is absent: {checkpoint}")
    if len(expected_checkpoint_sha256) != 64:
        raise ValueError("an independently recorded 64-character checkpoint SHA-256 is mandatory")
    actual_sha = sha256_file(checkpoint)
    if actual_sha.lower() != expected_checkpoint_sha256.lower():
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {actual_sha}")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping) or "ema_encoder" not in payload:
        raise RuntimeError("checkpoint does not contain the required ema_encoder key")
    encoder = build_vjepa21_vitb_encoder(source_root)
    encoder.load_state_dict(clean_backbone_state_dict(payload["ema_encoder"]), strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    metadata = {
        **source,
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": actual_sha,
        "checkpoint_key": "ema_encoder",
        "strict_load": "true",
    }
    return encoder, metadata


def reshape_dense_tokens(tokens: Tensor) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError("encoder output must have shape [B,N,D]")
    batch, token_count, dimension = tokens.shape
    grids = {
        576: EXPECTED_IMAGE_TOKEN_GRID,
        4608: EXPECTED_VIDEO_TOKEN_GRID,
    }
    if token_count not in grids or dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"unexpected encoder output {tuple(tokens.shape)}; expected [B,576,768] or [B,4608,768]"
        )
    return tokens.reshape(batch, *grids[token_count], dimension)


def extract_dense_tokens(
    encoder: nn.Module,
    video: Tensor,
    use_bfloat16: bool = True,
) -> Tensor:
    if video.ndim != 5 or tuple(video.shape[1:]) not in (EXPECTED_IMAGE_SHAPE, EXPECTED_VIDEO_SHAPE):
        raise ValueError("input must have shape [B,3,1,384,384] or [B,3,16,384,384]")
    try:
        parameter = next(encoder.parameters())
    except StopIteration as exc:
        raise ValueError("encoder has no parameters") from exc
    video = video.to(parameter.device, non_blocking=True)
    device_type = parameter.device.type
    autocast_enabled = use_bfloat16 and device_type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        tokens = encoder(video)
    return reshape_dense_tokens(tokens)
