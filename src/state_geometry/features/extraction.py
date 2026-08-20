from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from state_geometry.features.pooling import (
    full_token_pool,
    masks_to_tubelet_occupancy,
    weighted_region_pool,
)
from state_geometry.features.preprocess import (
    ResizeCropGeometry,
    preprocess_rgb_frames,
    rasterize_xywh_boxes,
    transform_binary_masks,
)
from state_geometry.features.vjepa import extract_dense_tokens


ALLOWED_POOLS = frozenset({"box", "mask", "full", "context_tokens"})
ALLOWED_INPUT_CONTROLS = frozenset({"original", "object_pixel_erased_mean"})


def resolve_workspace_path(workspace_root: str | Path, relative: object) -> Path:
    root = Path(workspace_root).resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError("a nonempty workspace-relative path is required")
    path = (root / Path(relative)).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path escapes workspace: {relative}")
    return path


def _decode_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(rgb)[None]


def _decode_video(path: Path, sampled_indices: Iterable[int]) -> torch.Tensor:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("video decoding requires the reported 'av' dependency") from exc
    requested = [int(value) for value in sampled_indices]
    if len(requested) != 16 or requested != sorted(set(requested)):
        raise ValueError("video extraction requires 16 sorted unique sampled_frame_indices")
    wanted = set(requested)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if len(decoded) == len(wanted):
                break
    missing = [index for index in requested if index not in decoded]
    if missing:
        raise RuntimeError(f"video ended before requested frames: {missing}")
    return torch.from_numpy(np.stack([decoded[index] for index in requested], axis=0))


def decode_observation(row: pd.Series, workspace_root: str | Path) -> torch.Tensor:
    path = resolve_workspace_path(workspace_root, row["media_relpath"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if row["media_type"] == "image":
        return _decode_image(path)
    if row["media_type"] == "video":
        return _decode_video(path, row["sampled_frame_indices"])
    raise ValueError(f"unsupported media_type: {row['media_type']}")


def _load_annotation_payload(row: pd.Series, workspace_root: str | Path, column: str) -> object:
    path = resolve_workspace_path(workspace_root, row[column])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "observations" in payload:
        payload = payload["observations"].get(str(row["observation_id"]))
    if payload is None:
        raise ValueError(f"annotation file has no entry for {row['observation_id']}")
    return payload


def load_boxes(row: pd.Series, workspace_root: str | Path, frame_count: int) -> list[tuple[float, float, float, float]]:
    if "bbox_xywh" in row.index and row["bbox_xywh"] is not None:
        raw = row["bbox_xywh"]
        boxes = [raw]
    elif "box_annotation_relpath" in row.index and pd.notna(row["box_annotation_relpath"]):
        payload = _load_annotation_payload(row, workspace_root, "box_annotation_relpath")
        raw = payload.get("boxes_xywh") if isinstance(payload, dict) else payload
        boxes = raw if raw and isinstance(raw[0], (list, tuple)) else [raw]
    else:
        raise ValueError(f"observation {row['observation_id']} has no aligned box")
    converted = [tuple(float(value) for value in box) for box in boxes]
    if len(converted) != frame_count or any(len(box) != 4 for box in converted):
        raise ValueError("box annotation must contain one xywh box per encoder input frame")
    return converted


def _polygon_components(value: object) -> list[list[float]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, (int, float)) for item in value):
        value = [value]
    return [
        [float(item) for item in component]
        for component in value
        if isinstance(component, list) and len(component) >= 6 and len(component) % 2 == 0
    ]


def rasterize_source_polygons(
    polygons_by_frame: list[object],
    height: int,
    width: int,
) -> torch.Tensor:
    masks: list[np.ndarray] = []
    for raw_components in polygons_by_frame:
        image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for component in _polygon_components(raw_components):
            points = list(zip(component[0::2], component[1::2]))
            draw.polygon(points, fill=1)
        masks.append(np.asarray(image, dtype=np.uint8).copy())
    return torch.from_numpy(np.stack(masks, axis=0)).bool()


def load_transformed_mask(
    row: pd.Series,
    workspace_root: str | Path,
    frame_count: int,
    geometry: ResizeCropGeometry,
) -> torch.Tensor:
    if "valid_segmentation_json" in row.index and pd.notna(row["valid_segmentation_json"]):
        polygons = [row["valid_segmentation_json"]]
    elif "mask_annotation_relpath" in row.index and pd.notna(row["mask_annotation_relpath"]):
        payload = _load_annotation_payload(row, workspace_root, "mask_annotation_relpath")
        polygons = payload.get("polygons_by_frame") if isinstance(payload, dict) else payload
        if frame_count == 1 and polygons and _polygon_components(polygons):
            polygons = [polygons]
    else:
        raise ValueError(f"observation {row['observation_id']} has no aligned mask")
    if not isinstance(polygons, list) or len(polygons) != frame_count:
        raise ValueError("mask annotation must contain polygons for every encoder input frame")
    source_masks = rasterize_source_polygons(
        polygons,
        height=geometry.source_height,
        width=geometry.source_width,
    )
    transformed = transform_binary_masks(source_masks, geometry)
    if not transformed.flatten(1).any(dim=1).all():
        raise ValueError("mask becomes empty in the encoder field of view")
    return transformed


def extract_observation_pools(
    encoder: torch.nn.Module,
    row: pd.Series,
    workspace_root: str | Path,
    pools: Iterable[str],
    input_control: str,
    use_bfloat16: bool = True,
) -> dict[str, np.ndarray]:
    requested = set(pools)
    if not requested or not requested.issubset(ALLOWED_POOLS):
        raise ValueError(f"unsupported pools: {sorted(requested - ALLOWED_POOLS)}")
    if input_control not in ALLOWED_INPUT_CONTROLS:
        raise ValueError(f"unsupported input control: {input_control}")
    frames = decode_observation(row, workspace_root)
    normalized, geometry = preprocess_rgb_frames(frames)
    frame_count = normalized.shape[1]
    boxes = load_boxes(row, workspace_root, frame_count)
    box_masks = rasterize_xywh_boxes(boxes, geometry)
    if input_control == "object_pixel_erased_mean":
        normalized = normalized.masked_fill(box_masks[None], 0.0)
    tokens = extract_dense_tokens(encoder, normalized[None], use_bfloat16=use_bfloat16)
    tubelet_size = 1 if frame_count == 1 else 2
    box_occupancy = masks_to_tubelet_occupancy(
        box_masks[None], tubelet_size=tubelet_size, patch_size=16
    )
    result: dict[str, np.ndarray] = {}
    if "box" in requested:
        result["box"] = weighted_region_pool(tokens, box_occupancy)[0].cpu().numpy()
    if "full" in requested:
        result["full"] = full_token_pool(tokens)[0].cpu().numpy()
    if "context_tokens" in requested:
        result["context_tokens"] = weighted_region_pool(tokens, 1.0 - box_occupancy)[0].cpu().numpy()
    if "mask" in requested:
        masks = load_transformed_mask(row, workspace_root, frame_count, geometry)
        occupancy = masks_to_tubelet_occupancy(
            masks[None], tubelet_size=tubelet_size, patch_size=16
        )
        result["mask"] = weighted_region_pool(tokens, occupancy)[0].cpu().numpy()
    return {name: np.asarray(values, dtype=np.float32) for name, values in result.items()}
