from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ResizeCropGeometry:
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    crop_size: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height


def eval_resize_crop_geometry(
    source_height: int,
    source_width: int,
    crop_size: int = 384,
) -> ResizeCropGeometry:
    """Reproduce the pinned V-JEPA evaluation resize/crop geometry.

    The official path resizes the short side to ``int(crop*256/224)`` and
    uses Python ``round`` for the center-crop offset. Pixel interpolation is
    handled separately so masks and boxes share this exact geometry.
    """
    if source_height <= 0 or source_width <= 0 or crop_size <= 0:
        raise ValueError("source dimensions and crop_size must be positive")
    short_side = int(crop_size * 256 / 224)
    if source_width < source_height:
        resized_width = short_side
        resized_height = int(short_side * source_height / source_width)
    else:
        resized_height = short_side
        resized_width = int(short_side * source_width / source_height)
    crop_left = int(round((resized_width - crop_size) / 2.0))
    crop_top = int(round((resized_height - crop_size) / 2.0))
    if crop_left < 0 or crop_top < 0:
        raise AssertionError("resize geometry is smaller than the requested crop")
    return ResizeCropGeometry(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        crop_size=crop_size,
    )


def _resize_crop(values: Tensor, geometry: ResizeCropGeometry, mode: str) -> Tensor:
    if values.ndim != 4:
        raise ValueError("values must have shape [T,C,H,W]")
    if tuple(values.shape[-2:]) != (geometry.source_height, geometry.source_width):
        raise ValueError("tensor dimensions do not match the recorded source geometry")
    resized = F.interpolate(
        values.float(),
        size=(geometry.resized_height, geometry.resized_width),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
        antialias=False,
    )
    top, left, size = geometry.crop_top, geometry.crop_left, geometry.crop_size
    return resized[..., top : top + size, left : left + size]


def preprocess_rgb_frames(
    frames: Tensor,
    crop_size: int = 384,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[Tensor, ResizeCropGeometry]:
    """Convert RGB frames [T,H,W,3] or [T,3,H,W] to [3,T,S,S]."""
    if frames.ndim != 4:
        raise ValueError("frames must have four dimensions")
    if frames.shape[-1] == 3:
        values = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] == 3:
        values = frames
    else:
        raise ValueError("frames must be RGB")
    if values.shape[0] not in (1, 16):
        raise ValueError("Phase 1 V-JEPA inputs must contain exactly 1 or 16 frames")
    geometry = eval_resize_crop_geometry(values.shape[-2], values.shape[-1], crop_size)
    values = values.float()
    if frames.dtype == torch.uint8 or float(values.max()) > 1.0:
        values = values / 255.0
    values = _resize_crop(values, geometry, mode="bilinear")
    mean_tensor = torch.tensor(mean, dtype=values.dtype, device=values.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=values.dtype, device=values.device).view(1, 3, 1, 1)
    values = (values - mean_tensor) / std_tensor
    return values.permute(1, 0, 2, 3).contiguous(), geometry


def transform_binary_masks(masks: Tensor, geometry: ResizeCropGeometry) -> Tensor:
    """Apply the recorded RGB spatial field of view to masks [T,H,W]."""
    if masks.ndim != 3:
        raise ValueError("masks must have shape [T,H,W]")
    transformed = _resize_crop(masks[:, None].float(), geometry, mode="nearest")
    return transformed[:, 0].ge(0.5)


def transform_xywh_box(
    box: tuple[float, float, float, float],
    geometry: ResizeCropGeometry,
) -> tuple[float, float, float, float]:
    """Map a COCO xywh box into the encoder crop and clip it to the field of view."""
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("box width and height must be positive")
    x0 = x * geometry.scale_x - geometry.crop_left
    y0 = y * geometry.scale_y - geometry.crop_top
    x1 = (x + width) * geometry.scale_x - geometry.crop_left
    y1 = (y + height) * geometry.scale_y - geometry.crop_top
    size = float(geometry.crop_size)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(size, x1), min(size, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box lies outside the encoder crop")
    return x0, y0, x1 - x0, y1 - y0


def rasterized_box_token_mass(
    transformed_box: tuple[float, float, float, float],
    patch_size: int = 16,
) -> float:
    """Return the exact occupancy mass produced by rasterize+patch averaging."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    x, y, width, height = transformed_box
    if width <= 0 or height <= 0:
        return 0.0
    raster_width = max(0, math.ceil(x + width) - math.floor(x))
    raster_height = max(0, math.ceil(y + height) - math.floor(y))
    return float(raster_width * raster_height) / float(patch_size * patch_size)


def rasterize_xywh_boxes(
    boxes: list[tuple[float, float, float, float]],
    geometry: ResizeCropGeometry,
) -> Tensor:
    """Rasterize one transformed source-space box per input frame."""
    if len(boxes) not in (1, 16):
        raise ValueError("exactly one box per image or sampled video frame is required")
    masks = torch.zeros(len(boxes), geometry.crop_size, geometry.crop_size, dtype=torch.bool)
    for index, box in enumerate(boxes):
        x, y, width, height = transform_xywh_box(box, geometry)
        left, top = int(torch.floor(torch.tensor(x))), int(torch.floor(torch.tensor(y)))
        right = int(torch.ceil(torch.tensor(x + width)))
        bottom = int(torch.ceil(torch.tensor(y + height)))
        masks[index, top:bottom, left:right] = True
    return masks
