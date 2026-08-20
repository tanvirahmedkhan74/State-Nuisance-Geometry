from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


@dataclass(frozen=True)
class VideoProbe:
    path: str
    codec: str
    width: int
    height: int
    sample_count: int
    duration_seconds: float
    fps: float


def _atoms(handle: BinaryIO, start: int, end: int) -> Iterator[tuple[str, int, int]]:
    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            break
        size, raw_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) != 8:
                raise ValueError("truncated extended MP4 atom")
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ValueError(f"invalid MP4 atom size at byte {position}")
        yield raw_type.decode("latin-1"), position + header_size, position + size
        position += size


def _children(handle: BinaryIO, container: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for kind, start, end in _atoms(handle, *container):
        result.setdefault(kind, []).append((start, end))
    return result


def probe_mp4(path: str | Path, relative_to: str | Path | None = None) -> VideoProbe:
    source = Path(path)
    size = source.stat().st_size
    with source.open("rb") as handle:
        top = _children(handle, (0, size))
        if "moov" not in top:
            raise ValueError("MP4 has no moov atom")
        moov = _children(handle, top["moov"][0])
        for trak_box in moov.get("trak", []):
            trak = _children(handle, trak_box)
            if "mdia" not in trak or "tkhd" not in trak:
                continue
            mdia = _children(handle, trak["mdia"][0])
            if "hdlr" not in mdia or "mdhd" not in mdia or "minf" not in mdia:
                continue
            handle.seek(mdia["hdlr"][0][0] + 8)
            if handle.read(4) != b"vide":
                continue

            tkhd_start, tkhd_end = trak["tkhd"][0]
            handle.seek(tkhd_end - 8)
            width_raw, height_raw = struct.unpack(">II", handle.read(8))
            width, height = width_raw >> 16, height_raw >> 16

            mdhd_start, _ = mdia["mdhd"][0]
            handle.seek(mdhd_start)
            version = handle.read(1)[0]
            handle.seek(mdhd_start + (20 if version == 1 else 12))
            timescale = struct.unpack(">I", handle.read(4))[0]
            duration = struct.unpack(">Q" if version == 1 else ">I", handle.read(8 if version == 1 else 4))[0]

            minf = _children(handle, mdia["minf"][0])
            stbl = _children(handle, minf["stbl"][0]) if "stbl" in minf else {}
            if "stsz" not in stbl:
                raise ValueError("video track has no stsz atom")
            stsz_start, _ = stbl["stsz"][0]
            handle.seek(stsz_start + 8)
            sample_count = struct.unpack(">I", handle.read(4))[0]
            codec = "unknown"
            if "stsd" in stbl:
                stsd_start, _ = stbl["stsd"][0]
                handle.seek(stsd_start + 12)
                codec = handle.read(4).decode("latin-1")
            seconds = duration / timescale if timescale else 0.0
            display = source.relative_to(relative_to).as_posix() if relative_to else source.as_posix()
            return VideoProbe(
                path=display,
                codec=codec,
                width=width,
                height=height,
                sample_count=sample_count,
                duration_seconds=seconds,
                fps=sample_count / seconds if seconds else 0.0,
            )
    raise ValueError("MP4 has no parseable video track")


def filesystem_inventory(root: str | Path) -> dict[str, int]:
    base = Path(root)
    payload_files = cache_files = payload_bytes = cache_bytes = 0
    partial_files = 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        in_cache = ".cache" in path.relative_to(base).parts
        if in_cache:
            cache_files += 1
            cache_bytes += path.stat().st_size
            if path.suffix.lower() in {".lock", ".incomplete"}:
                partial_files += 1
        else:
            payload_files += 1
            payload_bytes += path.stat().st_size
    return {
        "payload_files": payload_files,
        "payload_bytes": payload_bytes,
        "cache_files": cache_files,
        "cache_bytes": cache_bytes,
        "partial_or_lock_files": partial_files,
    }


def metadata_revisions(root: str | Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in Path(root).rglob("*.metadata"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
        counts[first or "<empty>"] += 1
    return dict(sorted(counts.items()))


def coco_summary(path: str | Path) -> dict[str, object]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = {str(image.get("id")) for image in images}
    category_ids = {str(category.get("id")) for category in data.get("categories", [])}
    missing_image_refs = 0
    invalid_bboxes = 0
    invalid_polygon_components = 0
    empty_valid_segmentations = 0
    for annotation in annotations:
        if str(annotation.get("image_id")) not in image_ids:
            missing_image_refs += 1
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_bboxes += 1
        if "segmentation" in annotation:
            components = annotation.get("segmentation") or []
            valid = 0
            for component in components if isinstance(components, list) else []:
                if isinstance(component, list) and len(component) >= 6 and len(component) % 2 == 0:
                    valid += 1
                else:
                    invalid_polygon_components += 1
            if components and valid == 0:
                empty_valid_segmentations += 1
    return {
        "path": source.as_posix(),
        "top_level_keys": sorted(data),
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(category_ids),
        "missing_image_references": missing_image_refs,
        "invalid_bboxes": invalid_bboxes,
        "invalid_polygon_components": invalid_polygon_components,
        "annotations_without_valid_polygon": empty_valid_segmentations,
    }


def video_tas_alignment(dataset_root: str | Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(dataset_root)
    video_root = root / "data" / "videos" / "0"
    tas_root = root / "data" / "annotations" / "tas"
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    label_counts: Counter[str] = Counter()
    def sequence_id(path: Path) -> str:
        stem = path.stem
        if not stem.startswith("sequence_"):
            raise ValueError(f"unexpected video filename: {path.name}")
        return stem.removeprefix("sequence_")

    for video in sorted(video_root.glob("*.mp4"), key=lambda p: int(sequence_id(p))):
        sequence = sequence_id(video)
        probe = probe_mp4(video, root)
        tas = tas_root / f"sequence.{sequence}.txt"
        labels = tas.read_text(encoding="utf-8").splitlines() if tas.exists() else []
        label_counts.update(labels)
        aligned = len(labels) == probe.sample_count
        if not aligned:
            errors.append(f"sequence {sequence}: video={probe.sample_count}, tas={len(labels)}")
        row = asdict(probe)
        row.update({"sequence_id": sequence, "tas_lines": len(labels), "tas_aligned": aligned})
        rows.append(row)
    return rows, {
        "videos": len(rows),
        "alignment_errors": errors,
        "tas_label_counts": dict(sorted(label_counts.items())),
    }
