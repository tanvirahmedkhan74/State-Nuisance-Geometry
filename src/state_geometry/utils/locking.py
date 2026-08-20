from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json


def begin_test_access(marker: str | Path, provenance: dict[str, Any]) -> None:
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The marker must survive even a write failure; do not remove it.
        raise


def complete_test_access(
    access_marker: str | Path,
    completion_marker: str | Path,
    result: dict[str, Any],
) -> None:
    access_path = Path(access_marker)
    if not access_path.exists():
        raise RuntimeError("cannot complete test evaluation before access marker exists")
    completion_path = Path(completion_marker)
    if completion_path.exists():
        raise FileExistsError(f"test completion marker already exists: {completion_path}")
    atomic_write_json(
        completion_path,
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
        },
    )
