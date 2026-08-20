"""Report missing Phase 1 dependencies without changing the environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    distribution: str
    minimum: str
    install_spec: str
    maximum_exclusive: str | None = None
    import_name: str | None = None


REQUIREMENTS = (
    Requirement("torch", "2.5.0", "torch>=2.5"),
    Requirement("torchvision", "0.20.0", "torchvision>=0.20"),
    Requirement("numpy", "1.26.0", "numpy>=1.26"),
    Requirement("pandas", "2.2.0", "pandas>=2.2"),
    Requirement("pyarrow", "15.0.0", "pyarrow>=15"),
    Requirement(
        "scikit-learn", "1.5.0", "scikit-learn>=1.5,<2", "2.0.0", "sklearn"
    ),
    Requirement("scipy", "1.13.0", "scipy>=1.13,<2", "2.0.0"),
    Requirement("av", "12.0.0", "av>=12,<17", "17.0.0"),
    Requirement("Pillow", "10.0.0", "Pillow>=10", import_name="PIL"),
    Requirement("PyYAML", "6.0.0", "PyYAML>=6", import_name="yaml"),
    Requirement("tqdm", "4.66.0", "tqdm>=4.66"),
    Requirement("matplotlib", "3.8.0", "matplotlib>=3.8"),
    Requirement(
        "huggingface-hub",
        "0.25.0",
        "huggingface-hub>=0.25",
        import_name="huggingface_hub",
    ),
    Requirement("timm", "1.0.0", "timm>=1.0,<2", "2.0.0"),
    Requirement("einops", "0.8.0", "einops>=0.8,<1", "1.0.0"),
    Requirement(
        "pycocotools",
        "2.0.8",
        "pycocotools>=2.0.8,<3",
        "3.0.0",
        "pycocotools.mask",
    ),
    Requirement("pytest", "8.0.0", "pytest>=8,<10", "10.0.0"),
)


def release_tuple(version: str) -> tuple[int, ...]:
    """Return a comparison tuple for ordinary PEP 440 release versions."""
    match = re.match(r"\s*(\d+(?:\.\d+)*)", version)
    return tuple(map(int, match.group(1).split("."))) if match else ()


def is_old(installed: str, minimum: str) -> bool:
    left = release_tuple(installed)
    right = release_tuple(minimum)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) < right + (0,) * (width - len(right))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Phase 1 dependencies without modifying the environment."
    )
    parser.add_argument(
        "--skip-env-name-check",
        action="store_true",
        help="Allow a Runpod/container Python not named 'llm'; package/runtime checks still run.",
    )
    args = parser.parse_args()

    installs: list[str] = []
    package_notes: list[str] = []
    runtime_errors: list[str] = []
    runtime_warnings: list[str] = []

    if sys.version_info < (3, 11):
        runtime_errors.append(
            f"Python {sys.version.split()[0]} is too old; pinned V-JEPA 2 requires Python >=3.11."
        )

    active_env = os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(sys.prefix)
    if not args.skip_env_name_check and active_env.lower() != "llm":
        runtime_errors.append(
            f"Active environment is {active_env!r}; run this script through the existing 'llm' environment."
        )

    for requirement in REQUIREMENTS:
        try:
            installed = importlib.metadata.version(requirement.distribution)
        except importlib.metadata.PackageNotFoundError:
            installs.append(requirement.install_spec)
            package_notes.append(f"missing: {requirement.distribution}")
            continue
        if is_old(installed, requirement.minimum):
            installs.append(requirement.install_spec)
            package_notes.append(
                f"update: {requirement.distribution} {installed} -> >={requirement.minimum}"
            )
        elif requirement.maximum_exclusive and not is_old(
            installed, requirement.maximum_exclusive
        ):
            installs.append(requirement.install_spec)
            package_notes.append(
                f"incompatible: {requirement.distribution} {installed}; "
                f"require {requirement.install_spec}"
            )

        try:
            importlib.import_module(
                requirement.import_name or requirement.distribution.replace("-", "_")
            )
        except Exception as exc:
            if requirement.install_spec not in installs:
                installs.append(requirement.install_spec)
                package_notes.append(
                    f"broken import: {requirement.distribution} {installed} ({exc})"
                )

    try:
        import torch

        if not torch.cuda.is_available():
            runtime_errors.append("PyTorch cannot see a CUDA GPU.")
        else:
            properties = torch.cuda.get_device_properties(0)
            gib = properties.total_memory / 1024**3
            bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if not bf16:
                runtime_warnings.append(
                    "CUDA is available, but BF16 is not supported; use FP16/FP32."
                )
            if gib < 8:
                runtime_warnings.append(
                    f"GPU memory is {gib:.1f} GiB; use inference batch 1 and run the planned OOM smoke test."
                )

        try:
            import torchvision

            from torchvision.ops import nms

            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
            scores = torch.tensor([1.0])
            nms(boxes, scores, 0.5)
        except Exception as exc:  # ABI/import failures are actionable even when metadata exists.
            runtime_errors.append(f"torch/torchvision compatibility check failed: {exc}")
    except Exception as exc:
        if not any(note.endswith("torch") for note in package_notes):
            runtime_errors.append(f"PyTorch import failed: {exc}")

    if installs:
        print("Necessary package installs/updates:")
        for note in package_notes:
            print(f"  - {note}")
        print("\nRun only this command in the existing environment:")
        quoted = " ".join(f'"{spec}"' for spec in installs)
        launcher = "python" if args.skip_env_name_check else "conda run -n llm python"
        print(f"  {launcher} -m pip install {quoted}")
        if any(spec.startswith(("torch>", "torchvision>")) for spec in installs):
            print(
                "  Note: for torch/torchvision, preserve a CUDA-enabled matched pair from the official PyTorch selector."
            )
    else:
        print("No package installs or updates are necessary.")

    if runtime_errors:
        print("\nRuntime blockers:")
        for message in runtime_errors:
            print(f"  - {message}")

    if runtime_warnings:
        print("\nRuntime warnings:")
        for message in runtime_warnings:
            print(f"  - {message}")

    return 1 if installs or runtime_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
