from pathlib import Path
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "phase1"
EGO_DIR = DATA_DIR / "EgoInteract"


def main():
    EGO_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading EgoInteract Phase-1 subset")
    print(f"Destination: {EGO_DIR}")
    print("=" * 70)

    snapshot_download(
        repo_id="EgoInteract/EgoInteract",
        repo_type="dataset",
        local_dir=str(EGO_DIR),

        # Phase-1 subset:
        # - all annotations
        # - all configs / metadata
        # - only video partition 0
        allow_patterns=[
            "README.md",
            "configs/**",
            "data/annotations/**",
            "data/videos/0/**",
        ],
    )

    print("\nDownload complete.")
    print(f"EgoInteract subset: {EGO_DIR}")


if __name__ == "__main__":
    main()