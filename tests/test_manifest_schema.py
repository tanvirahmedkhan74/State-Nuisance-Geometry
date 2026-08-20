import unittest

import pandas as pd

import _bootstrap  # noqa: F401
from state_geometry.data.schema import CURATED_REQUIRED_COLUMNS, validate_curated_manifest


class ManifestSchemaTests(unittest.TestCase):
    def valid_row(self) -> dict[str, object]:
        row = {column: "x" for column in CURATED_REQUIRED_COLUMNS}
        row.update(
            {
                "observation_id": "o1",
                "dataset_revision": "a" * 40,
                "source_video_id": "v1",
                "media_relpath": "data/videos/0/sequence_1.mp4",
                "media_type": "video",
                "physical_object_id": "object-track-1",
                "object_category_manual": "door",
                "state_family": "open_closed",
                "state_label": "open",
                "stable_segment_id": "segment-1",
                "transition_id": "transition-1",
                "start_frame": 10,
                "end_frame": 39,
                "sampled_frame_indices": [10 + round(index * 29 / 15) for index in range(16)],
                "fps": 30.0,
                "state_observable": True,
                "identity_verified": True,
                "box_annotation_relpath": "annotations/boxes/segment-1.json",
                "mask_available": False,
                "nuisance_tags": ["viewpoint"],
                "review_status": "approved",
                "media_sha256": "b" * 64,
                "perceptual_hash": "phash:123",
                "duplicate_group_id": "dup:o1",
            }
        )
        return row

    def test_valid_curated_row(self) -> None:
        self.assertEqual(validate_curated_manifest(pd.DataFrame([self.valid_row()])), [])

    def test_generic_state_and_windows_path_fail(self) -> None:
        row = self.valid_row()
        row["state_label"] = "before"
        row["media_relpath"] = r"C:\data\video.mp4"
        errors = validate_curated_manifest(pd.DataFrame([row]))
        self.assertTrue(any("generic" in error for error in errors))
        self.assertTrue(any("unsafe" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

