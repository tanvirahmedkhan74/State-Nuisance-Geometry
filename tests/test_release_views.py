import tempfile
import unittest
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from state_geometry.data.release_views import build_release_views
from state_geometry.data.schema import FEATURE_INPUT_ALLOWED_COLUMNS, validate_feature_inputs


class ReleaseViewTests(unittest.TestCase):
    def test_redaction_and_test_seal(self) -> None:
        records = []
        for index, split in enumerate(("train", "validation", "test")):
            row = {column: f"v{index}" for column in FEATURE_INPUT_ALLOWED_COLUMNS}
            row.update(
                {
                    "observation_id": f"o{index}",
                    "start_frame": 0,
                    "end_frame": 29,
                    "sampled_frame_indices": list(range(16)),
                    "fps": 30.0,
                    "mask_available": False,
                    "motion_quality_pass": True,
                    "split": split,
                    "state_label": "open" if index % 2 else "closed",
                }
            )
            records.append(row)
        observations = pd.DataFrame(records)
        triplets = pd.DataFrame(
            {
                "triplet_id": ["tr", "tv", "tt"],
                "split": ["train", "validation", "test"],
                "state_target": ["open", "closed", "open"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = build_release_views(observations, triplets, temporary)
            feature = pd.read_parquet(Path(temporary) / "feature_inputs.parquet")
            self.assertEqual(validate_feature_inputs(feature), [])
            self.assertNotIn("state_label", feature)
            self.assertIsNone(manifest["artifacts"]["sealed_test_targets.parquet"]["rows"])
            trainval = pd.read_parquet(Path(temporary) / "analysis_trainval.parquet")
            self.assertNotIn("test", set(trainval["split"]))


if __name__ == "__main__":
    unittest.main()

