import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from state_geometry.utils.locking import begin_test_access, complete_test_access


class TestLockTests(unittest.TestCase):
    def test_access_marker_is_exclusive_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            started = Path(temporary) / "started.lock"
            completed = Path(temporary) / "completed.lock"
            begin_test_access(started, {"selection": "abc"})
            self.assertTrue(started.exists())
            with self.assertRaises(FileExistsError):
                begin_test_access(started, {"selection": "abc"})
            complete_test_access(started, completed, {"metrics_sha256": "def"})
            self.assertTrue(started.exists())
            self.assertTrue(completed.exists())


if __name__ == "__main__":
    unittest.main()
