import unittest

import torch

import _bootstrap  # noqa: F401
from state_geometry.features.vjepa import clean_backbone_state_dict, reshape_dense_tokens


class VJEPAContractTests(unittest.TestCase):
    def test_checkpoint_prefix_cleaning(self) -> None:
        source = {"module.backbone.a": torch.ones(1), "module.b": torch.zeros(1)}
        cleaned = clean_backbone_state_dict(source)
        self.assertEqual(set(cleaned), {"a", "b"})
        self.assertEqual(set(source), {"module.backbone.a", "module.b"})

    def test_dense_token_shape(self) -> None:
        tokens = torch.randn(2, 4608, 768)
        self.assertEqual(tuple(reshape_dense_tokens(tokens).shape), (2, 8, 24, 24, 768))
        self.assertEqual(
            tuple(reshape_dense_tokens(torch.randn(1, 576, 768)).shape),
            (1, 1, 24, 24, 768),
        )
        with self.assertRaises(ValueError):
            reshape_dense_tokens(torch.randn(1, 100, 768))


if __name__ == "__main__":
    unittest.main()
