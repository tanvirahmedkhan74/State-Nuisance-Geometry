import unittest

import torch

import _bootstrap  # noqa: F401
from state_geometry.features.pooling import (
    full_token_pool,
    masks_to_tubelet_occupancy,
    weighted_region_pool,
)


class PoolingTests(unittest.TestCase):
    def test_shape_and_full_mask_invariant(self) -> None:
        masks = torch.ones(2, 16, 384, 384, dtype=torch.bool)
        occupancy = masks_to_tubelet_occupancy(masks)
        self.assertEqual(tuple(occupancy.shape), (2, 8, 24, 24))
        tokens = torch.randn(2, 8, 24, 24, 32, dtype=torch.bfloat16)
        torch.testing.assert_close(
            weighted_region_pool(tokens, occupancy), full_token_pool(tokens)
        )

    def test_empty_mask_fails(self) -> None:
        tokens = torch.randn(1, 8, 24, 24, 4)
        with self.assertRaises(ValueError):
            weighted_region_pool(tokens, torch.zeros(1, 8, 24, 24))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression test")
    def test_cpu_occupancy_pools_cuda_tokens(self) -> None:
        tokens = torch.randn(1, 1, 24, 24, 8, device="cuda", dtype=torch.bfloat16)
        occupancy = torch.ones(1, 1, 24, 24, device="cpu")
        pooled = weighted_region_pool(tokens, occupancy)
        self.assertEqual(pooled.device.type, "cuda")
        self.assertEqual(pooled.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
