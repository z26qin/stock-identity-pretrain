"""Shape tests for stock_id / masked_recon / none pretext paths and transfer_mode."""

from __future__ import annotations

import unittest

import torch

from model import StockTransformer


class ModelShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.B, self.T, self.F, self.N = 4, 30, 5, 8
        self.x = torch.randn(self.B, self.T, self.F)
        self.model = StockTransformer(
            n_features=self.F, num_stocks=self.N, d_model=64, nhead=4, num_layers=2, window=self.T
        )
        self.model.eval()

    def test_stock_id_head_shape(self) -> None:
        self.model.pretrain_task = "stock_id"
        y = self.model(self.x)
        self.assertEqual(tuple(y.shape), (self.B, self.N))

    def test_masked_recon_head_shape(self) -> None:
        self.model.pretrain_task = "mask_recon"
        mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        mask[:, 3:8] = True
        y = self.model(self.x, timestep_mask=mask)
        self.assertEqual(tuple(y.shape), (self.B, self.T, self.F))

    def test_none_pretext_uses_trend_head_shape(self) -> None:
        self.model.pretrain_task = "none"
        y = self.model(self.x)
        self.assertEqual(tuple(y.shape), (self.B,))

    def test_transfer_mode_frozen_and_full_shapes(self) -> None:
        self.model.apply_transfer_mode("frozen")
        self.assertEqual(self.model.transfer_mode, "frozen")
        y = self.model(self.x)
        self.assertEqual(tuple(y.shape), (self.B,))
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("trend_head") for n in trainable))

        self.model.apply_transfer_mode("full")
        self.assertEqual(self.model.transfer_mode, "full")
        y = self.model(self.x)
        self.assertEqual(tuple(y.shape), (self.B,))
        self.assertTrue(any(n.startswith("blocks") for n, p in self.model.named_parameters() if p.requires_grad))

    def test_linear_probe_encode_shape(self) -> None:
        self.model.apply_transfer_mode("linear_probe")
        h = self.model.encode(self.x)
        self.assertEqual(tuple(h.shape), (self.B, 64))


if __name__ == "__main__":
    unittest.main()
