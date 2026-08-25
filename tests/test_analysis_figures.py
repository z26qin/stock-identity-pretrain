"""analysis.py writes clustering / attention / cost figures into a directory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis import plot_cost_sensitivity, plot_equity_curves, produce_all_figures
from model import StockTransformer


class _DictLoader:
    """Minimal loader matching StockDataset batch keys."""

    def __init__(self, n=32, t=30, f=5, n_stocks=4):
        self.n = n
        rng = np.random.default_rng(0)
        self.x = torch.from_numpy(rng.normal(size=(n, t, f)).astype(np.float32))
        self.sector = torch.from_numpy(rng.integers(0, 3, size=n).astype(np.int64))
        self.y_id = torch.from_numpy(rng.integers(0, n_stocks, size=n).astype(np.int64))
        self.y_trend = torch.zeros(n)
        self.fwd_ret = torch.from_numpy(rng.normal(0, 0.01, size=n).astype(np.float32))
        self.mom = torch.zeros(n)
        self.date = torch.from_numpy(np.repeat(np.arange(8), n // 8 + 1)[:n].astype(np.int64))

    def __iter__(self):
        yield {
            "x": self.x,
            "sector": self.sector,
            "y_id": self.y_id,
            "y_trend": self.y_trend,
            "fwd_ret": self.fwd_ret,
            "mom": self.mom,
            "date": self.date,
        }


class AnalysisFigureTests(unittest.TestCase):
    def test_cost_and_equity_files(self) -> None:
        rng = np.random.default_rng(1)
        n, names = 80, 8
        dates = np.repeat(np.arange(10), names)
        tickers = np.tile(np.arange(names), 10)
        pred = {
            "date": dates,
            "y_id": tickers,
            "prob": rng.random(n),
            "fwd_ret": rng.normal(0, 0.01, size=n),
        }
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            plot_cost_sensitivity({"ours": pred, "scratch": pred}, out / "cost_sensitivity.png", min_names=4)
            plot_equity_curves({"ours": pred}, out / "equity_curves.png", min_names=4)
            self.assertTrue((out / "cost_sensitivity.png").is_file())
            self.assertTrue((out / "equity_curves.png").is_file())

    def test_produce_all_figures_with_tiny_model(self) -> None:
        model = StockTransformer(n_features=5, num_stocks=4, d_model=64, window=30)
        model.eval()
        loader = _DictLoader()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            rng = np.random.default_rng(2)
            pred = {
                "date": np.repeat(np.arange(10), 8),
                "y_id": np.tile(np.arange(8), 10),
                "prob": rng.random(80),
                "fwd_ret": rng.normal(0, 0.01, size=80),
            }
            report = produce_all_figures(
                model, None, loader, torch.device("cpu"),
                {0: "IT", 1: "Fin", 2: "HC"},
                preds_by_name={"ours": pred},
                out_dir=out, seed=0,
            )
            self.assertTrue((out / "umap_ours.png").is_file())
            self.assertTrue((out / "attn_ours.png").is_file())
            self.assertTrue((out / "cost_sensitivity.png").is_file())
            self.assertIn("ari_ours", report)


if __name__ == "__main__":
    unittest.main()
