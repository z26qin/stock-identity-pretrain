"""Scaler stats must be computed only from rows on or before the train split date."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import ALL_FEATURES, Config
from dataset import _fit_global_scaler, _split_bounds
from utils import apply_per_stock_zscore, compute_train_only_mean_std


def _shifted_feature_frame(start: str, end: str, train_end: str, post_mean: float) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    pre = idx <= pd.Timestamp(train_end)
    rng = np.random.default_rng(0)
    close_pct = np.where(pre, rng.normal(0.001, 0.01, len(idx)), rng.normal(post_mean, 0.01, len(idx)))
    vol = np.where(pre, 1.0e6, 9.0e6)
    feat = pd.DataFrame(
        {
            "Open_pct": close_pct,
            "High_pct": close_pct + 0.001,
            "Low_pct": close_pct - 0.001,
            "Close_pct": close_pct,
            "Volume_log": np.log1p(vol),
            "close": 100.0 * np.cumprod(1.0 + close_pct),
        },
        index=idx,
    )
    return feat


class LeakageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.cfg.start_date = "2020-01-01"
        self.cfg.train_end = "2021-12-31"
        self.cfg.val_end = "2022-12-31"
        self.cfg.end_date = "2023-12-31"
        self.lo, self.hi = _split_bounds(self.cfg)["train"]
        self.feat = _shifted_feature_frame(
            self.cfg.start_date, self.cfg.end_date, self.cfg.train_end, post_mean=0.08
        )

    def test_train_only_stats_ignore_post_split_rows(self) -> None:
        mean, std = compute_train_only_mean_std(
            self.feat, ALL_FEATURES, self.lo, self.hi, ddof=0
        )
        pre = self.feat.loc[self.feat.index <= self.hi, ALL_FEATURES]
        post_included = self.feat[ALL_FEATURES]
        np.testing.assert_allclose(mean.to_numpy(), pre.mean().to_numpy())
        self.assertFalse(
            np.allclose(mean.to_numpy(), post_included.mean().to_numpy()),
            "train-only mean should differ from full-sample mean when post-split distribution shifts",
        )
        self.assertGreater(self.feat.index.max(), self.hi)
        self.assertTrue((pre.index <= self.hi).all())

    def test_global_scaler_matches_pre_split_rows_only(self) -> None:
        frames = {"AAA": self.feat, "BBB": self.feat.copy()}
        scaler = _fit_global_scaler(frames, self.cfg)
        pooled = pd.concat(
            [
                f.loc[(f.index >= self.lo) & (f.index <= self.hi), ALL_FEATURES]
                for f in frames.values()
            ]
        )
        expected = StandardScaler().fit(pooled.to_numpy())
        np.testing.assert_allclose(scaler.mean_, expected.mean_)
        np.testing.assert_allclose(scaler.scale_, expected.scale_)

        all_rows = pd.concat([f[ALL_FEATURES] for f in frames.values()])
        leaked = StandardScaler().fit(all_rows.to_numpy())
        self.assertFalse(
            np.allclose(scaler.mean_, leaked.mean_),
            "scaler mean must not equal stats fitted on post-split rows",
        )

    def test_per_stock_zscore_uses_each_name_train_window(self) -> None:
        feat_b = _shifted_feature_frame(
            self.cfg.start_date, self.cfg.end_date, self.cfg.train_end, post_mean=0.12
        )
        frames = {"AAA": self.feat, "BBB": feat_b}
        scaled = apply_per_stock_zscore(frames, ALL_FEATURES, self.lo, self.hi)
        train = scaled["AAA"].loc[
            (scaled["AAA"].index >= self.lo) & (scaled["AAA"].index <= self.hi), ALL_FEATURES
        ]
        pct_cols = [c for c in ALL_FEATURES if c != "Volume_log"]
        np.testing.assert_allclose(
            train[pct_cols].mean().to_numpy(), np.zeros(len(pct_cols)), atol=1e-9
        )
        future = scaled["AAA"].loc[scaled["AAA"].index > self.hi, "Close_pct"]
        self.assertGreater(abs(float(future.mean())), 1.0)


if __name__ == "__main__":
    unittest.main()
