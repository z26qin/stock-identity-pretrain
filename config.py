"""Hyperparameters, US universe, GICS map, and experiment-protocol defaults."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CKPT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures"
RUNS_DIR = ROOT / "runs"

ALL_FEATURES = ["Open_pct", "High_pct", "Low_pct", "Close_pct", "Volume_log"]

US_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM",
    "ORCL", "INTC", "AMD", "AVGO", "CSCO", "QCOM", "IBM", "NOW", "INTU", "AMAT",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
    "MA", "SPGI", "PNC", "USB", "COF",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "LLY", "BMY", "AMGN",
    "KO", "PEP", "PG", "WMT", "COST", "MCD", "NKE", "SBUX", "HD", "DIS",
    "XOM", "CVX", "COP", "CAT", "HON", "UNP", "BA", "GE", "NEE", "LIN",
]

GICS_SECTOR: Dict[str, str] = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "GOOGL": "Communication Services", "AMZN": "Consumer Discretionary",
    "META": "Communication Services", "NVDA": "Information Technology",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    "ADBE": "Information Technology", "CRM": "Information Technology",
    "ORCL": "Information Technology", "INTC": "Information Technology",
    "AMD": "Information Technology", "AVGO": "Information Technology",
    "CSCO": "Information Technology", "QCOM": "Information Technology",
    "IBM": "Information Technology", "NOW": "Information Technology",
    "INTU": "Information Technology", "AMAT": "Information Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "SPGI": "Financials",
    "PNC": "Financials", "USB": "Financials", "COF": "Financials",
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care",
    "MRK": "Health Care", "ABBV": "Health Care", "TMO": "Health Care",
    "ABT": "Health Care", "LLY": "Health Care", "BMY": "Health Care",
    "AMGN": "Health Care",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "PG": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "DIS": "Communication Services",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "BA": "Industrials", "GE": "Industrials",
    "NEE": "Utilities", "LIN": "Materials",
}

SEEDS = (0, 1, 2, 3, 4)
LOGREG_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: list(US_TICKERS))
    start_date: str = "2016-01-01"
    end_date: str = "2024-12-31"
    train_end: str = "2021-12-31"
    val_end: str = "2022-12-31"

    window: int = 30
    momentum_days: int = 21
    drop_volume: bool = False
    per_stock_zscore: bool = False

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

    pretext: str = "stock_id"  # stock_id | mask_recon | none
    transfer_mode: str = "frozen"  # frozen | linear_probe | full
    mask_ratio: float = 0.15

    batch_size: int = 64
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    baseline_epochs: int = 50
    lr_pretrain: float = 1e-3
    lr_finetune: float = 1e-3
    lr_baseline: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    patience_pretrain: int = 15
    patience_finetune: int = 10
    num_workers: int = 0
    seed: int = 42

    pct_clip: float = 1.0
    min_rows_per_ticker: int = 400
    synthetic: bool = False

    n_bootstrap: int = 1000
    bootstrap_block: int = 5
    ls_decile: float = 0.10
    cost_bps: tuple = (0, 5, 10)

    @property
    def features(self) -> List[str]:
        if self.drop_volume:
            return [f for f in ALL_FEATURES if f != "Volume_log"]
        return list(ALL_FEATURES)

    @property
    def n_features(self) -> int:
        return len(self.features)

    def apply_quick(self) -> "Config":
        self.tickers = list(US_TICKERS[:16])
        self.start_date = "2020-01-01"
        self.train_end = "2022-12-31"
        self.val_end = "2023-06-30"
        self.pretrain_epochs = 8
        self.finetune_epochs = 5
        self.baseline_epochs = 5
        self.patience_pretrain = 4
        self.patience_finetune = 3
        self.min_rows_per_ticker = 200
        self.n_bootstrap = 200
        return self

    def apply_smoke(self) -> "Config":
        self.apply_quick()
        self.tickers = list(US_TICKERS[:8])
        self.synthetic = True
        self.pretrain_epochs = 2
        self.finetune_epochs = 2
        self.baseline_epochs = 2
        self.patience_pretrain = 2
        self.patience_finetune = 2
        self.n_bootstrap = 64
        return self

    def data_hash(self) -> str:
        payload = {
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "window": self.window,
            "momentum_days": self.momentum_days,
            "drop_volume": self.drop_volume,
            "per_stock_zscore": self.per_stock_zscore,
            "pct_clip": self.pct_clip,
            "min_rows_per_ticker": self.min_rows_per_ticker,
            "synthetic": self.synthetic,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(blob).hexdigest()[:12]

    def config_hash(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True).encode("utf-8")
        return hashlib.md5(blob).hexdigest()[:12]

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["features"] = self.features
        d["cost_bps"] = list(self.cost_bps)
        return d


def oat_experiments() -> List[Dict[str, Any]]:
    """One-at-a-time matrix from the default 'ours' cell, plus required baselines."""
    ours = dict(
        pretext="stock_id",
        transfer_mode="frozen",
        drop_volume=False,
        per_stock_zscore=False,
    )
    return [
        dict(name="majority", kind="classic"),
        dict(name="logreg", kind="classic"),
        dict(name="momentum", kind="classic"),
        dict(name="transformer_scratch", kind="transformer", pretext="none", transfer_mode="full",
             drop_volume=False, per_stock_zscore=False),
        dict(name="ours", kind="transformer", **ours),
        dict(name="ablate_pretext_mask", kind="transformer", **{**ours, "pretext": "mask_recon"}),
        dict(name="ablate_pretext_none", kind="transformer", **{**ours, "pretext": "none"}),
        dict(name="ablate_transfer_probe", kind="transformer", **{**ours, "transfer_mode": "linear_probe"}),
        dict(name="ablate_transfer_full", kind="transformer", **{**ours, "transfer_mode": "full"}),
        dict(name="ablate_no_volume", kind="transformer", **{**ours, "drop_volume": True}),
        dict(name="ablate_perstock_z", kind="transformer", **{**ours, "per_stock_zscore": True}),
    ]


def full_factorial_experiments() -> List[Dict[str, Any]]:
    rows = [
        dict(name="majority", kind="classic"),
        dict(name="logreg", kind="classic"),
        dict(name="momentum", kind="classic"),
    ]
    for pretext in ("stock_id", "mask_recon", "none"):
        for transfer_mode in ("frozen", "linear_probe", "full"):
            for drop_volume in (False, True):
                for per_stock_zscore in (False, True):
                    name = (
                        f"tf-pre{pretext}-tr{transfer_mode}"
                        f"-vol{int(not drop_volume)}-z{int(per_stock_zscore)}"
                    )
                    rows.append(
                        dict(
                            name=name,
                            kind="transformer",
                            pretext=pretext,
                            transfer_mode=transfer_mode,
                            drop_volume=drop_volume,
                            per_stock_zscore=per_stock_zscore,
                        )
                    )
    return rows
