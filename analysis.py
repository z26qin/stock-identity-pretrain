"""Embedding ARI, attention heatmaps, and long/short backtest figures → figures/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from config import FIGURES_DIR
from economics import _ann_sharpe, long_short_daily
from model import StockTransformer
from represent import run_representation
from utils import ensure_dirs


def plot_cost_sensitivity(
    preds_by_name: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
    bps: Sequence[int] = (0, 2, 5, 10, 15, 20),
    decile: float = 0.10,
    min_names: int = 4,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, pred in preds_by_name.items():
        daily = long_short_daily(
            pred["date"], pred["y_id"], pred["prob"], pred["fwd_ret"],
            decile=decile, min_names=min_names,
        )
        if daily.empty:
            continue
        sharpes = []
        for c in bps:
            net = daily["gross"].to_numpy() - daily["turnover"].to_numpy() * 2.0 * (c / 1e4)
            sharpes.append(_ann_sharpe(net))
        ax.plot(list(bps), sharpes, marker="o", label=name)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("per-side cost (bps)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Long/short Sharpe vs transaction cost")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_equity_curves(
    preds_by_name: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
    cost_bps: int = 5,
    decile: float = 0.10,
    min_names: int = 4,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, pred in preds_by_name.items():
        daily = long_short_daily(
            pred["date"], pred["y_id"], pred["prob"], pred["fwd_ret"],
            decile=decile, min_names=min_names,
        )
        if daily.empty:
            continue
        net = daily["gross"].to_numpy() - daily["turnover"].to_numpy() * 2.0 * (cost_bps / 1e4)
        equity = np.cumprod(1.0 + net)
        ax.plot(range(len(equity)), equity, label=name, lw=1.4)
    ax.set_xlabel("test day")
    ax.set_ylabel("growth of $1")
    ax.set_title(f"Long/short equity net of {cost_bps}bps per side")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def produce_all_figures(
    ours: Optional[StockTransformer],
    scratch: Optional[StockTransformer],
    loader: DataLoader,
    device,
    id_to_sector: Dict[int, str],
    preds_by_name: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    out_dir: Optional[Path] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Write clustering, attention, and economic figures into figures/."""
    ensure_dirs()
    out_dir = Path(out_dir) if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, float] = {}
    if ours is not None or scratch is not None:
        report.update(
            run_representation(
                ours, scratch, loader, device, id_to_sector, out_dir, seed=seed
            )
        )
    if preds_by_name:
        plot_cost_sensitivity(preds_by_name, out_dir / "cost_sensitivity.png")
        plot_equity_curves(preds_by_name, out_dir / "equity_curves.png")
        report["n_econ_models"] = float(len(preds_by_name))
    return report
