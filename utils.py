"""Download (AkShare + yfinance), synthetic OHLCV, plots, checkpoints, run logs."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import CKPT_DIR, FIGURES_DIR, OUTPUT_DIR, RAW_DIR, RUNS_DIR


OHLCV_MAP = {
    "open": ["open", "Open", "开盘", "开盘价"],
    "high": ["high", "High", "最高", "最高价"],
    "low": ["low", "Low", "最低", "最低价"],
    "close": ["close", "Close", "收盘", "收盘价", "adj close", "Adj Close"],
    "volume": ["volume", "Volume", "成交量"],
    "date": ["date", "Date", "日期", "time", "Time", "datetime"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs() -> None:
    from config import PROCESSED_DIR

    for path in (RAW_DIR, PROCESSED_DIR, CKPT_DIR, OUTPUT_DIR, FIGURES_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def log_run(exp_name: str, seed: int, payload: Dict[str, Any]) -> Path:
    path = RUNS_DIR / exp_name / f"{seed}.json"
    save_json(payload, path)
    return path


def save_preds(exp_name: str, seed: int, arrays: Dict[str, np.ndarray]) -> Path:
    path = RUNS_DIR / exp_name / f"{seed}_preds.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_preds(exp_name: str, seed: int) -> Dict[str, np.ndarray]:
    path = RUNS_DIR / exp_name / f"{seed}_preds.npz"
    blob = np.load(path, allow_pickle=False)
    return {k: blob[k] for k in blob.files}


def _pick_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return lowered[key]
    return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [str(col[0]).strip() for col in work.columns]
    else:
        work.columns = [str(c).strip() for c in work.columns]

    date_col = _pick_column(work, OHLCV_MAP["date"])
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col])
        work = work.set_index(date_col)
    else:
        work.index = pd.to_datetime(work.index)

    renamed = {}
    for std, aliases in OHLCV_MAP.items():
        if std == "date":
            continue
        col = _pick_column(work, aliases)
        if col is None:
            raise ValueError(f"Missing column for {std}; got {list(work.columns)}")
        renamed[col] = std
    out = work.rename(columns=renamed)[["open", "high", "low", "close", "volume"]]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.astype(float)


def _fetch_akshare(ticker: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
    return normalize_ohlcv(df)


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty frame for {ticker}")
    return normalize_ohlcv(df)


def fetch_us_stock(
    ticker: str,
    start: str,
    end: str,
    cache: bool = True,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    ensure_dirs()
    cache_path = RAW_DIR / f"{ticker}.csv"
    if cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start:end].copy()

    errors: List[str] = []
    df = None
    try:
        df = _fetch_akshare(ticker)
        time.sleep(sleep_s)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"akshare: {exc}")
        try:
            df = _fetch_yfinance(ticker, start, end)
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"yfinance: {exc2}")

    if df is None or df.empty:
        raise RuntimeError(f"Failed to download {ticker}: " + " | ".join(errors))
    if cache:
        df.to_csv(cache_path)
    return df.loc[start:end].copy()


def make_synthetic_ohlcv(
    ticker: str,
    ticker_id: int,
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """GBM + AR returns with a per-name volatility fingerprint."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed + ticker_id * 17)
    vol = 0.012 + 0.035 * ((ticker_id % 10) / 9.0)
    drift = 0.00015 * ((ticker_id % 7) - 3)
    phi = 0.08 + 0.12 * ((ticker_id % 5) / 4.0)
    noise = rng.normal(drift, vol, size=len(dates))
    rets = np.zeros_like(noise)
    for t in range(len(noise)):
        rets[t] = noise[t] + (phi * rets[t - 1] if t else 0.0)
    close = (40.0 + ticker_id) * np.cumprod(1.0 + rets)
    high = close * (1.0 + np.abs(rng.normal(0.004, 0.003, size=len(dates))))
    low = close * (1.0 - np.abs(rng.normal(0.004, 0.003, size=len(dates))))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.002, size=len(dates)))
    base_vol = 1.0e6 * (1.0 + 0.4 * (ticker_id % 6))
    volume = rng.lognormal(mean=np.log(base_vol), sigma=0.35, size=len(dates))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def fetch_universe(
    tickers: Sequence[str],
    start: str,
    end: str,
    min_rows: int,
    synthetic: bool = False,
) -> Dict[str, pd.DataFrame]:
    if synthetic:
        print("Using synthetic OHLCV (distinct volatility fingerprints per ticker).")
        return {
            ticker: make_synthetic_ohlcv(ticker, i, start, end)
            for i, ticker in enumerate(tickers)
        }

    from tqdm import tqdm

    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    for ticker in tqdm(list(tickers), desc="Downloading US stocks"):
        try:
            df = fetch_us_stock(ticker, start, end)
            if len(df) < min_rows:
                failed.append(f"{ticker} (only {len(df)} rows)")
                continue
            out[ticker] = df
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{ticker} ({exc})")
    if failed:
        print("Skipped / failed tickers:")
        for item in failed:
            print(f"  - {item}")
    if len(out) < 8:
        raise RuntimeError(
            f"Only {len(out)} tickers downloaded; need a working network "
            "for AkShare or yfinance. Or pass --synthetic."
        )
    return out


def compute_train_only_mean_std(
    feat: pd.DataFrame,
    columns: Sequence[str],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    ddof: int = 0,
) -> Tuple[pd.Series, pd.Series]:
    """Mean/std of `columns` using only rows with index in [lo, hi] (inclusive).

    Callers must pass the training split's last date as `hi` so validation/test
    rows never enter the scaler. ``ddof=0`` matches sklearn StandardScaler.
    """
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    train = feat.loc[(feat.index >= lo_ts) & (feat.index <= hi_ts), list(columns)]
    if train.empty:
        raise RuntimeError(f"No pre-split rows in [{lo_ts.date()}, {hi_ts.date()}] for scaler stats.")
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=ddof)
    std = std.mask(std == 0, 1.0)
    return mean, std


def apply_per_stock_zscore(
    frames: Dict[str, pd.DataFrame],
    columns: Sequence[str],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Z-score each name independently; stats fitted on [lo, hi] only."""
    out = {}
    for ticker, feat in frames.items():
        mean, std = compute_train_only_mean_std(feat, columns, lo, hi, ddof=1)
        scaled = feat.copy()
        scaled[list(columns)] = (feat[list(columns)] - mean) / std
        out[ticker] = scaled
    return out


def add_features(df: pd.DataFrame, pct_clip: float) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["Open_pct"] = df["open"].pct_change()
    feat["High_pct"] = df["high"].pct_change()
    feat["Low_pct"] = df["low"].pct_change()
    feat["Close_pct"] = df["close"].pct_change()
    feat["Volume_log"] = np.log1p(df["volume"].clip(lower=0))
    feat["close"] = df["close"]
    for col in ("Open_pct", "High_pct", "Low_pct", "Close_pct"):
        feat[col] = feat[col].replace([np.inf, -np.inf], np.nan).clip(-pct_clip, pct_clip)
    return feat.dropna()


def plot_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Path,
    keys: Sequence[Tuple[str, str]],
) -> None:
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (train_key, val_key) in zip(axes[0], keys):
        if train_key not in history:
            continue
        ax.plot(history[train_key], label="train")
        if val_key in history:
            ax.plot(history[val_key], label="val")
        ax.set_title(train_key.replace("train_", ""))
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison(rows: List[Dict[str, float]], save_path: Path, metric_keys=None) -> None:
    metric_keys = metric_keys or ["accuracy", "f1", "mcc", "auc"]
    names = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4.2 * len(metric_keys), 4.4))
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metric_keys):
        vals = [float(r[metric]) for r in rows]
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric.upper() if metric != "accuracy" else "Accuracy")
        ax.axhline(0.0 if metric == "mcc" else 0.5, color="black", lw=0.8, ls="--")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
