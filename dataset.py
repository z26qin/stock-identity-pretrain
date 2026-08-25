"""Sliding-window datasets with a time-based split (no shuffle across time)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import GICS_SECTOR, PROCESSED_DIR, Config
from utils import add_features, apply_per_stock_zscore, fetch_universe


class StockDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.x = torch.from_numpy(arrays["x"].astype(np.float32))
        self.y_trend = torch.from_numpy(arrays["y"].astype(np.float32))
        self.y_id = torch.from_numpy(arrays["id"].astype(np.int64))
        self.fwd_ret = torch.from_numpy(arrays["fwd"].astype(np.float32))
        self.mom = torch.from_numpy(arrays["mom"].astype(np.float32))
        self.date = torch.from_numpy(arrays["date"].astype(np.int64))
        self.sector = torch.from_numpy(arrays["sector"].astype(np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y_trend": self.y_trend[idx],
            "y_id": self.y_id[idx],
            "fwd_ret": self.fwd_ret[idx],
            "mom": self.mom[idx],
            "date": self.date[idx],
            "sector": self.sector[idx],
        }


@dataclass
class DataBundle:
    loaders: Dict[str, DataLoader]
    datasets: Dict[str, StockDataset]
    arrays: Dict[str, Dict[str, np.ndarray]]
    ticker_to_id: Dict[str, int]
    id_to_ticker: Dict[int, str]
    sector_to_id: Dict[str, int]
    id_to_sector: Dict[int, str]

    @property
    def num_stocks(self) -> int:
        return len(self.ticker_to_id)


def _split_bounds(cfg: Config) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(cfg.start_date)
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    end = pd.Timestamp(cfg.end_date)
    return {
        "train": (start, train_end),
        "val": (train_end + pd.Timedelta(days=1), val_end),
        "test": (val_end + pd.Timedelta(days=1), end),
    }


def _date_to_ord(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).normalize().toordinal())


def _windows_for_ticker(
    feat: pd.DataFrame,
    ticker_id: int,
    sector_id: int,
    cfg: Config,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> Dict[str, list]:
    """Keep a window ending at t only if t and t+1 both sit inside [lo, hi]."""
    out = {k: [] for k in ("x", "y", "id", "fwd", "mom", "date", "sector")}
    t_len = cfg.window
    mom_d = cfg.momentum_days
    cols = cfg.features
    sub = feat.loc[(feat.index >= lo) & (feat.index <= hi)]
    if len(sub) < t_len + 1:
        return out

    values = sub[cols].to_numpy(dtype=np.float32)
    close = sub["close"].to_numpy(dtype=np.float64)
    index = sub.index
    for i in range(t_len - 1, len(sub) - 1):
        window = values[i - t_len + 1 : i + 1]
        if np.isnan(window).any():
            continue
        fwd = close[i + 1] / close[i] - 1.0
        mom_i = i - (mom_d - 1)
        if mom_i < 0:
            continue
        mom = close[i] / close[mom_i] - 1.0
        out["x"].append(window)
        out["y"].append(int(close[i + 1] > close[i]))
        out["id"].append(ticker_id)
        out["fwd"].append(fwd)
        out["mom"].append(mom)
        out["date"].append(_date_to_ord(index[i]))
        out["sector"].append(sector_id)
    return out


def build_feature_frames(raw: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    frames = {}
    for ticker, df in raw.items():
        feat = add_features(df, pct_clip=cfg.pct_clip)
        if len(feat) >= cfg.window + cfg.momentum_days:
            frames[ticker] = feat
    return frames


def _fit_global_scaler(frames: Dict[str, pd.DataFrame], cfg: Config) -> StandardScaler:
    lo, hi = _split_bounds(cfg)["train"]
    chunks = []
    for feat in frames.values():
        sub = feat.loc[(feat.index >= lo) & (feat.index <= hi), cfg.features]
        if len(sub):
            chunks.append(sub.to_numpy(dtype=np.float64))
    if not chunks:
        raise RuntimeError("No training rows to fit the scaler.")
    scaler = StandardScaler()
    scaler.fit(np.concatenate(chunks, axis=0))
    return scaler


def _apply_global_scaler(
    frames: Dict[str, pd.DataFrame], scaler: StandardScaler, cfg: Config
) -> Dict[str, pd.DataFrame]:
    out = {}
    for ticker, feat in frames.items():
        scaled = feat.copy()
        scaled[cfg.features] = scaler.transform(feat[cfg.features].to_numpy())
        out[ticker] = scaled
    return out


def _assemble_split(
    frames: Dict[str, pd.DataFrame],
    ticker_to_id: Dict[str, int],
    sector_of: Dict[str, int],
    cfg: Config,
    split_name: str,
) -> Dict[str, np.ndarray]:
    lo, hi = _split_bounds(cfg)[split_name]
    bags = {k: [] for k in ("x", "y", "id", "fwd", "mom", "date", "sector")}
    for ticker, feat in frames.items():
        part = _windows_for_ticker(
            feat, ticker_to_id[ticker], sector_of[ticker], cfg, lo, hi
        )
        for k, vals in part.items():
            bags[k].extend(vals)
    if not bags["x"]:
        raise RuntimeError(f"Split '{split_name}' produced 0 windows.")
    return {
        "x": np.stack(bags["x"], axis=0),
        "y": np.asarray(bags["y"], dtype=np.int64),
        "id": np.asarray(bags["id"], dtype=np.int64),
        "fwd": np.asarray(bags["fwd"], dtype=np.float32),
        "mom": np.asarray(bags["mom"], dtype=np.float32),
        "date": np.asarray(bags["date"], dtype=np.int64),
        "sector": np.asarray(bags["sector"], dtype=np.int64),
    }


def _cache_paths(fp: str) -> Tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / f"windows_{fp}.npz", PROCESSED_DIR / f"meta_{fp}.json"


def build_or_load_arrays(cfg: Config, use_cache: bool = True):
    fp = cfg.data_hash()
    cache, meta_path = _cache_paths(fp)
    splits = ("train", "val", "test")
    if use_cache and cache.exists() and meta_path.exists():
        blob = np.load(cache, allow_pickle=False)
        if "train_date" in blob.files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = {}
            for split in splits:
                arrays[split] = {
                    "x": blob[f"{split}_x"],
                    "y": blob[f"{split}_y"],
                    "id": blob[f"{split}_id"],
                    "fwd": blob[f"{split}_fwd"],
                    "mom": blob[f"{split}_mom"],
                    "date": blob[f"{split}_date"],
                    "sector": blob[f"{split}_sector"],
                }
            return arrays, meta

    raw = fetch_universe(
        cfg.tickers, cfg.start_date, cfg.end_date, cfg.min_rows_per_ticker, cfg.synthetic
    )
    frames = build_feature_frames(raw, cfg)
    tickers = sorted(frames.keys())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    sectors = sorted({GICS_SECTOR.get(t, "Unknown") for t in tickers})
    sector_to_id = {s: i for i, s in enumerate(sectors)}
    sector_of = {t: sector_to_id[GICS_SECTOR.get(t, "Unknown")] for t in tickers}

    if cfg.per_stock_zscore:
        lo, hi = _split_bounds(cfg)["train"]
        frames = apply_per_stock_zscore(frames, cfg.features, lo, hi)
        scaler_mean, scaler_scale = [], []
    else:
        scaler = _fit_global_scaler(frames, cfg)
        frames = _apply_global_scaler(frames, scaler, cfg)
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()

    arrays = {}
    save_kwargs = {}
    for split in splits:
        arr = _assemble_split(frames, ticker_to_id, sector_of, cfg, split)
        arrays[split] = arr
        for key, val in arr.items():
            save_kwargs[f"{split}_{key}"] = val

    np.savez_compressed(cache, **save_kwargs)
    meta = {
        "fingerprint": fp,
        "tickers": tickers,
        "ticker_to_id": ticker_to_id,
        "sector_to_id": sector_to_id,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "n_train": int(arrays["train"]["x"].shape[0]),
        "n_val": int(arrays["val"]["x"].shape[0]),
        "n_test": int(arrays["test"]["x"].shape[0]),
        "drop_volume": cfg.drop_volume,
        "per_stock_zscore": cfg.per_stock_zscore,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Windows  train={meta['n_train']}  val={meta['n_val']}  "
        f"test={meta['n_test']}  stocks={len(tickers)}"
    )
    return arrays, meta


def get_dataloaders(
    cfg: Config,
    shuffle_train: bool = True,
    use_cache: bool = True,
) -> DataBundle:
    arrays, meta = build_or_load_arrays(cfg, use_cache=use_cache)
    datasets = {split: StockDataset(arrays[split]) for split in ("train", "val", "test")}
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=shuffle_train,
            num_workers=cfg.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        ),
    }
    ticker_to_id = {k: int(v) for k, v in meta["ticker_to_id"].items()}
    sector_to_id = {k: int(v) for k, v in meta["sector_to_id"].items()}
    return DataBundle(
        loaders=loaders,
        datasets=datasets,
        arrays=arrays,
        ticker_to_id=ticker_to_id,
        id_to_ticker={i: t for t, i in ticker_to_id.items()},
        sector_to_id=sector_to_id,
        id_to_sector={i: s for s, i in sector_to_id.items()},
    )
