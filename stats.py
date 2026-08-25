"""Block-bootstrap MCC intervals and paired win rates over test days."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import matthews_corrcoef


def _safe_mcc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    if y.size == 0 or len(np.unique(y)) < 2 or len(np.unique(pred)) < 1:
        return float("nan")
    return float(matthews_corrcoef(y, pred))


def _date_index(dates: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for d in np.unique(dates):
        out[int(d)] = np.flatnonzero(dates == d)
    return out


def iter_block_indices(
    dates: np.ndarray,
    n_bootstrap: int = 1000,
    block: int = 5,
    seed: int = 0,
) -> Iterable[np.ndarray]:
    """Yield row indices by resampling contiguous blocks of unique test days."""
    unique = np.sort(np.unique(dates))
    n_days = len(unique)
    if n_days == 0:
        return
    block = max(1, min(block, n_days))
    n_blocks = int(np.ceil(n_days / block))
    by_date = _date_index(dates)
    rng = np.random.default_rng(seed)
    max_start = n_days - block + 1
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        rows = []
        for s in starts:
            for d in unique[s : s + block]:
                rows.append(by_date[int(d)])
        yield np.concatenate(rows) if rows else np.array([], dtype=int)


def bootstrap_mcc(
    y: np.ndarray,
    prob: np.ndarray,
    dates: np.ndarray,
    n_bootstrap: int = 1000,
    block: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    stats = []
    for idx in iter_block_indices(dates, n_bootstrap, block, seed):
        stats.append(_safe_mcc(y[idx], prob[idx]))
    arr = np.asarray(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "n_valid": int(arr.size),
    }


def paired_win_rate(
    y: np.ndarray,
    dates: np.ndarray,
    ours_prob: np.ndarray,
    other_prob: np.ndarray,
    n_bootstrap: int = 1000,
    block: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    wins = 0
    n = 0
    for idx in iter_block_indices(dates, n_bootstrap, block, seed):
        a = _safe_mcc(y[idx], ours_prob[idx])
        b = _safe_mcc(y[idx], other_prob[idx])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        n += 1
        if a > b:
            wins += 1
    frac = float(wins / n) if n else float("nan")
    return {"win_rate": frac, "n_valid": n, "wins": wins}


def align_by_date_ticker(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]):
    key_a = a["date"].astype(np.int64) * 100_000 + a["y_id"].astype(np.int64)
    key_b = b["date"].astype(np.int64) * 100_000 + b["y_id"].astype(np.int64)
    pos_b = {int(k): i for i, k in enumerate(key_b)}
    ia, ib = [], []
    for i, k in enumerate(key_a):
        j = pos_b.get(int(k))
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}
