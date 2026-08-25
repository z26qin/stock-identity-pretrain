"""Daily long/short portfolio from P(up): Sharpe, drawdown, turnover, net of costs."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def _one_way_turnover(prev: Dict[int, float], curr: Dict[int, float]) -> float:
    names = set(prev) | set(curr)
    l1 = sum(abs(curr.get(n, 0.0) - prev.get(n, 0.0)) for n in names)
    return 0.5 * l1


def long_short_daily(
    dates: np.ndarray,
    ticker_ids: np.ndarray,
    prob: np.ndarray,
    fwd_ret: np.ndarray,
    decile: float = 0.10,
    min_names: int = 8,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": dates.astype(np.int64),
            "ticker": ticker_ids.astype(np.int64),
            "p": prob.astype(float),
            "r": fwd_ret.astype(float),
        }
    )
    rows = []
    prev_w: Dict[int, float] = {}
    for date, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        k = max(1, int(len(g) * decile))
        g = g.sort_values("p")
        short = g.iloc[:k]
        long = g.iloc[-k:]
        w: Dict[int, float] = {}
        lw, sw = 1.0 / k, -1.0 / k
        for t in long["ticker"]:
            w[int(t)] = lw
        for t in short["ticker"]:
            w[int(t)] = sw
        gross = float(long["r"].mean() - short["r"].mean())
        turn = _one_way_turnover(prev_w, w)
        rows.append({"date": int(date), "gross": gross, "turnover": turn})
        prev_w = w
    return pd.DataFrame(rows)


def long_short_backtest(
    dates: np.ndarray,
    ticker_ids: np.ndarray,
    prob: np.ndarray,
    fwd_ret: np.ndarray,
    decile: float = 0.10,
    cost_bps: Sequence[int] = (0, 5, 10),
    min_names: int = 8,
) -> Dict[str, float]:
    daily = long_short_daily(dates, ticker_ids, prob, fwd_ret, decile=decile, min_names=min_names)
    if daily.empty:
        return {
            "n_days": 0,
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "turnover": float("nan"),
            **{f"sharpe_{c}bps": float("nan") for c in cost_bps},
        }

    gross = daily["gross"].to_numpy(dtype=float)
    turn = daily["turnover"].to_numpy(dtype=float)
    out = {
        "n_days": int(len(daily)),
        "sharpe": _ann_sharpe(gross),
        "max_drawdown": _max_drawdown(gross),
        "turnover": float(turn.mean()),
        "mean_daily": float(gross.mean()),
    }
    for c in cost_bps:
        net = gross - turn * 2.0 * (c / 1e4)
        out[f"sharpe_{c}bps"] = _ann_sharpe(net)
        out[f"mean_daily_{c}bps"] = float(net.mean())
    return out


def _ann_sharpe(r: np.ndarray) -> float:
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252.0))


def _max_drawdown(r: np.ndarray) -> float:
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.clip(peak, 1e-12, None) - 1.0
    return float(dd.min()) if dd.size else float("nan")
