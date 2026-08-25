"""Classic baselines: majority class, flattened logistic regression, 1-month momentum."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import LOGREG_C_GRID
from dataset import DataBundle
from evaluate import compute_metrics


def _pack(split: Dict[str, np.ndarray], prob: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "y": split["y"].astype(np.float32),
        "prob": prob.astype(np.float32),
        "y_id": split["id"].astype(np.int64),
        "fwd_ret": split["fwd"].astype(np.float32),
        "mom": split["mom"].astype(np.float32),
        "date": split["date"].astype(np.int64),
        "sector": split["sector"].astype(np.int64),
    }


def run_majority(bundle: DataBundle) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict]:
    maj = int(bundle.arrays["train"]["y"].mean() >= 0.5)
    extra = {"majority_label": maj}
    val_p = np.full(len(bundle.arrays["val"]["y"]), float(maj))
    test_p = np.full(len(bundle.arrays["test"]["y"]), float(maj))
    val_m = compute_metrics(bundle.arrays["val"]["y"], val_p)
    test_pred = _pack(bundle.arrays["test"], test_p)
    return test_pred, val_m, extra


def run_momentum(bundle: DataBundle) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict]:
    val_p = (bundle.arrays["val"]["mom"] > 0).astype(np.float32)
    test_p = (bundle.arrays["test"]["mom"] > 0).astype(np.float32)
    val_m = compute_metrics(bundle.arrays["val"]["y"], val_p)
    return _pack(bundle.arrays["test"], test_p), val_m, {}


def run_logreg(bundle: DataBundle) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict]:
    def flat(split: str) -> Tuple[np.ndarray, np.ndarray]:
        x = bundle.arrays[split]["x"].reshape(len(bundle.arrays[split]["x"]), -1)
        y = bundle.arrays[split]["y"].astype(int)
        return x, y

    x_tr, y_tr = flat("train")
    x_va, y_va = flat("val")
    x_te, y_te = flat("test")
    best_c, best_mcc = LOGREG_C_GRID[0], -2.0
    for c in LOGREG_C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        clf.fit(x_tr, y_tr)
        p = clf.predict_proba(x_va)[:, 1]
        mcc = compute_metrics(y_va, p)["mcc"]
        if mcc > best_mcc:
            best_c, best_mcc = c, mcc
    clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")
    clf.fit(x_tr, y_tr)
    val_p = clf.predict_proba(x_va)[:, 1]
    test_p = clf.predict_proba(x_te)[:, 1]
    val_m = compute_metrics(y_va, val_p)
    extra = {"best_C": float(best_c), "n_features": int(x_tr.shape[1])}
    return _pack(bundle.arrays["test"], test_p), val_m, extra
