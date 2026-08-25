"""Accuracy, F1, MCC, AUC plus a shared train/eval epoch."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task: str = "trend",
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if task == "stock_id":
        y_pred = y_prob.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "auc": float("nan"),
        }

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_prob),
    }


def unpack_batch(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    buckets: Dict[str, list] = {
        "y": [],
        "prob": [],
        "y_id": [],
        "fwd_ret": [],
        "mom": [],
        "date": [],
        "sector": [],
    }
    for batch in loader:
        b = unpack_batch(batch, device)
        logits = model(b["x"])
        if task == "stock_id":
            buckets["y"].append(b["y_id"].cpu().numpy())
            buckets["prob"].append(torch.softmax(logits, dim=-1).cpu().numpy())
        else:
            buckets["y"].append(b["y_trend"].cpu().numpy())
            buckets["prob"].append(torch.sigmoid(logits).cpu().numpy())
        buckets["y_id"].append(b["y_id"].cpu().numpy())
        buckets["fwd_ret"].append(b["fwd_ret"].cpu().numpy())
        buckets["mom"].append(b["mom"].cpu().numpy())
        buckets["date"].append(b["date"].cpu().numpy())
        buckets["sector"].append(b["sector"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in buckets.items()}


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Dict[str, float]:
    pred = collect_predictions(model, loader, device, task)
    return compute_metrics(pred["y"], pred["prob"], task=task)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 1.0,
    mask_ratio: float = 0.15,
) -> Tuple[float, Dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    ys, probs = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            b = unpack_batch(batch, device)
            x = b["x"]
            if task == "mask_recon":
                timestep_mask = torch.rand(x.size(0), x.size(1), device=device) < mask_ratio
                if train:
                    timestep_mask[:, -1] = False
                recon = model(x, timestep_mask=timestep_mask)
                target = x
                denom = timestep_mask.unsqueeze(-1).float()
                se = (recon - target) ** 2
                loss = (se * denom).sum() / denom.sum().clamp_min(1.0)
                ys.append(np.array([0.0]))
                probs.append(np.array([0.0]))
            else:
                logits = model(x)
                if task == "stock_id":
                    loss = criterion(logits, b["y_id"])
                    ys.append(b["y_id"].detach().cpu().numpy())
                    probs.append(torch.softmax(logits.detach(), dim=-1).cpu().numpy())
                else:
                    loss = criterion(logits, b["y_trend"])
                    ys.append(b["y_trend"].detach().cpu().numpy())
                    probs.append(torch.sigmoid(logits.detach()).cpu().numpy())

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

    if task == "mask_recon":
        metrics = {"accuracy": float("nan"), "f1": float("nan"), "mcc": float("nan"), "auc": float("nan")}
    else:
        metrics = compute_metrics(np.concatenate(ys), np.concatenate(probs), task=task)
    return total_loss / max(n_batches, 1), metrics


def format_metrics(metrics: Dict[str, float], digits: int = 4) -> str:
    parts = []
    for key in ("accuracy", "f1", "mcc", "auc"):
        val = metrics.get(key, float("nan"))
        if key == "accuracy" and val == val:
            parts.append(f"acc={val * 100:.2f}%")
        else:
            parts.append(f"{key}={val:.{digits}f}")
    return " ".join(parts)
