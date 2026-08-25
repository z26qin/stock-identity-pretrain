"""Stage 1: stock-ID classification or masked-timestep reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from config import CKPT_DIR, OUTPUT_DIR, Config
from evaluate import format_metrics, run_epoch
from model import StockTransformer, build_model
from utils import plot_history, save_checkpoint


def run_pretrain(
    model: StockTransformer,
    loaders,
    cfg: Config,
    device: torch.device,
    ckpt_path: Path | None = None,
    plot_path: Path | None = None,
) -> Tuple[StockTransformer, Dict[str, List[float]]]:
    pretext = cfg.pretext
    if pretext == "none":
        return model, {}

    model.pretrain_task = pretext
    model.unfreeze_all()
    model.to(device)

    if pretext == "stock_id":
        criterion: nn.Module = nn.CrossEntropyLoss()
        monitor = "accuracy"
    elif pretext == "mask_recon":
        criterion = nn.MSELoss()
        monitor = "loss"
    else:
        raise ValueError(f"Unknown pretext: {pretext}")

    optimizer = AdamW(model.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_mcc": [],
        "val_mcc": [],
    }
    best_score = -1e18
    best_state = None
    patience = 0
    ckpt_path = ckpt_path or (CKPT_DIR / "pretrain_best.pt")

    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_loss, train_m = run_epoch(
            model, loaders["train"], device, task=pretext, criterion=criterion,
            optimizer=optimizer, max_grad_norm=cfg.max_grad_norm, mask_ratio=cfg.mask_ratio,
        )
        val_loss, val_m = run_epoch(
            model, loaders["val"], device, task=pretext, criterion=criterion,
            optimizer=None, max_grad_norm=cfg.max_grad_norm, mask_ratio=cfg.mask_ratio,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m.get("accuracy", float("nan")))
        history["val_acc"].append(val_m.get("accuracy", float("nan")))
        history["train_mcc"].append(train_m.get("mcc", float("nan")))
        history["val_mcc"].append(val_m.get("mcc", float("nan")))
        print(
            f"[pretrain/{pretext} {epoch:03d}/{cfg.pretrain_epochs}] "
            f"loss={train_loss:.4f}/{val_loss:.4f}  "
            f"train {format_metrics(train_m)}  val {format_metrics(val_m)}"
        )
        score = val_m["accuracy"] if monitor == "accuracy" else -val_loss
        improved = score > best_score
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            save_checkpoint(
                {"model_state": best_state, "num_stocks": model.num_stocks, "epoch": epoch,
                 "val_score": best_score, "pretext": pretext},
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= cfg.patience_pretrain:
                print(f"Early stopping pretrain at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    plot_history(
        history,
        title=f"Stage 1 — {pretext}",
        save_path=plot_path or (OUTPUT_DIR / "curves_pretrain.png"),
        keys=[("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_mcc", "val_mcc")],
    )
    return model, history


def main() -> None:
    from dataset import get_dataloaders
    from utils import get_device, set_seed

    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    bundle = get_dataloaders(cfg)
    model = build_model(cfg.n_features, bundle.num_stocks, cfg, device)
    run_pretrain(model, bundle.loaders, cfg, device)


if __name__ == "__main__":
    main()
