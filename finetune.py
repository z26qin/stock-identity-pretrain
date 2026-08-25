"""Stage 2: frozen encoder, linear probe, or full fine-tune on next-day direction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import AdamW

from config import CKPT_DIR, LOGREG_C_GRID, OUTPUT_DIR, Config
from evaluate import collect_predictions, compute_metrics, format_metrics, run_epoch, unpack_batch
from model import StockTransformer, build_model
from utils import load_checkpoint, plot_history, save_checkpoint


def load_pretrained_encoder(
    num_stocks: int,
    cfg: Config,
    device: torch.device,
    ckpt_path: Path | None = None,
) -> StockTransformer:
    ckpt_path = ckpt_path or (CKPT_DIR / "pretrain_best.pt")
    model = build_model(cfg.n_features, num_stocks, cfg, device=None)
    if cfg.pretext != "none":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pretrain checkpoint at {ckpt_path}")
        payload = load_checkpoint(ckpt_path, map_location="cpu")
        model.load_state_dict(payload["model_state"])
    return model.to(device)


@torch.no_grad()
def _embed(model: StockTransformer, loader, device: torch.device):
    model.eval()
    xs, ys = [], []
    extras = {k: [] for k in ("y_id", "fwd_ret", "mom", "date", "sector")}
    for batch in loader:
        b = unpack_batch(batch, device)
        h = model.encode(b["x"])
        xs.append(h.cpu().numpy())
        ys.append(b["y_trend"].cpu().numpy())
        for k in extras:
            extras[k].append(b[k].cpu().numpy())
    out = {"prob": None, "y": np.concatenate(ys), "embed": np.concatenate(xs)}
    for k, v in extras.items():
        out[k] = np.concatenate(v)
    return out


def run_linear_probe(
    model: StockTransformer,
    loaders,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float], float]:
    model.freeze_encoder()
    train = _embed(model, loaders["train"], device)
    val = _embed(model, loaders["val"], device)
    test = _embed(model, loaders["test"], device)
    best_c, best_mcc = LOGREG_C_GRID[0], -2.0
    for c in LOGREG_C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        clf.fit(train["embed"], train["y"].astype(int))
        p = clf.predict_proba(val["embed"])[:, 1]
        mcc = compute_metrics(val["y"], p, task="trend")["mcc"]
        if mcc > best_mcc:
            best_mcc, best_c = mcc, c
    clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")
    clf.fit(train["embed"], train["y"].astype(int))
    val_p = clf.predict_proba(val["embed"])[:, 1]
    test_p = clf.predict_proba(test["embed"])[:, 1]
    test["prob"] = test_p.astype(np.float32)
    return test, compute_metrics(val["y"], val_p), compute_metrics(test["y"], test_p), float(best_c)


def run_finetune(
    model: StockTransformer,
    loaders,
    cfg: Config,
    device: torch.device,
    ckpt_path: Path | None = None,
    freeze_encoder: bool = True,
    use_mlp: bool = False,
    epochs: int | None = None,
    lr: float | None = None,
    plot_title: str = "Stage 2 — trend fine-tuning",
    plot_path: Path | None = None,
) -> Tuple[StockTransformer, Dict[str, List[float]]]:
    model.pretrain_task = "trend_mlp" if use_mlp else "trend"
    head = "trend_mlp" if use_mlp else "trend_head"
    model.set_trainable(encoder=not freeze_encoder, head=head)
    model.to(device)

    n_epochs = cfg.finetune_epochs if epochs is None else epochs
    lr = cfg.lr_finetune if lr is None else lr
    plot_path = plot_path or (OUTPUT_DIR / "curves_finetune.png")
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trend-task trainable params: {sum(p.numel() for p in params):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_mcc", "val_mcc")}
    best_val = -2.0
    best_state = None
    patience = 0
    ckpt_path = ckpt_path or (CKPT_DIR / "finetune_best.pt")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_m = run_epoch(
            model, loaders["train"], device, task="trend", criterion=criterion,
            optimizer=optimizer, max_grad_norm=cfg.max_grad_norm,
        )
        val_loss, val_m = run_epoch(
            model, loaders["val"], device, task="trend", criterion=criterion,
            optimizer=None, max_grad_norm=cfg.max_grad_norm,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["train_mcc"].append(train_m["mcc"])
        history["val_mcc"].append(val_m["mcc"])
        print(
            f"[trend {epoch:03d}/{n_epochs}] loss={train_loss:.4f}/{val_loss:.4f}  "
            f"train {format_metrics(train_m)}  val {format_metrics(val_m)}"
        )
        if val_m["mcc"] > best_val:
            best_val = val_m["mcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            save_checkpoint(
                {"model_state": best_state, "num_stocks": model.num_stocks, "epoch": epoch,
                 "val_mcc": best_val, "freeze_encoder": freeze_encoder},
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= cfg.patience_finetune:
                print(f"Early stopping finetune at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    plot_history(
        history, title=plot_title, save_path=plot_path,
        keys=[("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_mcc", "val_mcc")],
    )
    return model, history


def run_transfer(
    model: StockTransformer,
    loaders,
    cfg: Config,
    device: torch.device,
    ckpt_path: Path | None = None,
    plot_path: Path | None = None,
):
    """Dispatch frozen head / linear probe / full fine-tune. Returns (model_or_None, test_pred, val_metrics, extra)."""
    mode = cfg.transfer_mode
    model.apply_transfer_mode(mode)
    if mode == "linear_probe":
        pred, val_m, test_m, best_c = run_linear_probe(model, loaders, device)
        return None, pred, val_m, {"test_metrics": test_m, "best_C": best_c}

    freeze = mode == "frozen"
    use_mlp = mode == "full"
    if mode == "frozen":
        title = "Stage 2 — frozen encoder"
    elif mode == "full":
        title = "Stage 2 — full fine-tune"
    else:
        raise ValueError(f"Unknown transfer_mode: {mode}")

    model, history = run_finetune(
        model, loaders, cfg, device,
        ckpt_path=ckpt_path, freeze_encoder=freeze, use_mlp=use_mlp,
        plot_title=title, plot_path=plot_path,
    )
    model.pretrain_task = "trend_mlp" if use_mlp else "trend"
    pred = collect_predictions(model, loaders["test"], device, task="trend")
    val_m = collect_predictions(model, loaders["val"], device, task="trend")
    return model, pred, compute_metrics(val_m["y"], val_m["prob"]), {"history": history}


def main() -> None:
    from dataset import get_dataloaders
    from utils import get_device, set_seed

    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    bundle = get_dataloaders(cfg)
    model = load_pretrained_encoder(bundle.num_stocks, cfg, device)
    run_transfer(model, bundle.loaders, cfg, device)


if __name__ == "__main__":
    main()
