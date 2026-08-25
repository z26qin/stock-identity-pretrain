"""Train the same trend model from scratch (no identity pre-training)."""

from __future__ import annotations

from pathlib import Path

from config import CKPT_DIR, OUTPUT_DIR, Config
from finetune import run_finetune
from model import build_model


def run_baseline(model, loaders, cfg: Config, device, ckpt_path: Path | None = None):
    return run_finetune(
        model,
        loaders,
        cfg,
        device,
        ckpt_path=ckpt_path or (CKPT_DIR / "baseline_best.pt"),
        freeze_encoder=False,
        use_mlp=True,
        epochs=cfg.baseline_epochs,
        lr=cfg.lr_baseline,
        plot_title="Baseline — trend prediction from scratch",
        plot_path=OUTPUT_DIR / "curves_baseline.png",
    )


def main() -> None:
    from dataset import get_dataloaders
    from utils import get_device, set_seed

    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()
    bundle = get_dataloaders(cfg)
    model = build_model(cfg.n_features, bundle.num_stocks, cfg, device)
    run_baseline(model, bundle.loaders, cfg, device)


if __name__ == "__main__":
    main()
