"""Single-run demo: identity pre-train + frozen fine-tune vs transformer from scratch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from config import OUTPUT_DIR, Config
from dataset import get_dataloaders
from evaluate import compute_metrics, format_metrics
from finetune import load_pretrained_encoder, run_transfer
from model import build_model
from pretrain import run_pretrain
from utils import ensure_dirs, get_device, plot_comparison, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock identity pre-training demo")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["all", "pretrain", "finetune", "baseline"],
        default="all",
    )
    return parser.parse_args()


def _eval_row(name: str, pred) -> Dict[str, float]:
    metrics = compute_metrics(pred["y"], pred["prob"])
    row = {"model": name, **metrics}
    print(f"TEST {name}: {format_metrics(metrics)}")
    return row


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.smoke:
        cfg.apply_smoke()
        print("Smoke mode: 8 synthetic names, 2 epochs.")
    elif args.quick:
        cfg.apply_quick()
        print("Quick mode: 16 US names, fewer epochs.")
    if args.synthetic:
        cfg.synthetic = True
        print("Synthetic data enabled.")

    set_seed(cfg.seed)
    ensure_dirs()
    device = get_device(args.device)
    print(f"Device: {device}")
    bundle = get_dataloaders(cfg, use_cache=not args.no_cache)
    print(
        f"{bundle.num_stocks} stocks | "
        f"train={len(bundle.datasets['train'])} "
        f"val={len(bundle.datasets['val'])} "
        f"test={len(bundle.datasets['test'])}"
    )

    rows: List[Dict[str, float]] = []
    ours_pred = None
    if args.stage in {"all", "pretrain", "finetune"}:
        if args.stage == "finetune":
            ours = load_pretrained_encoder(bundle.num_stocks, cfg, device)
        else:
            ours = build_model(cfg.n_features, bundle.num_stocks, cfg, device)
            ours, _ = run_pretrain(ours, bundle.loaders, cfg, device)
        if args.stage in {"all", "finetune"}:
            ours, ours_pred, _, _ = run_transfer(ours, bundle.loaders, cfg, device)

    if args.stage in {"all", "baseline"}:
        from dataclasses import replace

        from baseline import run_baseline
        from evaluate import collect_predictions

        scratch_cfg = replace(cfg, pretext="none", transfer_mode="full")
        scratch = build_model(cfg.n_features, bundle.num_stocks, cfg, device)
        scratch, _ = run_baseline(scratch, bundle.loaders, scratch_cfg, device)
        scratch.pretrain_task = "trend_mlp"
        scratch_pred = collect_predictions(scratch, bundle.loaders["test"], device, task="trend")
        rows.append(_eval_row("Baseline (no pre-train)", scratch_pred))

    if ours_pred is not None:
        rows.append(_eval_row("Ours (pre-train + fine-tune)", ours_pred))

    if rows:
        path = OUTPUT_DIR / "comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["model", "accuracy", "f1", "mcc", "auc"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fields})
        save_json(rows, OUTPUT_DIR / "metrics.json")
        if len(rows) >= 2:
            plot_comparison(rows, OUTPUT_DIR / "comparison.png")
        print(f"Wrote metrics to {OUTPUT_DIR}")
        print("For the full protocol (5 seeds, ablations, bootstrap, economics), run: python experiments.py")


if __name__ == "__main__":
    main()
