"""Run the experiment protocol and write results.md."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analysis import produce_all_figures
from baselines import run_logreg, run_majority, run_momentum
from config import (
    CKPT_DIR,
    FIGURES_DIR,
    OUTPUT_DIR,
    RUNS_DIR,
    SEEDS,
    Config,
    full_factorial_experiments,
    oat_experiments,
)
from dataset import DataBundle, get_dataloaders
from economics import long_short_backtest
from evaluate import compute_metrics, format_metrics
from finetune import run_transfer
from model import StockTransformer, build_model
from pretrain import run_pretrain
from stats import align_by_date_ticker, bootstrap_mcc, mean_std, paired_win_rate
from utils import (
    ensure_dirs,
    get_device,
    load_checkpoint,
    log_run,
    save_json,
    save_preds,
    set_seed,
)


CLASSIC = {
    "majority": run_majority,
    "logreg": run_logreg,
    "momentum": run_momentum,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Identity-pretrain experiment matrix")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 seed × 2 epochs, synthetic, OAT matrix")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--matrix", choices=["oat", "full"], default="oat")
    p.add_argument("--seeds", default="", help="comma-separated, default 0-4")
    p.add_argument("--skip-represent", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def _seeds(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.seeds.strip():
        return tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    if args.smoke:
        return (0,)
    return SEEDS


def _base_cfg(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.smoke:
        cfg.apply_smoke()
    elif args.quick:
        cfg.apply_quick()
    if args.synthetic:
        cfg.synthetic = True
    return cfg


def _spec_cfg(base: Config, spec: Dict[str, Any], seed: int) -> Config:
    cfg = replace(base, seed=seed)
    for key in ("pretext", "transfer_mode", "drop_volume", "per_stock_zscore"):
        if key in spec:
            setattr(cfg, key, spec[key])
    if "transfer" in spec and "transfer_mode" not in spec:
        cfg.transfer_mode = spec["transfer"]
    return cfg


def _get_bundle(cfg: Config, cache: Dict[str, DataBundle], use_cache: bool) -> DataBundle:
    key = cfg.data_hash()
    if key not in cache:
        print(f"Building windows fingerprint={key}  vol={not cfg.drop_volume}  z={cfg.per_stock_zscore}")
        cache[key] = get_dataloaders(cfg, use_cache=use_cache)
    return cache[key]


def _pred_arrays(pred: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keep = ("y", "prob", "y_id", "fwd_ret", "mom", "date", "sector")
    return {k: np.asarray(pred[k]) for k in keep if k in pred}


def run_classic(spec: Dict[str, Any], bundle: DataBundle):
    fn = CLASSIC[spec["name"]]
    pred, val_m, extra = fn(bundle)
    return pred, val_m, extra, None


def run_transformer(
    spec: Dict[str, Any],
    cfg: Config,
    bundle: DataBundle,
    device,
):
    model = build_model(cfg.n_features, bundle.num_stocks, cfg, device)
    ckpt_pre = CKPT_DIR / spec["name"] / f"pretrain_{cfg.seed}.pt"
    ckpt_ft = CKPT_DIR / spec["name"] / f"finetune_{cfg.seed}.pt"
    plot_pre = OUTPUT_DIR / spec["name"] / f"curves_pretrain_{cfg.seed}.png"
    plot_ft = OUTPUT_DIR / spec["name"] / f"curves_finetune_{cfg.seed}.png"
    if cfg.pretext != "none":
        model, _ = run_pretrain(
            model, bundle.loaders, cfg, device, ckpt_path=ckpt_pre, plot_path=plot_pre
        )
    model, pred, val_m, extra = run_transfer(
        model, bundle.loaders, cfg, device, ckpt_path=ckpt_ft, plot_path=plot_ft
    )
    return pred, val_m, extra, model


def run_one(
    spec: Dict[str, Any],
    base: Config,
    seed: int,
    device,
    bundles: Dict[str, DataBundle],
    use_cache: bool,
) -> Dict[str, Any]:
    cfg = _spec_cfg(base, spec, seed)
    set_seed(seed)
    bundle = _get_bundle(cfg, bundles, use_cache)
    t0 = time.perf_counter()
    if spec["kind"] == "classic":
        pred, val_m, extra, model = run_classic(spec, bundle)
    else:
        pred, val_m, extra, model = run_transformer(spec, cfg, bundle, device)
    wall = time.perf_counter() - t0
    test_m = compute_metrics(pred["y"], pred["prob"])
    econ = long_short_backtest(
        pred["date"], pred["y_id"], pred["prob"], pred["fwd_ret"],
        decile=cfg.ls_decile, cost_bps=cfg.cost_bps, min_names=max(4, bundle.num_stocks // 4),
    )
    payload = {
        "exp_name": spec["name"],
        "kind": spec["kind"],
        "seed": seed,
        "config_hash": cfg.config_hash(),
        "data_hash": cfg.data_hash(),
        "wall_time_sec": wall,
        "pretext": cfg.pretext if spec["kind"] != "classic" else spec["name"],
        "transfer_mode": cfg.transfer_mode if spec["kind"] != "classic" else spec["name"],
        "drop_volume": cfg.drop_volume,
        "per_stock_zscore": cfg.per_stock_zscore,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "economics": econ,
        "extra": {k: v for k, v in (extra or {}).items() if k != "history"},
        "n_test": int(len(pred["y"])),
        "num_stocks": bundle.num_stocks,
    }
    log_run(spec["name"], seed, payload)
    save_preds(spec["name"], seed, _pred_arrays(pred))
    print(
        f"[{spec['name']} seed={seed}] {format_metrics(test_m)}  "
        f"sharpe={econ.get('sharpe', float('nan')):.3f}  wall={wall:.1f}s"
    )
    return {"payload": payload, "model": model, "pred": pred, "cfg": cfg, "bundle": bundle}


def _collect_by_exp(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        out.setdefault(r["exp_name"], []).append(r)
    return out


def write_results_md(
    runs: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    cfg: Config,
    seeds: Tuple[int, ...],
    represent: Dict[str, float],
    path: Path,
) -> None:
    by_exp = _collect_by_exp(runs)
    names = [s["name"] for s in specs]
    lines: List[str] = []
    lines.append("# Experiment results")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Seeds: `{list(seeds)}`")
    lines.append(f"- Window T={cfg.window}, features default={cfg.features}")
    lines.append(f"- Split: train≤{cfg.train_end}, val≤{cfg.val_end}, test→{cfg.end_date}")
    lines.append(f"- Bootstrap: {cfg.n_bootstrap} resamples, block={cfg.bootstrap_block} test days")
    lines.append(f"- Synthetic: `{cfg.synthetic}`")
    if cfg.pretrain_epochs <= 2 or cfg.finetune_epochs <= 2:
        lines.append("")
        lines.append("> Smoke/short-epoch run. Classification signs and Sharpe are **not** the protocol result; rerun `python experiments.py` (5 seeds, full epochs) before claiming MCC or trading performance.")
    lines.append("")
    lines.append("## Classification (test, mean ± std over seeds)")
    lines.append("")
    lines.append("| Model | Accuracy | F1 | MCC | AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in names:
        rows = by_exp.get(name, [])
        if not rows:
            continue
        acc = mean_std([r["test_metrics"]["accuracy"] for r in rows])
        f1 = mean_std([r["test_metrics"]["f1"] for r in rows])
        mcc = mean_std([r["test_metrics"]["mcc"] for r in rows])
        auc = mean_std([r["test_metrics"]["auc"] for r in rows])
        fmt = lambda s: f"{s['mean']:.4f}" if s["n"] <= 1 else f"{s['mean']:.4f} ± {s['std']:.4f}"
        lines.append(f"| {name} | {fmt(acc)} | {fmt(f1)} | **{fmt(mcc)}** | {fmt(auc)} |")
    lines.append("")

    ours_rows = by_exp.get("ours", [])
    if ours_rows:
        ours_mcc = mean_std([r["test_metrics"]["mcc"] for r in ours_rows])["mean"]
        lines.append("## Did ours beat the required baselines?")
        lines.append("")
        for base_name in ("majority", "logreg", "momentum", "transformer_scratch"):
            other = by_exp.get(base_name, [])
            if not other:
                lines.append(f"- `{base_name}`: not run")
                continue
            other_mcc = mean_std([r["test_metrics"]["mcc"] for r in other])["mean"]
            flag = "yes" if ours_mcc > other_mcc else "no"
            lines.append(
                f"- vs `{base_name}`: ours MCC {ours_mcc:.4f} vs {other_mcc:.4f} → **{flag}**"
            )
        lines.append("")

    seed0 = seeds[0]
    lines.append(f"## Block-bootstrap MCC 95% CI (seed={seed0})")
    lines.append("")
    lines.append("| Model | mean | 2.5% | 97.5% |")
    lines.append("|---|---:|---:|---:|")
    from utils import load_preds

    preds0: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        pred_path = RUNS_DIR / name / f"{seed0}_preds.npz"
        if not pred_path.exists():
            continue
        preds0[name] = load_preds(name, seed0)
        ci = bootstrap_mcc(
            preds0[name]["y"], preds0[name]["prob"], preds0[name]["date"],
            n_bootstrap=cfg.n_bootstrap, block=cfg.bootstrap_block, seed=seed0,
        )
        lines.append(f"| {name} | {ci['mean']:.4f} | {ci['lo']:.4f} | {ci['hi']:.4f} |")
    lines.append("")

    if "ours" in preds0:
        lines.append(f"## Paired bootstrap: fraction of samples where ours MCC > baseline (seed={seed0})")
        lines.append("")
        lines.append("| Baseline | P(ours > baseline) |")
        lines.append("|---|---:|")
        for base_name in ("majority", "logreg", "momentum", "transformer_scratch"):
            if base_name not in preds0:
                lines.append(f"| {base_name} | n/a |")
                continue
            ia, ib = align_by_date_ticker(preds0["ours"], preds0[base_name])
            if len(ia) == 0:
                lines.append(f"| {base_name} | n/a |")
                continue
            wr = paired_win_rate(
                preds0["ours"]["y"][ia],
                preds0["ours"]["date"][ia],
                preds0["ours"]["prob"][ia],
                preds0[base_name]["prob"][ib],
                n_bootstrap=cfg.n_bootstrap,
                block=cfg.bootstrap_block,
                seed=seed0,
            )
            lines.append(f"| {base_name} | {wr['win_rate']:.3f} |")
        lines.append("")

    lines.append("## Ablations (test MCC)")
    lines.append("")
    lines.append("| Cell | pretext | transfer | volume | per-stock z | MCC |")
    lines.append("|---|---|---|---|---|---:|")
    for spec in specs:
        if spec["kind"] != "transformer":
            continue
        rows = by_exp.get(spec["name"], [])
        if not rows:
            continue
        mcc = mean_std([r["test_metrics"]["mcc"] for r in rows])
        fmt = f"{mcc['mean']:.4f}" if mcc["n"] <= 1 else f"{mcc['mean']:.4f} ± {mcc['std']:.4f}"
        vol = "no" if spec.get("drop_volume") else "yes"
        z = "on" if spec.get("per_stock_zscore") else "off"
        lines.append(
            f"| {spec['name']} | {spec.get('pretext')} | {spec.get('transfer_mode')} | {vol} | {z} | {fmt} |"
        )
    lines.append("")

    lines.append("## Representation")
    lines.append("")
    if represent:
        for k, v in represent.items():
            lines.append(f"- `{k}`: {v:.4f}")
        lines.append("- UMAP: `figures/umap_ours.png`, `figures/umap_scratch.png`")
        lines.append("- Attention: `figures/attn_ours.png`, `figures/attn_scratch.png`")
        lines.append("- Economics: `figures/cost_sensitivity.png`, `figures/equity_curves.png`")
    else:
        lines.append("_Skipped._")
    lines.append("")

    lines.append("## Economics (long top-decile P(up) / short bottom-decile, equal weight)")
    lines.append("")
    lines.append("| Model | Sharpe | Max DD | Turnover | Sharpe 5bps | Sharpe 10bps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in names:
        rows = by_exp.get(name, [])
        if not rows:
            continue
        sh = mean_std([r["economics"]["sharpe"] for r in rows])
        dd = mean_std([r["economics"]["max_drawdown"] for r in rows])
        to = mean_std([r["economics"]["turnover"] for r in rows])
        s5 = mean_std([r["economics"].get("sharpe_5bps", float("nan")) for r in rows])
        s10 = mean_std([r["economics"].get("sharpe_10bps", float("nan")) for r in rows])
        fmt = lambda s: f"{s['mean']:.3f}" if s["n"] <= 1 else f"{s['mean']:.3f} ± {s['std']:.3f}"
        lines.append(f"| {name} | {fmt(sh)} | {fmt(dd)} | {fmt(to)} | {fmt(s5)} | {fmt(s10)} |")
    lines.append("")
    lines.append("Per-side costs charge `2 × one-way turnover × bps` on each day's weight change.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cfg = _base_cfg(args)
    seeds = _seeds(args)
    specs = full_factorial_experiments() if args.matrix == "full" else oat_experiments()
    device = get_device(args.device)
    print(f"Device={device}  seeds={seeds}  n_experiments={len(specs)}")

    bundles: Dict[str, DataBundle] = {}
    runs: List[Dict[str, Any]] = []
    keep_models: Dict[str, Tuple[Optional[StockTransformer], DataBundle]] = {}
    keep_preds: Dict[str, Dict[str, np.ndarray]] = {}
    represent_stats: Dict[str, float] = {}

    for seed in seeds:
        for spec in specs:
            result = run_one(spec, cfg, seed, device, bundles, use_cache=not args.no_cache)
            runs.append(result["payload"])
            if seed == seeds[0]:
                keep_preds[spec["name"]] = result["pred"]
            if seed == seeds[0] and spec["name"] in {"ours", "transformer_scratch"}:
                keep_models[spec["name"]] = (result["model"], result["bundle"])

    if not args.skip_represent and "ours" in keep_models:
        try:
            ours_model, bundle = keep_models["ours"]
            scratch_model = keep_models.get("transformer_scratch", (None, bundle))[0]
            if ours_model is None:
                ours_model = _reload("ours", seeds[0], keep_models["ours"][1], cfg, device)
            if scratch_model is None and "transformer_scratch" in keep_models:
                scratch_model = _reload(
                    "transformer_scratch", seeds[0], keep_models["transformer_scratch"][1], cfg, device
                )
            econ_names = ("majority", "logreg", "momentum", "transformer_scratch", "ours")
            represent_stats = produce_all_figures(
                ours_model,
                scratch_model,
                bundle.loaders["test"],
                device,
                bundle.id_to_sector,
                preds_by_name={k: keep_preds[k] for k in econ_names if k in keep_preds},
                out_dir=FIGURES_DIR,
                seed=seeds[0],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Representation analysis failed: {exc}")

    save_json({"runs": runs, "seeds": list(seeds)}, RUNS_DIR / "index.json")
    write_results_md(runs, specs, cfg, seeds, represent_stats, ROOT_RESULTS)


def _reload(name: str, seed: int, bundle: DataBundle, base: Config, device) -> Optional[StockTransformer]:
    path = CKPT_DIR / name / f"finetune_{seed}.pt"
    if not path.exists():
        return None
    spec = next(s for s in oat_experiments() if s["name"] == name)
    cfg = _spec_cfg(base, spec, seed)
    model = build_model(cfg.n_features, bundle.num_stocks, cfg, device=None)
    payload = load_checkpoint(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    model.pretrain_task = "trend_mlp" if cfg.transfer_mode == "full" else "none"
    return model.to(device)


ROOT_RESULTS = Path(__file__).resolve().parent / "results.md"


if __name__ == "__main__":
    main()
