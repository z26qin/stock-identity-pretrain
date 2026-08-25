# Experiment results

## Setup

- Seeds: `[0]`
- Window T=30, features default=['Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_log']
- Split: train≤2022-12-31, val≤2023-06-30, test→2024-12-31
- Bootstrap: 64 resamples, block=5 test days
- Synthetic: `True`

> Smoke/short-epoch run. Classification signs and Sharpe are **not** the protocol result; rerun `python experiments.py` (5 seeds, full epochs) before claiming MCC or trading performance.

## Classification (test, mean ± std over seeds)

| Model | Accuracy | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| majority | 0.5218 | 0.0000 | **0.0000** | 0.5000 |
| logreg | 0.5269 | 0.4922 | **0.0501** | 0.5370 |
| momentum | 0.5193 | 0.4727 | **0.0332** | 0.5165 |
| transformer_scratch | 0.5283 | 0.0618 | **0.0494** | 0.5505 |
| ours | 0.4938 | 0.3756 | **-0.0287** | 0.4872 |
| ablate_pretext_mask | 0.5162 | 0.2877 | **0.0080** | 0.5209 |
| ablate_pretext_none | 0.5097 | 0.2398 | **-0.0130** | 0.4964 |
| ablate_transfer_probe | 0.5318 | 0.4687 | **0.0562** | 0.5468 |
| ablate_transfer_full | 0.5300 | 0.1865 | **0.0422** | 0.5408 |
| ablate_no_volume | 0.5383 | 0.5115 | **0.0740** | 0.5503 |
| ablate_perstock_z | 0.5269 | 0.2619 | **0.0332** | 0.5236 |

## Did ours beat the required baselines?

- vs `majority`: ours MCC -0.0287 vs 0.0000 → **no**
- vs `logreg`: ours MCC -0.0287 vs 0.0501 → **no**
- vs `momentum`: ours MCC -0.0287 vs 0.0332 → **no**
- vs `transformer_scratch`: ours MCC -0.0287 vs 0.0494 → **no**

## Block-bootstrap MCC 95% CI (seed=0)

| Model | mean | 2.5% | 97.5% |
|---|---:|---:|---:|
| majority | 0.0000 | 0.0000 | 0.0000 |
| logreg | 0.0474 | 0.0118 | 0.0807 |
| momentum | 0.0322 | -0.0010 | 0.0577 |
| transformer_scratch | 0.0516 | 0.0224 | 0.0800 |
| ours | -0.0306 | -0.0696 | 0.0062 |
| ablate_pretext_mask | 0.0075 | -0.0245 | 0.0354 |
| ablate_pretext_none | -0.0114 | -0.0391 | 0.0171 |
| ablate_transfer_probe | 0.0577 | 0.0315 | 0.0867 |
| ablate_transfer_full | 0.0396 | 0.0016 | 0.0738 |
| ablate_no_volume | 0.0734 | 0.0488 | 0.1047 |
| ablate_perstock_z | 0.0360 | 0.0094 | 0.0716 |

## Paired bootstrap: fraction of samples where ours MCC > baseline (seed=0)

| Baseline | P(ours > baseline) |
|---|---:|
| majority | 0.062 |
| logreg | 0.000 |
| momentum | 0.016 |
| transformer_scratch | 0.016 |

## Ablations (test MCC)

| Cell | pretext | transfer | volume | per-stock z | MCC |
|---|---|---|---|---|---:|
| transformer_scratch | none | full | yes | off | 0.0494 |
| ours | stock_id | frozen | yes | off | -0.0287 |
| ablate_pretext_mask | mask_recon | frozen | yes | off | 0.0080 |
| ablate_pretext_none | none | frozen | yes | off | -0.0130 |
| ablate_transfer_probe | stock_id | linear_probe | yes | off | 0.0562 |
| ablate_transfer_full | stock_id | full | yes | off | 0.0422 |
| ablate_no_volume | stock_id | frozen | no | off | 0.0740 |
| ablate_perstock_z | stock_id | frozen | yes | on | 0.0332 |

## Representation

- `ari_ours`: 0.1703
- `ari_scratch`: 0.0588
- UMAP: `outputs/umap_ours.png`, `outputs/umap_scratch.png`
- Attention: `outputs/attn_ours.png`, `outputs/attn_scratch.png`

## Economics (long top-decile P(up) / short bottom-decile, equal weight)

| Model | Sharpe | Max DD | Turnover | Sharpe 5bps | Sharpe 10bps |
|---|---:|---:|---:|---:|---:|
| majority | -1.218 | -0.916 | 0.003 | -1.219 | -1.220 |
| logreg | 2.163 | -0.426 | 1.738 | 1.531 | 0.901 |
| momentum | -0.540 | -0.763 | 0.348 | -0.694 | -0.848 |
| transformer_scratch | 3.828 | -0.253 | 1.547 | 3.266 | 2.703 |
| ours | -0.520 | -0.739 | 0.721 | -0.784 | -1.048 |
| ablate_pretext_mask | 0.943 | -0.402 | 1.290 | 0.457 | -0.030 |
| ablate_pretext_none | 0.380 | -0.477 | 1.417 | -0.190 | -0.761 |
| ablate_transfer_probe | 2.607 | -0.275 | 1.398 | 2.110 | 1.613 |
| ablate_transfer_full | 3.007 | -0.366 | 1.354 | 2.517 | 2.027 |
| ablate_no_volume | 3.365 | -0.353 | 1.464 | 2.821 | 2.278 |
| ablate_perstock_z | 0.943 | -0.403 | 1.707 | 0.209 | -0.524 |

Per-side costs charge `2 × one-way turnover × bps` on each day's weight change.
