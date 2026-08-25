"""Encoder embeddings: UMAP, k-means ARI vs GICS, attention-by-position heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

from evaluate import unpack_batch
from model import StockTransformer
from utils import ensure_dirs


@torch.no_grad()
def extract_embeddings(model: StockTransformer, loader: DataLoader, device) -> Dict[str, np.ndarray]:
    model.eval()
    hs, sectors, ids = [], [], []
    for batch in loader:
        b = unpack_batch(batch, device)
        h = model.encode(b["x"])
        hs.append(h.cpu().numpy())
        sectors.append(b["sector"].cpu().numpy())
        ids.append(b["y_id"].cpu().numpy())
    return {
        "embed": np.concatenate(hs),
        "sector": np.concatenate(sectors),
        "y_id": np.concatenate(ids),
    }


def _umap_2d(x: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(x)
    except Exception as exc:  # noqa: BLE001
        print(f"UMAP unavailable ({exc}); falling back to PCA.")
        return PCA(n_components=2, random_state=seed).fit_transform(x)


def sector_ari(embed: np.ndarray, sector: np.ndarray, seed: int = 0) -> float:
    n_clusters = int(len(np.unique(sector)))
    if n_clusters < 2 or len(embed) < n_clusters:
        return float("nan")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(embed)
    return float(adjusted_rand_score(sector, pred))


def plot_umap(
    xy: np.ndarray,
    labels: np.ndarray,
    names: Dict[int, str],
    title: str,
    save_path: Path,
) -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.7, label=names.get(int(lab), str(lab)))
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=2, frameon=False)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def attention_by_position(
    model: StockTransformer,
    loader: DataLoader,
    device,
    max_batches: int = 16,
) -> np.ndarray:
    model.eval()
    acc = None
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = unpack_batch(batch, device)
        attn = model.last_token_attention(b["x"])  # (L, T)
        acc = attn if acc is None else acc + attn
        n += 1
    if acc is None:
        return np.zeros((len(model.blocks), 1), dtype=np.float32)
    return (acc / n).cpu().numpy()


def plot_attention_heatmap(attn: np.ndarray, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    im = ax.imshow(attn, aspect="auto", cmap="magma")
    ax.set_xlabel("position (0 = oldest, T-1 = last token)")
    ax.set_ylabel("layer")
    ax.set_yticks(range(attn.shape[0]))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_representation(
    ours: Optional[StockTransformer],
    scratch: Optional[StockTransformer],
    loader: DataLoader,
    device,
    id_to_sector: Dict[int, str],
    out_dir: Path,
    seed: int = 0,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, float] = {}
    for name, model in (("ours", ours), ("scratch", scratch)):
        if model is None:
            continue
        bundle = extract_embeddings(model, loader, device)
        xy = _umap_2d(bundle["embed"], seed=seed)
        plot_umap(
            xy, bundle["sector"], id_to_sector,
            title=f"UMAP of encoder embeddings ({name})",
            save_path=out_dir / f"umap_{name}.png",
        )
        report[f"ari_{name}"] = sector_ari(bundle["embed"], bundle["sector"], seed=seed)
        attn = attention_by_position(model, loader, device)
        plot_attention_heatmap(
            attn,
            title=f"Last-token attention by position ({name})",
            save_path=out_dir / f"attn_{name}.png",
        )
        np.save(out_dir / f"attn_{name}.npy", attn)
    return report
