"""Transformer Encoder with stock-ID, reconstruction, and trend heads."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            h, h, h, need_weights=need_weights, average_attn_weights=True
        )
        x = x + self.dropout_attn(attn_out)
        ff = self.lin2(self.act(self.lin1(self.norm2(x))))
        x = x + self.dropout_ff(ff)
        return x, attn_w if need_weights else None


class StockTransformer(nn.Module):
    """
    Shared encoder + task heads.

    ``pretrain_task``: ``stock_id`` | ``mask_recon`` | ``none`` (trend head, no extra pretext).
    ``transfer_mode``: ``frozen`` | ``linear_probe`` | ``full``.
    Last-token pooling for classification heads.
    """

    def __init__(
        self,
        n_features: int,
        num_stocks: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        window: int = 30,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.num_stocks = num_stocks
        self.n_features = n_features
        self.pretrain_task = "stock_id"
        self.transfer_mode = "frozen"

        self.input_proj = nn.Linear(n_features, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max(window * 4, 128), dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.stock_id_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_stocks))
        self.recon_head = nn.Linear(d_model, n_features)
        self.trend_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.trend_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_seq(
        self,
        x: torch.Tensor,
        timestep_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: (B, T, F) -> token states (B, T, d). Optional bool mask True=masked."""
        z = self.input_proj(x)
        if timestep_mask is not None:
            z = torch.where(timestep_mask.unsqueeze(-1), self.mask_token.expand_as(z), z)
        z = self.pos_enc(z)
        attns: List[torch.Tensor] = []
        for block in self.blocks:
            z, w = block(z, need_weights=need_weights)
            if w is not None:
                attns.append(w)
        z = self.norm(z)
        return z, attns

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode_seq(x)
        return z[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        timestep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z, _ = self.encode_seq(x, timestep_mask=timestep_mask)
        last = z[:, -1, :]
        if self.pretrain_task == "stock_id":
            return self.stock_id_head(last)
        if self.pretrain_task == "mask_recon":
            return self.recon_head(z)
        if self.pretrain_task in {"none", "trend"}:
            return self.trend_head(last).squeeze(-1)
        if self.pretrain_task == "trend_mlp":
            return self.trend_mlp(last).squeeze(-1)
        return self.trend_head(last).squeeze(-1)

    def apply_transfer_mode(self, mode: str) -> None:
        """frozen | linear_probe | full — sets which parameters train after pretraining."""
        mode = mode.replace("-", "_")
        self.transfer_mode = mode
        if mode == "linear_probe":
            self.pretrain_task = "none"
            self.set_trainable(encoder=False, head="trend_head")
        elif mode == "frozen":
            self.pretrain_task = "none"
            self.set_trainable(encoder=False, head="trend_head")
        elif mode == "full":
            self.pretrain_task = "trend_mlp"
            self.set_trainable(encoder=True, head="trend_mlp")
        else:
            raise ValueError(f"Unknown transfer_mode: {mode}")

    def freeze_encoder(self, train_prefixes: tuple = ("trend_head",)) -> None:
        self.set_trainable(encoder=False, head=train_prefixes[0])

    def set_trainable(self, encoder: bool, head: str = "trend_head") -> None:
        for name, param in self.named_parameters():
            if name.startswith(head):
                param.requires_grad = True
            elif name.startswith("stock_id_head") or name.startswith("recon_head"):
                param.requires_grad = False
            elif name.startswith("trend_head") or name.startswith("trend_mlp"):
                param.requires_grad = False
            else:
                param.requires_grad = encoder

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def last_token_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Mean last-token attention over batch. Returns (n_layers, T)."""
        _, attns = self.encode_seq(x, need_weights=True)
        rows = [w[:, -1, :].mean(dim=0) for w in attns]
        return torch.stack(rows, dim=0)


def build_model(n_features: int, num_stocks: int, cfg, device=None) -> StockTransformer:
    model = StockTransformer(
        n_features=n_features,
        num_stocks=num_stocks,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        window=cfg.window,
    )
    if device is not None:
        model = model.to(device)
    return model
