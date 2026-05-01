from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sprint_week_utils import TransformerXTPaths, utc_now_iso, write_json


class SimpleMessagePassing(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.neigh_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N] with row-normalized weights
        self_term = self.self_proj(x)
        neigh_term = torch.matmul(adj, self.neigh_proj(x))
        out = torch.relu(self_term + neigh_term)
        return self.norm(out)


class SkaterOnlyContextEncoder(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, out_dim: int = 96, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        first_in = in_dim
        for _ in range(layers):
            self.layers.append(SimpleMessagePassing(first_in, hidden_dim))
            first_in = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = node_mask.unsqueeze(-1).float()
        x = node_feats * mask_expanded
        for layer in self.layers:
            x = layer(x, adj)
            x = x * mask_expanded

        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def knn_adjacency(xy: np.ndarray, k: int = 3) -> np.ndarray:
    n = xy.shape[0]
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nbrs = np.argsort(dist[i])[:k]
        adj[i, nbrs] = 1.0
    adj = np.maximum(adj, adj.T)

    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def _mask_adjacency(adj: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
    # Vacant slots should not send or receive graph messages.
    keep = node_mask.astype(np.float32)
    masked = adj * keep[:, None] * keep[None, :]
    row_sum = masked.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return masked / row_sum


def tracking_row_to_graph(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 12 team-anchored slots (Home/Away 0..5). Slot 5 is extra-skater capacity, not a goalie slot.
    slots = []
    mask = []
    for side_flag, side in enumerate(["Home", "Away"]):
        for i in range(6):
            x = float(row.get(f"{side}_Track_{i}_X", 0.0))
            y = float(row.get(f"{side}_Track_{i}_Y", 0.0))
            vx = float(row.get(f"{side}_Track_{i}_Vel_X", 0.0))
            vy = float(row.get(f"{side}_Track_{i}_Vel_Y", 0.0))
            possessing = float(row.get(f"{side}_Track_{i}_is_possessing_team", 0.0))
            primary_actor = float(
                row.get(
                    f"{side}_Track_{i}_is_primary_actor",
                    row.get(f"{side}_Track_{i}_is_actor", 0.0),
                )
            )
            slot_vacant = float(row.get(f"{side}_Track_{i}_slot_vacant", 0.0))
            dist = float(np.sqrt(x * x + y * y))
            slots.append([x, y, vx, vy, dist, float(side_flag), possessing, primary_actor])
            mask.append(0.0 if slot_vacant >= 0.5 else 1.0)

    feats = np.asarray(slots, dtype=np.float32)
    node_mask = np.asarray(mask, dtype=np.float32)
    xy = feats[:, :2]
    adj = _mask_adjacency(knn_adjacency(xy, k=3), node_mask)
    return feats, adj, node_mask


def _resolve_tracking_tensor_path(base_dir: Path) -> Path:
    path = base_dir / "Data" / "Tensor-Ready Data" / "Final Datasets" / "tracking_tensor_pinned.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing tracking tensor input: {path}")
    return path


def run_inspection(base_dir: Path, sample_rows: int = 128) -> dict:
    tracking_path = _resolve_tracking_tensor_path(base_dir)
    df = pd.read_parquet(tracking_path).head(sample_rows)

    feat_list = []
    adj_list = []
    mask_list = []
    for row in df.itertuples(index=False):
        f, a, m = tracking_row_to_graph(pd.Series(row._asdict()))
        feat_list.append(f)
        adj_list.append(a)
        mask_list.append(m)

    node_feats = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    adj = torch.tensor(np.stack(adj_list), dtype=torch.float32)
    mask = torch.tensor(np.stack(mask_list), dtype=torch.float32)

    model = SkaterOnlyContextEncoder(in_dim=node_feats.shape[-1])
    with torch.no_grad():
        emb = model(node_feats, adj, mask)

    return {
        "generated_at_utc": utc_now_iso(),
        "tracking_input": str(tracking_path),
        "tracking_rows_checked": int(len(df)),
        "node_count": int(node_feats.shape[1]),
        "node_feature_dim": int(node_feats.shape[2]),
        "embedding_dim": int(emb.shape[1]),
        "adjacency_density_mean": float(adj.mean().item()),
        "mean_active_nodes": float(mask.sum(dim=1).mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skater-only GNN context encoder helper")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument("--sample-rows", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = TransformerXTPaths(args.base_dir, run_label=args.run_label)
    paths.ensure_all()

    inspection = run_inspection(args.base_dir, sample_rows=args.sample_rows)
    out_path = paths.inspections_dir / "gnn_tracking_inspection.json"
    write_json(out_path, inspection)

    print(f"Saved: {out_path}")
    print(inspection)


if __name__ == "__main__":
    main()
