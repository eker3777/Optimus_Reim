from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.gnn.gnn_context_encoder import SkaterOnlyContextEncoder, tracking_row_to_graph
from sprint_week_utils import TransformerXTPaths, utc_now_iso, write_csv, write_json


class TransformerGNNFusionHead(nn.Module):
    def __init__(self, event_dim: int, gnn_dim: int, hidden_dim: int = 128, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(event_dim + gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, event_feat: torch.Tensor, gnn_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([event_feat, gnn_feat], dim=-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 transformer + skater-only GNN integration scaffold")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument("--dry-run-rows", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve_tensor_input_paths(base_dir: Path) -> tuple[Path, Path]:
    tensor_path = base_dir / "Data" / "Tensor-Ready Data" / "Final Datasets" / "tensor_ready_dataset.parquet"
    track_path = base_dir / "Data" / "Tensor-Ready Data" / "Final Datasets" / "tracking_tensor_pinned.parquet"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing tensor-ready dataset: {tensor_path}")
    if not track_path.exists():
        raise FileNotFoundError(f"Missing tracking tensor dataset: {track_path}")
    return tensor_path, track_path


def run_dry_integration(base_dir: Path, dry_run_rows: int) -> dict:
    tensor_path, track_path = _resolve_tensor_input_paths(base_dir)

    read_rows = max(int(dry_run_rows), 1) * 4
    events = pd.read_parquet(tensor_path).head(read_rows)
    tracking = pd.read_parquet(track_path).head(read_rows)

    merged = events.merge(tracking, on=["game_id", "sl_event_id"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between tensor_ready_dataset and tracking_tensor_pinned")
    merged = merged.head(dry_run_rows).copy()

    event_feature_candidates = [
        "x_adj",
        "y_adj",
        "distance_to_net_event",
        "angle_to_net_event",
        "period_time_remaining",
        "score_differential_poss",
        "n_skaters_poss",
        "n_skaters_def",
        "net_empty_poss",
        "net_empty_def",
        "home_team_poss",
    ]
    event_cols = [c for c in event_feature_candidates if c in merged.columns]
    if not event_cols:
        raise KeyError("No expected event feature columns found for dry integration")

    event_feat_np = merged[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    event_feats = torch.tensor(event_feat_np, dtype=torch.float32)

    feat_list: list[np.ndarray] = []
    adj_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    for row in merged.itertuples(index=False):
        feats, adj, mask = tracking_row_to_graph(pd.Series(row._asdict()))
        feat_list.append(feats)
        adj_list.append(adj)
        mask_list.append(mask)

    node_feats = torch.tensor(np.stack(feat_list), dtype=torch.float32)
    adj = torch.tensor(np.stack(adj_list), dtype=torch.float32)
    mask = torch.tensor(np.stack(mask_list), dtype=torch.float32)

    gnn = SkaterOnlyContextEncoder(in_dim=node_feats.shape[-1], hidden_dim=64, out_dim=96)
    gnn_emb = gnn(node_feats, adj, mask)

    fusion = TransformerGNNFusionHead(event_dim=event_feats.shape[1], gnn_dim=96, hidden_dim=128, out_dim=3)
    logits = fusion(event_feats, gnn_emb)

    target = torch.tensor(merged["target"].fillna(2).astype(int).clip(0, 2).to_numpy(), dtype=torch.long)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    loss = nn.CrossEntropyLoss()(logits, target)

    pred = torch.argmax(logits, dim=1)
    acc = float((pred == target).float().mean().item())

    return {
        "generated_at_utc": utc_now_iso(),
        "tensor_input": str(tensor_path),
        "tracking_input": str(track_path),
        "rows_requested": int(dry_run_rows),
        "rows_used": int(len(merged)),
        "event_feature_columns": event_cols,
        "event_feature_dim": int(event_feats.shape[1]),
        "node_count": int(node_feats.shape[1]),
        "node_feature_dim": int(node_feats.shape[2]),
        "mean_active_nodes": float(mask.sum(dim=1).mean().item()),
        "gnn_embedding_dim": int(gnn_emb.shape[1]),
        "logits_dim": int(logits.shape[1]),
        "dry_run_cross_entropy": float(loss.item()),
        "dry_run_accuracy": acc,
        "status": "dry_run_success",
    }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    paths = TransformerXTPaths(base_dir, run_label=args.run_label)
    paths.ensure_all()

    output = args.output or (paths.metrics_dir / "transformer_gnn_metrics.csv")
    summary = run_dry_integration(base_dir, args.dry_run_rows)

    df = pd.DataFrame([summary])
    write_csv(output, df)
    write_json(paths.logs_dir / "transformer_gnn_run_summary.json", summary)

    print(f"Saved: {output}")
    print(summary)


if __name__ == "__main__":
    main()
