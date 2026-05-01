from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from scripts.gnn.gnn_context_encoder import SkaterOnlyContextEncoder
from sprint_week_utils import TransformerXTPaths, write_json


def _latest_pipeline_run_dir(base_dir: Path) -> Path:
    runs_root = base_dir / "Data" / "Pipeline Runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Pipeline Runs directory not found: {runs_root}")

    runs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under: {runs_root}")

    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _pipeline_phase_dir(base_dir: Path, phase_name: str) -> Path:
    run_dir = _latest_pipeline_run_dir(base_dir)
    phase_dir = run_dir / phase_name
    if not phase_dir.exists():
        raise FileNotFoundError(f"Expected pipeline phase directory not found: {phase_dir}")
    return phase_dir


def _normalize_prob_rows(y_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, eps, 1.0)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return y_prob / row_sums


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    except ValueError:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "token_logloss": float("nan"),
            "aucpr_class_0": float("nan"),
            "aucpr_class_1": float("nan"),
            "aucpr_goal_mean": float("nan"),
            "mean_prob_0": float("nan"),
            "mean_prob_1": float("nan"),
            "mean_prob_2": float("nan"),
            "n_eval_tokens": 0,
        }

    y_prob = _normalize_prob_rows(y_prob)
    metrics: Dict[str, float] = {
        "token_logloss": _safe_logloss(y_true, y_prob),
        "mean_prob_0": float(np.mean(y_prob[:, 0])),
        "mean_prob_1": float(np.mean(y_prob[:, 1])),
        "mean_prob_2": float(np.mean(y_prob[:, 2])),
        "n_eval_tokens": int(len(y_true)),
    }

    for class_idx in (0, 1):
        binary_true = (y_true == class_idx).astype(np.int64)
        if binary_true.min() == binary_true.max():
            ap = float("nan")
        else:
            ap = float(average_precision_score(binary_true, y_prob[:, class_idx]))
        metrics[f"aucpr_class_{class_idx}"] = ap

    rare_auc = [metrics.get("aucpr_class_0", float("nan")), metrics.get("aucpr_class_1", float("nan"))]
    rare_auc = [x for x in rare_auc if np.isfinite(x)]
    metrics["aucpr_goal_mean"] = float(np.mean(rare_auc)) if rare_auc else float("nan")
    return metrics


def _build_balanced_alpha(counts: np.ndarray, balance_power: float, max_ratio: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    inv = np.power(counts.sum() / counts, float(balance_power))
    inv = inv / np.min(inv)
    inv = np.clip(inv, 1.0, float(max_ratio))
    return inv.astype(np.float32)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


@dataclass
class GNNOnlyConfig:
    seed: int = 42
    n_folds: int = 5
    num_epochs: int = 30
    batch_size: int = 512
    eval_batch_size: int = 1024
    lr: float = 8e-5
    weight_decay: float = 3e-4
    dropout: float = 0.15
    gnn_hidden_dim: int = 64
    gnn_embedding_dim: int = 96
    gnn_layers: int = 2
    gnn_node_feature_dim: int = 8
    gnn_graph_variant: str = "actor_rel"
    gnn_feats_path: Optional[Path] = None
    gnn_adj_path: Optional[Path] = None
    gnn_mask_path: Optional[Path] = None
    loss_mode: str = "ce_balanced"
    label_smoothing: float = 0.0
    focal_gamma: float = 1.5
    loss_balance_power: float = 0.35
    loss_max_ratio: float = 3.0
    early_stopping_patience: int = 8
    lr_scheduler: str = "cosine"
    resume_from_checkpoint: bool = False
    use_tensorboard: bool = True
    num_workers: int = 0
    include_eos: bool = False
    export_encoder_weights: bool = True
    sample_rows: int = 0

    base_dir: Path = Path(__file__).resolve().parents[1]

    input_dir_override: Optional[Path] = None
    results_dir_override: Optional[Path] = None
    models_dir_override: Optional[Path] = None
    tensorboard_dir_override: Optional[Path] = None

    @property
    def input_dir(self) -> Path:
        if self.input_dir_override is not None:
            return Path(self.input_dir_override)
        return _pipeline_phase_dir(self.base_dir, "phase3")

    @property
    def results_dir(self) -> Path:
        if self.results_dir_override is not None:
            return Path(self.results_dir_override)
        return self.base_dir / "Results" / "Transformer_xT" / "gnn_only"

    @property
    def models_dir(self) -> Path:
        if self.models_dir_override is not None:
            return Path(self.models_dir_override)
        return self.base_dir / "Models" / "Transformer_xT" / "gnn_only" / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        if self.tensorboard_dir_override is not None:
            return Path(self.tensorboard_dir_override)
        return self.base_dir / "TensorBoard" / "Transformer_xT_Training"


class GNNOnlyDataset(Dataset):
    def __init__(
        self,
        row_ids: np.ndarray,
        targets: np.ndarray,
        graph_row_indexer: np.ndarray,
        feats: np.ndarray,
        adj: np.ndarray,
        mask: np.ndarray,
    ):
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.graph_rows = np.asarray(graph_row_indexer[self.row_ids], dtype=np.int64)

        self.feats = feats
        self.adj = adj
        self.mask = mask

    def __len__(self) -> int:
        return int(len(self.row_ids))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        graph_idx = int(self.graph_rows[idx])
        feats = torch.from_numpy(np.asarray(self.feats[graph_idx], dtype=np.float32))
        adj = torch.from_numpy(np.asarray(self.adj[graph_idx], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(self.mask[graph_idx], dtype=np.bool_))
        target = torch.tensor(int(self.targets[self.row_ids[idx]]), dtype=torch.long)
        row_id = torch.tensor(int(self.row_ids[idx]), dtype=torch.long)

        return {
            "feats": feats,
            "adj": adj,
            "mask": mask,
            "target": target,
            "row_id": row_id,
        }


class StandaloneGNNModel(nn.Module):
    def __init__(self, cfg: GNNOnlyConfig):
        super().__init__()
        self.gnn_encoder = SkaterOnlyContextEncoder(
            in_dim=cfg.gnn_node_feature_dim,
            hidden_dim=cfg.gnn_hidden_dim,
            out_dim=cfg.gnn_embedding_dim,
            layers=cfg.gnn_layers,
        )
        hidden = max(32, cfg.gnn_embedding_dim // 2)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gnn_embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, feats: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.gnn_encoder(feats, adj, mask)
        return self.classifier(emb)


def _resolve_graph_paths(cfg: GNNOnlyConfig) -> Tuple[Path, Path, Path]:
    if cfg.gnn_feats_path is not None and cfg.gnn_adj_path is not None and cfg.gnn_mask_path is not None:
        return Path(cfg.gnn_feats_path), Path(cfg.gnn_adj_path), Path(cfg.gnn_mask_path)

    variant = str(cfg.gnn_graph_variant).strip().lower()
    variant_map = {
        "base": ("base_feats.npy", "base_adj.npy", "base_mask.npy"),
        "actor_rel": ("actor_rel_feats.npy", "actor_rel_adj.npy", "actor_rel_mask.npy"),
        "actor_rel_ctx": ("actor_rel_ctx_feats.npy", "actor_rel_ctx_adj.npy", "actor_rel_ctx_mask.npy"),
        "actor_emph": ("actor_emph_feats.npy", "actor_emph_adj.npy", "actor_emph_mask.npy"),
    }
    if variant not in variant_map:
        raise ValueError(f"Unknown gnn_graph_variant: {variant}")

    f_name, a_name, m_name = variant_map[variant]
    return cfg.input_dir / f_name, cfg.input_dir / a_name, cfg.input_dir / m_name


def _normalize_graph_keys(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = ["game_id", "sl_event_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required key columns: {missing}")

    keys = df[required].copy()
    keys["game_id"] = keys["game_id"].astype(str)
    keys["sl_event_id"] = pd.to_numeric(keys["sl_event_id"], errors="coerce").astype("Int64")

    bad = keys["sl_event_id"].isna()
    if bad.any():
        examples = keys.loc[bad, required].head(5).to_dict("records")
        raise RuntimeError(f"{label} has invalid sl_event_id values. examples={examples}")

    dup = keys.duplicated(required, keep=False)
    if dup.any():
        examples = keys.loc[dup, required].head(5).to_dict("records")
        raise RuntimeError(f"{label} has duplicate keys. examples={examples}")

    return keys


def _load_mmap_graph_data(df: pd.DataFrame, cfg: GNNOnlyConfig) -> Dict[str, Any]:
    feats_path, adj_path, mask_path = _resolve_graph_paths(cfg)
    for p in [feats_path, adj_path, mask_path]:
        if not p.exists():
            raise RuntimeError(f"Missing GNN graph artifact: {p}")

    feats = np.load(feats_path, mmap_mode="r")
    adj = np.load(adj_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")

    if feats.ndim != 3 or feats.shape[1] != 12 or feats.shape[2] <= 0:
        raise RuntimeError(f"Unexpected feats shape: {feats.shape}. Expected [N, 12, F] where F > 0.")
    if adj.ndim != 3 or adj.shape[1:] != (12, 12):
        raise RuntimeError(f"Unexpected adj shape: {adj.shape}. Expected [N, 12, 12].")
    if mask.ndim != 2 or mask.shape[1] != 12:
        raise RuntimeError(f"Unexpected mask shape: {mask.shape}. Expected [N, 12].")
    if not (feats.shape[0] == adj.shape[0] == mask.shape[0]):
        raise RuntimeError(f"Graph row mismatch feats={feats.shape} adj={adj.shape} mask={mask.shape}")

    tensor_path = cfg.input_dir / "tensor_ready_dataset.parquet"
    tensor_df = pd.read_parquet(tensor_path, columns=["game_id", "sl_event_id"])
    tensor_keys = _normalize_graph_keys(tensor_df, label="tensor_ready_dataset")
    if len(tensor_keys) != int(feats.shape[0]):
        raise RuntimeError(
            "Graph row count must match tensor_ready_dataset rows. "
            f"graph_rows={int(feats.shape[0]):,} tensor_ready_rows={len(tensor_keys):,}"
        )

    df_keys = _normalize_graph_keys(df, label="gnn_only_training_df")
    graph_index = pd.MultiIndex.from_frame(tensor_keys[["game_id", "sl_event_id"]])
    df_index = pd.MultiIndex.from_frame(df_keys[["game_id", "sl_event_id"]])
    row_indexer = graph_index.get_indexer(df_index).astype(np.int64, copy=False)

    if (row_indexer < 0).any():
        missing_mask = row_indexer < 0
        missing_count = int(missing_mask.sum())
        examples = df_keys.loc[missing_mask, ["game_id", "sl_event_id"]].head(5).to_dict("records")
        raise RuntimeError(
            "Missing graph rows for training dataframe keys. "
            f"missing_rows={missing_count:,} examples={examples}"
        )

    print(
        "Loaded mmap graph tensors: "
        f"feats={feats.shape} adj={adj.shape} mask={mask.shape} "
        f"| variant={cfg.gnn_graph_variant}"
    )

    cfg.gnn_node_feature_dim = int(feats.shape[2])

    return {
        "feats": feats,
        "adj": adj,
        "mask": mask,
        "row_indexer": row_indexer,
        "node_feature_dim": int(feats.shape[2]),
        "paths": {
            "feats": str(feats_path),
            "adj": str(adj_path),
            "mask": str(mask_path),
        },
    }


def _build_game_level_folds(df: pd.DataFrame, cfg: GNNOnlyConfig) -> List[Dict[str, Any]]:
    unique_games = df["game_id"].dropna().astype(str).unique()
    if len(unique_games) < cfg.n_folds:
        raise RuntimeError(
            f"Need at least n_folds={cfg.n_folds} unique games, got {len(unique_games)}"
        )

    if "event_type" in df.columns:
        event_goal_mask = df["event_type"].astype(str).str.strip().str.lower().eq("goal")
        game_goal_counts = df.loc[event_goal_mask].groupby("game_id").size()
        goal_source_label = "event_type == goal"
    elif "is_eos" in df.columns:
        eos_flag = pd.to_numeric(df["is_eos"], errors="coerce").fillna(0.0) > 0.5
        goal_token_mask = eos_flag & df["target"].isin([0, 1])
        game_goal_counts = df.loc[goal_token_mask].groupby("game_id").size()
        goal_source_label = "is_eos == 1 and target in {0,1}"
    else:
        goal_token_mask = df["target"].isin([0, 1])
        game_goal_counts = df.loc[goal_token_mask].groupby("game_id").size()
        goal_source_label = "target in {0,1}"

    goal_strata: List[int] = []
    for game_id in unique_games:
        goals = int(game_goal_counts.get(game_id, 0))
        if goals <= 3:
            stratum = 0
        elif goals <= 6:
            stratum = 1
        else:
            stratum = 2
        goal_strata.append(stratum)

    goal_strata_arr = np.asarray(goal_strata, dtype=np.int64)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    print(
        f"Creating {cfg.n_folds}-fold game-level splits for GNN-only "
        f"(goal strata source: {goal_source_label})"
    )

    fold_splits: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(unique_games, goal_strata_arr)):
        train_games = unique_games[tr_idx]
        val_games = unique_games[va_idx]

        train_df = df[df["game_id"].isin(train_games)]
        val_df = df[df["game_id"].isin(val_games)]

        train_counts = np.bincount(train_df["target"].to_numpy(dtype=np.int64), minlength=3)
        val_counts = np.bincount(val_df["target"].to_numpy(dtype=np.int64), minlength=3)

        print(
            f"Fold {fold_idx + 1}: train_games={len(train_games):,} val_games={len(val_games):,} "
            f"| train_counts={train_counts.tolist()} val_counts={val_counts.tolist()}"
        )

        fold_splits.append(
            {
                "fold": fold_idx,
                "train_game_ids": train_games,
                "val_game_ids": val_games,
                "train_class_counts_raw": train_counts.tolist(),
                "val_class_counts_raw": val_counts.tolist(),
            }
        )

    return fold_splits


def _resolve_criterion(cfg: GNNOnlyConfig, train_counts: np.ndarray, device: torch.device) -> Tuple[nn.Module, np.ndarray, str]:
    alpha_np = _build_balanced_alpha(
        train_counts,
        balance_power=cfg.loss_balance_power,
        max_ratio=cfg.loss_max_ratio,
    )
    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    loss_mode = str(cfg.loss_mode).strip().lower()
    if loss_mode == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
        desc = f"CrossEntropyLoss(unweighted, label_smoothing={cfg.label_smoothing:.4f})"
    elif loss_mode == "ce_balanced":
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=float(cfg.label_smoothing))
        desc = f"CrossEntropyLoss(weighted, alpha={alpha_np.tolist()}, label_smoothing={cfg.label_smoothing:.4f})"
    elif loss_mode == "focal":
        criterion = FocalCrossEntropyLoss(gamma=cfg.focal_gamma, alpha=None)
        desc = f"FocalCrossEntropyLoss(gamma={cfg.focal_gamma:.2f}, alpha=None)"
    elif loss_mode == "focal_balanced":
        criterion = FocalCrossEntropyLoss(gamma=cfg.focal_gamma, alpha=alpha)
        desc = f"FocalCrossEntropyLoss(gamma={cfg.focal_gamma:.2f}, alpha={alpha_np.tolist()})"
    else:
        raise ValueError(f"Unsupported loss_mode: {cfg.loss_mode}")

    return criterion, alpha_np, desc


def _train_one_epoch(
    model: StandaloneGNNModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        feats = batch["feats"].to(device)
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feats, adj, mask)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(1, n_batches)


def _evaluate_model(
    model: StandaloneGNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    n_batches = 0

    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_row_ids: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            row_id = batch["row_id"].to(device)

            logits = model(feats, adj, mask)
            loss = criterion(logits, target)
            running_loss += float(loss.item())
            n_batches += 1

            probs = F.softmax(logits, dim=-1)
            all_targets.append(target.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_row_ids.append(row_id.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3), dtype=np.float32)
    row_ids = np.concatenate(all_row_ids, axis=0) if all_row_ids else np.zeros((0,), dtype=np.int64)

    return running_loss / max(1, n_batches), y_true, y_prob, row_ids


def _is_better_checkpoint(
    curr_aucpr: float,
    curr_logloss: float,
    best_aucpr: float,
    best_logloss: float,
    aucpr_delta: float = 1e-8,
) -> bool:
    if not np.isfinite(curr_aucpr):
        return False
    if not np.isfinite(best_aucpr):
        return True
    if curr_aucpr > (best_aucpr + aucpr_delta):
        return True
    if curr_aucpr < (best_aucpr - aucpr_delta):
        return False

    curr_ll = curr_logloss if np.isfinite(curr_logloss) else np.inf
    best_ll = best_logloss if np.isfinite(best_logloss) else np.inf
    return curr_ll < best_ll


def _run_cv_gnn_only(
    df: pd.DataFrame,
    cfg: GNNOnlyConfig,
    fold_splits: List[Dict[str, Any]],
    graph_data: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    targets = df["target"].to_numpy(dtype=np.int64)
    all_oof: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    exported_encoder_paths: List[str] = []

    row_indexer = graph_data["row_indexer"]
    feats = graph_data["feats"]
    adj = graph_data["adj"]
    mask = graph_data["mask"]
    cfg.gnn_node_feature_dim = int(graph_data.get("node_feature_dim", cfg.gnn_node_feature_dim))

    for fold in fold_splits:
        fold_idx = int(fold["fold"])
        train_games = set(str(x) for x in fold["train_game_ids"])
        val_games = set(str(x) for x in fold["val_game_ids"])

        train_rows = np.where(df["game_id"].astype(str).isin(train_games).to_numpy())[0]
        val_rows = np.where(df["game_id"].astype(str).isin(val_games).to_numpy())[0]

        if len(train_rows) == 0 or len(val_rows) == 0:
            raise RuntimeError(
                f"Fold {fold_idx + 1} has empty train/val rows. train={len(train_rows)}, val={len(val_rows)}"
            )

        train_ds = GNNOnlyDataset(train_rows, targets, row_indexer, feats, adj, mask)
        val_ds = GNNOnlyDataset(val_rows, targets, row_indexer, feats, adj, mask)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        cfg_fold = copy.deepcopy(cfg)
        model = StandaloneGNNModel(cfg_fold).to(device)

        train_counts = np.bincount(targets[train_rows], minlength=3)
        criterion, alpha_np, loss_desc = _resolve_criterion(cfg_fold, train_counts, device)
        print(f"Fold {fold_idx + 1} loss: {loss_desc}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_fold.lr, weight_decay=cfg_fold.weight_decay)
        if cfg_fold.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_fold.num_epochs)
        elif cfg_fold.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, cfg_fold.num_epochs // 3),
                gamma=0.1,
            )
        else:
            scheduler = None

        fold_writer: Optional[SummaryWriter] = None
        if cfg_fold.use_tensorboard:
            fold_log_dir = cfg_fold.tensorboard_dir / f"gnn_only_fold_{fold_idx + 1}"
            fold_log_dir.mkdir(parents=True, exist_ok=True)
            fold_writer = SummaryWriter(log_dir=str(fold_log_dir))

        best_path = cfg_fold.models_dir / f"gnn_only_with_tracking_fold{fold_idx}_best.pth"
        last_path = cfg_fold.models_dir / f"gnn_only_with_tracking_fold{fold_idx}_last.pth"

        start_epoch = 0
        best_epoch = -1
        best_aucpr = -np.inf
        best_logloss = np.inf
        patience = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_encoder_state: Optional[Dict[str, torch.Tensor]] = None

        if cfg_fold.resume_from_checkpoint and last_path.exists():
            checkpoint = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_epoch = int(checkpoint.get("best_epoch", -1))
            best_aucpr = float(checkpoint.get("best_aucpr_goal_mean", -np.inf))
            best_logloss = float(checkpoint.get("best_logloss", np.inf))
            patience = int(checkpoint.get("patience_counter", 0))
            print(f"Resumed fold {fold_idx + 1} from epoch {start_epoch}")

        for epoch in range(start_epoch, cfg_fold.num_epochs):
            t0 = time.perf_counter()
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, y_true, y_prob, _ = _evaluate_model(model, val_loader, criterion, device)
            val_metrics = _compute_metrics(y_true, y_prob)
            goal_aucpr = float(val_metrics.get("aucpr_goal_mean", float("nan")))
            token_logloss = float(val_metrics.get("token_logloss", float("nan")))
            epoch_s = time.perf_counter() - t0

            improved = _is_better_checkpoint(
                curr_aucpr=goal_aucpr,
                curr_logloss=token_logloss,
                best_aucpr=best_aucpr,
                best_logloss=best_logloss,
            )

            if improved:
                best_aucpr = goal_aucpr
                best_logloss = token_logloss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_encoder_state = copy.deepcopy(model.gnn_encoder.state_dict())
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_epoch": best_epoch,
                        "best_aucpr_goal_mean": best_aucpr,
                        "best_logloss": best_logloss,
                        "config": asdict(cfg_fold),
                    },
                    best_path,
                )
            else:
                patience += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_aucpr_goal_mean": best_aucpr,
                    "best_logloss": best_logloss,
                    "patience_counter": patience,
                    "config": asdict(cfg_fold),
                },
                last_path,
            )

            if scheduler is not None:
                scheduler.step()

            if fold_writer is not None:
                fold_writer.add_scalar("train/loss", float(train_loss), epoch)
                fold_writer.add_scalar("val/loss", float(val_loss), epoch)
                fold_writer.add_scalar("val/token_logloss", token_logloss, epoch)
                fold_writer.add_scalar("val/aucpr_goal_mean", goal_aucpr, epoch)
                fold_writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)

            print(
                f"Fold {fold_idx + 1} | Epoch {epoch + 1:02d}/{cfg_fold.num_epochs:02d} ({epoch_s:.1f}s) "
                f"| train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"| token_logloss={token_logloss:.5f} aucpr_goal_mean={goal_aucpr:.5f}"
            )

            if patience >= cfg_fold.early_stopping_patience:
                print(f"Early stopping fold {fold_idx + 1} at epoch {epoch + 1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        if cfg_fold.export_encoder_weights and best_encoder_state is not None:
            enc_path = cfg_fold.models_dir / f"gnn_only_encoder_with_tracking_fold{fold_idx}_best.pth"
            torch.save(best_encoder_state, enc_path)
            exported_encoder_paths.append(str(enc_path))

        if fold_writer is not None:
            fold_writer.close()

        _, y_true, y_prob, row_ids = _evaluate_model(model, val_loader, criterion, device)
        y_prob = _normalize_prob_rows(y_prob)

        fold_pred = df.iloc[row_ids][[c for c in ["game_id", "sl_event_id", "target", "event_type", "period"] if c in df.columns]].copy()
        fold_pred["P_actor_goal"] = y_prob[:, 0]
        fold_pred["P_opp_goal"] = y_prob[:, 1]
        fold_pred["P_no_goal"] = y_prob[:, 2]
        fold_pred["fold"] = fold_idx
        fold_pred["variant_name"] = "with_tracking"
        fold_pred["model_variant"] = "GNN_ONLY"
        all_oof.append(fold_pred)

        fold_metric = _compute_metrics(y_true, y_prob)
        fold_metric["fold"] = fold_idx
        fold_metric["best_epoch"] = best_epoch + 1
        fold_metric["best_aucpr_goal_mean"] = best_aucpr
        fold_metric["best_logloss"] = best_logloss
        fold_metric["loss_mode"] = str(cfg_fold.loss_mode)
        fold_metric["loss_alpha_0"] = float(alpha_np[0])
        fold_metric["loss_alpha_1"] = float(alpha_np[1])
        fold_metric["loss_alpha_2"] = float(alpha_np[2])
        fold_metrics.append(fold_metric)

    preds = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    if len(metrics_df):
        metrics_df = metrics_df.sort_values("fold").reset_index(drop=True)

    return preds, metrics_df, exported_encoder_paths


def _apply_transformer_xt_paths(cfg: GNNOnlyConfig, base_dir: Path, run_label: str, tensorboard_dir_override: Optional[Path]) -> Dict[str, str]:
    paths = TransformerXTPaths(base_dir=base_dir, run_label=run_label)
    paths.ensure_all()

    cfg.input_dir_override = _pipeline_phase_dir(base_dir, "phase3")
    cfg.results_dir_override = paths.run_results_dir
    cfg.models_dir_override = paths.checkpoints_dir
    cfg.tensorboard_dir_override = (
        Path(tensorboard_dir_override)
        if tensorboard_dir_override is not None
        else base_dir / "TensorBoard" / "Transformer_xT_Training"
    )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": str(cfg.results_dir),
        "models_dir": str(cfg.models_dir),
        "tensorboard_dir": str(cfg.tensorboard_dir),
    }


def _write_run_manifest(
    cfg: GNNOnlyConfig,
    run_label: str,
    path_info: Dict[str, str],
    graph_paths: Dict[str, str],
    metrics: pd.DataFrame,
    encoder_paths: List[str],
) -> Path:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_label": run_label,
        "variant": "gnn_only_with_tracking",
        "results_dir": path_info["results_dir"],
        "models_dir": path_info["models_dir"],
        "tensorboard_dir": path_info["tensorboard_dir"],
        "graph_paths": graph_paths,
        "gnn_graph_variant": cfg.gnn_graph_variant,
        "metrics_rows": int(len(metrics)) if isinstance(metrics, pd.DataFrame) else 0,
        "exported_encoder_paths": encoder_paths,
        "config": asdict(cfg),
    }
    logs_dir = cfg.results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "training_manifest_gnn_only_with_tracking.json"
    write_json(out_path, manifest)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone Phase 6 GNN encoder with transformer-parity folds")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--variant", type=str, default="with_tracking", choices=["with_tracking"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-embedding-dim", type=int, default=96)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-graph-variant", type=str, default="actor_rel")
    parser.add_argument("--gnn-feats-path", type=Path, default=None)
    parser.add_argument("--gnn-adj-path", type=Path, default=None)
    parser.add_argument("--gnn-mask-path", type=Path, default=None)
    parser.add_argument("--loss-mode", type=str, default="ce_balanced", choices=["ce", "ce_balanced", "focal", "focal_balanced"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--loss-balance-power", type=float, default=0.35)
    parser.add_argument("--loss-max-ratio", type=float, default=3.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--export-encoder-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = GNNOnlyConfig()
    cfg.base_dir = args.base_dir.resolve()
    cfg.seed = int(args.seed)
    cfg.n_folds = int(args.n_folds)
    cfg.num_epochs = int(args.num_epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.eval_batch_size = int(args.eval_batch_size)
    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.dropout = float(args.dropout)
    cfg.gnn_hidden_dim = int(args.gnn_hidden_dim)
    cfg.gnn_embedding_dim = int(args.gnn_embedding_dim)
    cfg.gnn_layers = int(args.gnn_layers)
    cfg.gnn_graph_variant = str(args.gnn_graph_variant)
    cfg.gnn_feats_path = Path(args.gnn_feats_path) if args.gnn_feats_path is not None else None
    cfg.gnn_adj_path = Path(args.gnn_adj_path) if args.gnn_adj_path is not None else None
    cfg.gnn_mask_path = Path(args.gnn_mask_path) if args.gnn_mask_path is not None else None
    cfg.loss_mode = str(args.loss_mode)
    cfg.label_smoothing = float(args.label_smoothing)
    cfg.focal_gamma = float(args.focal_gamma)
    cfg.loss_balance_power = float(args.loss_balance_power)
    cfg.loss_max_ratio = float(args.loss_max_ratio)
    cfg.early_stopping_patience = int(args.early_stopping_patience)
    cfg.lr_scheduler = str(args.lr_scheduler)
    cfg.resume_from_checkpoint = bool(args.resume)
    cfg.use_tensorboard = not bool(args.no_tensorboard)
    cfg.num_workers = int(args.num_workers)
    cfg.sample_rows = int(args.sample_rows)
    cfg.include_eos = bool(args.include_eos)
    cfg.export_encoder_weights = bool(args.export_encoder_weights)

    path_info = _apply_transformer_xt_paths(
        cfg,
        base_dir=cfg.base_dir,
        run_label=str(args.run_label),
        tensorboard_dir_override=args.tensorboard_dir,
    )

    print(f"GNN-only results dir: {cfg.results_dir}")
    print(f"GNN-only models dir: {cfg.models_dir}")

    dataset_path = cfg.input_dir / "tensor_ready_dataset.parquet"
    print(f"Loading training table: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    required_cols = ["game_id", "sl_event_id", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"tensor_ready_dataset missing required columns: {missing}")

    df["game_id"] = df["game_id"].astype(str)
    df["sl_event_id"] = pd.to_numeric(df["sl_event_id"], errors="coerce").astype("Int64")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(2).astype(np.int64).clip(0, 2)

    if not cfg.include_eos and "is_eos" in df.columns:
        eos_flag = pd.to_numeric(df["is_eos"], errors="coerce").fillna(0.0) > 0.5
        n_drop = int(eos_flag.sum())
        if n_drop > 0:
            print(f"Excluding EOS rows for parity with transformer windows: {n_drop:,}")
            df = df.loc[~eos_flag].copy()

    df = df.dropna(subset=["game_id", "sl_event_id"]).copy()

    if cfg.sample_rows > 0:
        df = df.head(cfg.sample_rows).copy()
        print(f"Sample mode enabled: using first {len(df):,} rows")

    if len(df) == 0:
        raise RuntimeError("No rows available for GNN-only training after filtering.")

    graph_data = _load_mmap_graph_data(df, cfg)
    fold_splits = _build_game_level_folds(df, cfg)

    preds, metrics, encoder_paths = _run_cv_gnn_only(df, cfg, fold_splits, graph_data)

    preds_path = cfg.results_dir / "oof_phase6_gnn_only_with_tracking.parquet"
    metrics_path = cfg.results_dir / "metrics_phase6_gnn_only_with_tracking.parquet"
    if not preds.empty:
        preds.to_parquet(preds_path, index=False)
    if not metrics.empty:
        metrics.to_parquet(metrics_path, index=False)

    manifest_path = _write_run_manifest(
        cfg=cfg,
        run_label=str(args.run_label),
        path_info=path_info,
        graph_paths=graph_data["paths"],
        metrics=metrics,
        encoder_paths=encoder_paths,
    )

    print("Standalone GNN training complete.")
    print(f"Predictions: {preds_path if preds_path.exists() else 'not written'}")
    print(f"Metrics: {metrics_path if metrics_path.exists() else 'not written'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
