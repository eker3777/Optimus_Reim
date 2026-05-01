# Phase 6 Optimus Reim trainer (3-class net impact).
#
# This script is an exported training runtime used by the consolidated notebook.
# It is intentionally self-contained so it can be launched directly from CLI.

import copy
import argparse
import gc
import json
import logging
import random
import sys
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SPEED_FROM_LAST_EVENT_MIN_DT_SECONDS = 0.25
SPEED_FROM_LAST_EVENT_HARD_CAP_FTPS = 120.0

try:
    from IPython.display import display
except Exception:
    def display(obj):
        try:
            if isinstance(obj, pd.DataFrame):
                print(obj.head(10).to_string(index=False))
            else:
                print(obj)
        except Exception:
            print(obj)


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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from phase6_dataloader import (
    Phase6SequenceDataset,
    phase6_collate_fn,
    phase6_seed_dataloader_worker,
)


def _register_checkpoint_compat_aliases() -> None:
    """Register legacy __main__ aliases needed to unpickle old checkpoints."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    aliases = {
        "OptimusReimConfig": globals().get("OptimusReimConfig"),
    }
    for alias_name, alias_obj in aliases.items():
        if alias_obj is not None and not hasattr(main_module, alias_name):
            setattr(main_module, alias_name, alias_obj)


def _torch_load_checkpoint_compat(path: Path, map_location: torch.device | str) -> Any:
    """Load checkpoints saved from script scope where classes were pickled under __main__."""
    _register_checkpoint_compat_aliases()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except AttributeError as ex:
        msg = str(ex)
        if "Can't get attribute" in msg and "__main__" in msg:
            _register_checkpoint_compat_aliases()
            return torch.load(path, map_location=map_location, weights_only=False)
        raise

# GNN context encoder for with_tracking variant
class SimpleMessagePassing(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, hidden_dim)
        self.neigh_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
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


def _knn_adjacency(xy: np.ndarray, k: int = 3) -> np.ndarray:
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
    keep = node_mask.astype(np.float32)
    masked = adj * keep[:, None] * keep[None, :]
    row_sum = masked.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return masked / row_sum


def _tracking_row_to_graph(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract tracking data from a row and compute graph structure for GNN.
    
    Returns:
        feats: (12, 8) array with [x, y, vx, vy, dist, side_flag, possessing, actor]
        adj: (12, 12) adjacency matrix with row-normalized weights
        node_mask: (12,) mask for vacant slots (0=vacant, 1=active)
    """
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
    adj = _mask_adjacency(_knn_adjacency(xy, k=3), node_mask)
    return feats, adj, node_mask


def _normalize_gnn_graph_variant(gnn_graph_variant: str = 'base') -> str:
    variant = str(gnn_graph_variant).strip().lower()
    if not variant:
        variant = 'base'
    valid = {'base', 'actor_rel', 'actor_rel_ctx', 'actor_emph'}
    if variant not in valid:
        raise ValueError(f"gnn_graph_variant must be one of {sorted(valid)}, got {gnn_graph_variant}")
    return variant


def _resolve_coordinate_mode(model_variant: str, gnn_graph_variant: str = 'base') -> str:
    variant = str(model_variant).strip().lower()
    graph_variant = _normalize_gnn_graph_variant(gnn_graph_variant)
    if variant in {'with_tracking', 'b'}:
        return 'adjusted' if graph_variant in {'actor_rel', 'actor_rel_ctx'} else 'absolute'
    return 'adjusted'


def _resolve_coordinate_columns(model_variant: str, gnn_graph_variant: str = 'base') -> Tuple[str, str, List[str]]:
    mode = _resolve_coordinate_mode(model_variant=model_variant, gnn_graph_variant=gnn_graph_variant)
    if mode == 'adjusted':
        return 'x_adj', 'y_adj', ['dest_x_adj', 'dest_y_adj']
    return 'x', 'y', []


def _get_variant_continuous_cols(
    model_variant: str,
    all_cols: List[str],
    gnn_graph_variant: str = 'base',
) -> List[str]:
    """Return continuous column list for the given variant.
    
    Coordinate selection:
    - events_only -> adjusted coordinates
    - with_tracking + actor_rel/actor_rel_ctx -> adjusted coordinates
    - with_tracking + base/actor_emph -> absolute coordinates
    """
    x_col, y_col, dest_cols = _resolve_coordinate_columns(
        model_variant=model_variant,
        gnn_graph_variant=gnn_graph_variant,
    )
    base_candidates = [
        x_col,
        y_col,
        *dest_cols,
        'distance_to_net_event',
        'angle_to_net_event',
        'period_time_remaining',
        'score_differential_actor',
        'n_skaters_actor',
        'n_skaters_opp',
        'time_since_last_event',
        'distance_from_last_event',
        'speed_from_last_event',
        'goalie_angle_change',
    ]
    
    return [c for c in base_candidates if c in all_cols]

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check for Intel GPU (Arc B580)
try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except Exception:
    ipex = None

# Hard fail-fast policy: Phase 6 must run on Intel XPU (B580) and never on CPU.
if not hasattr(torch, 'xpu'):
    raise RuntimeError(
        'Phase 6 requires Intel XPU runtime, but torch.xpu is not available. '
        'Install/activate an Intel XPU-enabled PyTorch environment.'
    )
if not torch.xpu.is_available():
    raise RuntimeError(
        'Phase 6 requires Intel XPU, but no XPU device is available. '
        'Training is blocked to prevent CPU fallback.'
    )

device = torch.device('xpu')
print(f"Using device: {device}")
if device.type == 'xpu':
    print(f"Device name: {torch.xpu.get_device_name(0)}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

@dataclass
class OptimusReimConfig:
    seed: int = 42
    n_folds: int = 5
    max_seq_length: int = 128
    window_stride: Optional[int] = None
    min_window_tokens: int = 10
    batch_size: int = 64
    eval_batch_size: int = 128
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    use_tracking: bool = True
    gnn_embedding_dim: int = 96  # Output dimension of SkaterOnlyContextEncoder
    gnn_graph_variant: str = "base"
    gnn_feats_path: Optional[Path] = None
    gnn_adj_path: Optional[Path] = None
    gnn_mask_path: Optional[Path] = None
    gnn_emb_dropout: float = 0.0
    gnn_emb_noise_std: float = 0.0
    gnn_emb_mask_rate: float = 0.0
    gnn_proj_dim: int = 0
    gnn_bottleneck_dim: int = 0
    gnn_node_feature_dim: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: Optional[bool] = None
    persistent_workers: bool = False
    debug_validate_items: bool = False
    print_epoch_timing_breakdown: bool = True
    clip_grad_norm: float = 1.0
    class_weight_max_ratio: float = 20.0
    optimus_shrinkage_k: int = 200
    model_variants: Tuple[str, ...] = ('events_only', 'with_tracking')
    
    # Training enhancements
    early_stopping_patience: int = 10
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', or 'none'
    resume_from_checkpoint: bool = False
    save_last_checkpoint: bool = True
    use_tensorboard: bool = True

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def input_dir(self) -> Path:
        return _pipeline_phase_dir(self.base_dir, "phase3")

    @property
    def tracking_input_dir(self) -> Path:
        return _pipeline_phase_dir(self.base_dir, "phase2")

    @property
    def models_dir(self) -> Path:
        return self.base_dir / 'Models' / 'Transformer_OptimusReim'

    @property
    def results_dir(self) -> Path:
        return self.base_dir / 'Results' / 'phase6_optimus_reim'

    @property
    def tensorboard_dir(self) -> Path:
        return self.base_dir / 'TensorBoard' / 'phase6_optimus_reim'


config = OptimusReimConfig()


def _read_startup_hint(env_name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(env_name)
    if value is None:
        return default
    value = str(value).strip()
    if value == '':
        return default
    return value
    return default


def _canonical_variant_for_feature_engineering(model_variant_arg: Optional[str]) -> str:
    raw = str(model_variant_arg or 'events_only').strip().lower()
    if raw in {'with_tracking', 'b'}:
        return 'with_tracking'
    if raw in {'both'}:
        return 'with_tracking'
    return 'events_only'


def _variant_requests_tracking(model_variant_arg: Optional[str]) -> bool:
    raw = str(model_variant_arg or 'events_only').strip().lower()
    return raw in {'with_tracking', 'b', 'both'}


# Startup mode is supplied by the lightweight entry script via environment hints.
_cli_model_variant = _read_startup_hint('HALO_PHASE6_MODEL_VARIANT', 'events_only')
_feature_engineering_variant = _canonical_variant_for_feature_engineering(_cli_model_variant)
_tracking_requested_at_startup = _variant_requests_tracking(_cli_model_variant)
_cli_graph_variant = _read_startup_hint('HALO_PHASE6_GNN_GRAPH_VARIANT', config.gnn_graph_variant)
config.gnn_graph_variant = _normalize_gnn_graph_variant(str(_cli_graph_variant or config.gnn_graph_variant))

for p in [config.models_dir, config.results_dir, config.tensorboard_dir]:
    p.mkdir(parents=True, exist_ok=True)


print(f'Using device: {device}')
print(f'Results dir: {config.results_dir}')
print(f'Max seq length: {config.max_seq_length} | Window stride: {config.window_stride or config.max_seq_length // 2}')
print(f'Min tokens per window: {config.min_window_tokens} | Optimus shrinkage K: {config.optimus_shrinkage_k}')
print(
    'Feature-engineering coordinate scope: '
    f'variant={_feature_engineering_variant} '
    f'gnn_graph_variant={config.gnn_graph_variant}'
)
if str(_cli_model_variant).strip().lower() == 'both':
    print('Note: --model-variant=both uses with_tracking coordinate mode for shared physics features.')
print(
    'Startup data scope: '
    f'tracking_merge_enabled={_tracking_requested_at_startup} '
    f'(model_variant={_cli_model_variant})'
)

# ## 1) Load Data + Build Actor-Relative 3-Class Target
# 
# Target mapping used in this notebook:
# - `0`: no team scores (no-goal sequence outcome)
# - `1`: actor's team scores
# - `2`: opponent team scores

dataset_path = config.input_dir / 'tensor_ready_dataset.parquet'
tracking_path_candidates = [
    config.tracking_input_dir / 'tracking_absolute_pinned.parquet',
    config.tracking_input_dir / 'tracking_tensor_pinned.parquet',
    config.input_dir / 'tracking_absolute_pinned.parquet',
    config.input_dir / 'tracking_tensor_pinned.parquet'
]
tracking_path = next((p for p in tracking_path_candidates if p.exists()), tracking_path_candidates[0])
events_path_candidates = [
    config.input_dir / 'events_with_embedding_indices.parquet',
    config.tracking_input_dir / 'events_phase2_enriched.parquet',
    config.base_dir / 'HALO Hackathon Data' / 'events.parquet',
    config.input_dir / 'events.parquet'
]
events_path = next((p for p in events_path_candidates if p.exists()), events_path_candidates[0])
embeddings_path = config.input_dir / 'text_embeddings.npy'

print(f'Loading dataset from: {dataset_path}')
tensor_source_df = pd.read_parquet(dataset_path, engine='pyarrow')
df_events = tensor_source_df.copy()

if _tracking_requested_at_startup:
    if not tracking_path.exists():
        raise FileNotFoundError(
            'with_tracking variant requested, but tracking tensor file was not found. '
            f'Expected one of: {[str(p) for p in tracking_path_candidates]}'
        )

    print(f'Loading tracking tensor from: {tracking_path}')
    df_tracking = pd.read_parquet(tracking_path, engine='pyarrow')

    merge_keys = [k for k in ['game_id', 'sl_event_id'] if k in df_events.columns and k in df_tracking.columns]
    if len(merge_keys) != 2:
        raise RuntimeError('Expected merge keys [game_id, sl_event_id] in both base and tracking datasets.')

    df_events['game_id'] = df_events['game_id'].astype(str)
    df_tracking['game_id'] = df_tracking['game_id'].astype(str)
    df_events['sl_event_id'] = pd.to_numeric(df_events['sl_event_id'], errors='coerce').astype('Int64')
    df_tracking['sl_event_id'] = pd.to_numeric(df_tracking['sl_event_id'], errors='coerce').astype('Int64')

    right_before = len(df_tracking)
    df_tracking = df_tracking.sort_values(merge_keys).drop_duplicates(subset=merge_keys, keep='last')
    print(f'Tracking rows deduped by keys: {right_before:,} -> {len(df_tracking):,}')

    base_cols = set(df_events.columns)
    trk_cols = [c for c in df_tracking.columns if c not in merge_keys and c not in base_cols]
    if len(trk_cols) == 0:
        print('No new tracking columns detected to merge (already present).')
    else:
        df_events = df_events.merge(df_tracking[merge_keys + trk_cols], on=merge_keys, how='left', validate='many_to_one')
        print(f'Merged tracking columns: {len(trk_cols)}')
else:
    print('Skipping tracking tensor load/merge at startup (events_only scope).')

if events_path.exists() and (('x' not in df_events.columns) or ('y' not in df_events.columns)):
    print(f'Backfilling x/y from events parquet: {events_path}')
    df_events_xy = pd.read_parquet(events_path, engine='pyarrow')
    xy_keys = [k for k in ['game_id', 'sl_event_id'] if k in df_events.columns and k in df_events_xy.columns]
    if len(xy_keys) == 2:
        df_events_xy['game_id'] = df_events_xy['game_id'].astype(str)
        df_events_xy['sl_event_id'] = pd.to_numeric(df_events_xy['sl_event_id'], errors='coerce').astype('Int64')

        xy_cols = [c for c in ['x', 'y'] if c in df_events_xy.columns]
        if len(xy_cols) > 0:
            xy_map = df_events_xy[xy_keys + xy_cols].drop_duplicates(subset=xy_keys, keep='last')
            df_events = df_events.merge(xy_map, on=xy_keys, how='left', suffixes=('', '_evtxy'))
            for col in xy_cols:
                evt_col = f'{col}_evtxy'
                if evt_col in df_events.columns:
                    if col in df_events.columns:
                        df_events[col] = pd.to_numeric(df_events[col], errors='coerce').fillna(pd.to_numeric(df_events[evt_col], errors='coerce'))
                    else:
                        df_events[col] = pd.to_numeric(df_events[evt_col], errors='coerce')
                    df_events.drop(columns=[evt_col], inplace=True)
            del xy_map
    del df_events_xy
    gc.collect()

# One-row-per-event contract for model input
key_cols = ['game_id', 'sl_event_id']
dup_before = int(df_events.duplicated(subset=key_cols).sum())
if dup_before > 0:
    print(f'Deduplicating repeated event keys in base table: {dup_before:,}')
    keep_sort = [c for c in ['game_id', 'period', 'game_event_id', 'sequence_event_id', 'sl_event_id'] if c in df_events.columns]
    if len(keep_sort):
        df_events = df_events.sort_values(keep_sort).reset_index(drop=True)
    df_events = df_events.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)

dup_after = int(df_events.duplicated(subset=key_cols).sum())
if dup_after != 0:
    raise RuntimeError(f'Duplicate event keys remain after deduplication: {dup_after}')

print(f'Loaded rows: {len(df_events):,}')

if embeddings_path.exists():
    text_embeddings = np.load(embeddings_path)
else:
    text_embeddings = np.zeros((1, 384), dtype=np.float32)

# Rename any legacy context columns before validating required actor/opp schema
legacy_to_actor_opp = {
    'net_empty_poss': 'net_empty_actor',
    'net_empty_def': 'net_empty_opp',
    'home_team_poss': 'home_team_actor'
}
for old_col, new_col in legacy_to_actor_opp.items():
    if old_col in df_events.columns and new_col not in df_events.columns:
        df_events = df_events.rename(columns={old_col: new_col})

required_actor_opp_cols = [
    'score_differential_actor', 'n_skaters_actor', 'n_skaters_opp',
    'net_empty_actor', 'net_empty_opp', 'home_team_actor'
]
missing_actor_opp = [c for c in required_actor_opp_cols if c not in df_events.columns]
if missing_actor_opp:
    raise RuntimeError(f'Missing required actor/opp context columns: {missing_actor_opp}')

df_events['text_embedding_idx'] = pd.to_numeric(df_events['text_embedding_idx'], errors='coerce').fillna(0).astype(np.int64)
df_events['text_embedding_idx'] = df_events['text_embedding_idx'].clip(lower=0, upper=max(0, text_embeddings.shape[0] - 1))

display(df_events.head())

df_events['target'] = pd.to_numeric(df_events['target'], errors='coerce').fillna(0).astype(np.int64)

target_counts = df_events['target'].value_counts(dropna=False).sort_index()
print('Target class distribution (0=actor goal, 1=opp goal, 2=no goal):')
display(target_counts.rename('count').to_frame())

import numpy as np
import pandas as pd

print("=" * 80)
print("FEATURE ENGINEERING: TEMPORAL & SPATIAL CONTEXT")
print("=" * 80)

print(f"\nEngineering contextual features for {len(df_events):,} events...")

# --- 0) EVENT NORMALIZATION + DYNAMIC VOCAB ---
sort_cols = [
    c for c in ['game_id', 'period', 'game_event_id', 'sequence_event_id', 'period_time', 'sl_event_id']
    if c in df_events.columns
]
if len(sort_cols):
    df_events = df_events.sort_values(sort_cols).reset_index(drop=True)

event_source_col = 'event_type_clean' if 'event_type_clean' in df_events.columns else 'event_type'
if event_source_col in df_events.columns:
    event_type_norm = df_events[event_source_col].astype(str).str.strip().str.lower()
else:
    event_type_norm = pd.Series('', index=df_events.index, dtype='object')

event_type_norm = event_type_norm.replace({'end_of_period': 'whistle', 'nan': ''})
df_events['event_type'] = event_type_norm

dynamic_tokens = sorted(set(t for t in event_type_norm.dropna().tolist() if isinstance(t, str) and t != ''))
event_type_to_id = {tok: int(i + 1) for i, tok in enumerate(dynamic_tokens)}
event_type_pad_token = 0
df_events['event_type_id'] = df_events['event_type'].map(event_type_to_id).fillna(event_type_pad_token).astype(int)

print("\nToken normalization summary:")
print(f"  - source column: {event_source_col}")
print("  - mapped end_of_period -> whistle")
print(f"  - dynamic vocabulary size: {len(dynamic_tokens):,}")
print(f"  - includes 'save' token: {'save' in event_type_to_id}")
print(f"  - event_type_id padding token used for prior_event_type_id: {event_type_pad_token}")

# --- 1) RELATIVE CONTEXT ---
if len(sort_cols):
    df_events = df_events.sort_values(sort_cols).reset_index(drop=True)

period_time_sec = pd.to_numeric(df_events.get('period_time', np.nan), errors='coerce')
if period_time_sec.isna().any():
    period_time_str = df_events.get('period_time', pd.Series(np.nan, index=df_events.index)).astype(str)
    mmss = period_time_str.str.extract(r'^(\d{1,2}):(\d{2})$')
    parsed = pd.to_numeric(mmss[0], errors='coerce') * 60 + pd.to_numeric(mmss[1], errors='coerce')
    period_time_sec = period_time_sec.fillna(parsed)
df_events['period_time_sec_tmp'] = period_time_sec.astype(np.float32)

physics_mode = _resolve_coordinate_mode(
    _feature_engineering_variant,
    gnn_graph_variant=getattr(config, 'gnn_graph_variant', 'base'),
)
physics_x_col, physics_y_col, _ = _resolve_coordinate_columns(
    _feature_engineering_variant,
    gnn_graph_variant=getattr(config, 'gnn_graph_variant', 'base'),
)
x_col = physics_x_col if physics_x_col in df_events.columns else None
y_col = physics_y_col if physics_y_col in df_events.columns else None
if x_col is None or y_col is None:
    raise RuntimeError(
        'Expected spatial columns for configured coordinate mode. '
        f'mode={physics_mode} expected=({physics_x_col}, {physics_y_col}) '
        f'available={[c for c in ["x", "y", "x_adj", "y_adj"] if c in df_events.columns]}'
    )
print(
    'Physics coordinate mode: '
    f'{physics_mode} (x_col={x_col}, y_col={y_col}, '
    f'variant={_feature_engineering_variant}, gnn_graph_variant={getattr(config, "gnn_graph_variant", "base")})'
)

# Respect period boundaries for relative features when period is available.
feature_group_cols = ['game_id']
if 'period' in df_events.columns:
    feature_group_cols.append('period')
feature_group_cols = [c for c in feature_group_cols if c in df_events.columns]
if not feature_group_cols:
    feature_group_cols = ['game_id']

feature_group = df_events.groupby(feature_group_cols, sort=False)
period_start_mask = feature_group.cumcount().eq(0)

df_events['prior_event_type'] = feature_group['event_type'].shift(1)
df_events['prior_event_type_id'] = feature_group['event_type_id'].shift(1)
feature_source_map: Dict[str, str] = {}

if 'time_since_last_event' in df_events.columns:
    feature_source_map['time_since_last_event'] = 'preexisting'
else:
    raise RuntimeError(
        'Missing required upstream feature: time_since_last_event. '
        'Re-run Phase 2/3 to materialize canonical tensor-ready temporal deltas.'
    )

if 'distance_from_last_event' in df_events.columns:
    feature_source_map['distance_from_last_event'] = 'preexisting'
else:
    x_diff_tmp_col = '_x_diff_from_last_event_tmp'
    y_diff_tmp_col = '_y_diff_from_last_event_tmp'
    df_events[x_diff_tmp_col] = feature_group[x_col].diff()
    df_events[y_diff_tmp_col] = feature_group[y_col].diff()
    df_events['distance_from_last_event'] = np.sqrt(
        df_events[x_diff_tmp_col] ** 2 +
        df_events[y_diff_tmp_col] ** 2
    )
    df_events = df_events.drop(columns=[x_diff_tmp_col, y_diff_tmp_col], errors='ignore')
    feature_source_map['distance_from_last_event'] = 'computed_runtime'

df_events['prior_event_type_id'] = (
    pd.to_numeric(df_events['prior_event_type_id'], errors='coerce')
    .fillna(event_type_pad_token)
    .astype(int)
 )
df_events['time_since_last_event'] = pd.to_numeric(df_events['time_since_last_event'], errors='coerce')
df_events['distance_from_last_event'] = pd.to_numeric(df_events['distance_from_last_event'], errors='coerce')

# Guard against malformed ordering artifacts while preserving in-period continuity.
df_events['time_since_last_event'] = df_events['time_since_last_event'].clip(lower=0.0)
df_events['distance_from_last_event'] = df_events['distance_from_last_event'].clip(lower=0.0)

time_median = float(df_events['time_since_last_event'].median()) if df_events['time_since_last_event'].notna().any() else 0.0
dist_median = float(df_events['distance_from_last_event'].median()) if df_events['distance_from_last_event'].notna().any() else 0.0
df_events['time_since_last_event'] = df_events['time_since_last_event'].fillna(time_median)
df_events['distance_from_last_event'] = df_events['distance_from_last_event'].fillna(dist_median)
df_events.loc[period_start_mask, 'time_since_last_event'] = 0.0
df_events.loc[period_start_mask, 'distance_from_last_event'] = 0.0

# --- 2) PHYSICS DERIVATIVES (Velocity & Displacement) ---
if 'speed_from_last_event' in df_events.columns:
    feature_source_map['speed_from_last_event'] = 'preexisting'
elif 'pre_event_speed' in df_events.columns:
    df_events['speed_from_last_event'] = pd.to_numeric(df_events['pre_event_speed'], errors='coerce')
    feature_source_map['speed_from_last_event'] = 'legacy_alias_pre_event_speed'
else:
    speed_dt = df_events['time_since_last_event'].clip(lower=float(SPEED_FROM_LAST_EVENT_MIN_DT_SECONDS))
    df_events['speed_from_last_event'] = df_events['distance_from_last_event'] / speed_dt
    feature_source_map['speed_from_last_event'] = 'computed_runtime'
df_events['speed_from_last_event'] = pd.to_numeric(df_events['speed_from_last_event'], errors='coerce')
if df_events['speed_from_last_event'].notna().any():
    # Cap extreme speeds with a robust quantile and a fixed physics ceiling.
    q99_cap = float(df_events['speed_from_last_event'].quantile(0.99))
    if np.isfinite(q99_cap):
        speed_cap = float(min(q99_cap, SPEED_FROM_LAST_EVENT_HARD_CAP_FTPS))
    else:
        speed_cap = float(SPEED_FROM_LAST_EVENT_HARD_CAP_FTPS)
    df_events['speed_from_last_event'] = df_events['speed_from_last_event'].clip(lower=0.0, upper=speed_cap)
else:
    df_events['speed_from_last_event'] = 0.0
df_events.loc[period_start_mask, 'speed_from_last_event'] = 0.0

if 'goalie_angle_change' in df_events.columns:
    feature_source_map['goalie_angle_change'] = 'preexisting'
else:
    raise RuntimeError(
        'Missing required upstream feature: goalie_angle_change. '
        'Re-run Phase 2/3 to materialize shot/deflection-only goalie angle deltas.'
    )
df_events['goalie_angle_change'] = pd.to_numeric(df_events['goalie_angle_change'], errors='coerce')
if df_events['goalie_angle_change'].notna().any():
    goalie_angle_median = float(df_events['goalie_angle_change'].median())
    df_events['goalie_angle_change'] = df_events['goalie_angle_change'].fillna(goalie_angle_median)
else:
    df_events['goalie_angle_change'] = 0.0
df_events.loc[period_start_mask, 'goalie_angle_change'] = 0.0

# --- 3) EXPLICIT STATE-CHANGE TOKENS (stand out to model) ---
df_events['is_whistle_token'] = df_events['event_type'].eq('whistle').astype(np.float32)
df_events['is_goal_token'] = df_events['event_type'].eq('goal').astype(np.float32)

df_events['is_eos'] = ((df_events['is_whistle_token'] > 0) | (df_events['is_goal_token'] > 0)).astype(np.float32)

df_events = df_events.drop(columns=['period_time_sec_tmp'], errors='ignore')

if 'distance_to_net' not in df_events.columns:
    raise RuntimeError(
        'Missing required upstream feature: distance_to_net. '
        'Re-run Phase 2/3 to materialize canonical tensor-ready distance_to_net.'
    )

print('\nFeature source summary (preexisting vs computed):')
for feature_name in ['time_since_last_event', 'distance_from_last_event', 'speed_from_last_event', 'goalie_angle_change']:
    print(f"  - {feature_name}: {feature_source_map.get(feature_name, 'unknown')}")
print(f"  - relative grouping columns: {feature_group_cols}")
print(
    f"  - speed_from_last_event caps: min_dt={SPEED_FROM_LAST_EVENT_MIN_DT_SECONDS:.2f}s, "
    f"hard_cap={SPEED_FROM_LAST_EVENT_HARD_CAP_FTPS:.1f} ft/s"
)

print(f"\nâœ“ Added contextual & physics features:")
print(f"  - prior_event_type_id (event type before this event; padded with event_type_id pad token)")
print(f"  - time_since_last_event (seconds since last event)")
print(f"  - distance_from_last_event (distance from last event location)")
print(f"  - distance_to_net (absolute distance to goal)")
print(f"  - angle_to_net_event (absolute angle to goal)")
print(f"  - speed_from_last_event (proxy for puck velocity)")
print(f"  - goalie_angle_change (proxy for lateral goalie movement)")
print(f"  - is_whistle_token (explicit whistle state-change token)")
print(f"  - is_goal_token (explicit goal state-change token)")

print(f"\nFeature statistics (All Events):")
print(f"  distance_to_net:")
print(f"    Mean: {df_events['distance_to_net'].mean():.2f} ft")
print(f"    Min: {df_events['distance_to_net'].min():.2f} ft")
print(f"  angle_to_net_event:")
print(f"    Mean: {df_events['angle_to_net_event'].mean():.2f} rad ({np.degrees(df_events['angle_to_net_event'].mean()):.2f} deg)")
print(f"    Max: {df_events['angle_to_net_event'].max():.2f} rad ({np.degrees(df_events['angle_to_net_event'].max()):.2f} deg)")
print(f"  goalie_angle_change:")
print(f"    Mean: {df_events['goalie_angle_change'].mean():.2f} rad ({np.degrees(df_events['goalie_angle_change'].mean()):.2f} deg)")
print(f"    Max: {df_events['goalie_angle_change'].max():.2f} rad ({np.degrees(df_events['goalie_angle_change'].max()):.2f} deg)")
print(f"  speed_from_last_event:")
print(f"    Mean: {df_events['speed_from_last_event'].mean():.2f} ft/s")
print(f"    Max: {df_events['speed_from_last_event'].max():.2f} ft/s")
print(f"\nDistance from last event:")
print(f"  Mean: {df_events['distance_from_last_event'].mean():.2f} ft")
print(f"  Max: {df_events['distance_from_last_event'].max():.2f} ft")
print(f"Time since last event:")
print(f"  Mean: {df_events['time_since_last_event'].mean():.2f} seconds")
print(f"  Max: {df_events['time_since_last_event'].max():.2f} seconds")
print(f"\nEOS token counts:")
print(f"  whistle: {int(df_events['is_whistle_token'].sum()):,}")
print(f"  goal: {int(df_events['is_goal_token'].sum()):,}")
print(f"  total is_eos=1: {int(df_events['is_eos'].sum()):,}")


# ## 2) Feature Schema + Sequence Dataset
# 
# Phase 6 uses all valid timesteps in each sequence window. No shot masking is applied in loss or metrics.

categorical_cols = [c for c in ['event_type_id', 'outcome_id', 'period_id'] if c in df_events.columns]

all_event_cols = df_events.columns.tolist()
continuous_cols_events_only = _get_variant_continuous_cols('events_only', all_event_cols)
continuous_cols_with_tracking = _get_variant_continuous_cols(
    'with_tracking',
    all_event_cols,
    gnn_graph_variant=getattr(config, 'gnn_graph_variant', 'base'),
)

continuous_cols = []
for col in continuous_cols_events_only + continuous_cols_with_tracking:
    if col not in continuous_cols:
        continuous_cols.append(col)

legacy_dropped_candidates = ['x_adj', 'y_adj', 'dest_x_adj', 'dest_y_adj']
present_legacy_candidates = [c for c in legacy_dropped_candidates if c in df_events.columns]

binary_candidates = [
    'net_empty_actor', 'net_empty_opp', 'home_team_actor', 'is_eos'
 ]
binary_cols = [c for c in binary_candidates if c in df_events.columns]

tracking_cols = [
    c for c in df_events.columns
    if c.startswith(('Home_Track_', 'Away_Track_', 'home_track_', 'away_track_'))
]

# Backward/forward compatibility for actor flag naming:
# - legacy: *_is_actor
# - current: *_is_primary_actor
for side in ['Home', 'Away']:
    for slot in range(6):
        legacy_actor_col = f'{side}_Track_{slot}_is_actor'
        primary_actor_col = f'{side}_Track_{slot}_is_primary_actor'

        if legacy_actor_col not in df_events.columns and primary_actor_col in df_events.columns:
            df_events[legacy_actor_col] = pd.to_numeric(df_events[primary_actor_col], errors='coerce').fillna(0.0)
        if primary_actor_col not in df_events.columns and legacy_actor_col in df_events.columns:
            df_events[primary_actor_col] = pd.to_numeric(df_events[legacy_actor_col], errors='coerce').fillna(0.0)

expected_tracking = []
for side in ['Home', 'Away']:
    for slot in range(6):
        expected_tracking.extend([
            f'{side}_Track_{slot}_X',
            f'{side}_Track_{slot}_Y',
            f'{side}_Track_{slot}_Vel_X',
            f'{side}_Track_{slot}_Vel_Y',
            f'{side}_Track_{slot}_is_present',
            f'{side}_Track_{slot}_is_actor',
            f'{side}_Track_{slot}_is_consistent'
        ])
missing_tracking = [c for c in expected_tracking if c not in df_events.columns]
TRACKING_SCHEMA_COMPLETE = len(missing_tracking) == 0
if not TRACKING_SCHEMA_COMPLETE:
    print(
        'Warning: expected pinned tracking columns are missing. '
        f'first10={missing_tracking[:10]} | total_missing={len(missing_tracking)}. '
        'Events-only training can continue, but with_tracking will fail until Phase 2 tracking is refreshed.'
    )

missing_cont = [c for c in ['period_time'] if c not in df_events.columns]
if len(missing_cont):
    print(f'Warning: expected continuous columns missing: {missing_cont}')

required_events_only_cont = [
    x_col,
    y_col,
    'distance_to_net',
    'angle_to_net_event',
    'period_time_remaining',
    'score_differential_actor',
    'n_skaters_actor',
    'n_skaters_opp',
    'time_since_last_event',
    'distance_from_last_event',
    'speed_from_last_event',
    'goalie_angle_change',
]
missing_required_events_only_cont = [c for c in required_events_only_cont if c and c not in df_events.columns]
if len(missing_required_events_only_cont):
    raise RuntimeError(
        'Missing required events_only continuous features after feature resolution: '
        f'{missing_required_events_only_cont}'
    )

events_only_coord_mode = _resolve_coordinate_mode('events_only', gnn_graph_variant='actor_rel')
with_tracking_coord_mode = _resolve_coordinate_mode(
    'with_tracking',
    gnn_graph_variant=getattr(config, 'gnn_graph_variant', 'base'),
)

print(f'Categorical cols: {len(categorical_cols)}')
print(f'Continuous cols (events_only): {len(continuous_cols_events_only)}')
print(f'Continuous cols (with_tracking): {len(continuous_cols_with_tracking)}')
print(f'Binary cols: {len(binary_cols)}')
print(f'Tracking cols: {len(tracking_cols)}')
print(f'Expected tracking cols present: {len(expected_tracking)}')
print(f'Coordinate mode (events_only): {events_only_coord_mode}')
print(
    'Coordinate mode (with_tracking): '
    f'{with_tracking_coord_mode} '
    f'| gnn_graph_variant={getattr(config, "gnn_graph_variant", "base")}'
)
if len(present_legacy_candidates):
    print(f'Adjusted coordinate columns available: {present_legacy_candidates}')

print('\nCategorical columns selected:')
for c in categorical_cols:
    print(f'  - {c}')

print(f'\nContinuous columns selected (events_only | mode={events_only_coord_mode}):')
for c in continuous_cols_events_only:
    print(f'  - {c}')

print(
    '\nContinuous columns selected '
    f'(with_tracking | mode={with_tracking_coord_mode} | '
    f'gnn_graph_variant={getattr(config, "gnn_graph_variant", "base")}):'
)
for c in continuous_cols_with_tracking:
    print(f'  - {c}')

print('\nBinary columns selected:')
for c in binary_cols:
    print(f'  - {c}')

print('\nNote: Numeric coercion/imputation is applied in the forensic sanitization cell below.')

# ## 2.1) Data Quality Verification
# 
# Verify all input features are finite and properly bounded before training.

# Pre-clean missingness check (reload raw parquet so current in-memory cleaning does not affect counts)
raw_df = df_events.copy()

print('=' * 80)
print('PRE-CLEAN MISSINGNESS (NaN ONLY)')
print('=' * 80)

def group_missing_report(df, cols, name):
    if len(cols) == 0:
        print(f'{name}: no columns found\n')
        return
    na_per_col = df[cols].isna().sum().sort_values(ascending=False)
    total_na = int(na_per_col.sum())
    rows_with_any_na = int(df[cols].isna().any(axis=1).sum())
    print(f'{name}:')
    print(f'  Total NaN values: {total_na:,}')
    print(f'  Rows with any NaN: {rows_with_any_na:,}')
    if total_na > 0:
        print('  Top columns with NaN:')
        for col, c in na_per_col[na_per_col > 0].head(10).items():
            print(f'    {col:35s}: {int(c):,}')
    else:
        print('  No NaN values detected')
    print()

group_missing_report(raw_df, categorical_cols, 'Categorical')
group_missing_report(raw_df, continuous_cols, 'Continuous')
group_missing_report(raw_df, binary_cols, 'Binary')


def numeric_with_nan(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


print('=' * 80)
print('FORENSIC INPUT SANITIZATION (TRACKING-FIRST)')
print('=' * 80)

home_cols = [col for col in df_events.columns if col.startswith(('Home_Track_', 'home_track_'))]
away_cols = [col for col in df_events.columns if col.startswith(('Away_Track_', 'away_track_'))]
tracking_cols = home_cols + away_cols

print(f'Rows: {len(df_events):,}')
print(f'Tracking columns: {len(tracking_cols)} (home={len(home_cols)}, away={len(away_cols)})')

print('\n' + '=' * 80)
print('TRACKING DATA FORENSICS (BEFORE IMPUTATION)')
print('=' * 80)

if len(tracking_cols) > 0:
    # Process columns in chunks to avoid memory consolidation spike during apply()
    chunk_size = 10  # Process 10 columns at a time
    chunks = [tracking_cols[i:i+chunk_size] for i in range(0, len(tracking_cols), chunk_size)]
    
    tracking_missing_by_col = {}
    tracking_problem_mask = pd.Series(False, index=df_events.index)
    
    for chunk in chunks:
        chunk_numeric = df_events[chunk].apply(numeric_with_nan)
        tracking_missing_by_col.update(chunk_numeric.isna().sum().to_dict())
        tracking_problem_mask = tracking_problem_mask | chunk_numeric.isna().any(axis=1)
    
    tracking_missing_by_col = pd.Series(tracking_missing_by_col).sort_values(ascending=False)
    total_tracking_missing = int(tracking_missing_by_col.sum())
    n_problem_rows = int(tracking_problem_mask.sum())

    print(f'Total non-finite tracking values: {total_tracking_missing:,}')
    print(f'Rows with any non-finite tracking value: {n_problem_rows:,}')

    top_cols = tracking_missing_by_col[tracking_missing_by_col > 0].head(20)
    if len(top_cols) > 0:
        print('\nTop problematic tracking columns:')
        for col, count in top_cols.items():
            pct = count / len(df_events) * 100
            print(f'  {col:35s}: {int(count):8,} ({pct:6.2f}%)')

print('\n' + '=' * 80)
print('IMPUTING MISSING TRACKING DATA FOR TENSOR COMPLETENESS')
print('=' * 80)

print('\n' + '=' * 80)
print('HOME TRACKING (presence flag semantics preserved)')
print('=' * 80)

if len(home_cols) > 0:
    home_numeric = df_events[home_cols].apply(numeric_with_nan)
    mask_missing_home = home_numeric.isna().any(axis=1)
    num_imputed_home = int(mask_missing_home.sum())
    print(f'Rows with missing home tracking data: {num_imputed_home:,}')

    df_events[home_cols] = home_numeric

    if num_imputed_home > 0:
        home_xy_cols = [c for c in home_cols if c.endswith('_X') or c.endswith('_Y')]
        home_vel_cols = [c for c in home_cols if ('Vel_X' in c) or ('Vel_Y' in c)]
        for col in home_xy_cols + home_vel_cols:
            df_events.loc[mask_missing_home, col] = df_events.loc[mask_missing_home, col].fillna(0.0)

print('\n' + '=' * 80)
print('AWAY TRACKING (presence flag semantics preserved)')
print('=' * 80)

if len(away_cols) > 0:
    away_numeric = df_events[away_cols].apply(numeric_with_nan)
    missing_away = away_numeric.isna().sum()
    total_missing_away = int(missing_away.sum())
    print(f'Total missing values across away columns: {total_missing_away:,}')

    df_events[away_cols] = away_numeric

    if total_missing_away > 0:
        position_distance_cols = [c for c in away_cols if c.endswith('_X') or c.endswith('_Y')]
        velocity_cols = [c for c in away_cols if ('Vel_X' in c) or ('Vel_Y' in c)]

        for col in position_distance_cols:
            df_events[col] = df_events[col].fillna(0.0)
        for col in velocity_cols:
            df_events[col] = df_events[col].fillna(0.0)

        remaining_away_na = int(df_events[away_cols].apply(numeric_with_nan).isna().sum().sum())
        if remaining_away_na > 0:
            df_events[away_cols] = df_events[away_cols].apply(numeric_with_nan).fillna(0.0)

print('\n' + '=' * 80)
print('FILLING MISSING MASK VALUES')
print('=' * 80)

mask_cols = [c for c in df_events.columns if c.endswith('_mask')]
if len(mask_cols) > 0:
    missing_masks = df_events[mask_cols].apply(numeric_with_nan).isna().sum()
    total_missing_masks = int(missing_masks.sum())
    print(f'Total missing mask values across {len(mask_cols)} columns: {total_missing_masks:,}')
    if total_missing_masks > 0:
        for col in mask_cols:
            s = numeric_with_nan(df_events[col])
            df_events[col] = s.fillna(0.0).astype(np.float32)

print('\n' + '=' * 80)
print('SANITIZING NON-TRACKING MODEL INPUT COLUMNS')
print('=' * 80)

for col in categorical_cols:
    df_events[col] = numeric_with_nan(df_events[col]).fillna(0).astype(np.int64)

for col in binary_cols:
    df_events[col] = numeric_with_nan(df_events[col]).fillna(0.0).astype(np.float32)

for col in continuous_cols:
    s = numeric_with_nan(df_events[col])
    med = s.median()
    fill_val = float(med) if pd.notna(med) else 0.0
    df_events[col] = s.fillna(fill_val).astype(np.float32)

if len(tracking_cols) > 0:
    # Process tracking columns in chunks to avoid consolidation memory spike
    chunk_size = 10
    chunks = [tracking_cols[i:i+chunk_size] for i in range(0, len(tracking_cols), chunk_size)]
    for chunk in chunks:
        df_events[chunk] = df_events[chunk].apply(numeric_with_nan).fillna(0.0).astype(np.float32)

# Ensure target is consistent/int
if 'target' not in df_events.columns:
    raise RuntimeError("Expected 'target' column not found in df_events.")
df_events['target'] = pd.to_numeric(df_events['target'], errors='coerce').fillna(2).astype(np.int64)

# Clear dataset cache so new datasets rebuild from cleaned df_events
if 'OptimusReimSequenceDataset' in globals():
    OptimusReimSequenceDataset._shared_cache.clear()
    print('âœ“ Cleared OptimusReimSequenceDataset shared cache')
if 'Phase6SequenceDataset' in globals():
    Phase6SequenceDataset._shared_cache.clear()
    print('âœ“ Cleared Phase6SequenceDataset shared cache')


def count_nonfinite(df, cols):
    if len(cols) == 0:
        return 0, 0
    arr = df[cols].to_numpy()
    bad = ~np.isfinite(arr)
    return int(bad.sum()), int(bad.any(axis=1).sum())

nf_cat, rows_cat = count_nonfinite(df_events, categorical_cols)
nf_cont, rows_cont = count_nonfinite(df_events, continuous_cols)
nf_bin, rows_bin = count_nonfinite(df_events, binary_cols)
nf_track, rows_track = count_nonfinite(df_events, tracking_cols)

total_issues = nf_cat + nf_cont + nf_bin + nf_track
print('\nPost-clean non-finite counts:')
print(f'  categorical: {nf_cat:,} values | {rows_cat:,} rows')
print(f'  continuous : {nf_cont:,} values | {rows_cont:,} rows')
print(f'  binary     : {nf_bin:,} values | {rows_bin:,} rows')
print(f'  tracking   : {nf_track:,} values | {rows_track:,} rows')

if total_issues == 0:
    print('\nâœ“ All model input columns are finite and ready for tensor conversion')
else:
    raise RuntimeError('Non-finite values remain after forensic sanitization.')

# EOS check for filtering pipeline
if 'is_eos' in df_events.columns:
    eos_rows = int((pd.to_numeric(df_events['is_eos'], errors='coerce').fillna(0) > 0).sum())
    print(f'\nEOS rows present in raw events table: {eos_rows:,} (will be excluded in dataset windows)')

# Target distribution summary
target_counts = df_events['target'].value_counts(sort=False).sort_index()
print('\nTarget class distribution (0=actor goal, 1=opp goal, 2=no goal):')
for cls, cnt in target_counts.items():
    pct = float(cnt) / max(1, len(df_events)) * 100.0
    print(f'  Class {int(cls)}: {int(cnt):,} ({pct:.2f}%)')

print('\n' + '=' * 80)
print('ORDER PROVENANCE ASSERTIONS (PHASE 2/3 CONTRACT)')
print('=' * 80)

def _phase6_period_time_seconds(df: pd.DataFrame) -> pd.Series:
    if 'period_time_sec' in df.columns:
        s = pd.to_numeric(df['period_time_sec'], errors='coerce')
    elif 'period_time' in df.columns:
        s = pd.to_numeric(df['period_time'], errors='coerce')
    else:
        s = pd.Series(np.nan, index=df.index, dtype='float64')

    bad = s.isna()
    if bad.any() and 'period_time' in df.columns:
        mmss = df['period_time'].astype(str).str.extract(r'^(\d{1,2}):(\d{2})$')
        parsed = pd.to_numeric(mmss[0], errors='coerce') * 60 + pd.to_numeric(mmss[1], errors='coerce')
        s.loc[bad] = parsed.loc[bad]
    return s.astype(float)

def _order_anomaly_table(df: pd.DataFrame, threshold: float = -1.0) -> pd.DataFrame:
    if 'game_id' not in df.columns:
        return pd.DataFrame()

    work = df[['game_id']].copy()
    if 'sequence_id' in df.columns:
        work['sequence_id'] = df['sequence_id']
    else:
        work['sequence_id'] = -1

    work['period_time_sec__tmp'] = _phase6_period_time_seconds(df)
    work['delta_sec'] = work.groupby(['game_id', 'sequence_id'], sort=False)['period_time_sec__tmp'].diff()

    anomalies = work[work['delta_sec'] < threshold].copy()
    if len(anomalies) == 0:
        return anomalies

    keep_cols = ['game_id', 'sequence_id', 'period_time_sec__tmp', 'delta_sec']
    for c in ['sl_event_id', 'game_event_id', 'period', 'period_time', 'event_type']:
        if c in df.columns:
            anomalies[c] = df.loc[anomalies.index, c].values
            keep_cols.append(c)
    return anomalies[keep_cols]

order_anomalies = _order_anomaly_table(df_events, threshold=-1.0)
n_order_anomalies = int(len(order_anomalies))
print(f'Rows with delta_sec < -1.0: {n_order_anomalies:,}')

if n_order_anomalies > 0:
    print('Top ordering anomalies (first 20):')
    display(order_anomalies.head(20))
    raise RuntimeError(
        'Order provenance assertion failed: detected temporal inversions (delta_sec < -1). '
        'Phase 6 expects native order unless abnormal drift is explicitly repaired upstream.'
    )

df_events.attrs['order_provenance'] = {
    'policy': 'preserve_native_order_unless_delta_lt_minus1',
    'checked_grouping': ['game_id', 'sequence_id'] if 'sequence_id' in df_events.columns else ['game_id'],
    'anomaly_threshold_sec': -1.0,
    'anomaly_rows': n_order_anomalies
}
print('âœ“ Order provenance contract asserted and stored in df_events.attrs["order_provenance"]')

class OptimusReimSequenceDataset(Dataset):
    _shared_cache: Dict = {}
    _cache_schema_version: int = 2

    def __init__(
        self,
        df: pd.DataFrame,
        text_embeddings: np.ndarray,
        cfg: OptimusReimConfig,
        game_ids: Optional[List] = None,
        sequences: Optional[List[Tuple]] = None,
        threat_vectors: Optional[np.ndarray] = None,
        threat_row_indexer: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.text_embeddings = text_embeddings.astype(np.float32, copy=False)
        self.stride = cfg.window_stride if (cfg.window_stride is not None and cfg.window_stride > 0) else max(1, cfg.max_seq_length // 2)
        self.min_window_tokens = max(1, int(cfg.min_window_tokens))

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.tracking_cols = tracking_cols if cfg.use_tracking else []
        self.threat_vectors = threat_vectors
        self.threat_row_indexer = threat_row_indexer

        if cfg.use_tracking:
            if self.threat_vectors is None or self.threat_row_indexer is None:
                raise RuntimeError(
                    'Threat vectors are required for with_tracking. '
                    'Run Phase 3 threat export to generate phase3/threat_vectors.npy.'
                )
            if len(self.threat_row_indexer) != len(df):
                raise RuntimeError(
                    'Threat row indexer size does not match df_events. '
                    f'expected={len(df):,} got={len(self.threat_row_indexer):,}'
                )
            if self.threat_vectors.ndim != 2:
                raise RuntimeError(
                    f'Threat vectors must be 2D [N, F]. got={self.threat_vectors.shape}'
                )
            if int(self.threat_vectors.shape[1]) != 36:
                raise RuntimeError(
                    'Threat vectors must have 36 features. '
                    f'got={int(self.threat_vectors.shape[1])}'
                )

        cache_key = (
            id(df), tuple(self.categorical_cols), tuple(self.continuous_cols),
            tuple(self.binary_cols), tuple(self.tracking_cols), cfg.max_seq_length,
            int(self._cache_schema_version)
        )

        if cache_key not in self._shared_cache:
            game_indices = df.groupby(['game_id'], sort=False).indices
            period_time_series = df['period_time'] if 'period_time' in df.columns else pd.Series(np.nan, index=df.index)
            period_time_sec = pd.to_numeric(
                df['period_time_sec'] if 'period_time_sec' in df.columns else period_time_series,
                errors='coerce'
            )
            bad_period_time = period_time_sec.isna()
            if bad_period_time.any():
                mmss = period_time_series.astype(str).str.extract(r'^(\d{1,2}):(\d{2})$')
                parsed = pd.to_numeric(mmss[0], errors='coerce') * 60 + pd.to_numeric(mmss[1], errors='coerce')
                period_time_sec.loc[bad_period_time] = parsed.loc[bad_period_time]

            event_type_norm = (
                df['event_type'].astype(str).str.strip().str.lower().to_numpy(copy=False)
                if 'event_type' in df.columns
                else np.full(len(df), '', dtype=object)
            )
            meta_cols = [
                c for c in [
                    'game_id', 'sl_event_id', 'game_event_id', 'period', 'period_time',
                    'sequence_id', 'sequence_event_id', 'team_id', 'player_id', 'event_type', 'outcome', 'detail'
                ]
                if c in df.columns
            ]
            meta_arrays = {
                col: df[col].to_numpy(copy=False)
                for col in meta_cols
            }
            payload = {
                'game_indices': game_indices,
                'cat': df[self.categorical_cols].to_numpy(copy=False).astype(np.int64) if self.categorical_cols else np.zeros((len(df), 0), dtype=np.int64),
                'cont': df[self.continuous_cols].to_numpy(copy=False).astype(np.float32) if self.continuous_cols else np.zeros((len(df), 0), dtype=np.float32),
                'bin': df[self.binary_cols].to_numpy(copy=False).astype(np.float32) if self.binary_cols else np.zeros((len(df), 0), dtype=np.float32),
                'track': df[self.tracking_cols].to_numpy(copy=False).astype(np.float32) if self.tracking_cols else np.zeros((len(df), 0), dtype=np.float32),
                'target': df['target'].to_numpy(copy=False).astype(np.int64),
                'text_idx': df['text_embedding_idx'].to_numpy(copy=False).astype(np.int64),
                'meta_cols': tuple(meta_cols),
                'meta_arrays': meta_arrays,
                'is_eos': (df['is_eos'].to_numpy(copy=False).astype(np.int64) if 'is_eos' in df.columns else np.zeros(len(df), dtype=np.int64)),
                'period_time_sec': period_time_sec.to_numpy(copy=False).astype(np.float32),
                'event_type_norm': event_type_norm
            }
            self._shared_cache[cache_key] = payload

        self.backend = self._shared_cache[cache_key]

        if game_ids is not None:
            game_keys = [g for g in list(game_ids) if g in self.backend['game_indices']]
        elif sequences is not None:
            fallback_games = [seq[0] for seq in list(sequences) if len(seq) > 0]
            game_keys = [g for g in fallback_games if g in self.backend['game_indices']]
        else:
            game_keys = list(self.backend['game_indices'].keys())

        self.samples: List[Dict] = []
        self.filtered_all_eos_games = 0
        self.split_on_large_gap_games = 0
        self.faceoff_gap_rows_dropped = 0
        self.max_gap_seconds = 10.0

        def _split_on_large_gaps(row_idx: np.ndarray) -> List[np.ndarray]:
            if len(row_idx) <= 1:
                return [row_idx]

            def _gap_breaks(idx: np.ndarray) -> np.ndarray:
                t = self.backend['period_time_sec'][idx]
                dt = np.diff(t)
                finite = np.isfinite(t)
                valid_dt = finite[:-1] & finite[1:]
                return np.where(valid_dt & (dt > self.max_gap_seconds))[0] + 1

            breaks = _gap_breaks(row_idx)

            if len(breaks) > 0:
                first_break = int(breaks[0])
                if first_break == 2 and len(row_idx) > 2:
                    row_idx = row_idx[2:]
                    self.faceoff_gap_rows_dropped += 2
                    if len(row_idx) <= 1:
                        return [row_idx] if len(row_idx) else []
                    breaks = _gap_breaks(row_idx)

            if len(breaks) == 0:
                return [row_idx]

            chunks = []
            start = 0
            for b in breaks:
                seg = row_idx[start:b]
                if len(seg) > 0:
                    chunks.append(seg)
                start = int(b)
            tail = row_idx[start:]
            if len(tail) > 0:
                chunks.append(tail)
            return chunks

        for game_key in game_keys:
            base_idx = np.asarray(self.backend['game_indices'][game_key], dtype=np.int64)
            eos = self.backend['is_eos'][base_idx]
            clean_idx = base_idx[eos == 0]

            if len(clean_idx) == 0:
                self.filtered_all_eos_games += 1
                continue

            game_segments = _split_on_large_gaps(clean_idx)
            if len(game_segments) > 1:
                self.split_on_large_gap_games += 1

            for seg_row_idx in game_segments:
                if len(seg_row_idx) < self.min_window_tokens:
                    continue

                if len(seg_row_idx) <= self.cfg.max_seq_length:
                    self.samples.append({
                        'game_key': game_key,
                        'row_idx': seg_row_idx,
                        'chunk_start': 0,
                        'chunk_index': 0,
                        'is_first_chunk': 1,
                        'drop_prefix_tokens': 0
                    })
                else:
                    chunk_counter = 0
                    stop = len(seg_row_idx) - self.cfg.max_seq_length + self.stride
                    for start in range(0, max(stop, 1), self.stride):
                        chunk = seg_row_idx[start:start + self.cfg.max_seq_length]
                        if len(chunk) < self.min_window_tokens:
                            continue
                        is_first = 1 if start == 0 else 0
                        drop_prefix = 0 if is_first else min(self.stride, len(chunk))
                        self.samples.append({
                            'game_key': game_key,
                            'row_idx': chunk,
                            'chunk_start': int(start),
                            'chunk_index': int(chunk_counter),
                            'is_first_chunk': int(is_first),
                            'drop_prefix_tokens': int(drop_prefix)
                        })
                        chunk_counter += 1

        if len(self.samples) == 0:
            raise RuntimeError('No non-EOS windows available. Check is_eos filtering and source data quality.')

        self.filtered_all_eos_sequences = self.filtered_all_eos_games
        self.split_on_large_gap_sequences = self.split_on_large_gap_games

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        row_idx = np.asarray(sample['row_idx'], dtype=np.int64)

        eos_slice = self.backend['is_eos'][row_idx]
        if np.any(eos_slice != 0):
            raise RuntimeError('EOS token leakage detected in dataset sample. Expected all row_idx to be non-EOS.')

        seq_len = len(row_idx)
        max_len = self.cfg.max_seq_length

        cat = np.zeros((max_len, self.backend['cat'].shape[1]), dtype=np.int64)
        cont = np.zeros((max_len, self.backend['cont'].shape[1]), dtype=np.float32)
        binv = np.zeros((max_len, self.backend['bin'].shape[1]), dtype=np.float32)
        track = np.zeros((max_len, self.backend['track'].shape[1]), dtype=np.float32)
        text = np.zeros((max_len, self.text_embeddings.shape[1]), dtype=np.float32)
        threat_dim = int(self.threat_vectors.shape[1]) if self.threat_vectors is not None else 0
        threat_arr = np.zeros((max_len, threat_dim), dtype=np.float32)
        target = np.full((max_len,), -100, dtype=np.int64)
        valid_mask = np.zeros((max_len,), dtype=bool)

        if self.backend['cat'].shape[1] > 0:
            cat[:seq_len] = self.backend['cat'][row_idx]
        if self.backend['cont'].shape[1] > 0:
            cont[:seq_len] = self.backend['cont'][row_idx]
        if self.backend['bin'].shape[1] > 0:
            binv[:seq_len] = self.backend['bin'][row_idx]
        if self.backend['track'].shape[1] > 0:
            track[:seq_len] = self.backend['track'][row_idx]
        if self.threat_vectors is not None and self.threat_row_indexer is not None:
            threat_idx = self.threat_row_indexer[row_idx]
            if np.any(threat_idx < 0) or np.any(threat_idx >= len(self.threat_vectors)):
                raise RuntimeError('Threat index out of bounds in dataset window.')
            threat_arr[:seq_len] = self.threat_vectors[threat_idx]

        text_idx = self.backend['text_idx'][row_idx]
        if np.any(text_idx < 0) or np.any(text_idx >= len(self.text_embeddings)):
            raise RuntimeError('text_embedding_idx out of bounds in dataset window.')
        text[:seq_len] = self.text_embeddings[text_idx]
        target[:seq_len] = self.backend['target'][row_idx]
        valid_mask[:seq_len] = True

        if getattr(self.cfg, 'debug_validate_items', False):
            if (
                (self.backend['cont'].shape[1] > 0 and not np.isfinite(cont[:seq_len]).all()) or
                (self.backend['bin'].shape[1] > 0 and not np.isfinite(binv[:seq_len]).all()) or
                (self.backend['track'].shape[1] > 0 and not np.isfinite(track[:seq_len]).all()) or
                (not np.isfinite(text[:seq_len]).all()) or
                (self.threat_vectors is not None and not np.isfinite(threat_arr[:seq_len]).all())
            ):
                raise RuntimeError('Non-finite values detected in model inputs for a dataset sample.')

        meta_slice = {
            'meta_cols': self.backend['meta_cols'],
            'meta_arrays': {col: self.backend['meta_arrays'][col][row_idx] for col in self.backend['meta_cols']},
            'chunk_start': int(sample['chunk_start']),
            'chunk_index': int(sample['chunk_index']),
            'is_first_chunk': int(sample['is_first_chunk']),
            'drop_prefix_tokens': int(sample['drop_prefix_tokens']),
            'seq_len': int(seq_len),
        }

        return {
            'categorical': torch.from_numpy(cat),
            'continuous': torch.from_numpy(cont),
            'binary': torch.from_numpy(binv),
            'tracking': torch.from_numpy(track),
            'text_emb': torch.from_numpy(text),
            'threat_vec': torch.from_numpy(threat_arr),
            'target': torch.from_numpy(target),
            'valid_mask': torch.from_numpy(valid_mask),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'meta': meta_slice
        }


def collate_fn(batch):
    out = {
        'categorical': torch.stack([b['categorical'] for b in batch]),
        'continuous': torch.stack([b['continuous'] for b in batch]),
        'binary': torch.stack([b['binary'] for b in batch]),
        'tracking': torch.stack([b['tracking'] for b in batch]),
        'text_emb': torch.stack([b['text_emb'] for b in batch]),
        'threat_vec': torch.stack([b['threat_vec'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
        'seq_len': torch.stack([b['seq_len'] for b in batch]),
        'meta': [b['meta'] for b in batch]
    }
    return out


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AlibiCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads for ALiBi attention.')

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self._alibi_bias_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._alibi_cache_max_entries = 8

    def _alibi_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (int(seq_len), str(device), str(dtype))
        cached = self._alibi_bias_cache.get(cache_key)
        if cached is not None:
            return cached

        distance = torch.arange(seq_len, device=device)[None, :] - torch.arange(seq_len, device=device)[:, None]
        distance = torch.clamp(distance, max=0).to(dtype)

        slopes = torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            -torch.arange(1, self.n_heads + 1, device=device, dtype=dtype) * (8.0 / self.n_heads)
        ).view(1, self.n_heads, 1, 1)

        bias = distance.view(1, 1, seq_len, seq_len) * slopes
        if len(self._alibi_bias_cache) >= self._alibi_cache_max_entries:
            self._alibi_bias_cache.clear()
        self._alibi_bias_cache[cache_key] = bias
        return bias

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self._alibi_bias(seq_len, scores.device, scores.dtype)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float('-inf'))

        if valid_mask is not None:
            key_padding_mask = (~valid_mask).view(bsz, 1, 1, seq_len)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return out


class OptimusReimBlock(nn.Module):
    def __init__(self, cfg: OptimusReimConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = AlibiCausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), valid_mask)
        x = x + self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class OptimusReimModel(nn.Module):
    def __init__(
        self,
        cfg: OptimusReimConfig,
        cat_cardinalities: Dict[str, int],
        text_dim: int,
        n_cont: int,
        n_bin: int,
        n_track: int,
        n_threat: int = 36,
    ):
        super().__init__()
        self.cfg = cfg
        self.threat_input_dim = int(n_threat)
        self.threat_proj = nn.Linear(self.threat_input_dim, 16)

        self.cat_cols = list(cat_cardinalities.keys())
        self.cat_embeds = nn.ModuleDict()
        cat_total = 0
        for col, card in cat_cardinalities.items():
            emb_dim = min(32, max(4, int(np.sqrt(card + 1))))
            self.cat_embeds[col] = nn.Embedding(card + 1, emb_dim)
            cat_total += emb_dim

        in_dim = cat_total + text_dim + n_cont + n_bin + n_track + 16
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )

        self.blocks = nn.ModuleList([OptimusReimBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, 3)

    def forward(self, categorical, continuous, binary, tracking, text_emb, valid_mask, threat_vec=None):
        cat_chunks = []
        for i, col in enumerate(self.cat_cols):
            x = categorical[:, :, i].clamp_min(0)
            x = x.clamp_max(self.cat_embeds[col].num_embeddings - 1)
            cat_chunks.append(self.cat_embeds[col](x))

        cat_block = torch.cat(cat_chunks, dim=-1) if len(cat_chunks) > 0 else torch.zeros((categorical.size(0), categorical.size(1), 0), device=categorical.device)

        if threat_vec is None:
            threat_vec = torch.zeros(
                (text_emb.size(0), text_emb.size(1), self.threat_input_dim),
                device=text_emb.device,
                dtype=text_emb.dtype,
            )
        threat_proj = self.threat_proj(threat_vec)

        feats = torch.cat([cat_block, continuous, binary, tracking, text_emb, threat_proj], dim=-1)
        h = self.input_proj(feats)

        for block in self.blocks:
            h = block(h, valid_mask)

        z = self.norm(h)
        logits = self.classifier(z)
        return logits


def compute_class_weights_from_sequences(df: pd.DataFrame, sequence_keys: List[Tuple], max_ratio: float = 20.0):
    mask = df.set_index(['game_id', 'sequence_id']).index.isin(sequence_keys)
    arr = df.loc[mask, 'target'].to_numpy(dtype=np.int64)
    counts = np.bincount(arr, minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = counts.sum() / (3.0 * counts)
    inv = inv / inv.min()
    inv = np.clip(inv, 1.0, max_ratio)
    return torch.tensor(inv, dtype=torch.float32)

# ## 4) 5-Fold Training (All-Timestep Supervision)
# 
# This training loop supervises all valid non-padded tokens. It does not use the Phase 5 shot-only masking pattern.
# 
# **Dual-model setup (separate run cells):**
# - **Model A** = `events_only` (tracking disabled)
# - **Model B** = `with_tracking` (tracking enabled)
# 
# **Production features:**
# - TensorBoard logging (scalars per epoch + hyperparameters)
# - Learning rate scheduling (Cosine Annealing)
# - Early stopping with patience counter
# - Best + last checkpoint saving
# - Checkpoint resumption support
# - Detailed per-class validation metrics
# - Fold verification before training
# - Progress timing and logging

print('=' * 80)
print('CREATE STRATIFIED K-FOLD SPLITS FOR TRANSFORMER (GAME-TIMELINE UNITS)')
print('=' * 80)

unique_games = df_events['game_id'].dropna().unique()
print(f'\nTotal unique games in events data: {len(unique_games):,}')

if 'event_type' in df_events.columns:
    event_goal_mask = df_events['event_type'].astype(str).str.strip().str.lower().eq('goal')
    game_goal_counts = df_events.loc[event_goal_mask].groupby('game_id').size()
    goal_source_label = 'event_type == goal'
    total_goal_tokens = int(event_goal_mask.sum())
else:
    if 'is_eos' in df_events.columns:
        eos_flag = pd.to_numeric(df_events['is_eos'], errors='coerce').fillna(0.0) > 0.5
    else:
        eos_flag = pd.Series(False, index=df_events.index)
    goal_token_mask = eos_flag & df_events['target'].isin([0, 1])
    game_goal_counts = df_events.loc[goal_token_mask].groupby('game_id').size()
    goal_source_label = 'is_eos == 1 and target in {0,1}'
    total_goal_tokens = int(goal_token_mask.sum())

goal_strata = []
for game_id in unique_games:
    goals = int(game_goal_counts.get(game_id, 0))
    if goals <= 3:
        stratum = 0
    elif goals <= 6:
        stratum = 1
    else:
        stratum = 2
    goal_strata.append(stratum)
goal_strata = np.array(goal_strata)

print(f'\nTotal goal rows for stratification ({goal_source_label}): {total_goal_tokens:,}')
print(f'Game stratification by total goals ({goal_source_label}):')
for stratum in range(3):
    count = int(np.sum(goal_strata == stratum))
    pct = count / max(1, len(goal_strata)) * 100
    if stratum == 0:
        label = 'Low (0-3 goals)'
    elif stratum == 1:
        label = 'Medium (4-6 goals)'
    else:
        label = 'High (7+ goals)'
    print(f'  {label:20s}: {count:4d} games ({pct:5.1f}%)')

skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
fold_splits = []

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(unique_games, goal_strata)):
    train_games = unique_games[tr_idx]
    val_games = unique_games[va_idx]

    train_df = df_events[df_events['game_id'].isin(train_games)]
    val_df = df_events[df_events['game_id'].isin(val_games)]

    train_targets = pd.to_numeric(train_df['target'], errors='coerce').fillna(2).astype(int)
    val_targets = pd.to_numeric(val_df['target'], errors='coerce').fillna(2).astype(int)

    train_counts = np.bincount(train_targets.clip(0, 2), minlength=3)
    val_counts = np.bincount(val_targets.clip(0, 2), minlength=3)

    fold_splits.append({
        'fold': fold_idx,
        'train_game_ids': train_games,
        'val_game_ids': val_games,
        'train_class_counts_raw': train_counts.tolist(),
        'val_class_counts_raw': val_counts.tolist()
    })

    print(f'\nFold {fold_idx + 1}:')
    print(f'  Train games: {len(train_games):,} | raw rows: {len(train_df):,}')
    print(f'  Val games:   {len(val_games):,} | raw rows: {len(val_df):,}')
    print(f'  Train class counts (raw rows): {train_counts.tolist()}')
    print(f'  Val class counts   (raw rows): {val_counts.tolist()}')

print(f'\nâœ“ Created {config.n_folds}-fold cross-validation splits using game-timeline units')

# ## 4.1) Fold Verification Helper
# 
# Quick sanity check before training to catch data/pipeline issues early.

def verify_fold_pipeline(model, dataloader, device, fold_label: str):
    """Verify fold data pipeline and model forward pass before training."""
    model.eval()

    t0 = time.perf_counter()
    batch = next(iter(dataloader))
    first_batch_s = time.perf_counter() - t0
    non_blocking = bool(getattr(dataloader, 'pin_memory', False))

    categorical = batch['categorical'].to(device, non_blocking=non_blocking)
    continuous = batch['continuous'].to(device, non_blocking=non_blocking)
    binary = batch['binary'].to(device, non_blocking=non_blocking)
    text_emb = batch['text_emb'].to(device, non_blocking=non_blocking)
    tracking = batch['tracking'].to(device, non_blocking=non_blocking)
    valid_mask = batch['valid_mask'].to(device, non_blocking=non_blocking)
    threat_batch = batch['threat_vec']
    threat_vec = (
        threat_batch.to(device, non_blocking=non_blocking)
        if threat_batch.shape[-1] > 0
        else None
    )

    t1 = time.perf_counter()
    with torch.no_grad():
        logits = model(categorical, continuous, binary, tracking, text_emb, valid_mask, threat_vec=threat_vec)
    first_fwd_s = time.perf_counter() - t1

    valid_tokens = (batch['target'] != -100).sum().item()
    n_sequences = len(batch['meta'])

    print(f'\n[Verification] {fold_label}')
    print(f'  First batch load time: {first_batch_s:.2f}s')
    print(f'  First forward pass: {first_fwd_s:.2f}s')
    print(f'  Batch shape: {tuple(batch["categorical"].shape)}')
    print(f'  Sequences in batch: {n_sequences}')
    print(f'  Valid tokens: {valid_tokens:,}')
    print(f'  Output shape: {tuple(logits.shape)}')

    if 'is_eos' in binary_cols:
        eos_idx = binary_cols.index('is_eos')
        if batch['binary'].shape[-1] > eos_idx:
            eos_visible = int((batch['binary'][:, :, eos_idx] > 0.5).sum().item())
            if eos_visible > 0:
                raise RuntimeError(f'EOS tokens detected in training batch: {eos_visible}. Expected filtered-out windows only.')
            print('  âœ“ EOS filtering check passed (no EOS tokens in batch)')

    if not torch.isfinite(logits).all():
        print('  âš  WARNING: Non-finite logits detected in forward pass')
        valid_logits = logits[valid_mask]
        if valid_logits.numel() > 0:
            nonfinite_count = (~torch.isfinite(valid_logits)).sum().item()
            print(f'    Non-finite valid logits: {nonfinite_count}')
    else:
        print('  âœ“ All logits are finite')

    if first_batch_s > 10.0:
        print('  âš  WARNING: Slow first batch (>10s) - possible data pipeline issue')

    model.train()


print('âœ“ Fold verification function defined')

from tqdm.auto import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, fold_label: str = '', epoch_idx: int = 0, num_epochs: int = 0, use_tqdm: bool = True):
    model.train()
    running_loss = 0.0
    n_batches = 0

    iterator = loader
    non_blocking = bool(getattr(loader, 'pin_memory', False))
    if use_tqdm:
        iterator = tqdm(
            loader,
            desc=f'{fold_label} | Train {epoch_idx + 1:02d}/{num_epochs:02d}',
            leave=False,
            dynamic_ncols=True
        )

    for batch in iterator:
        categorical = batch['categorical'].to(device, non_blocking=non_blocking)
        continuous = batch['continuous'].to(device, non_blocking=non_blocking)
        binary = batch['binary'].to(device, non_blocking=non_blocking)
        tracking = batch['tracking'].to(device, non_blocking=non_blocking)
        text_emb = batch['text_emb'].to(device, non_blocking=non_blocking)
        target = batch['target'].to(device, non_blocking=non_blocking)
        valid_mask = batch['valid_mask'].to(device, non_blocking=non_blocking)
        threat_batch = batch['threat_vec']
        threat_vec = (
            threat_batch.to(device, non_blocking=non_blocking)
            if threat_batch.shape[-1] > 0
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, continuous, binary, tracking, text_emb, valid_mask, threat_vec=threat_vec)
        loss = criterion(logits.view(-1, 3), target.view(-1))
        loss.backward()

        if config.clip_grad_norm is not None and config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

        if use_tqdm and (n_batches % 10 == 0):
            iterator.set_postfix({'loss': f'{running_loss / max(1, n_batches):.4f}'})

    return running_loss / max(1, n_batches)


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float('nan')
    y_prob = _normalize_prob_rows(y_prob)
    return float(log_loss(y_true, y_prob, labels=[0, 1, 2]))


def _normalize_prob_rows(y_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if y_prob.size == 0:
        return y_prob.astype(np.float64, copy=False)

    p = np.asarray(y_prob, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    row_sum = p.sum(axis=1, keepdims=True)
    bad = (row_sum <= eps) | (~np.isfinite(row_sum))
    if np.any(bad):
        p[bad.reshape(-1), :] = 1.0 / p.shape[1]
        row_sum = p.sum(axis=1, keepdims=True)

    return p / np.clip(row_sum, eps, None)


def _top_bucket_precision_recall(y_true: np.ndarray, class_prob: np.ndarray, class_idx: int, top_pct: float = 5.0) -> Dict[str, float]:
    n = int(len(class_prob))
    if n == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'count': 0,
            'threshold': float('nan')
        }

    k = max(1, int(np.ceil((top_pct / 100.0) * n)))
    top_idx = np.argpartition(class_prob, -k)[-k:]
    top_scores = class_prob[top_idx]
    threshold = float(np.min(top_scores)) if len(top_scores) else float('nan')

    is_correct_top = (y_true[top_idx] == class_idx)
    correct = int(np.sum(is_correct_top))
    true_total = int(np.sum(y_true == class_idx))

    precision = float(correct) / float(max(1, k))
    recall = float(correct) / float(max(1, true_total))

    return {
        'precision': precision,
        'recall': recall,
        'count': int(k),
        'threshold': threshold
    }


def _build_meta_view_dict(meta_payload, drop_prefix: int, take: int) -> Optional[Dict[str, np.ndarray]]:
    if take <= drop_prefix:
        return None

    if isinstance(meta_payload, pd.DataFrame):
        meta_view = meta_payload.iloc[drop_prefix:take].reset_index(drop=True)
        if len(meta_view) == 0:
            return None
        return {col: meta_view[col].to_numpy(copy=False) for col in meta_view.columns}

    if not isinstance(meta_payload, dict):
        return None

    cols = tuple(meta_payload.get('meta_cols', ()))
    arrays = meta_payload.get('meta_arrays', {})
    view = {col: np.asarray(arrays[col])[drop_prefix:take] for col in cols if col in arrays}
    n_rows = 0
    if view:
        n_rows = len(next(iter(view.values())))
    if n_rows == 0:
        return None

    view['chunk_start'] = np.full((n_rows,), int(meta_payload.get('chunk_start', 0)), dtype=np.int64)
    view['chunk_index'] = np.full((n_rows,), int(meta_payload.get('chunk_index', 0)), dtype=np.int64)
    view['is_first_chunk'] = np.full((n_rows,), int(meta_payload.get('is_first_chunk', 0)), dtype=np.int64)
    view['drop_prefix_tokens'] = np.full((n_rows,), int(meta_payload.get('drop_prefix_tokens', 0)), dtype=np.int64)
    return view


def _concat_meta_parts(meta_parts: List[Dict[str, np.ndarray]]) -> pd.DataFrame:
    if len(meta_parts) == 0:
        return pd.DataFrame()

    all_cols: List[str] = []
    for part in meta_parts:
        for col in part.keys():
            if col not in all_cols:
                all_cols.append(col)

    data = {}
    for col in all_cols:
        col_arrays = [np.asarray(part[col]) for part in meta_parts if col in part and len(part[col]) > 0]
        if len(col_arrays) == 0:
            continue
        data[col] = np.concatenate(col_arrays, axis=0)

    if len(data) == 0:
        return pd.DataFrame()
    return pd.DataFrame(data)


@torch.no_grad()
def infer_tokens(
    model,
    loader,
    fold_idx: int,
    progress_every: int = 0,
    context_label: Optional[str] = None,
    return_stats: bool = False,
):
    model.eval()
    rows: List[pd.DataFrame] = []
    eos_idx = binary_cols.index('is_eos') if 'is_eos' in binary_cols else None
    rows_collected = 0
    batch_count = 0
    model_forward_s = 0.0
    assembly_s = 0.0
    concat_s = 0.0
    context = context_label or f'fold_{fold_idx + 1}'
    start_ts = time.perf_counter()
    first_batch_reported = False

    try:
        total_batches = int(len(loader))
    except Exception:
        total_batches = 0
    non_blocking = bool(getattr(loader, 'pin_memory', False))

    for batch in loader:
        batch_count += 1
        if progress_every > 0 and not first_batch_reported:
            first_batch_secs = time.perf_counter() - start_ts
            suffix = f'/{total_batches}' if total_batches > 0 else ''
            print(
                f'[{context}] first batch ready in {first_batch_secs:.1f}s '
                f'| batch {batch_count}{suffix}'
            )
            first_batch_reported = True

        categorical = batch['categorical'].to(device, non_blocking=non_blocking)
        continuous = batch['continuous'].to(device, non_blocking=non_blocking)
        binary = batch['binary'].to(device, non_blocking=non_blocking)
        tracking = batch['tracking'].to(device, non_blocking=non_blocking)
        text_emb = batch['text_emb'].to(device, non_blocking=non_blocking)
        valid_mask = batch['valid_mask'].to(device, non_blocking=non_blocking)
        target = batch['target'].to(device, non_blocking=non_blocking)
        threat_batch = batch['threat_vec']
        threat_vec = (
            threat_batch.to(device, non_blocking=non_blocking)
            if threat_batch.shape[-1] > 0
            else None
        )

        model_t0 = time.perf_counter()
        logits = model(categorical, continuous, binary, tracking, text_emb, valid_mask, threat_vec=threat_vec)
        model_forward_s += (time.perf_counter() - model_t0)

        assemble_t0 = time.perf_counter()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        targ = target.cpu().numpy()
        vm = valid_mask.cpu().numpy()
        bin_np = batch['binary'].cpu().numpy()

        batch_meta_parts: List[Dict[str, np.ndarray]] = []
        batch_target_parts: List[np.ndarray] = []
        batch_prob_parts: List[np.ndarray] = []

        for b_ix in range(len(batch['meta'])):
            meta_payload = batch['meta'][b_ix]
            keep = vm[b_ix].astype(bool)

            if eos_idx is not None and bin_np.shape[-1] > eos_idx:
                eos_tok = bin_np[b_ix, :, eos_idx] > 0.5
                if int((keep & eos_tok).sum()) > 0:
                    raise RuntimeError('EOS leakage detected in inference path.')
                keep = keep & (~eos_tok)

            n_keep = int(keep.sum())
            if n_keep == 0:
                continue

            p = probs[b_ix][keep]
            y = targ[b_ix][keep]

            if isinstance(meta_payload, pd.DataFrame):
                meta_len = len(meta_payload)
                drop_prefix = int(meta_payload['drop_prefix_tokens'].iloc[0]) if (meta_len and 'drop_prefix_tokens' in meta_payload.columns) else 0
            elif isinstance(meta_payload, dict):
                meta_len = int(meta_payload.get('seq_len', 0))
                drop_prefix = int(meta_payload.get('drop_prefix_tokens', 0))
            else:
                meta_len = 0
                drop_prefix = 0

            take = min(meta_len, n_keep)
            p = p[:take]
            y = y[:take]

            if drop_prefix > 0:
                if take <= drop_prefix:
                    continue
                p = p[drop_prefix:]
                y = y[drop_prefix:]

            meta_view = _build_meta_view_dict(meta_payload, drop_prefix, take)

            if meta_view is None:
                continue

            batch_meta_parts.append(meta_view)
            batch_target_parts.append(y.astype(np.int64, copy=False))
            batch_prob_parts.append(p.astype(np.float32, copy=False))
            rows_collected += int(len(y))

        if batch_meta_parts:
            concat_t0 = time.perf_counter()
            batch_df = _concat_meta_parts(batch_meta_parts)
            batch_targets = np.concatenate(batch_target_parts, axis=0)
            batch_probs = np.concatenate(batch_prob_parts, axis=0)
            concat_s += (time.perf_counter() - concat_t0)

            if len(batch_df) != len(batch_targets) or len(batch_targets) != len(batch_probs):
                raise RuntimeError('Inference output shape mismatch while assembling prediction frame.')

            batch_df['fold'] = int(fold_idx)
            batch_df['target'] = batch_targets
            batch_df['P_actor_goal'] = batch_probs[:, 0].astype(float, copy=False)
            batch_df['P_opp_goal'] = batch_probs[:, 1].astype(float, copy=False)
            batch_df['P_no_goal'] = batch_probs[:, 2].astype(float, copy=False)
            batch_df['pred_class'] = batch_probs.argmax(axis=1).astype(np.int64, copy=False)
            rows.append(batch_df)

        assembly_s += (time.perf_counter() - assemble_t0)

        if progress_every > 0 and (batch_count % int(progress_every) == 0):
            elapsed = time.perf_counter() - start_ts
            rate = rows_collected / max(elapsed, 1e-6)
            suffix = f'/{total_batches}' if total_batches > 0 else ''
            print(
                f'[{context}] progress: batch {batch_count}{suffix} '
                f'| rows={rows_collected:,} | elapsed={elapsed:.1f}s | rate={rate:.1f} rows/s'
            )

    if progress_every > 0:
        elapsed = time.perf_counter() - start_ts
        suffix = f'/{total_batches}' if total_batches > 0 else ''
        print(
            f'[{context}] complete: batch {batch_count}{suffix} '
            f'| rows={rows_collected:,} | elapsed={elapsed:.1f}s'
        )

    if len(rows) == 0:
        out_df = pd.DataFrame()
    else:
        out_df = pd.concat(rows, ignore_index=True)

    if not return_stats:
        return out_df

    stats = {
        'rows_collected': int(rows_collected),
        'batches': int(batch_count),
        'elapsed_s': float(time.perf_counter() - start_ts),
        'model_forward_s': float(model_forward_s),
        'assembly_s': float(assembly_s),
        'concat_s': float(concat_s),
    }
    return out_df, stats


# --- Phase 6 loss upgrade: class-collapse mitigation ---
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if alpha is not None:
            self.register_buffer('alpha', alpha.to(dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f'Expected logits shape [N, C], got {tuple(logits.shape)}')
        if target.ndim != 1:
            raise ValueError(f'Expected target shape [N], got {tuple(target.shape)}')

        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits_v = logits[valid]
        target_v = target[valid]

        log_probs = F.log_softmax(logits_v, dim=-1)
        row_idx = torch.arange(logits_v.size(0), device=logits_v.device)
        log_pt = log_probs[row_idx, target_v]
        pt = torch.exp(log_pt)

        focal_factor = (1.0 - pt).clamp(min=0.0) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits_v.device)[target_v]
            loss = -alpha_t * focal_factor * log_pt
        else:
            loss = -focal_factor * log_pt

        return loss.mean()


def _count_targets_from_dataset_windows(dataset: OptimusReimSequenceDataset) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    target_arr = dataset.backend['target']
    for sample in dataset.samples:
        idx = np.asarray(sample['row_idx'], dtype=np.int64)
        if len(idx) == 0:
            continue
        vals = target_arr[idx]
        vals = vals[(vals >= 0) & (vals < 3)]
        if len(vals) == 0:
            continue
        counts += np.bincount(vals, minlength=3)
    return counts


def _build_balanced_alpha(
    class_counts: np.ndarray,
    balance_power: float = 1.0,
    max_ratio: float = 12.0,
    eps: float = 1.0
) -> np.ndarray:
    counts = class_counts.astype(np.float64).copy()
    counts = np.maximum(counts, eps)

    inv = (counts.sum() / (len(counts) * counts)) ** float(balance_power)
    inv = inv / np.min(inv)
    inv = np.clip(inv, 1.0, float(max_ratio))
    return inv.astype(np.float32)


def _format_class_counts_line(prefix: str, counts: np.ndarray) -> str:
    total = int(np.sum(counts))
    if total <= 0:
        return f"{prefix} [0:0 (0.0%), 1:0 (0.0%), 2:0 (0.0%)]"
    p0 = 100.0 * float(counts[0]) / total
    p1 = 100.0 * float(counts[1]) / total
    p2 = 100.0 * float(counts[2]) / total
    return f"{prefix} [0:{int(counts[0]):,} ({p0:5.1f}%), 1:{int(counts[1]):,} ({p1:5.1f}%), 2:{int(counts[2]):,} ({p2:5.1f}%)]"


def _baseline_entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return float('nan')
    probs = np.clip(counts.astype(np.float64) / total, eps, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def _safe_skill_score(model_logloss: float, baseline_entropy: float) -> float:
    if not np.isfinite(model_logloss):
        return float('nan')
    if not np.isfinite(baseline_entropy) or baseline_entropy <= 0.0:
        return float('nan')
    return float(1.0 - (model_logloss / baseline_entropy))


def _event_key_candidates(df: pd.DataFrame) -> List[List[str]]:
    cands = []
    if all(c in df.columns for c in ['game_id', 'sl_event_id']):
        cands.append(['game_id', 'sl_event_id'])
    if all(c in df.columns for c in ['game_id', 'game_event_id']):
        cands.append(['game_id', 'game_event_id'])
    return cands


def _resolve_event_key_cols(df: pd.DataFrame) -> List[str]:
    candidates = _event_key_candidates(df)
    if len(candidates) == 0:
        raise RuntimeError('No valid event-key candidates found. Need game_id + sl_event_id or game_id + game_event_id.')

    best_cols = None
    best_null = None
    for cols in candidates:
        null_count = int(df[cols].isna().any(axis=1).sum())
        if best_null is None or null_count < best_null:
            best_null = null_count
            best_cols = cols

    if best_cols is None:
        raise RuntimeError('Could not resolve event-key columns.')
    return best_cols


def _deduplicate_event_predictions(pred_df: pd.DataFrame, context_label: str = '') -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(pred_df) == 0:
        return pred_df.copy(), {'raw_rows': 0, 'dedup_rows': 0, 'duplicates_removed': 0, 'duplicate_rate': 0.0, 'key_cols': []}

    key_cols = _resolve_event_key_cols(pred_df)
    work = pred_df.copy()

    null_rows = int(work[key_cols].isna().any(axis=1).sum())
    if null_rows > 0:
        raise RuntimeError(f'{context_label} has {null_rows:,} rows with null event keys in {key_cols}.')

    sort_cols = [c for c in ['game_id', 'period', 'game_event_id', 'period_time_sec', 'sequence_event_id', 'chunk_start', 'chunk_index', 'sl_event_id'] if c in work.columns]
    work['_dedup_order'] = np.arange(len(work), dtype=np.int64)
    if len(sort_cols):
        work = work.sort_values(sort_cols + ['_dedup_order']).reset_index(drop=True)

    raw_rows = int(len(work))
    dedup = work.drop_duplicates(key_cols, keep='last').copy()
    dedup = dedup.drop(columns=['_dedup_order'], errors='ignore').reset_index(drop=True)
    dedup_rows = int(len(dedup))
    duplicates_removed = raw_rows - dedup_rows
    duplicate_rate = float(duplicates_removed) / float(max(1, raw_rows))

    audit = {
        'raw_rows': raw_rows,
        'dedup_rows': dedup_rows,
        'duplicates_removed': int(duplicates_removed),
        'duplicate_rate': float(duplicate_rate),
        'key_cols': key_cols
    }
    return dedup, audit


def _coerce_sl_event_id_key(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.isna().any():
        bad_examples = series[numeric.isna()].head(5).tolist()
        raise RuntimeError(f'{label} has invalid sl_event_id values. examples={bad_examples}')

    # Keep IDs numeric and stable for joins without forcing integer-only values.
    return pd.Series(np.round(numeric.astype(np.float64), 6), index=series.index).astype('Float64')


def _prepare_source_tensor_for_oof(source_df: pd.DataFrame) -> pd.DataFrame:
    required = ['game_id', 'sl_event_id']
    missing = [c for c in required if c not in source_df.columns]
    if missing:
        raise RuntimeError(f'Source tensor dataset missing required key columns: {missing}')

    src = source_df.copy()
    src['game_id'] = src['game_id'].astype(str)
    src['sl_event_id'] = _coerce_sl_event_id_key(src['sl_event_id'], 'Source tensor dataset')

    null_keys = int(src[['game_id', 'sl_event_id']].isna().any(axis=1).sum())
    if null_keys > 0:
        raise RuntimeError(f'Source tensor dataset has {null_keys:,} rows with null keys [game_id, sl_event_id].')

    dup_keys = int(src.duplicated(['game_id', 'sl_event_id'], keep=False).sum())
    if dup_keys > 0:
        raise RuntimeError(f'Source tensor dataset has duplicate keys: {dup_keys:,} rows for [game_id, sl_event_id].')

    return src


def _attach_source_tensor_columns(pred_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    pred = pred_df.copy()
    pred['game_id'] = pred['game_id'].astype(str)
    pred['sl_event_id'] = _coerce_sl_event_id_key(pred['sl_event_id'], 'Prediction frame')

    null_pred_keys = int(pred[['game_id', 'sl_event_id']].isna().any(axis=1).sum())
    if null_pred_keys > 0:
        raise RuntimeError(f'Prediction frame has {null_pred_keys:,} rows with null keys before source-column join.')

    src_payload = source_df.copy()
    src_payload['_source_tensor_present'] = 1

    merged = pred.merge(src_payload, on=['game_id', 'sl_event_id'], how='left', suffixes=('', '__src'), validate='many_to_one')
    missing_src = int(merged['_source_tensor_present'].isna().sum())
    if missing_src > 0:
        raise RuntimeError(
            'Raw OOF source-column join failed: '
            f'{missing_src:,} prediction rows were not found in tensor_ready_dataset by [game_id, sl_event_id].'
        )

    merged = merged.drop(columns=['_source_tensor_present'])

    source_cols = list(source_df.columns)
    pred_only_cols = [c for c in merged.columns if c not in source_cols]
    return merged[source_cols + pred_only_cols]


def _audit_raw_oof_source_contract(raw_oof_df: pd.DataFrame, source_df: pd.DataFrame) -> Dict[str, Any]:
    if raw_oof_df.empty:
        return {
            'raw_rows': 0,
            'source_column_count': int(len(source_df.columns)),
            'source_columns_missing_in_raw': 0,
            'raw_key_rows_missing_in_source': 0,
            'raw_duplicate_key_rows': 0,
            'raw_key_columns': ['game_id', 'sl_event_id'],
        }

    required_keys = ['game_id', 'sl_event_id']
    for key in required_keys:
        if key not in raw_oof_df.columns:
            raise RuntimeError(f'Raw OOF contract failed: missing required key column: {key}')

    source_cols = list(source_df.columns)
    missing_source_cols = [c for c in source_cols if c not in raw_oof_df.columns]
    if missing_source_cols:
        raise RuntimeError(
            'Raw OOF contract failed: missing source tensor columns in export. '
            f'Missing {len(missing_source_cols)} columns (sample): {missing_source_cols[:10]}'
        )

    raw_key_cols = ['game_id', 'sl_event_id']
    for opt in ['model_variant', 'variant_name']:
        if opt in raw_oof_df.columns:
            raw_key_cols.append(opt)

    raw_dups = int(raw_oof_df.duplicated(raw_key_cols, keep=False).sum())
    if raw_dups > 0:
        raise RuntimeError(
            'Raw OOF contract failed: duplicate rows detected for raw key columns '
            f'{raw_key_cols}: {raw_dups:,} rows.'
        )

    raw_keys = raw_oof_df[['game_id', 'sl_event_id']].drop_duplicates().copy()
    raw_keys['game_id'] = raw_keys['game_id'].astype(str)
    raw_keys['sl_event_id'] = _coerce_sl_event_id_key(raw_keys['sl_event_id'], 'Raw OOF contract raw keys')

    source_keys = source_df[['game_id', 'sl_event_id']].drop_duplicates().copy()
    source_keys['game_id'] = source_keys['game_id'].astype(str)
    source_keys['sl_event_id'] = _coerce_sl_event_id_key(source_keys['sl_event_id'], 'Raw OOF contract source keys')

    key_check = raw_keys.merge(source_keys, on=['game_id', 'sl_event_id'], how='left', indicator=True)
    raw_missing_keys = int((key_check['_merge'] == 'left_only').sum())
    if raw_missing_keys > 0:
        raise RuntimeError(
            'Raw OOF contract failed: raw export contains keys not present in source tensor '
            f'dataset: {raw_missing_keys:,} rows.'
        )

    return {
        'raw_rows': int(len(raw_oof_df)),
        'raw_unique_event_keys': int(len(raw_keys)),
        'source_rows': int(len(source_df)),
        'source_column_count': int(len(source_cols)),
        'source_columns_missing_in_raw': 0,
        'raw_key_rows_missing_in_source': 0,
        'raw_duplicate_key_rows': 0,
        'raw_key_columns': raw_key_cols,
    }


def _run_pretraining_gates(
    df: pd.DataFrame,
    train_games: np.ndarray,
    val_games: np.ndarray,
    train_ds: OptimusReimSequenceDataset,
    val_ds: OptimusReimSequenceDataset,
    fold_label: str
 ) -> Dict[str, float]:
    failures = []

    fold_df = df[df['game_id'].isin(np.concatenate([train_games, val_games]))].copy()
    key_cols = _resolve_event_key_cols(fold_df)

    null_key_rows = int(fold_df[key_cols].isna().any(axis=1).sum())
    if null_key_rows > 0:
        failures.append(f'Null event keys in fold data: {null_key_rows:,} rows for keys {key_cols}')

    dup_key_rows = int(fold_df.duplicated(key_cols, keep=False).sum())
    if dup_key_rows > 0:
        failures.append(f'Duplicate event keys in fold data: {dup_key_rows:,} rows for keys {key_cols}')

    work = fold_df[['game_id']].copy()
    work['period_time_sec__tmp'] = _phase6_period_time_seconds(fold_df)
    if 'sequence_id' in fold_df.columns:
        work['sequence_id'] = fold_df['sequence_id']
        grp_cols = ['game_id', 'sequence_id']
    else:
        grp_cols = ['game_id']
    work['delta_sec'] = work.groupby(grp_cols, sort=False)['period_time_sec__tmp'].diff()
    n_inv = int((work['delta_sec'] < -1.0).sum())
    if n_inv > 0:
        failures.append(f'Ordering inversion gate failed: {n_inv:,} rows with delta_sec < -1.0')

    train_counts = _count_targets_from_dataset_windows(train_ds)
    val_counts = _count_targets_from_dataset_windows(val_ds)
    if int(train_counts[0]) <= 0 or int(train_counts[1]) <= 0:
        failures.append(f'Train fold missing rare class support after windowing: counts={train_counts.tolist()}')
    if int(val_counts[0]) <= 0 or int(val_counts[1]) <= 0:
        failures.append(f'Val fold missing rare class support after windowing: counts={val_counts.tolist()}')

    for ds_name, ds in [('train', train_ds), ('val', val_ds)]:
        backend = ds.backend
        for arr_name in ['cont', 'bin', 'track']:
            arr = backend[arr_name]
            if arr.size and not np.isfinite(arr).all():
                failures.append(f'Non-finite values in {ds_name} backend[{arr_name}]')
        text_idx = backend['text_idx']
        if np.any(text_idx < 0) or np.any(text_idx >= len(ds.text_embeddings)):
            failures.append(f'text_embedding_idx out of bounds in {ds_name} dataset')

    overlap_train = int(sum(int(s.get('drop_prefix_tokens', 0) > 0) for s in train_ds.samples))
    overlap_val = int(sum(int(s.get('drop_prefix_tokens', 0) > 0) for s in val_ds.samples))

    if failures:
        detail = '\n  - ' + '\n  - '.join(failures)
        raise RuntimeError(f'Pre-training gates failed for {fold_label}:{detail}')

    return {
        'train_counts': train_counts.tolist(),
        'val_counts': val_counts.tolist(),
        'overlap_windows_train': overlap_train,
        'overlap_windows_val': overlap_val,
        'event_key_cols': key_cols,
        'ordering_inversions': n_inv
    }


def _is_better_checkpoint(
    curr_aucpr: float,
    curr_skill: float,
    curr_logloss: float,
    best_aucpr: float,
    best_skill: float,
    best_logloss: float,
    aucpr_delta: float = 1e-8,
    eps: float = 1e-8
) -> bool:
    if not np.isfinite(curr_aucpr):
        return False
    if not np.isfinite(best_aucpr):
        return True
    if curr_aucpr > (best_aucpr + aucpr_delta):
        return True
    if curr_aucpr < (best_aucpr - aucpr_delta):
        return False

    curr_skill_cmp = curr_skill if np.isfinite(curr_skill) else -np.inf
    best_skill_cmp = best_skill if np.isfinite(best_skill) else -np.inf
    if curr_skill_cmp > (best_skill_cmp + eps):
        return True
    if curr_skill_cmp < (best_skill_cmp - eps):
        return False

    curr_ll_cmp = curr_logloss if np.isfinite(curr_logloss) else np.inf
    best_ll_cmp = best_logloss if np.isfinite(best_logloss) else np.inf
    return curr_ll_cmp < (best_ll_cmp - eps)


def _sanitize_continuous_columns(df: pd.DataFrame, cols: List[str]) -> None:
    if not cols:
        return
    for col in cols:
        s = numeric_with_nan(df[col])
        med = s.median()
        fill_val = float(med) if pd.notna(med) else 0.0
        df[col] = s.fillna(fill_val).astype(np.float32)


def _resolve_graph_paths(cfg: OptimusReimConfig) -> tuple[Path, Path, Path]:
    if cfg.gnn_feats_path is not None and cfg.gnn_adj_path is not None and cfg.gnn_mask_path is not None:
        return Path(cfg.gnn_feats_path), Path(cfg.gnn_adj_path), Path(cfg.gnn_mask_path)

    variant = str(getattr(cfg, 'gnn_graph_variant', 'base')).strip().lower()
    variant_map = {
        'base': ('base_feats.npy', 'base_adj.npy', 'base_mask.npy'),
        'actor_rel': ('actor_rel_feats.npy', 'actor_rel_adj.npy', 'actor_rel_mask.npy'),
        'actor_rel_ctx': ('actor_rel_ctx_feats.npy', 'actor_rel_ctx_adj.npy', 'actor_rel_ctx_mask.npy'),
        'actor_emph': ('actor_emph_feats.npy', 'actor_emph_adj.npy', 'actor_emph_mask.npy'),
    }
    if variant not in variant_map:
        raise ValueError(f'Unknown gnn_graph_variant: {variant}')

    f_name, a_name, m_name = variant_map[variant]
    return (cfg.input_dir / f_name, cfg.input_dir / a_name, cfg.input_dir / m_name)


def _graph_feature_names_for_variant(variant: str, node_feature_dim: int) -> List[str]:
    key = str(variant).strip().lower()
    if key == 'actor_rel_ctx' and int(node_feature_dim) == 10:
        return [
            'x_rel_to_actor',
            'y_rel_to_actor',
            'vx_rel_to_actor',
            'vy_rel_to_actor',
            'dist_to_actor',
            'dist_to_net_abs',
            'angle_to_net_abs',
            'is_teammate',
            'is_possessing_team',
            'is_primary_actor',
        ]
    if int(node_feature_dim) == 8:
        return ['x', 'y', 'vx', 'vy', 'dist', 'side_flag', 'is_possessing_team', 'is_primary_actor']
    return [f'feature_{i}' for i in range(int(node_feature_dim))]


def _normalize_graph_keys(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = ['game_id', 'sl_event_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'{label} missing required key columns: {missing}')

    keys = df[required].copy()
    keys['game_id'] = keys['game_id'].astype(str)
    keys['sl_event_id'] = _coerce_sl_event_id_key(keys['sl_event_id'], label)

    bad = keys['sl_event_id'].isna()
    if bad.any():
        examples = keys.loc[bad, required].head(5).to_dict('records')
        raise RuntimeError(f'{label} has invalid sl_event_id values. examples={examples}')

    dup = keys.duplicated(required, keep=False)
    if dup.any():
        examples = keys.loc[dup, required].head(5).to_dict('records')
        raise RuntimeError(f'{label} has duplicate keys. examples={examples}')

    return keys


def _load_threat_vectors(df: pd.DataFrame, cfg: OptimusReimConfig) -> dict[str, Any]:
    threat_path = cfg.input_dir / 'threat_vectors.npy'
    for p in [threat_path]:
        if not p.exists():
            raise RuntimeError(f'Missing threat artifact: {p}')

    threat = np.load(threat_path)

    if threat.ndim != 2:
        raise RuntimeError(f'Unexpected threat shape: {threat.shape}. Expected [N, 36].')
    if int(threat.shape[1]) != 36:
        raise RuntimeError(f'Unexpected threat width: {threat.shape}. Expected [N, 36].')
    if threat.dtype != np.float32:
        raise RuntimeError(f'Expected threat dtype float32, got {threat.dtype}')

    tensor_path = cfg.input_dir / 'tensor_ready_dataset.parquet'
    tensor_df = pd.read_parquet(tensor_path, columns=['game_id', 'sl_event_id'])
    tensor_keys = _normalize_graph_keys(tensor_df, label='tensor_ready_dataset')
    if len(tensor_keys) != int(threat.shape[0]):
        raise RuntimeError(
            'Threat row count must match tensor_ready_dataset rows. '
            f'threat_rows={int(threat.shape[0]):,} tensor_ready_rows={len(tensor_keys):,}'
        )

    df_keys = _normalize_graph_keys(df, label='training dataframe')
    graph_index = pd.MultiIndex.from_frame(tensor_keys[['game_id', 'sl_event_id']])
    df_index = pd.MultiIndex.from_frame(df_keys[['game_id', 'sl_event_id']])
    row_indexer = graph_index.get_indexer(df_index)

    if (row_indexer < 0).any():
        missing_mask = row_indexer < 0
        missing_count = int(missing_mask.sum())
        examples = df_keys.loc[missing_mask, ['game_id', 'sl_event_id']].head(5).to_dict('records')
        raise RuntimeError(
            'Missing threat rows for training dataframe keys. '
            f'missing_rows={missing_count:,} examples={examples}'
        )

    row_indexer = row_indexer.astype(np.int64, copy=False)
    print(
        'Loaded threat vectors in memory: '
        f'threat={threat.shape} '
        f'| dtype={threat.dtype}'
    )

    return {
        'threat': threat,
        'row_indexer': row_indexer,
        'paths': {
            'threat': str(threat_path),
        },
        'threat_dim': int(threat.shape[1]),
    }


def run_cv_phase6(
    df: pd.DataFrame,
    cfg: OptimusReimConfig,
    model_variant: str,
    threat_data: Optional[dict[str, Any]] = None,
    source_tensor_df: Optional[pd.DataFrame] = None,
):
    """Run CV training on full-game sliding windows with rare-class checkpointing."""
    work_df = df.copy(deep=False)
    event_key_cols = _resolve_event_key_cols(work_df)
    dup_mask = work_df.duplicated(event_key_cols, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        sort_cols = [c for c in ['game_id', 'period', 'game_event_id', 'period_time_sec', 'sequence_event_id', 'sl_event_id'] if c in work_df.columns]
        work_df['_dedupe_order'] = np.arange(len(work_df), dtype=np.int64)
        if len(sort_cols):
            work_df = work_df.sort_values(sort_cols + ['_dedupe_order']).reset_index(drop=True)
        before_rows = int(len(work_df))
        work_df = work_df.drop_duplicates(event_key_cols, keep='last').drop(columns=['_dedupe_order'], errors='ignore').reset_index(drop=True)
        after_rows = int(len(work_df))
        print(f'Applied deterministic event-key dedupe before CV: removed {before_rows - after_rows:,} rows using keys {event_key_cols}')
    variant_map = {
        'A': 'events_only',
        'B': 'with_tracking',
        'events_only': 'events_only',
        'with_tracking': 'with_tracking'
    }
    canonical_variant = variant_map.get(model_variant, model_variant)
    if canonical_variant not in {'events_only', 'with_tracking'}:
        raise ValueError(f'Unsupported model_variant: {model_variant}')

    if source_tensor_df is None:
        raise RuntimeError('run_cv_phase6 requires source_tensor_df to preserve full source columns in raw OOF output.')

    source_tensor_prepared = _prepare_source_tensor_for_oof(source_tensor_df)

    total_epochs = int(cfg.num_epochs)
    if total_epochs < 0:
        raise ValueError(f'num_epochs must be >= 0, got {total_epochs}')

    inference_only_mode = total_epochs == 0
    if inference_only_mode:
        if not bool(getattr(cfg, 'resume_from_checkpoint', False)):
            raise ValueError(
                'num_epochs=0 (inference-only mode) requires resume_from_checkpoint=True '
                'so model weights can be loaded from checkpoints.'
            )
        print(
            'Inference-only checkpoint mode enabled (num_epochs=0): '
            'skipping training epochs and generating OOF predictions from resumed checkpoints.'
        )

    loss_mode = str(getattr(cfg, 'loss_mode', 'ce_balanced')).strip().lower()
    if loss_mode not in {'ce', 'ce_balanced', 'focal', 'focal_balanced'}:
        loss_mode = 'ce_balanced'
    fallback_to_ce_balanced = bool(getattr(cfg, 'loss_fallback_to_ce_balanced', True))

    focal_gamma = float(getattr(cfg, 'focal_gamma', 2.0))
    balance_power = float(getattr(cfg, 'loss_balance_power', 1.0))
    loss_max_ratio = float(getattr(cfg, 'loss_max_ratio', 12.0))
    label_smoothing = float(getattr(cfg, 'label_smoothing', 0.0))
    early_stop_warmup = int(getattr(cfg, 'early_stop_warmup', 8))
    aucpr_patience_delta = float(getattr(cfg, 'aucpr_patience_delta', 1e-4))
    rare_prob_mass_floor = float(getattr(cfg, 'rare_class_prob_mass_floor', 0.025))
    no_goal_prob_ceiling = float(getattr(cfg, 'no_goal_prob_ceiling', 0.985))
    collapse_patience = int(getattr(cfg, 'collapse_patience', 4))
    use_tqdm = True

    all_oof = []
    fold_metrics = []

    use_tracking_variant = canonical_variant == 'with_tracking'
    if use_tracking_variant and not TRACKING_SCHEMA_COMPLETE:
        raise RuntimeError(
            'with_tracking requested, but expected pinned tracking columns are missing. '
            f'first10={missing_tracking[:10]} | total_missing={len(missing_tracking)}'
        )

    # Use variant-specific continuous feature lists
    cfg_infer = copy.copy(cfg)
    cfg_infer.use_tracking = use_tracking_variant

    graph_variant = str(getattr(cfg_infer, 'gnn_graph_variant', 'base')) if use_tracking_variant else 'base'
    coord_mode_variant = _resolve_coordinate_mode(canonical_variant, gnn_graph_variant=graph_variant)
    continuous_cols_variant = _get_variant_continuous_cols(
        canonical_variant,
        work_df.columns.tolist(),
        gnn_graph_variant=graph_variant,
    )
    for col in continuous_cols_variant:
        if col in work_df.columns:
            work_df[col] = work_df[col].copy()
    _sanitize_continuous_columns(work_df, continuous_cols_variant)

    cat_card = {c: int(work_df[c].max()) + 1 for c in categorical_cols}
    n_cont = len(continuous_cols_variant)
    n_bin = len(binary_cols)
    n_track = len(tracking_cols) if use_tracking_variant else 0
    n_threat = 36
    if use_tracking_variant and threat_data is None:
        threat_data = _load_threat_vectors(work_df, cfg)

    print('\nFeature schema for this run:')
    print(f'  variant: {canonical_variant}')
    print(f'  coordinate mode: {coord_mode_variant} (gnn_graph_variant={graph_variant})')
    print(f'  continuous cols ({len(continuous_cols_variant)}): {continuous_cols_variant}')
    if use_tracking_variant:
        print('  Threat conditioning active (GNN removed)')
        if threat_data is not None:
            print(f"  threat path: {threat_data['paths']['threat']}")
            print(f"  threat dim: {threat_data['threat_dim']}")

    print('Class-weight config for this run:')
    print(f'  mode: {loss_mode}')
    print(f'  fallback_to_ce_balanced: {fallback_to_ce_balanced}')
    print(f'  focal_gamma: {focal_gamma:.2f}')
    print(f'  balance_power: {balance_power:.2f}')
    print(f'  loss_max_ratio: {loss_max_ratio:.1f}')
    print(f'  label_smoothing: {label_smoothing:.4f}')
    print(f'  early_stop_warmup: {early_stop_warmup}')
    print(f'  aucpr_patience_delta: {aucpr_patience_delta:.6f}')
    print(f'  rare_class_prob_mass_floor: {rare_prob_mass_floor:.4f}')
    print(f'  no_goal_prob_ceiling: {no_goal_prob_ceiling:.4f}')
    print(f'  collapse_patience: {collapse_patience}')
    print(f'Configured num_epochs: {total_epochs}')

    writer_root = None
    if cfg.use_tensorboard:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer_root = cfg.tensorboard_dir / f'{model_variant}_{canonical_variant}_{run_timestamp}'
        writer_root.mkdir(parents=True, exist_ok=True)
        print(f'\nTensorBoard LIVE log root (Model {model_variant}): {writer_root}')
        print('Open TensorBoard with:')
        print(f'  tensorboard --logdir="{writer_root}"')

    for f in fold_splits:
        fold_idx = f['fold']
        print(f'\n{"=" * 80}')
        print(f'TRAINING MODEL {model_variant} ({canonical_variant.upper()}) - FOLD {fold_idx + 1}/{cfg.n_folds}')
        print(f'{"=" * 80}')

        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.use_tracking = use_tracking_variant

        train_games = f['train_game_ids']
        val_games = f['val_game_ids']

        try:
            train_ds = Phase6SequenceDataset(
                work_df,
                text_embeddings,
                max_seq_length=cfg_fold.max_seq_length,
                window_stride=cfg_fold.window_stride,
                min_window_tokens=cfg_fold.min_window_tokens,
                use_tracking=use_tracking_variant,
                debug_validate_items=bool(getattr(cfg_fold, 'debug_validate_items', False)),
                categorical_cols=list(categorical_cols),
                continuous_cols=list(continuous_cols_variant),
                binary_cols=list(binary_cols),
                tracking_cols=list(tracking_cols),
                game_ids=train_games,
                threat_vectors=(threat_data['threat'] if use_tracking_variant and threat_data is not None else None),
                threat_row_indexer=(threat_data['row_indexer'] if use_tracking_variant and threat_data is not None else None),
            )
            val_ds = Phase6SequenceDataset(
                work_df,
                text_embeddings,
                max_seq_length=cfg_fold.max_seq_length,
                window_stride=cfg_fold.window_stride,
                min_window_tokens=cfg_fold.min_window_tokens,
                use_tracking=use_tracking_variant,
                debug_validate_items=bool(getattr(cfg_fold, 'debug_validate_items', False)),
                categorical_cols=list(categorical_cols),
                continuous_cols=list(continuous_cols_variant),
                binary_cols=list(binary_cols),
                tracking_cols=list(tracking_cols),
                game_ids=val_games,
                threat_vectors=(threat_data['threat'] if use_tracking_variant and threat_data is not None else None),
                threat_row_indexer=(threat_data['row_indexer'] if use_tracking_variant and threat_data is not None else None),
            )
        except RuntimeError as exc:
            if 'No non-EOS windows available' in str(exc):
                print(f'Skipping fold {fold_idx + 1}: no non-EOS windows after filtering.')
                continue
            raise

        print(f'Filtered all-EOS games | train={train_ds.filtered_all_eos_games:,}, val={val_ds.filtered_all_eos_games:,}')
        print(f'Dataset windows | train={len(train_ds):,}, val={len(val_ds):,}')

        gate_report = _run_pretraining_gates(
            df=work_df,
            train_games=train_games,
            val_games=val_games,
            train_ds=train_ds,
            val_ds=val_ds,
            fold_label=f'fold {fold_idx + 1}'
        )
        print(f"Pre-training gates passed | key={gate_report['event_key_cols']} | order_inversions={gate_report['ordering_inversions']}")
        print(f"  overlap windows (train/val): {gate_report['overlap_windows_train']:,} / {gate_report['overlap_windows_val']:,}")

        train_counts = _count_targets_from_dataset_windows(train_ds)
        val_counts = _count_targets_from_dataset_windows(val_ds)
        val_baseline_entropy = _baseline_entropy_from_counts(val_counts)
        print('  ' + _format_class_counts_line('Train class dist:', train_counts))
        print('  ' + _format_class_counts_line('Val class dist:  ', val_counts))
        print(f'  Val baseline entropy: {val_baseline_entropy:.6f}')

        alpha_np = _build_balanced_alpha(
            train_counts,
            balance_power=balance_power,
            max_ratio=loss_max_ratio
        )
        alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

        if loss_mode == 'ce':
            criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
            loss_desc = f'CrossEntropyLoss(unweighted, label_smoothing={label_smoothing:.4f})'
        elif loss_mode == 'ce_balanced':
            criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=alpha, label_smoothing=label_smoothing)
            loss_desc = f'CrossEntropyLoss(weighted, alpha={alpha_np.tolist()}, label_smoothing={label_smoothing:.4f})'
        elif loss_mode == 'focal':
            criterion = FocalCrossEntropyLoss(gamma=focal_gamma, alpha=None, ignore_index=-100)
            loss_desc = f'FocalCrossEntropyLoss(gamma={focal_gamma:.2f}, alpha=None)'
        else:
            criterion = FocalCrossEntropyLoss(gamma=focal_gamma, alpha=alpha, ignore_index=-100)
            loss_desc = f'FocalCrossEntropyLoss(gamma={focal_gamma:.2f}, alpha={alpha_np.tolist()})'

        print(f'  Loss: {loss_desc}')

        loader_num_workers = max(0, int(cfg.num_workers))
        loader_pin_memory = bool(cfg.pin_memory) if cfg.pin_memory is not None else (device.type in {'cuda', 'xpu'})
        loader_kwargs = {
            'num_workers': loader_num_workers,
            'collate_fn': phase6_collate_fn,
            'pin_memory': loader_pin_memory,
        }
        if loader_num_workers > 0:
            loader_kwargs['worker_init_fn'] = phase6_seed_dataloader_worker
            loader_kwargs['persistent_workers'] = bool(cfg.persistent_workers)
            if int(cfg.prefetch_factor) > 0:
                loader_kwargs['prefetch_factor'] = int(cfg.prefetch_factor)
            if sys.platform.startswith('win'):
                loader_kwargs['multiprocessing_context'] = 'spawn'

        train_generator = torch.Generator()
        train_generator.manual_seed(int(cfg.seed) + int(fold_idx))

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        model = OptimusReimModel(cfg_fold, cat_card, text_embeddings.shape[1], n_cont, n_bin, n_track, n_threat=n_threat).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_fold.lr, weight_decay=cfg_fold.weight_decay)

        if cfg.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif cfg.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 3), gamma=0.1)
        else:
            scheduler = None

        writer = None
        if writer_root is not None:
            fold_log_dir = writer_root / f'fold_{fold_idx + 1}'
            writer = SummaryWriter(log_dir=str(fold_log_dir))
            print(f'TensorBoard log dir (Model {model_variant}, Fold {fold_idx + 1}): {fold_log_dir}')

        best_ckpt_path = cfg.models_dir / f'optimus_reim_{model_variant}_fold{fold_idx}_best.pth'
        last_ckpt_path = cfg.models_dir / f'optimus_reim_{model_variant}_fold{fold_idx}_last.pth'

        best_state = None
        best_epoch = -1
        best_aucpr = -np.inf
        best_skill_score = -np.inf
        best_logloss = np.inf
        patience_counter = 0
        collapse_counter = 0
        start_epoch = 0
        fold_history = []

        if cfg.resume_from_checkpoint and last_ckpt_path.exists():
            checkpoint = _torch_load_checkpoint_compat(last_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_epoch = checkpoint.get('best_epoch', -1)
            best_aucpr = checkpoint.get('best_aucpr_goal_mean', -np.inf)
            best_skill_score = checkpoint.get('best_skill_score', -np.inf)
            best_logloss = checkpoint.get('best_logloss', np.inf)
            patience_counter = checkpoint.get('patience_counter', 0)
            fold_history = checkpoint.get('history', [])
            print(f'â†» Resumed from checkpoint at epoch {start_epoch}')
        else:
            if inference_only_mode:
                raise FileNotFoundError(
                    'Inference-only mode requires an existing checkpoint per fold. '
                    f'Missing checkpoint: {last_ckpt_path}'
                )
            print('â†» Checkpoint resume disabled for this run; training fold from scratch.')

        verify_fold_pipeline(model, train_loader, device, f'{model_variant} Fold {fold_idx + 1}')
        fold_tag = f'Model {model_variant} | Fold {fold_idx + 1}'

        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.perf_counter()

            train_t0 = time.perf_counter()
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                fold_label=fold_tag, epoch_idx=epoch, num_epochs=total_epochs, use_tqdm=use_tqdm
            )
            train_s = time.perf_counter() - train_t0

            if (not np.isfinite(tr_loss)) and fallback_to_ce_balanced and loss_mode in {'focal', 'focal_balanced'}:
                print('  Non-finite train loss detected under focal mode; falling back to class-balanced CE.')
                criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=alpha, label_smoothing=label_smoothing)
                loss_mode = 'ce_balanced'

            infer_t0 = time.perf_counter()
            epoch_pred_raw, infer_stats = infer_tokens(
                model,
                val_loader,
                fold_idx,
                context_label=f'val_fold_{fold_idx + 1}_epoch_{epoch + 1}',
                return_stats=True,
            )
            infer_s = time.perf_counter() - infer_t0

            dedup_t0 = time.perf_counter()
            epoch_pred, dedup_audit = _deduplicate_event_predictions(epoch_pred_raw, context_label=f'val_fold_{fold_idx}')
            dedup_s = time.perf_counter() - dedup_t0

            metrics_t0 = time.perf_counter()
            val_m = _metrics_from_prediction_frame(epoch_pred, detailed=True)
            metrics_s = time.perf_counter() - metrics_t0
            val_m['val_loss'] = float('nan')
            val_m['n_eval_tokens_raw'] = int(dedup_audit['raw_rows'])
            val_m['n_eval_tokens_dedup'] = int(dedup_audit['dedup_rows'])
            val_m['eval_duplicates_removed'] = int(dedup_audit['duplicates_removed'])
            val_m['eval_duplicate_rate'] = float(dedup_audit['duplicate_rate'])

            goal_aucpr = float(val_m.get('aucpr_goal_mean', float('nan')))
            skill_score = _safe_skill_score(val_m['token_logloss'], val_baseline_entropy)
            val_m['skill_score'] = skill_score
            rare_prob_mass = float(val_m.get('mean_prob_0', 0.0) + val_m.get('mean_prob_1', 0.0))
            rare_mass_ok = rare_prob_mass >= rare_prob_mass_floor
            no_goal_prob = float(val_m.get('mean_prob_2', 0.0))
            no_goal_ok = no_goal_prob <= no_goal_prob_ceiling
            collapse_ok = rare_mass_ok and no_goal_ok
            epoch_time = time.perf_counter() - epoch_start

            improved = False
            if collapse_ok:
                improved = _is_better_checkpoint(
                    curr_aucpr=goal_aucpr,
                    curr_skill=skill_score,
                    curr_logloss=float(val_m.get('token_logloss', float('nan'))),
                    best_aucpr=best_aucpr,
                    best_skill=best_skill_score,
                    best_logloss=best_logloss,
                    aucpr_delta=aucpr_patience_delta
                )
            if collapse_ok:
                collapse_counter = 0
            else:
                collapse_counter += 1
                print(f'  âš  Collapse guard active: rare_mass_ok={rare_mass_ok}, no_goal_ok={no_goal_ok} (mean_prob_2={no_goal_prob:.4f})')
                if collapse_counter >= collapse_patience and loss_mode in {'focal', 'focal_balanced'}:
                    print('  âš  Persistent no-goal collapse detected. Auto-switching loss to ce_balanced for stability.')
                    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=alpha, label_smoothing=label_smoothing)
                    loss_mode = 'ce_balanced'
                    collapse_counter = 0

            if writer is not None:
                writer.add_scalar('train/loss', tr_loss, epoch)
                writer.add_scalar('val/00_aucpr_goal_mean', goal_aucpr, epoch)
                writer.add_scalar('val/01_skill_score', skill_score, epoch)
                writer.add_scalar('val/token_logloss', val_m['token_logloss'], epoch)
                writer.add_scalar('val/mean_prob_0', val_m.get('mean_prob_0', 0.0), epoch)
                writer.add_scalar('val/mean_prob_1', val_m.get('mean_prob_1', 0.0), epoch)
                writer.add_scalar('val/mean_prob_2', val_m.get('mean_prob_2', 0.0), epoch)
                writer.add_scalar('val/rare_prob_mass', rare_prob_mass, epoch)
                writer.add_scalar('val/eval_duplicate_rate', val_m.get('eval_duplicate_rate', 0.0), epoch)
                writer.add_scalar('val/top5_actor_precision', val_m.get('top5_actor_precision', 0.0), epoch)
                writer.add_scalar('val/top5_actor_recall', val_m.get('top5_actor_recall', 0.0), epoch)
                writer.add_scalar('val/top5_opp_precision', val_m.get('top5_opp_precision', 0.0), epoch)
                writer.add_scalar('val/top5_opp_recall', val_m.get('top5_opp_recall', 0.0), epoch)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

            print(
                f'Epoch {epoch + 1:02d}/{total_epochs:02d} ({epoch_time:.1f}s) | '
                f'train_loss={tr_loss:.4f} | val_logloss={val_m["token_logloss"]:.4f} | '
                f'aucpr_goal_mean={goal_aucpr:.4f} | skill={skill_score:.6f} | '
                f'rare_prob_mass={rare_prob_mass:.4f} ({"OK" if rare_mass_ok else "LOW"}) | '
                f'mean_prob_2={no_goal_prob:.4f} ({"OK" if no_goal_ok else "HIGH"})'
            )
            print(
                f"  Mean prob by class: [0:{val_m.get('mean_prob_0', 0.0):.4f}, "
                f"1:{val_m.get('mean_prob_1', 0.0):.4f}, 2:{val_m.get('mean_prob_2', 0.0):.4f}]"
            )
            print(
                f"  Top5% actor -> precision={val_m.get('top5_actor_precision', 0.0):.4f}, recall={val_m.get('top5_actor_recall', 0.0):.4f} | "
                f"Top5% opp -> precision={val_m.get('top5_opp_precision', 0.0):.4f}, recall={val_m.get('top5_opp_recall', 0.0):.4f}"
            )
            print(
                f"  Baseline entropy={val_baseline_entropy:.6f} | "
                f"Skill Score = 1 - (logloss / baseline_entropy) = {skill_score:.6f}"
            )
            print(
                f"  Eval de-dup: raw={val_m.get('n_eval_tokens_raw', 0):,}, dedup={val_m.get('n_eval_tokens_dedup', 0):,}, "
                f"dup_rate={val_m.get('eval_duplicate_rate', 0.0):.2%}"
            )

            fold_history.append({
                'epoch': epoch,
                'train_loss': tr_loss,
                'val_loss': val_m['val_loss'],
                'val_logloss': val_m['token_logloss'],
                'val_aucpr_goal_mean': goal_aucpr,
                'skill_score': skill_score,
                'rare_prob_mass': rare_prob_mass,
                'mean_prob_2': no_goal_prob,
                'rare_mass_ok': bool(rare_mass_ok),
                'no_goal_ok': bool(no_goal_ok),
                'eval_duplicate_rate': val_m.get('eval_duplicate_rate', float('nan')),
                'time_train_s': float(train_s),
                'time_infer_s': float(infer_s),
                'time_dedup_s': float(dedup_s),
                'time_metrics_s': float(metrics_s),
                'infer_model_forward_s': float(infer_stats.get('model_forward_s', float('nan'))),
                'infer_assembly_s': float(infer_stats.get('assembly_s', float('nan'))),
                'infer_concat_s': float(infer_stats.get('concat_s', float('nan')))
            })

            checkpoint_s = 0.0
            if cfg.save_last_checkpoint:
                ckpt_t0 = time.perf_counter()
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_epoch': best_epoch,
                    'best_aucpr_goal_mean': best_aucpr,
                    'best_skill_score': best_skill_score,
                    'best_logloss': best_logloss,
                    'patience_counter': patience_counter,
                    'history': fold_history,
                    'config': cfg_fold
                }
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint_data, last_ckpt_path)
                checkpoint_s += (time.perf_counter() - ckpt_t0)

            if improved:
                best_epoch = epoch
                best_aucpr = goal_aucpr
                best_skill_score = skill_score
                best_logloss = float(val_m.get('token_logloss', float('nan')))
                best_state = copy.deepcopy(model.state_dict())

                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_logloss': best_logloss,
                    'val_aucpr_goal_mean': best_aucpr,
                    'baseline_entropy': val_baseline_entropy,
                    'skill_score': best_skill_score,
                    'rare_prob_mass': rare_prob_mass,
                    'selection_metric': 'aucpr_goal_mean_then_skill_then_logloss_with_collapse_floor',
                    'config': cfg_fold
                }
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                best_ckpt_t0 = time.perf_counter()
                torch.save(checkpoint_data, best_ckpt_path)
                checkpoint_s += (time.perf_counter() - best_ckpt_t0)
                patience_counter = 0
            else:
                if (epoch + 1) <= max(0, early_stop_warmup):
                    patience_counter = 0
                else:
                    patience_counter += 1

            if scheduler is not None:
                scheduler.step()

            if getattr(cfg, 'print_epoch_timing_breakdown', True):
                print(
                    f"  Timing (s): train={train_s:.2f} | infer={infer_s:.2f} "
                    f"(model={infer_stats.get('model_forward_s', 0.0):.2f}, assemble={infer_stats.get('assembly_s', 0.0):.2f}, concat={infer_stats.get('concat_s', 0.0):.2f}) "
                    f"| dedup={dedup_s:.2f} | metrics={metrics_s:.2f} | checkpoint={checkpoint_s:.2f}"
                )

            if patience_counter >= cfg.early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        if best_ckpt_path.exists():
            best_checkpoint = _torch_load_checkpoint_compat(best_ckpt_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
        elif best_state is not None:
            model.load_state_dict(best_state)
        else:
            # Allow continuation for short/smoke runs where strict selection criteria
            # may never emit a best checkpoint within limited epochs.
            print(
                'Warning: no best checkpoint available for OOF inference after fold training. '
                f'Proceeding with current in-memory weights for fold {fold_idx + 1}. '
                f'Expected checkpoint path: {best_ckpt_path}'
            )

        if writer is not None:
            writer.add_hparams(
                {
                    'd_model': cfg_fold.d_model,
                    'n_heads': cfg_fold.n_heads,
                    'n_layers': cfg_fold.n_layers,
                    'batch_size': cfg_fold.batch_size,
                    'lr': cfg_fold.lr,
                    'dropout': cfg_fold.dropout,
                    'loss_mode': loss_mode,
                    'focal_gamma': focal_gamma,
                    'label_smoothing': label_smoothing,
                    'selection_metric': 'aucpr_goal_mean_then_skill_then_logloss_with_collapse_floor'
                },
                {
                    'best_aucpr_goal_mean': best_aucpr,
                    'best_skill_score': best_skill_score,
                    'best_epoch': best_epoch + 1
                }
            )
            writer.close()

        fold_pred_raw = infer_tokens(model, val_loader, fold_idx)
        fold_pred, fold_dedup_audit = _deduplicate_event_predictions(fold_pred_raw, context_label=f'oof_fold_{fold_idx}')
        fold_pred = _attach_source_tensor_columns(fold_pred, source_tensor_prepared)
        fold_pred['model_variant'] = model_variant
        fold_pred['variant_name'] = canonical_variant
        all_oof.append(fold_pred)

        fm = _metrics_from_prediction_frame(fold_pred, detailed=True)
        fm['fold'] = fold_idx
        fm['model_variant'] = model_variant
        fm['variant_name'] = canonical_variant
        fm['best_epoch'] = best_epoch + 1
        fm['best_aucpr_goal_mean'] = best_aucpr
        fm['best_skill_score'] = best_skill_score
        fm['best_logloss'] = best_logloss
        fm['baseline_entropy'] = val_baseline_entropy
        fm['loss_mode'] = loss_mode
        fm['focal_gamma'] = focal_gamma
        fm['loss_alpha_0'] = float(alpha_np[0])
        fm['loss_alpha_1'] = float(alpha_np[1])
        fm['loss_alpha_2'] = float(alpha_np[2])
        fm['oof_raw_rows'] = int(fold_dedup_audit['raw_rows'])
        fm['oof_dedup_rows'] = int(fold_dedup_audit['dedup_rows'])
        fm['oof_duplicate_rate'] = float(fold_dedup_audit['duplicate_rate'])
        fold_metrics.append(fm)

    if len(fold_metrics) == 0:
        raise RuntimeError('All folds were skipped or failed before producing metrics; no OOF outputs were generated.')

    preds = pd.concat(all_oof, ignore_index=True) if len(all_oof) else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    if len(metrics_df):
        metrics_df = metrics_df.sort_values('fold').reset_index(drop=True)
    return preds, metrics_df


print('Loaded full-game CV pipeline with de-dup metrics, rare-goal checkpointing, and pre-training gates')

# Lean metric schema override (no F1 / no >0.5-threshold prediction metrics)
PHASE6_RELEVANT_METRIC_KEYS = [
    'token_logloss',
    'aucpr_class_0',
    'aucpr_class_1',
    'aucpr_goal_mean',
    'mean_prob_0',
    'mean_prob_1',
    'mean_prob_2',
    'top5_actor_precision',
    'top5_actor_recall',
    'top5_opp_precision',
    'top5_opp_recall',
    'n_eval_tokens'
]

def compute_token_metrics(y_true: np.ndarray, y_prob: np.ndarray, detailed: bool = False) -> Dict[str, float]:
    y_prob = _normalize_prob_rows(y_prob)

    metrics = {
        'token_logloss': _safe_logloss(y_true, y_prob)
    }

    mean_probs = y_prob.mean(axis=0) if len(y_prob) else np.array([0.0, 0.0, 0.0], dtype=np.float64)
    metrics['mean_prob_0'] = float(mean_probs[0])
    metrics['mean_prob_1'] = float(mean_probs[1])
    metrics['mean_prob_2'] = float(mean_probs[2])

    for class_idx in [0, 1]:
        binary_true = (y_true == class_idx).astype(np.int64)
        if binary_true.min() == binary_true.max():
            ap = float('nan')
        else:
            ap = float(average_precision_score(binary_true, y_prob[:, class_idx]))
        metrics[f'aucpr_class_{class_idx}'] = ap

    rare_auc = [metrics.get('aucpr_class_0', float('nan')), metrics.get('aucpr_class_1', float('nan'))]
    rare_auc = [x for x in rare_auc if np.isfinite(x)]
    metrics['aucpr_goal_mean'] = float(np.mean(rare_auc)) if len(rare_auc) else float('nan')

    return metrics


def _metrics_from_prediction_frame(pred_df: pd.DataFrame, detailed: bool = True) -> Dict[str, float]:
    default_vals = {k: float('nan') for k in PHASE6_RELEVANT_METRIC_KEYS}
    default_vals['n_eval_tokens'] = 0

    if len(pred_df) == 0:
        return default_vals

    keep_cols = ['target', 'P_actor_goal', 'P_opp_goal', 'P_no_goal']
    for c in keep_cols:
        if c not in pred_df.columns:
            raise RuntimeError(f'Missing required prediction column for metrics: {c}')

    valid_rows = pred_df[keep_cols].dropna()
    if len(valid_rows) == 0:
        return default_vals

    y_true = valid_rows['target'].astype(int).to_numpy()
    y_prob = valid_rows[['P_actor_goal', 'P_opp_goal', 'P_no_goal']].to_numpy()
    y_prob = _normalize_prob_rows(y_prob)

    metrics = compute_token_metrics(y_true, y_prob, detailed=False)
    metrics['n_eval_tokens'] = int(len(y_true))

    actor_top = _top_bucket_precision_recall(y_true, y_prob[:, 0], class_idx=0, top_pct=5.0)
    opp_top = _top_bucket_precision_recall(y_true, y_prob[:, 1], class_idx=1, top_pct=5.0)
    metrics['top5_actor_precision'] = float(actor_top['precision'])
    metrics['top5_actor_recall'] = float(actor_top['recall'])
    metrics['top5_opp_precision'] = float(opp_top['precision'])
    metrics['top5_opp_recall'] = float(opp_top['recall'])

    pruned = {k: metrics.get(k, default_vals.get(k, float('nan'))) for k in PHASE6_RELEVANT_METRIC_KEYS}
    return pruned

print('âœ“ Lean metric schema active (actor/opp-only names)')


def _apply_transformer_xt_paths(base_dir: Path, run_label: str, tensorboard_dir_override: Optional[Path] = None) -> dict:
    from sprint_week_utils import TransformerXTPaths

    paths = TransformerXTPaths(base_dir=base_dir, run_label=run_label)
    paths.ensure_all()

    config._input_dir_override = _pipeline_phase_dir(base_dir, "phase3")
    config._tracking_input_dir_override = _pipeline_phase_dir(base_dir, "phase2")
    config._models_dir_override = paths.checkpoints_dir
    config._results_dir_override = paths.run_results_dir
    config._tensorboard_dir_override = (
        Path(tensorboard_dir_override)
        if tensorboard_dir_override is not None
        else base_dir / "TensorBoard" / "phase6_optimus_reim"
    )

    OptimusReimConfig.input_dir = property(lambda self: Path(getattr(self, "_input_dir_override", _pipeline_phase_dir(Path(self.base_dir), "phase3"))))
    OptimusReimConfig.tracking_input_dir = property(lambda self: Path(getattr(self, "_tracking_input_dir_override", _pipeline_phase_dir(Path(self.base_dir), "phase2"))))
    OptimusReimConfig.models_dir = property(lambda self: Path(getattr(self, "_models_dir_override", Path(self.base_dir) / "Models" / "Transformer_OptimusReim")))
    OptimusReimConfig.results_dir = property(lambda self: Path(getattr(self, "_results_dir_override", Path(self.base_dir) / "Results" / "phase6_optimus_reim")))
    OptimusReimConfig.tensorboard_dir = property(lambda self: Path(getattr(self, "_tensorboard_dir_override", Path(self.base_dir) / "TensorBoard" / "phase6_optimus_reim")))

    for p in [config.models_dir, config.results_dir, config.tensorboard_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": str(config.results_dir),
        "models_dir": str(config.models_dir),
        "tensorboard_dir": str(config.tensorboard_dir),
    }


def _apply_shared_training_knobs() -> None:
    config.num_epochs = 100
    config.lr = 8e-5
    config.early_stopping_patience = 10
    config.resume_from_checkpoint = True
    config.early_stop_mode = "skill_or_aucpr"
    config.early_stop_warmup = 10
    config.aucpr_patience_delta = 1e-4
    config.skill_tolerance = 0.01
    config.loss_mode = "ce"
    config.focal_gamma = 1.5
    config.loss_balance_power = 0.35
    config.loss_max_ratio = 3.0
    config.label_smoothing = 0.005
    config.loss_fallback_to_ce_balanced = True
    config.rare_class_prob_mass_floor = 0.025
    config.no_goal_prob_ceiling = 0.985
    config.collapse_patience = 4
    config.use_tensorboard = True


def _print_run_config(label: str) -> None:
    print(f"Model {label} run config:")
    print(f"  num_epochs: {config.num_epochs}")
    print(f"  lr: {config.lr}")
    print(f"  early_stopping_patience: {config.early_stopping_patience}")
    print(f"  resume_from_checkpoint: {config.resume_from_checkpoint}")
    print(f"  early_stop_mode: {getattr(config, 'early_stop_mode', 'n/a')} | early_stop_warmup: {getattr(config, 'early_stop_warmup', 'n/a')}")
    print(f"  aucpr_patience_delta: {getattr(config, 'aucpr_patience_delta', 'n/a')} | skill_tolerance: {getattr(config, 'skill_tolerance', 'n/a')}")
    print(f"  d_model: {config.d_model} | n_heads: {config.n_heads} | n_layers: {config.n_layers}")
    print(f"  batch_size: {config.batch_size} | eval_batch_size: {config.eval_batch_size}")
    print(
        "  num_workers: {workers} | prefetch_factor: {prefetch} | pin_memory: {pin} | persistent_workers: {persistent}".format(
            workers=getattr(config, 'num_workers', 'n/a'),
            prefetch=getattr(config, 'prefetch_factor', 'n/a'),
            pin=getattr(config, 'pin_memory', 'n/a'),
            persistent=getattr(config, 'persistent_workers', 'n/a'),
        )
    )
    print(f"  debug_validate_items: {getattr(config, 'debug_validate_items', 'n/a')}")
    print(f"  print_epoch_timing_breakdown: {getattr(config, 'print_epoch_timing_breakdown', 'n/a')}")
    print(f"  weight_decay: {getattr(config, 'weight_decay', 'n/a')} | dropout: {getattr(config, 'dropout', 'n/a')}")
    print(f"  loss_mode: {getattr(config, 'loss_mode', 'n/a')} | focal_gamma: {getattr(config, 'focal_gamma', 'n/a')}")
    print(f"  loss_balance_power: {getattr(config, 'loss_balance_power', 'n/a')} | loss_max_ratio: {getattr(config, 'loss_max_ratio', 'n/a')}")
    print(f"  label_smoothing: {getattr(config, 'label_smoothing', 'n/a')}")
    print(
        "  gnn_emb_dropout: {dropout} | gnn_emb_noise_std: {noise} | gnn_emb_mask_rate: {mask} | gnn_proj_dim: {proj} | gnn_bottleneck_dim: {bottleneck}".format(
            dropout=getattr(config, 'gnn_emb_dropout', 'n/a'),
            noise=getattr(config, 'gnn_emb_noise_std', 'n/a'),
            mask=getattr(config, 'gnn_emb_mask_rate', 'n/a'),
            proj=getattr(config, 'gnn_proj_dim', 'n/a'),
            bottleneck=getattr(config, 'gnn_bottleneck_dim', 'n/a')
        )
    )
    print(f"  gnn_node_feature_dim: {getattr(config, 'gnn_node_feature_dim', 'n/a')}")
    print(
        "  gnn_graph_variant: {variant} | gnn_feats_path: {feats} | gnn_adj_path: {adj} | gnn_mask_path: {mask}".format(
            variant=getattr(config, 'gnn_graph_variant', 'n/a'),
            feats=getattr(config, 'gnn_feats_path', 'n/a'),
            adj=getattr(config, 'gnn_adj_path', 'n/a'),
            mask=getattr(config, 'gnn_mask_path', 'n/a'),
        )
    )
    print(f"  rare_class_prob_mass_floor: {getattr(config, 'rare_class_prob_mass_floor', 'n/a')}")
    print(f"  no_goal_prob_ceiling: {getattr(config, 'no_goal_prob_ceiling', 'n/a')}")
    print(f"  collapse_patience: {getattr(config, 'collapse_patience', 'n/a')}")


def _write_run_manifest(base_dir: Path, run_label: str, variant: str, path_info: dict, metrics: pd.DataFrame) -> None:
    from sprint_week_utils import write_json

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_label": run_label,
        "variant": variant,
        "results_dir": path_info["results_dir"],
        "models_dir": path_info["models_dir"],
        "tensorboard_dir": path_info["tensorboard_dir"],
        "metrics_rows": int(len(metrics)) if isinstance(metrics, pd.DataFrame) else 0,
    }
    logs_dir = Path(path_info["results_dir"]) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(logs_dir / f"training_manifest_{variant}.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 6 Optimus Reim transformer")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--model-variant", type=str, default="events_only", choices=["events_only", "with_tracking", "both", "A", "B"])
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=None)
    parser.add_argument("--debug-validate-items", dest="debug_validate_items", action="store_true")
    parser.add_argument("--no-debug-validate-items", dest="debug_validate_items", action="store_false")
    parser.set_defaults(debug_validate_items=None)
    parser.add_argument("--timing-breakdown", dest="print_epoch_timing_breakdown", action="store_true")
    parser.add_argument("--no-timing-breakdown", dest="print_epoch_timing_breakdown", action="store_false")
    parser.set_defaults(print_epoch_timing_breakdown=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--loss-mode", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stop-mode", type=str, default=None)
    parser.add_argument("--early-stop-warmup", type=int, default=None)
    parser.add_argument("--aucpr-patience-delta", type=float, default=None)
    parser.add_argument("--skill-tolerance", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--loss-balance-power", type=float, default=None)
    parser.add_argument("--loss-max-ratio", type=float, default=None)
    parser.add_argument("--rare-class-prob-mass-floor", type=float, default=None)
    parser.add_argument("--no-goal-prob-ceiling", type=float, default=None)
    parser.add_argument("--collapse-patience", type=int, default=None)
    parser.add_argument("--gnn-emb-dropout", type=float, default=None)
    parser.add_argument("--gnn-emb-noise-std", type=float, default=None)
    parser.add_argument("--gnn-emb-mask-rate", type=float, default=None)
    parser.add_argument("--gnn-proj-dim", type=int, default=None)
    parser.add_argument("--gnn-bottleneck-dim", type=int, default=None)
    parser.add_argument("--gnn-embedding-dim", type=int, default=None)
    parser.add_argument("--gnn-graph-variant", type=str, default=None)
    parser.add_argument("--gnn-feats-path", type=Path, default=None)
    parser.add_argument("--gnn-adj-path", type=Path, default=None)
    parser.add_argument("--gnn-mask-path", type=Path, default=None)
    parser.add_argument(
        "--smoke-preflight-only",
        action="store_true",
        help="Run startup data assembly checks, then exit before training.",
    )
    return parser.parse_args()


def _recommended_num_workers() -> int:
    cpu_count = int(os.cpu_count() or 4)
    if sys.platform.startswith("win"):
        # Keep worker count conservative on Windows spawn to avoid process thrash.
        return max(2, min(4, cpu_count // 2 if cpu_count > 2 else 2))
    return max(2, min(8, cpu_count // 2 if cpu_count > 2 else 2))


def _apply_runtime_loader_defaults(args: argparse.Namespace) -> None:
    if args.num_workers is None:
        config.num_workers = int(_recommended_num_workers())
    if args.prefetch_factor is None:
        config.prefetch_factor = int(max(2, int(getattr(config, 'prefetch_factor', 2))))
    if args.pin_memory is None:
        config.pin_memory = bool(device.type in {'cuda', 'xpu'})
    if args.persistent_workers is None:
        config.persistent_workers = bool(config.num_workers > 0)


def run_variant(variant: str, source_tensor_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_variant = "A" if variant in {"events_only", "A"} else "B"
    _print_run_config(model_variant)
    preds, metrics = run_cv_phase6(df_events, config, model_variant=model_variant, source_tensor_df=source_tensor_df)
    return preds, metrics


def main() -> None:
    # Ensure Windows subprocess entry is initialized before any worker spawns.
    if sys.platform.startswith("win"):
        mp.freeze_support()

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_dir = args.base_dir.resolve()
    run_label = args.run_label

    _apply_shared_training_knobs()
    if args.num_epochs is not None:
        config.num_epochs = int(args.num_epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.eval_batch_size is not None:
        config.eval_batch_size = int(args.eval_batch_size)
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)
    if args.prefetch_factor is not None:
        config.prefetch_factor = int(args.prefetch_factor)
    if args.pin_memory is not None:
        config.pin_memory = bool(args.pin_memory)
    if args.persistent_workers is not None:
        config.persistent_workers = bool(args.persistent_workers)
    if args.debug_validate_items is not None:
        config.debug_validate_items = bool(args.debug_validate_items)
    if args.print_epoch_timing_breakdown is not None:
        config.print_epoch_timing_breakdown = bool(args.print_epoch_timing_breakdown)
    if args.lr is not None:
        config.lr = float(args.lr)
    if args.d_model is not None:
        config.d_model = int(args.d_model)
    if args.n_heads is not None:
        config.n_heads = int(args.n_heads)
    if args.n_layers is not None:
        config.n_layers = int(args.n_layers)
    if args.weight_decay is not None:
        config.weight_decay = float(args.weight_decay)
    if args.dropout is not None:
        config.dropout = float(args.dropout)
    if args.max_seq_length is not None:
        if int(args.max_seq_length) <= 0:
            raise ValueError(f"--max-seq-length must be > 0, got {args.max_seq_length}")
        config.max_seq_length = int(args.max_seq_length)
    if args.window_stride is not None:
        if int(args.window_stride) <= 0:
            raise ValueError(f"--window-stride must be > 0, got {args.window_stride}")
        config.window_stride = int(args.window_stride)
    if args.resume:
        config.resume_from_checkpoint = True
    if args.loss_mode is not None:
        config.loss_mode = str(args.loss_mode)
    if args.label_smoothing is not None:
        config.label_smoothing = float(args.label_smoothing)
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = int(args.early_stopping_patience)
    if args.early_stop_mode is not None:
        config.early_stop_mode = str(args.early_stop_mode)
    if args.early_stop_warmup is not None:
        config.early_stop_warmup = int(args.early_stop_warmup)
    if args.aucpr_patience_delta is not None:
        config.aucpr_patience_delta = float(args.aucpr_patience_delta)
    if args.skill_tolerance is not None:
        config.skill_tolerance = float(args.skill_tolerance)
    if args.focal_gamma is not None:
        config.focal_gamma = float(args.focal_gamma)
    if args.loss_balance_power is not None:
        config.loss_balance_power = float(args.loss_balance_power)
    if args.loss_max_ratio is not None:
        config.loss_max_ratio = float(args.loss_max_ratio)
    if args.rare_class_prob_mass_floor is not None:
        config.rare_class_prob_mass_floor = float(args.rare_class_prob_mass_floor)
    if args.no_goal_prob_ceiling is not None:
        config.no_goal_prob_ceiling = float(args.no_goal_prob_ceiling)
    if args.collapse_patience is not None:
        config.collapse_patience = int(args.collapse_patience)
    if args.gnn_emb_dropout is not None:
        config.gnn_emb_dropout = float(args.gnn_emb_dropout)
    if args.gnn_emb_noise_std is not None:
        config.gnn_emb_noise_std = float(args.gnn_emb_noise_std)
    if args.gnn_emb_mask_rate is not None:
        config.gnn_emb_mask_rate = float(args.gnn_emb_mask_rate)
    if args.gnn_proj_dim is not None:
        config.gnn_proj_dim = int(args.gnn_proj_dim)
    if args.gnn_bottleneck_dim is not None:
        config.gnn_bottleneck_dim = int(args.gnn_bottleneck_dim)
    if args.gnn_embedding_dim is not None:
        config.gnn_embedding_dim = int(args.gnn_embedding_dim)
    if args.gnn_graph_variant is not None:
        config.gnn_graph_variant = str(args.gnn_graph_variant)
    if args.gnn_feats_path is not None:
        config.gnn_feats_path = Path(args.gnn_feats_path)
    if args.gnn_adj_path is not None:
        config.gnn_adj_path = Path(args.gnn_adj_path)
    if args.gnn_mask_path is not None:
        config.gnn_mask_path = Path(args.gnn_mask_path)

    _apply_runtime_loader_defaults(args)

    if config.gnn_emb_dropout < 0 or config.gnn_emb_dropout > 1:
        raise ValueError(f"gnn_emb_dropout must be within [0, 1], got {config.gnn_emb_dropout}")
    if config.gnn_emb_mask_rate < 0 or config.gnn_emb_mask_rate > 1:
        raise ValueError(f"gnn_emb_mask_rate must be within [0, 1], got {config.gnn_emb_mask_rate}")
    if config.gnn_emb_noise_std < 0:
        raise ValueError(f"gnn_emb_noise_std must be >= 0, got {config.gnn_emb_noise_std}")
    if config.gnn_embedding_dim <= 0:
        raise ValueError(f"gnn_embedding_dim must be > 0, got {config.gnn_embedding_dim}")
    if config.gnn_bottleneck_dim < 0:
        raise ValueError(f"gnn_bottleneck_dim must be >= 0, got {config.gnn_bottleneck_dim}")
    if config.gnn_proj_dim > 0 and config.gnn_bottleneck_dim > 0:
        raise ValueError('Use either gnn_proj_dim or gnn_bottleneck_dim, not both.')
    if str(config.gnn_graph_variant).strip().lower() not in {'base', 'actor_rel', 'actor_rel_ctx', 'actor_emph'}:
        raise ValueError(f"gnn_graph_variant must be one of {{'base', 'actor_rel', 'actor_rel_ctx', 'actor_emph'}}, got {config.gnn_graph_variant}")

    graph_override_flags = [
        config.gnn_feats_path is not None,
        config.gnn_adj_path is not None,
        config.gnn_mask_path is not None,
    ]
    if any(graph_override_flags) and not all(graph_override_flags):
        raise ValueError(
            'When overriding graph paths, provide all of --gnn-feats-path, --gnn-adj-path, and --gnn-mask-path.'
        )
    if config.dropout < 0 or config.dropout >= 1:
        raise ValueError(f"dropout must be within [0, 1), got {config.dropout}")
    if config.d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {config.d_model}")
    if config.n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {config.n_heads}")
    if config.n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {config.n_layers}")
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {config.num_workers}")
    if config.prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be > 0, got {config.prefetch_factor}")
    if config.d_model % config.n_heads != 0:
        raise ValueError(
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

    tensorboard_dir_override = None
    if args.tensorboard_dir is not None:
        tensorboard_dir_override = args.tensorboard_dir
        if not tensorboard_dir_override.is_absolute():
            tensorboard_dir_override = (base_dir / tensorboard_dir_override).resolve()

    path_info = _apply_transformer_xt_paths(
        base_dir,
        run_label,
        tensorboard_dir_override=tensorboard_dir_override,
    )

    global df_events
    sample_mode = bool(args.sample_rows and args.sample_rows > 0)
    if sample_mode:
        df_events = df_events.head(int(args.sample_rows)).copy()
        print(f"Sample mode enabled: using first {len(df_events):,} rows")

    if bool(args.smoke_preflight_only):
        smoke_summary = {
            'generated_at_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            'run_label': run_label,
            'model_variant': args.model_variant,
            'smoke_preflight_only': True,
            'results_dir': str(config.results_dir),
            'models_dir': str(config.models_dir),
            'rows_loaded': int(len(df_events)),
            'event_vocab_size': int(df_events['event_type_id'].nunique()) if 'event_type_id' in df_events.columns else 0,
        }
        logs_dir = config.results_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        smoke_summary_path = logs_dir / 'phase6_smoke_preflight_summary.json'
        with smoke_summary_path.open('w', encoding='utf-8') as f:
            json.dump(smoke_summary, f, indent=2)
        print(f'Smoke preflight summary saved: {smoke_summary_path}')
        print('Smoke preflight-only mode complete. Skipping training and ensemble inference.')
        return

    variants = []
    if args.model_variant in {"events_only", "A"}:
        variants = ["events_only"]
    elif args.model_variant in {"with_tracking", "B"}:
        variants = ["with_tracking"]
    else:
        variants = ["events_only", "with_tracking"]

    all_preds = []
    all_metrics = []

    source_tensor_for_oof = _prepare_source_tensor_for_oof(tensor_source_df)

    for variant in variants:
        preds, metrics = run_variant(variant, source_tensor_for_oof)

        all_preds.append(preds)
        all_metrics.append(metrics)

        _write_run_manifest(base_dir, run_label, variant, path_info, metrics)

    combined_preds = pd.concat([x for x in all_preds if isinstance(x, pd.DataFrame) and not x.empty], ignore_index=True) if any(isinstance(x, pd.DataFrame) and not x.empty for x in all_preds) else pd.DataFrame()
    combined_metrics = pd.concat([x for x in all_metrics if isinstance(x, pd.DataFrame) and not x.empty], ignore_index=True) if any(isinstance(x, pd.DataFrame) and not x.empty for x in all_metrics) else pd.DataFrame()

    if not combined_preds.empty:
        raw_oof_contract_audit = _audit_raw_oof_source_contract(combined_preds, source_tensor_for_oof)
        print(f'Raw OOF source-column contract audit: {raw_oof_contract_audit}')
        combined_preds.to_parquet(config.results_dir / "raw_oof_predictions.parquet", index=False)
    if not combined_metrics.empty:
        combined_metrics.to_parquet(config.results_dir / "metrics_phase6_training.parquet", index=False)

    print("Training complete.")
    print(f"Results dir: {config.results_dir}")
    print(f"Models dir: {config.models_dir}")
    print(
        "Raw OOF export complete. Run scripts/postprocess_phase6_outputs.py "
        f"--base-dir {base_dir} --run-label {run_label}"
    )


if __name__ == "__main__":
    main()
