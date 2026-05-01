from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ATTACK_NET_X = 89.0
ATTACK_NET_Y = 0.0

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json
    from .run_resolver import require_artifacts_exist, resolve_run_label
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json
    from run_resolver import require_artifacts_exist, resolve_run_label

def _knn_adjacency(xy: np.ndarray, k: int = 3) -> np.ndarray:
    n = int(xy.shape[0])
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


def _tracking_row_to_graph_cpu(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pure numpy conversion: no torch/xpu/cuda usage in exporter math.
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

NODE_X = 0
NODE_Y = 1
NODE_VX = 2
NODE_VY = 3
NODE_DIST = 4
NODE_SIDE = 5
NODE_POSSESS = 6
NODE_ACTOR = 7

ACTOR_REL_CTX_FEATURE_NAMES = [
    "x_rel_to_actor",
    "y_rel_to_actor",
    "vx_rel_to_actor",
    "vy_rel_to_actor",
    "dist_to_actor",
    "dist_to_net_abs",
    "angle_to_net_abs",
    "is_teammate",
    "is_possessing_team",
    "is_primary_actor",
]

GRAPH_VARIANTS = {
    "base": {
        "name": "gnn_graph_base",
        "feats_attr": "phase3_gnn_base_feats_output",
        "adj_attr": "phase3_gnn_base_adj_output",
        "mask_attr": "phase3_gnn_base_mask_output",
        "description": "baseline",
    },
    "actor_rel": {
        "name": "gnn_graph_actor_rel",
        "feats_attr": "phase3_gnn_actor_rel_feats_output",
        "adj_attr": "phase3_gnn_actor_rel_adj_output",
        "mask_attr": "phase3_gnn_actor_rel_mask_output",
        "description": "actor-relative coordinates",
    },
    "actor_rel_ctx": {
        "name": "gnn_graph_actor_rel_ctx",
        "feats_attr": "phase3_gnn_actor_rel_ctx_feats_output",
        "adj_attr": "phase3_gnn_actor_rel_ctx_adj_output",
        "mask_attr": "phase3_gnn_actor_rel_ctx_mask_output",
        "description": "actor-relative coordinates with net and team context",
    },
}

THREAT_NEIGHBOR_SPEC = (
    ("tm", 1),
    ("tm", 2),
    ("opp", 1),
    ("opp", 2),
)


def _to_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _threat_feature_names() -> list[str]:
    names = [
        "actor_speed",
        "actor_distance_to_net",
        "actor_angle_to_net",
        "actor_is_missing",
    ]
    for role, slot in THREAT_NEIGHBOR_SPEC:
        stem = f"{role}_{slot}"
        names.extend(
            [
                f"{stem}_rel_x",
                f"{stem}_rel_y",
                f"{stem}_vx_rel",
                f"{stem}_vy_rel",
                f"{stem}_distance_to_actor",
                f"{stem}_angle_to_actor",
                f"{stem}_speed_rel",
                f"{stem}_is_missing",
            ]
        )
    return names


def _safe_max_neighbor_distance(df_tracking_aligned: pd.DataFrame) -> float:
    observed: list[np.ndarray] = []
    for role, slot in THREAT_NEIGHBOR_SPEC:
        dist_col = f"{role}_{slot}_distance"
        present_col = f"{role}_{slot}_is_present"
        dist = _to_numeric_series(df_tracking_aligned, dist_col, default=np.nan).to_numpy(dtype=float)
        if present_col in df_tracking_aligned.columns:
            present = _to_numeric_series(df_tracking_aligned, present_col, default=0.0).to_numpy(dtype=float) >= 0.5
            dist = dist[present]
        dist = dist[np.isfinite(dist)]
        if dist.size > 0:
            observed.append(dist)

    if len(observed) == 0:
        return 200.0

    merged = np.concatenate(observed, axis=0)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return 200.0

    max_dist = float(np.max(merged))
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        return 200.0
    return max_dist


def _build_threat_vector_matrix(
    *,
    df_tracking_aligned: pd.DataFrame,
    df_tensor_aligned: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    feature_names = _threat_feature_names()
    n_rows = len(df_tracking_aligned)
    n_features = len(feature_names)
    matrix = np.zeros((n_rows, n_features), dtype=np.float32)

    safe_max_distance = _safe_max_neighbor_distance(df_tracking_aligned)

    actor_vx = _to_numeric_series(df_tracking_aligned, "actor_vx_rel", default=0.0).to_numpy(dtype=float)
    actor_vy = _to_numeric_series(df_tracking_aligned, "actor_vy_rel", default=0.0).to_numpy(dtype=float)
    actor_speed = np.sqrt(np.maximum(actor_vx**2 + actor_vy**2, 0.0))

    dist_col = _first_present_column(df_tensor_aligned, ["distance_to_net_event", "distance_to_net"])
    angle_col = _first_present_column(df_tensor_aligned, ["angle_to_net_event", "angle_to_net"])
    if dist_col is None:
        actor_dist_to_net = np.zeros(n_rows, dtype=float)
    else:
        actor_dist_to_net = _to_numeric_series(df_tensor_aligned, dist_col, default=0.0).to_numpy(dtype=float)
    if angle_col is None:
        actor_angle_to_net = np.zeros(n_rows, dtype=float)
    else:
        actor_angle_to_net = _to_numeric_series(df_tensor_aligned, angle_col, default=0.0).to_numpy(dtype=float)

    if "actor_is_missing" in df_tracking_aligned.columns:
        actor_missing = _to_numeric_series(df_tracking_aligned, "actor_is_missing", default=0.0).to_numpy(dtype=float)
    elif "actor_is_present" in df_tracking_aligned.columns:
        actor_present = _to_numeric_series(df_tracking_aligned, "actor_is_present", default=0.0).to_numpy(dtype=float)
        actor_missing = 1.0 - (actor_present >= 0.5).astype(float)
    else:
        actor_missing = np.zeros(n_rows, dtype=float)

    matrix[:, 0] = np.nan_to_num(actor_speed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    matrix[:, 1] = np.nan_to_num(actor_dist_to_net, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    matrix[:, 2] = np.nan_to_num(actor_angle_to_net, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    matrix[:, 3] = np.nan_to_num(actor_missing, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)

    offset = 4
    for role, slot in THREAT_NEIGHBOR_SPEC:
        stem = f"{role}_{slot}"
        rel_x = _to_numeric_series(df_tracking_aligned, f"{stem}_rel_x", default=0.0).to_numpy(dtype=float)
        rel_y = _to_numeric_series(df_tracking_aligned, f"{stem}_rel_y", default=0.0).to_numpy(dtype=float)
        vx_rel = _to_numeric_series(df_tracking_aligned, f"{stem}_vx_rel", default=0.0).to_numpy(dtype=float)
        vy_rel = _to_numeric_series(df_tracking_aligned, f"{stem}_vy_rel", default=0.0).to_numpy(dtype=float)
        dist = _to_numeric_series(df_tracking_aligned, f"{stem}_distance", default=np.nan).to_numpy(dtype=float)

        if f"{stem}_is_present" in df_tracking_aligned.columns:
            present = _to_numeric_series(df_tracking_aligned, f"{stem}_is_present", default=0.0).to_numpy(dtype=float) >= 0.5
        elif f"{stem}_is_missing" in df_tracking_aligned.columns:
            missing_raw = _to_numeric_series(df_tracking_aligned, f"{stem}_is_missing", default=1.0).to_numpy(dtype=float)
            present = ~(missing_raw >= 0.5)
        else:
            present = np.isfinite(dist) & (dist > 0.0)

        dist_fallback = np.sqrt(np.maximum(rel_x**2 + rel_y**2, 0.0))
        use_fallback = (~np.isfinite(dist)) | (dist < 0.0)
        dist = np.where(use_fallback, dist_fallback, dist)
        angle = np.arctan2(rel_y, rel_x)
        speed = np.sqrt(np.maximum(vx_rel**2 + vy_rel**2, 0.0))

        missing = (~present).astype(float)
        rel_x = np.where(present, rel_x, 0.0)
        rel_y = np.where(present, rel_y, 0.0)
        vx_rel = np.where(present, vx_rel, 0.0)
        vy_rel = np.where(present, vy_rel, 0.0)
        dist = np.where(present, dist, safe_max_distance)
        angle = np.where(present, angle, 0.0)
        speed = np.where(present, speed, 0.0)

        block = np.column_stack(
            [
                rel_x,
                rel_y,
                vx_rel,
                vy_rel,
                dist,
                angle,
                speed,
                missing,
            ]
        ).astype(np.float32, copy=False)
        matrix[:, offset : offset + block.shape[1]] = np.nan_to_num(
            block,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        offset += block.shape[1]

    meta = {
        "rows": int(n_rows),
        "features": int(n_features),
        "safe_max_distance": float(safe_max_distance),
        "distance_source_column": dist_col,
        "angle_source_column": angle_col,
    }
    return matrix, feature_names, meta


def _zscore_standardize_features(
    matrix: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.ndim != 2:
        raise RuntimeError(f"Threat matrix must be 2D. got shape={matrix.shape}")

    mean = matrix.mean(axis=0, dtype=np.float64)
    std = matrix.std(axis=0, dtype=np.float64)
    std_safe = np.where(std < eps, 1.0, std)

    standardized = ((matrix.astype(np.float64, copy=False) - mean) / std_safe).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise RuntimeError("Threat vectors contain non-finite values after z-score standardization.")

    post_mean = standardized.mean(axis=0, dtype=np.float64)
    post_std = standardized.std(axis=0, dtype=np.float64)

    scaler_meta = {
        "eps": float(eps),
        "mean": mean.astype(np.float64).tolist(),
        "std": std.astype(np.float64).tolist(),
        "std_safe": std_safe.astype(np.float64).tolist(),
        "near_constant_feature_count": int((std < eps).sum()),
        "post_mean_abs_max": float(np.max(np.abs(post_mean))) if post_mean.size else 0.0,
        "post_std_min": float(np.min(post_std)) if post_std.size else 0.0,
        "post_std_max": float(np.max(post_std)) if post_std.size else 0.0,
    }
    return standardized, scaler_meta


def _normalize_event_keys(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = ["game_id", "sl_event_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required key columns: {missing}")

    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["sl_event_id"] = pd.to_numeric(out["sl_event_id"], errors="coerce").astype("Int64")

    bad = out["sl_event_id"].isna()
    if bad.any():
        examples = out.loc[bad, ["game_id", "sl_event_id"]].head(5).to_dict("records")
        raise RuntimeError(
            f"{label} has invalid sl_event_id values after numeric coercion. examples={examples}"
        )

    dup = out.duplicated(["game_id", "sl_event_id"], keep=False)
    if dup.any():
        examples = out.loc[dup, ["game_id", "sl_event_id"]].head(5).to_dict("records")
        raise RuntimeError(
            f"{label} contains duplicate (game_id, sl_event_id) keys. examples={examples}"
        )

    return out


def _select_actor_index(feats: np.ndarray, node_mask: np.ndarray) -> int | None:
    active = node_mask
    if not np.any(active):
        return None

    actor_idx = np.where((feats[:, NODE_ACTOR] >= 0.5) & active)[0]
    if len(actor_idx) > 0:
        return int(actor_idx[0])

    poss_idx = np.where((feats[:, NODE_POSSESS] >= 0.5) & active)[0]
    if len(poss_idx) > 0:
        return int(poss_idx[0])

    fallback = np.where(active)[0]
    return int(fallback[0]) if len(fallback) else None


def _apply_actor_relative(feats: np.ndarray, node_mask: np.ndarray, actor_idx: int | None) -> np.ndarray:
    if actor_idx is None:
        return feats

    adjusted = feats.copy()
    anchor = adjusted[actor_idx, [NODE_X, NODE_Y, NODE_VX, NODE_VY]]
    adjusted[:, NODE_X] = adjusted[:, NODE_X] - anchor[0]
    adjusted[:, NODE_Y] = adjusted[:, NODE_Y] - anchor[1]
    adjusted[:, NODE_VX] = adjusted[:, NODE_VX] - anchor[2]
    adjusted[:, NODE_VY] = adjusted[:, NODE_VY] - anchor[3]
    adjusted[:, NODE_DIST] = np.sqrt(adjusted[:, NODE_X] ** 2 + adjusted[:, NODE_Y] ** 2)
    return adjusted


def _build_actor_relative_context_features(
    feats: np.ndarray,
    node_mask: np.ndarray,
    actor_idx: int | None,
) -> np.ndarray:
    if actor_idx is None:
        active = np.where(node_mask)[0]
        actor_idx = int(active[0]) if len(active) else 0

    actor_x = float(feats[actor_idx, NODE_X])
    actor_y = float(feats[actor_idx, NODE_Y])
    actor_vx = float(feats[actor_idx, NODE_VX])
    actor_vy = float(feats[actor_idx, NODE_VY])
    actor_side = float(feats[actor_idx, NODE_SIDE])

    rel_x = feats[:, NODE_X] - actor_x
    rel_y = feats[:, NODE_Y] - actor_y
    rel_vx = feats[:, NODE_VX] - actor_vx
    rel_vy = feats[:, NODE_VY] - actor_vy

    dist_to_actor = np.sqrt(rel_x ** 2 + rel_y ** 2)
    dist_to_net_abs = np.sqrt((feats[:, NODE_X] - ATTACK_NET_X) ** 2 + (feats[:, NODE_Y] - ATTACK_NET_Y) ** 2)
    angle_to_net_abs = np.degrees(np.arctan2(ATTACK_NET_Y - feats[:, NODE_Y], ATTACK_NET_X - feats[:, NODE_X]))

    is_teammate = (np.isclose(feats[:, NODE_SIDE], actor_side)).astype(np.float32)
    is_possessing_team = feats[:, NODE_POSSESS].astype(np.float32, copy=False)
    is_primary_actor = feats[:, NODE_ACTOR].astype(np.float32, copy=False)

    ctx_feats = np.stack(
        [
            rel_x.astype(np.float32, copy=False),
            rel_y.astype(np.float32, copy=False),
            rel_vx.astype(np.float32, copy=False),
            rel_vy.astype(np.float32, copy=False),
            dist_to_actor.astype(np.float32, copy=False),
            dist_to_net_abs.astype(np.float32, copy=False),
            angle_to_net_abs.astype(np.float32, copy=False),
            is_teammate,
            is_possessing_team,
            is_primary_actor,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if len(node_mask):
        inactive = ~node_mask
        if inactive.any():
            ctx_feats[inactive] = 0.0

    return ctx_feats


def _node_feature_dim_for_variant(variant: str) -> int:
    if variant == "actor_rel_ctx":
        return len(ACTOR_REL_CTX_FEATURE_NAMES)
    return 8


def _feature_names_for_variant(variant: str) -> list[str]:
    if variant == "actor_rel_ctx":
        return list(ACTOR_REL_CTX_FEATURE_NAMES)
    return ["x", "y", "vx", "vy", "dist", "side_flag", "is_possessing_team", "is_primary_actor"]


def _assert_actor_rel_ctx_row_invariants(feats: np.ndarray, node_mask: np.ndarray) -> None:
    if feats.shape != (12, len(ACTOR_REL_CTX_FEATURE_NAMES)):
        raise RuntimeError(
            "actor_rel_ctx features must be shaped [12, 10]. "
            f"got={tuple(int(x) for x in feats.shape)}"
        )
    if not np.isfinite(feats).all():
        raise RuntimeError("actor_rel_ctx row has non-finite values.")

    dist_actor = feats[:, 4]
    dist_net = feats[:, 5]
    angle_net = feats[:, 6]
    is_teammate = feats[:, 7]

    active = node_mask.astype(bool)
    if np.any(dist_actor[active] < 0) or np.any(dist_net[active] < 0):
        raise RuntimeError("actor_rel_ctx row has negative distances.")
    if np.any(np.abs(angle_net[active]) > 180.0 + 1e-6):
        raise RuntimeError("actor_rel_ctx row has angle_to_net_abs outside [-180, 180].")
    if np.any((is_teammate[active] != 0.0) & (is_teammate[active] != 1.0)):
        raise RuntimeError("actor_rel_ctx row has non-binary is_teammate values.")


def _align_tracking_to_tensor_order(
    df_tracking: pd.DataFrame,
    df_tensor: pd.DataFrame,
) -> pd.DataFrame:
    tracking = _normalize_event_keys(df_tracking, label="tracking_absolute_pinned.parquet")
    tensor = _normalize_event_keys(df_tensor, label="tensor_ready_dataset.parquet")

    key_cols = ["game_id", "sl_event_id"]
    tracking_index = pd.MultiIndex.from_frame(tracking[key_cols])
    tensor_index = pd.MultiIndex.from_frame(tensor[key_cols])

    indexer = tracking_index.get_indexer(tensor_index)
    if (indexer < 0).any():
        missing_mask = indexer < 0
        missing_count = int(missing_mask.sum())
        examples = tensor.loc[missing_mask, key_cols].head(5).to_dict("records")
        raise RuntimeError(
            "Tracking rows missing for tensor-ready events. "
            f"missing_rows={missing_count:,} examples={examples}"
        )

    aligned = tracking.iloc[indexer].reset_index(drop=True)
    if len(aligned) != len(tensor):
        raise RuntimeError(
            "Aligned tracking row count mismatch with tensor-ready dataset. "
            f"tracking_aligned={len(aligned):,} tensor_ready={len(tensor):,}"
        )

    if not aligned[key_cols].equals(tensor[key_cols].reset_index(drop=True)):
        raise RuntimeError("Aligned tracking keys do not match tensor-ready key order.")

    return aligned


def _build_and_save_graph_arrays(
    *,
    df_tracking_aligned: pd.DataFrame,
    variant: str,
    feats_path: Path,
    adj_path: Path,
    mask_path: Path,
    progress_every: int = 50000,
    flush_every: int = 20000,
) -> dict[str, Any]:
    n_rows = len(df_tracking_aligned)
    node_feat_dim = _node_feature_dim_for_variant(variant)
    if n_rows == 0:
        np.save(feats_path, np.zeros((0, 12, node_feat_dim), dtype=np.float32), allow_pickle=False)
        np.save(adj_path, np.zeros((0, 12, 12), dtype=np.float32), allow_pickle=False)
        np.save(mask_path, np.zeros((0, 12), dtype=np.bool_), allow_pickle=False)
        return {
            "rows": 0,
            "feats_shape": (0, 12, node_feat_dim),
            "adj_shape": (0, 12, 12),
            "mask_shape": (0, 12),
            "dtype_feats": "float32",
            "dtype_adj": "float32",
            "dtype_mask": "bool",
            "feature_names": _feature_names_for_variant(variant),
        }

    first_row = next(df_tracking_aligned.itertuples(index=False))
    first_feats, first_adj, first_mask = _tracking_row_to_graph_cpu(pd.Series(first_row._asdict()))
    first_mask_bool = first_mask >= 0.5
    first_actor_idx = _select_actor_index(first_feats, first_mask_bool)
    if variant == "actor_rel":
        first_feats = _apply_actor_relative(first_feats, first_mask_bool, first_actor_idx)
    elif variant == "actor_rel_ctx":
        first_feats = _build_actor_relative_context_features(first_feats, first_mask_bool, first_actor_idx)
        _assert_actor_rel_ctx_row_invariants(first_feats, first_mask_bool)

    n_nodes = int(first_feats.shape[0])
    n_node_feats = int(first_feats.shape[1])

    feats_mm = np.lib.format.open_memmap(
        feats_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, n_nodes, n_node_feats),
    )
    adj_mm = np.lib.format.open_memmap(
        adj_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, n_nodes, n_nodes),
    )
    mask_mm = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(n_rows, n_nodes),
    )

    for i, row in enumerate(df_tracking_aligned.itertuples(index=False), start=1):
        feats, adj, node_mask = _tracking_row_to_graph_cpu(pd.Series(row._asdict()))
        node_mask_bool = (node_mask >= 0.5)
        actor_idx = _select_actor_index(feats, node_mask_bool)

        if variant == "actor_rel":
            feats = _apply_actor_relative(feats, node_mask_bool, actor_idx)
        elif variant == "actor_rel_ctx":
            feats = _build_actor_relative_context_features(feats, node_mask_bool, actor_idx)
            _assert_actor_rel_ctx_row_invariants(feats, node_mask_bool)

        row_ix = i - 1
        feats_mm[row_ix] = feats.astype(np.float32, copy=False)
        adj_mm[row_ix] = adj.astype(np.float32, copy=False)
        mask_mm[row_ix] = node_mask_bool.astype(np.bool_, copy=False)

        if i == 1 or (i % progress_every == 0) or i == n_rows:
            pct = 100.0 * (float(i) / float(max(1, n_rows)))
            print(f"  Built graph rows {i:,}/{n_rows:,} ({pct:5.1f}%)")

        if i % flush_every == 0:
            feats_mm.flush()
            adj_mm.flush()
            mask_mm.flush()

    feats_mm.flush()
    adj_mm.flush()
    mask_mm.flush()

    feats_shape = tuple(int(x) for x in feats_mm.shape)
    adj_shape = tuple(int(x) for x in adj_mm.shape)
    mask_shape = tuple(int(x) for x in mask_mm.shape)
    dtype_feats = str(feats_mm.dtype)
    dtype_adj = str(adj_mm.dtype)
    dtype_mask = str(mask_mm.dtype)

    del feats_mm
    del adj_mm
    del mask_mm
    gc.collect()

    return {
        "rows": int(n_rows),
        "feats_shape": feats_shape,
        "adj_shape": adj_shape,
        "mask_shape": mask_shape,
        "dtype_feats": dtype_feats,
        "dtype_adj": dtype_adj,
        "dtype_mask": dtype_mask,
        "feature_names": _feature_names_for_variant(variant),
    }


def _update_feature_definitions(
    path: Path,
    *,
    variant_key: str,
    feats_path: Path,
    adj_path: Path,
    mask_path: Path,
    shape: tuple[int, int, int],
    key: str,
) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    payload[key] = {
        "variant": variant_key,
        "feats_path": str(feats_path),
        "adj_path": str(adj_path),
        "mask_path": str(mask_path),
        "feats_shape": [int(shape[0]), int(shape[1]), int(shape[2])],
        "adj_shape": [int(shape[0]), int(shape[1]), int(shape[1])],
        "mask_shape": [int(shape[0]), int(shape[1])],
        "dtypes": {
            "feats": "float32",
            "adj": "float32",
            "mask": "bool",
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _update_feature_definitions_threat_vectors(
    path: Path,
    *,
    threat_path: Path,
    scaler_path: Path,
    n_rows: int,
    n_features: int,
    feature_names: list[str],
) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    payload["threat_vectors"] = {
        "variant": "micro_spatial_v1",
        "path": str(threat_path),
        "shape": [int(n_rows), int(n_features)],
        "dtype": "float32",
        "scaler_path": str(scaler_path),
        "feature_names": list(feature_names),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_phase3_threat_vectors(
    base_dir: Path,
    *,
    run_label: str = "run_current",
    config: DataPrepConfig | None = None,
) -> dict[str, Any]:
    _ = config or DataPrepConfig()
    resolved_run_label = resolve_run_label(base_dir, run_label)
    paths = DataPrepPaths(base_dir, run_label=resolved_run_label)
    paths.ensure_dirs()

    require_artifacts_exist(
        [paths.phase2_tracking_event_relative_output, paths.phase3_tensor_ready_dataset_output],
        stage_name="Phase 3 threat-vector export",
        run_root=paths.runs_root_dir,
    )

    df_tracking = read_parquet(paths.phase2_tracking_event_relative_output)
    df_tensor_ready = read_parquet(paths.phase3_tensor_ready_dataset_output)

    aligned_tracking = _align_tracking_to_tensor_order(df_tracking, df_tensor_ready)
    tensor_norm = _normalize_event_keys(df_tensor_ready, label="tensor_ready_dataset.parquet").reset_index(drop=True)

    key_cols = ["game_id", "sl_event_id"]
    if not aligned_tracking[key_cols].equals(tensor_norm[key_cols]):
        raise RuntimeError("Threat-vector alignment key order mismatch after normalization.")

    threat_raw, feature_names, threat_meta = _build_threat_vector_matrix(
        df_tracking_aligned=aligned_tracking,
        df_tensor_aligned=tensor_norm,
    )
    threat_z, scaler_meta = _zscore_standardize_features(threat_raw, eps=1e-6)

    threat_path = paths.phase3_threat_vectors_output
    scaler_path = paths.phase3_threat_vector_scaler_output

    mm = np.lib.format.open_memmap(
        threat_path,
        mode="w+",
        dtype=np.float32,
        shape=threat_z.shape,
    )
    mm[:] = threat_z
    mm.flush()
    del mm
    gc.collect()

    scaler_payload = {
        "generated_at_utc": utc_now_iso(),
        "run_label": resolved_run_label,
        "feature_names": list(feature_names),
        "n_rows": int(threat_z.shape[0]),
        "n_features": int(threat_z.shape[1]),
        "scaler": scaler_meta,
        "raw_matrix": {
            "mean_abs_max": float(np.max(np.abs(threat_raw.mean(axis=0, dtype=np.float64)))) if threat_raw.size else 0.0,
            "std_min": float(np.min(threat_raw.std(axis=0, dtype=np.float64))) if threat_raw.size else 0.0,
            "std_max": float(np.max(threat_raw.std(axis=0, dtype=np.float64))) if threat_raw.size else 0.0,
        },
    }
    write_json(scaler_path, scaler_payload)

    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase3_threat_vector_export",
        output_path=threat_path,
        rows=int(threat_z.shape[0]),
        columns=list(feature_names),
        extra={
            "gate_status": "pass",
            "threat_vector_shape": [int(threat_z.shape[0]), int(threat_z.shape[1])],
            "dtype": "float32",
            "scaler_path": str(scaler_path),
            "safe_max_distance": float(threat_meta["safe_max_distance"]),
            "distance_source_column": threat_meta.get("distance_source_column"),
            "angle_source_column": threat_meta.get("angle_source_column"),
            "zscore_post_mean_abs_max": float(scaler_meta["post_mean_abs_max"]),
            "zscore_post_std_min": float(scaler_meta["post_std_min"]),
            "zscore_post_std_max": float(scaler_meta["post_std_max"]),
            "near_constant_feature_count": int(scaler_meta["near_constant_feature_count"]),
        },
    )

    _update_feature_definitions_threat_vectors(
        paths.phase3_feature_definitions_output,
        threat_path=threat_path,
        scaler_path=scaler_path,
        n_rows=int(threat_z.shape[0]),
        n_features=int(threat_z.shape[1]),
        feature_names=feature_names,
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "run_label": resolved_run_label,
        "rows": int(threat_z.shape[0]),
        "features": int(threat_z.shape[1]),
        "alignment": {
            "tensor_ready_path": str(paths.phase3_tensor_ready_dataset_output),
            "tracking_relative_path": str(paths.phase2_tracking_event_relative_output),
            "key_columns": ["game_id", "sl_event_id"],
            "status": "pass",
        },
        "threat_vectors_path": str(threat_path),
        "scaler_path": str(scaler_path),
        "feature_names": feature_names,
        "safe_max_distance": float(threat_meta["safe_max_distance"]),
        "zscore": {
            "post_mean_abs_max": float(scaler_meta["post_mean_abs_max"]),
            "post_std_min": float(scaler_meta["post_std_min"]),
            "post_std_max": float(scaler_meta["post_std_max"]),
            "near_constant_feature_count": int(scaler_meta["near_constant_feature_count"]),
        },
    }
    write_json(paths.phase3_threat_vector_summary_output, summary)
    return summary


def run_phase3_gnn_embeddings(
    base_dir: Path,
    *,
    run_label: str = "run_current",
    config: DataPrepConfig | None = None,
    variants: list[str] | None = None,
) -> dict:
    _ = config or DataPrepConfig()
    resolved_run_label = resolve_run_label(base_dir, run_label)
    paths = DataPrepPaths(base_dir, run_label=resolved_run_label)
    paths.ensure_dirs()

    require_artifacts_exist(
        [paths.phase2_tracking_absolute_output, paths.phase3_tensor_ready_dataset_output],
        stage_name="Phase 3 GNN graph export",
        run_root=paths.runs_root_dir,
    )

    df_tracking = read_parquet(paths.phase2_tracking_absolute_output)
    df_tensor_ready = read_parquet(paths.phase3_tensor_ready_dataset_output)
    aligned_tracking = _align_tracking_to_tensor_order(df_tracking, df_tensor_ready)

    print(f"Exporting raw graph components for {len(aligned_tracking):,} tensor-ready rows")

    variant_list = variants or ["base"]
    normalized = []
    for item in variant_list:
        key = str(item).strip().lower()
        if key == "all":
            normalized = list(GRAPH_VARIANTS.keys())
            break
        normalized.append(key)

    unknown = [v for v in normalized if v not in GRAPH_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown GNN graph variants: {unknown}")

    outputs: dict[str, dict[str, Any]] = {}
    for variant in normalized:
        meta = GRAPH_VARIANTS[variant]
        feats_path = getattr(paths, meta["feats_attr"])
        adj_path = getattr(paths, meta["adj_attr"])
        mask_path = getattr(paths, meta["mask_attr"])

        print(f"\n[Variant {variant}] {meta['description']}")

        artifact_info = _build_and_save_graph_arrays(
            df_tracking_aligned=aligned_tracking,
            variant=variant,
            feats_path=feats_path,
            adj_path=adj_path,
            mask_path=mask_path,
        )

        append_manifest_record(
            paths.prep_logs_dir / "data_prep_manifest.json",
            name=f"phase3_gnn_graph_export_{variant}",
            output_path=feats_path,
            rows=int(artifact_info["rows"]),
            columns=["node_features", "adjacency", "node_mask"],
            extra={
                "gate_status": "pass",
                "variant": variant,
                "adj_path": str(adj_path),
                "mask_path": str(mask_path),
                "feats_shape": list(artifact_info["feats_shape"]),
                "adj_shape": list(artifact_info["adj_shape"]),
                "mask_shape": list(artifact_info["mask_shape"]),
                "dtype_feats": str(artifact_info["dtype_feats"]),
                "dtype_adj": str(artifact_info["dtype_adj"]),
                "dtype_mask": str(artifact_info["dtype_mask"]),
                "feature_names": list(artifact_info["feature_names"]),
            },
        )

        _update_feature_definitions(
            paths.phase3_feature_definitions_output,
            variant_key=variant,
            feats_path=feats_path,
            adj_path=adj_path,
            mask_path=mask_path,
            shape=tuple(artifact_info["feats_shape"]),
            key=meta["name"],
        )

        outputs[variant] = {
            "rows": int(artifact_info["rows"]),
            "feats_shape": [int(x) for x in artifact_info["feats_shape"]],
            "adj_shape": [int(x) for x in artifact_info["adj_shape"]],
            "mask_shape": [int(x) for x in artifact_info["mask_shape"]],
            "feats_path": str(feats_path),
            "adj_path": str(adj_path),
            "mask_path": str(mask_path),
            "dtype_feats": str(artifact_info["dtype_feats"]),
            "dtype_adj": str(artifact_info["dtype_adj"]),
            "dtype_mask": str(artifact_info["dtype_mask"]),
            "feature_names": list(artifact_info["feature_names"]),
        }
        print(f"Saved raw graph tensors: {feats_path.name}, {adj_path.name}, {mask_path.name}")

    summary = {
        "generated_at_utc": utc_now_iso(),
        "run_label": resolved_run_label,
        "rows": int(len(aligned_tracking)),
        "alignment": {
            "tensor_ready_path": str(paths.phase3_tensor_ready_dataset_output),
            "tracking_path": str(paths.phase2_tracking_absolute_output),
            "key_columns": ["game_id", "sl_event_id"],
            "status": "pass",
        },
        "variants": outputs,
    }

    summary_path = paths.prep_logs_dir / "phase3_gnn_embeddings_summary.json"
    write_json(summary_path, summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 raw GNN graph component export")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated list: base, actor_rel, actor_rel_ctx, or 'all'",
    )
    parser.add_argument(
        "--export-threat-vectors",
        action="store_true",
        help="Also export standardized flat threat vectors from phase2 tracking_event_relative.",
    )
    parser.add_argument(
        "--threat-only",
        action="store_true",
        help="Export threat vectors only and skip graph tensor exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    if args.threat_only:
        summary = run_phase3_threat_vectors(
            args.base_dir,
            run_label=args.run_label,
        )
        print(json.dumps(summary, indent=2))
        return

    graph_summary = run_phase3_gnn_embeddings(
        args.base_dir,
        run_label=args.run_label,
        variants=variants,
    )
    print(json.dumps(graph_summary, indent=2))

    if args.export_threat_vectors:
        threat_summary = run_phase3_threat_vectors(
            args.base_dir,
            run_label=args.run_label,
        )
        print(json.dumps(threat_summary, indent=2))


if __name__ == "__main__":
    main()
