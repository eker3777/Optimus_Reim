from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json, write_parquet
    from .validation import (
        combine_gate_reports,
        assert_no_duplicate_columns,
        assert_target_domain,
        require_columns,
        validate_phase2_phase3_key_parity,
        validate_phase3_outputs,
    )
    from .run_resolver import require_artifacts_exist, resolve_run_label
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json, write_parquet
    from validation import (
        combine_gate_reports,
        assert_no_duplicate_columns,
        assert_target_domain,
        require_columns,
        validate_phase2_phase3_key_parity,
        validate_phase3_outputs,
    )
    from run_resolver import require_artifacts_exist, resolve_run_label


CLASS_POSS_GOAL = 0
CLASS_DEF_GOAL = 1
CLASS_NO_GOAL = 2

CATEGORICAL_FEATURES = ["event_type", "outcome", "period"]
TEXT_FEATURES = ["embedding_text_clean"]
CONTINUOUS_EVENT_FEATURES_RAW = [
    "x",
    "y",
    "x_adj",
    "y_adj",
    "dest_x_adj",
    "dest_y_adj",
    "period_time",
    "time_since_last_event",
    "distance_to_net",
    "distance_to_net_event",
    "angle_to_net_event",
    "distance_from_last_event",
    "speed_from_last_event",
    "angle_from_last_event",
    "goalie_angle_change",
]
CONTINUOUS_POSSESSION_FEATURES = [
    "period_time_remaining",
    "score_differential_actor",
    "score_differential_poss",
    "n_skaters_actor",
    "n_skaters_poss",
    "n_skaters_opp",
    "n_skaters_def",
]
CONTINUOUS_EVENT_FEATURES = CONTINUOUS_EVENT_FEATURES_RAW + CONTINUOUS_POSSESSION_FEATURES
BINARY_EVENT_FEATURES = ["net_empty_poss", "net_empty_def", "home_team_poss"]

ACTOR_FEATURES = [
    "actor_rel_x",
    "actor_rel_y",
    "actor_vx_std",
    "actor_vy_std",
    "actor_mask",
    "actor_is_imputed",
]
TEAMMATE_FEATURES: list[str] = []
for _i in range(1, 6):
    TEAMMATE_FEATURES.extend(
        [
            f"tm_{_i}_rel_x",
            f"tm_{_i}_rel_y",
            f"tm_{_i}_vx_std",
            f"tm_{_i}_vy_std",
            f"tm_{_i}_distance",
            f"tm_{_i}_mask",
        ]
    )

OPPONENT_FEATURES: list[str] = []
for _i in range(1, 7):
    OPPONENT_FEATURES.extend(
        [
            f"opp_{_i}_rel_x",
            f"opp_{_i}_rel_y",
            f"opp_{_i}_vx_std",
            f"opp_{_i}_vy_std",
            f"opp_{_i}_distance",
            f"opp_{_i}_mask",
        ]
    )

TRACKING_FEATURES = ACTOR_FEATURES + TEAMMATE_FEATURES + OPPONENT_FEATURES

PHASE3_REQUIRED_PHASE2_COLS = [
    "game_id",
    "period",
    "period_time",
    "sequence_id",
    "event_type",
    "team_id",
    "player_id",
    "detail",
    "sl_event_id",
    "game_time_sec",
    "is_boundary_event",
    "is_end_of_period",
]

LEAK_REGEX = re.compile(
    r"\bgoal\b|\bgoals\b|\bon\s*net\b|\bonnet\b|\bwith\s*goal\b|\bwith\s*shot\s*on\s*net(?:\s*whistle)?\b",
    flags=re.IGNORECASE,
)
BLOCKED_COMPACT_TOKENS = {"withgoal", "withshotonnet", "withshotonnetwhistle"}
RESERVED_EVENT_TYPE_TOKENS = {"defensive_deflection", "save"}


def _sorted_unique(values: pd.Series) -> list[Any]:
    out = values.dropna().unique().tolist()
    out.sort(key=lambda v: str(v))
    return out


def _safe_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _prepare_phase3_input_frame(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()

    if "is_end_of_period" not in df.columns and "is_end_of_period_event" in df.columns:
        df["is_end_of_period"] = pd.to_numeric(df["is_end_of_period_event"], errors="coerce").fillna(0).astype(np.int8)

    tracking_aliases: dict[str, str] = {
        "actor_vx_rel": "actor_vx_std",
        "actor_vy_rel": "actor_vy_std",
        "actor_is_present": "actor_mask",
    }
    for i in range(1, 6):
        tracking_aliases[f"tm_{i}_vx_rel"] = f"tm_{i}_vx_std"
        tracking_aliases[f"tm_{i}_vy_rel"] = f"tm_{i}_vy_std"
        tracking_aliases[f"tm_{i}_is_present"] = f"tm_{i}_mask"
    for i in range(1, 7):
        tracking_aliases[f"opp_{i}_vx_rel"] = f"opp_{i}_vx_std"
        tracking_aliases[f"opp_{i}_vy_rel"] = f"opp_{i}_vy_std"
        tracking_aliases[f"opp_{i}_is_present"] = f"opp_{i}_mask"

    for src_col, dst_col in tracking_aliases.items():
        if dst_col not in df.columns and src_col in df.columns:
            df[dst_col] = df[src_col]

    return df


def _assign_next_time_by_group(
    base_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    value_col: str,
    total_rows: int,
) -> np.ndarray:
    out = np.full(total_rows, np.nan, dtype=float)
    if base_df.empty or ref_df.empty:
        return out

    ref_groups = {
        key: grp[value_col].to_numpy(dtype=float)
        for key, grp in ref_df.groupby(["game_id", "period"], sort=False)
    }

    for key, grp in base_df.groupby(["game_id", "period"], sort=False):
        ref_times = ref_groups.get(key)
        if ref_times is None or len(ref_times) == 0:
            continue

        t = grp["period_time"].to_numpy(dtype=float)
        idx = np.searchsorted(ref_times, t, side="right")
        valid = idx < len(ref_times)
        if not np.any(valid):
            continue

        row_ids = grp["_row_id"].to_numpy(dtype=np.int64)[valid]
        out[row_ids] = ref_times[idx[valid]]

    return out


def _assign_next_goal_by_group(
    base_df: pd.DataFrame,
    goals_df: pd.DataFrame,
    total_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_time = np.full(total_rows, np.nan, dtype=float)
    out_team = np.empty(total_rows, dtype=object)
    out_team[:] = np.nan

    if base_df.empty or goals_df.empty:
        return out_time, out_team

    grouped_goals = {
        key: (
            grp["next_goal_time"].to_numpy(dtype=float),
            grp["next_goal_team_id"].to_numpy(dtype=object),
        )
        for key, grp in goals_df.groupby(["game_id", "period"], sort=False)
    }

    for key, grp in base_df.groupby(["game_id", "period"], sort=False):
        goal_pack = grouped_goals.get(key)
        if goal_pack is None:
            continue

        goal_times, goal_teams = goal_pack
        if len(goal_times) == 0:
            continue

        t = grp["period_time"].to_numpy(dtype=float)
        idx = np.searchsorted(goal_times, t, side="right")
        valid = idx < len(goal_times)
        if not np.any(valid):
            continue

        row_ids = grp["_row_id"].to_numpy(dtype=np.int64)[valid]
        out_time[row_ids] = goal_times[idx[valid]]
        out_team[row_ids] = goal_teams[idx[valid]]

    return out_time, out_team


def _select_embedding_device() -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "cpu", "torch-not-installed"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_name = "xpu"
        try:
            device_name = str(torch.xpu.get_device_name(0))
        except Exception:
            pass
        return "xpu", device_name

    if torch.cuda.is_available():
        device_name = "cuda"
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        return "cuda", device_name

    return "cpu", "cpu"


def _validate_phase3_input_contract(
    df_events: pd.DataFrame,
    df_games: pd.DataFrame,
    *,
    require_tracking_features: bool,
) -> dict[str, Any]:
    required_phase3_inputs = (
        PHASE3_REQUIRED_PHASE2_COLS
        + [
            "description_clean",
            "flags_clean",
            "outcome",
            "sl_xg_all_shots",
            "score_differential_home",
            "score_differential_away",
            "n_home_skaters",
            "n_away_skaters",
            "is_home_net_empty",
            "is_away_net_empty",
        ]
        + TEXT_FEATURES
        + CONTINUOUS_EVENT_FEATURES_RAW
    )
    if require_tracking_features:
        required_phase3_inputs += TRACKING_FEATURES
    required_phase3_inputs = sorted(set(required_phase3_inputs))
    require_columns(df_events, required_phase3_inputs, "phase2_events_for_phase3")
    require_columns(df_games, ["game_id", "home_team_id"], "games")

    reception_count = int((df_events["event_type"].astype(str).str.lower() == "reception").sum())
    reception_presence_status = "present" if reception_count > 0 else "absent"

    actor_cols = [c for c in df_events.columns if c.startswith("actor_")]
    tm_cols = [c for c in df_events.columns if c.startswith("tm_")]
    opp_cols = [c for c in df_events.columns if c.startswith("opp_")]

    return {
        "input_rows": int(len(df_events)),
        "input_columns": int(len(df_events.columns)),
        "tracking_required": bool(require_tracking_features),
        "actor_columns": int(len(actor_cols)),
        "teammate_columns": int(len(tm_cols)),
        "opponent_columns": int(len(opp_cols)),
        "reception_events": reception_count,
        "reception_presence_status": reception_presence_status,
    }


def _add_possession_relative_features(df_events: pd.DataFrame, df_games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    home_team_lookup = df_games.set_index("game_id")["home_team_id"].to_dict()
    df["home_team_id"] = df["game_id"].map(home_team_lookup)

    if df["home_team_id"].isna().any():
        missing_games = int(df.loc[df["home_team_id"].isna(), "game_id"].nunique())
        raise KeyError(f"Missing home_team_id lookup for {missing_games} game_id values")

    df["home_team_poss"] = (df["team_id"] == df["home_team_id"]).astype(np.int8)

    df["period_time"] = pd.to_numeric(df["period_time"], errors="coerce")
    # Phase 3 Fix
    period_num = pd.to_numeric(df["period"], errors="coerce").fillna(1)
    period_length_sec = np.where(period_num <= 3, 1200.0, 300.0)
    df["period_time_remaining"] = (period_length_sec - df["period_time"]).clip(0.0, 1200.0)

    df["score_differential_poss"] = np.where(
        df["home_team_poss"] == 1,
        pd.to_numeric(df["score_differential_home"], errors="coerce"),
        pd.to_numeric(df["score_differential_away"], errors="coerce"),
    )

    df["n_skaters_poss"] = np.where(
        df["home_team_poss"] == 1,
        pd.to_numeric(df["n_home_skaters"], errors="coerce"),
        pd.to_numeric(df["n_away_skaters"], errors="coerce"),
    )
    df["n_skaters_def"] = np.where(
        df["home_team_poss"] == 1,
        pd.to_numeric(df["n_away_skaters"], errors="coerce"),
        pd.to_numeric(df["n_home_skaters"], errors="coerce"),
    )

    # Canonical actor/opponent aliases for downstream model inputs.
    df["score_differential_actor"] = pd.to_numeric(df["score_differential_poss"], errors="coerce")
    df["n_skaters_actor"] = pd.to_numeric(df["n_skaters_poss"], errors="coerce")
    df["n_skaters_opp"] = pd.to_numeric(df["n_skaters_def"], errors="coerce")

    # Canonical distance alias for downstream model inputs.
    if "distance_to_net" not in df.columns and "distance_to_net_event" in df.columns:
        df["distance_to_net"] = pd.to_numeric(df["distance_to_net_event"], errors="coerce")

    net_empty_poss = np.where(
        df["home_team_poss"] == 1,
        pd.to_numeric(df["is_home_net_empty"], errors="coerce"),
        pd.to_numeric(df["is_away_net_empty"], errors="coerce"),
    )
    net_empty_def = np.where(
        df["home_team_poss"] == 1,
        pd.to_numeric(df["is_away_net_empty"], errors="coerce"),
        pd.to_numeric(df["is_home_net_empty"], errors="coerce"),
    )
    # Guard against NaN -> int RuntimeWarnings from malformed net-empty flags.
    df["net_empty_poss"] = pd.Series(pd.to_numeric(net_empty_poss, errors="coerce"), index=df.index).fillna(0).astype(np.int8)
    df["net_empty_def"] = pd.Series(pd.to_numeric(net_empty_def, errors="coerce"), index=df.index).fillna(0).astype(np.int8)

    summary = {
        "home_possession_events": int(df["home_team_poss"].sum()),
        "away_possession_events": int((1 - df["home_team_poss"]).sum()),
        "period_time_remaining_min": float(df["period_time_remaining"].min()),
        "period_time_remaining_max": float(df["period_time_remaining"].max()),
        "score_differential_poss_mean": float(pd.to_numeric(df["score_differential_poss"], errors="coerce").mean()),
        "score_differential_actor_mean": float(pd.to_numeric(df["score_differential_actor"], errors="coerce").mean()),
    }
    return df, summary


def _sanitize_embedding_text(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()
    require_columns(df, ["embedding_text_clean"], "phase3_events")

    cleaned = _safe_str_series(df["embedding_text_clean"])
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.replace(r" for (missed|blocked) ;", " for ;", regex=True, case=False)
    cleaned = cleaned.str.strip()

    compact_series = cleaned.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    leak_mask = cleaned.str.contains(LEAK_REGEX, na=False) | compact_series.isin(BLOCKED_COMPACT_TOKENS)
    n_leaks = int(leak_mask.sum())
    if n_leaks > 0:
        sample = cleaned[leak_mask].head(5).tolist()
        raise ValueError(
            "Leakage tokens detected in embedding_text_clean after sanitization. "
            f"Rows flagged: {n_leaks}. Sample: {sample}"
        )

    df["embedding_text_clean"] = cleaned
    summary = {
        "non_empty_rows": int(cleaned.ne("").sum()),
        "unique_values": int(cleaned.nunique()),
        "leak_rows": n_leaks,
    }
    return df, summary


def _encode_sentence_embeddings(
    df_events: pd.DataFrame,
    cfg: DataPrepConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    df = df_events.copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for Phase 3 embedding generation. "
            "Install it in the active environment before running Phase 3."
        ) from exc
    except OSError as exc:
        err = str(exc)
        if "WinError 4551" in err or "Application Control policy has blocked this file" in err:
            raise RuntimeError(
                "Phase 3 embedding import failed because Windows Application Control blocked a PyTorch DLL "
                "(typically torch_xpu_ops_aten.dll). This is an endpoint policy issue, not a notebook/script "
                "permission issue. Ask IT/security to allowlist the blocked DLL path under your virtual environment "
                "or switch this environment to CPU-only PyTorch wheels."
            ) from exc
        raise

    device, device_name = _select_embedding_device()
    unique_texts = pd.unique(_safe_str_series(df["embedding_text_clean"]))

    print(
        "Phase 3 embeddings: "
        f"device={device} ({device_name}), model={cfg.phase3_embedding_model_name}, "
        f"batch_size={int(cfg.phase3_embedding_batch_size)}, unique_texts={int(len(unique_texts)):,}"
    )

    model = SentenceTransformer(cfg.phase3_embedding_model_name, device=device)
    unique_embeddings = model.encode(
        unique_texts.tolist(),
        batch_size=max(1, int(cfg.phase3_embedding_batch_size)),
        show_progress_bar=True,
        normalize_embeddings=bool(cfg.phase3_embedding_normalize),
        convert_to_numpy=True,
    ).astype(np.float32)

    text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
    df["text_embedding_idx"] = _safe_str_series(df["embedding_text_clean"]).map(text_to_idx).astype(np.int32)

    embeddings_path = output_dir / "text_embeddings.npy"
    np.save(embeddings_path, unique_embeddings)

    map_df = pd.DataFrame(
        {
            "text_embedding_idx": np.arange(len(unique_texts), dtype=np.int32),
            "embedding_text_clean": unique_texts,
        }
    )
    map_path = output_dir / "text_embedding_idx_map.parquet"
    write_parquet(map_path, map_df)

    summary = {
        "device": device,
        "device_name": device_name,
        "model": cfg.phase3_embedding_model_name,
        "batch_size": int(cfg.phase3_embedding_batch_size),
        "unique_texts": int(len(unique_texts)),
        "embedding_rows": int(unique_embeddings.shape[0]),
        "embedding_dim": int(unique_embeddings.shape[1]),
        "embeddings_output": str(embeddings_path),
        "map_output": str(map_path),
    }

    del model
    del unique_embeddings
    del text_to_idx
    gc.collect()

    return df, int(len(unique_texts)), summary


def _build_categorical_vocabs(df_events: pd.DataFrame) -> tuple[dict[str, dict[Any, int]], dict[str, Any]]:
    vocabs: dict[str, dict[Any, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        unique_values = _sorted_unique(df_events[feature])
        if feature == "event_type":
            present = {str(v).strip().lower() for v in unique_values}
            for reserved in sorted(RESERVED_EVENT_TYPE_TOKENS):
                if reserved not in present:
                    unique_values.append(reserved)
            unique_values.sort(key=lambda v: str(v))
        vocab: dict[Any, int] = {"<PAD>": 0, "<UNK>": 1}
        for idx, value in enumerate(unique_values, start=2):
            vocab[value] = idx
        vocabs[feature] = vocab

    summary = {
        "vocab_sizes": {feature: int(len(vocab)) for feature, vocab in vocabs.items()},
    }
    return vocabs, summary


def _apply_categorical_ids(df_events: pd.DataFrame, vocabs: dict[str, dict[Any, int]]) -> pd.DataFrame:
    df = df_events.copy()
    for feature in CATEGORICAL_FEATURES:
        df[f"{feature}_id"] = df[feature].map(vocabs[feature]).fillna(1).astype(np.int32)
    return df


def _add_eos_tokens(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()
    terminator_events = ["whistle", "goal", "end_of_period"]

    df["is_last_in_seq"] = False
    last_event_idx = df.groupby(["game_id", "sequence_id"], sort=False).tail(1).index
    df.loc[last_event_idx, "is_last_in_seq"] = True

    df["is_eos"] = (
        df["is_last_in_seq"]
        & df["event_type"].astype(str).str.lower().isin(terminator_events)
    ).astype(np.int8)

    summary = {
        "last_event_rows": int(df["is_last_in_seq"].sum()),
        "eos_rows": int(df["is_eos"].sum()),
        "eos_rate": float(df["is_eos"].mean()),
    }
    return df, summary


def _strict_event_dedupe(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    if "sl_event_id" in df.columns:
        df["_sl_event_id_norm"] = pd.to_numeric(df["sl_event_id"], errors="coerce").astype("Float64").round(6)
    if "period_time" in df.columns:
        df["_period_time_norm"] = (
            pd.to_numeric(df["period_time"], errors="coerce")
            .mul(1000)
            .round()
            .astype("Int64")
        )

    dedupe_keys_preferred = [
        "game_id",
        "period",
        "sequence_id",
        "_sl_event_id_norm",
        "event_type",
        "team_id",
        "player_id",
        "_period_time_norm",
    ]
    dedupe_keys = [c for c in dedupe_keys_preferred if c in df.columns]
    if len(dedupe_keys) < 4:
        raise ValueError(f"Not enough dedupe keys available for strict dedupe: {dedupe_keys}")

    rows_before = len(df)
    dup_mask = df.duplicated(subset=dedupe_keys, keep="first")
    duplicate_rows = int(dup_mask.sum())
    duplicate_groups = int(df.groupby(dedupe_keys, dropna=False).size().gt(1).sum())

    if duplicate_rows > 0:
        df = df.loc[~dup_mask].copy().reset_index(drop=True)

    df = df.drop(columns=["_sl_event_id_norm", "_period_time_norm"], errors="ignore")

    summary = {
        "rows_before": int(rows_before),
        "rows_after": int(len(df)),
        "duplicate_rows_dropped": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "dedupe_keys": dedupe_keys,
    }
    return df, summary


def _build_actor_relative_targets(df_events: pd.DataFrame, cfg: DataPrepConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    require_columns(
        df,
        [
            "game_id",
            "period",
            "sequence_id",
            "period_time",
            "event_type",
            "team_id",
            "game_time_sec",
            "sequence_event_id",
            "is_boundary_event",
            "is_end_of_period",
        ],
        "phase3_target_input",
    )

    df["period_time"] = pd.to_numeric(df["period_time"], errors="coerce")
    df["sequence_event_id"] = pd.to_numeric(df["sequence_event_id"], errors="coerce")
    missing_sequence_event_id_rows = int(df["sequence_event_id"].isna().sum())
    if missing_sequence_event_id_rows > 0:
        raise ValueError(
            "Phase 3 contract violation: sequence_event_id contains missing/non-numeric values. "
            f"rows={missing_sequence_event_id_rows}"
        )

    df["sequence_event_id"] = df["sequence_event_id"].astype(np.int64)
    seq_keys = ["game_id", "period", "sequence_id"]
    delta = df.groupby(seq_keys, sort=False)["period_time"].diff()
    threshold = float(cfg.sequence_disorder_threshold_seconds)
    abnormal_mask = delta < threshold
    abnormal_rows = int(abnormal_mask.sum())

    sort_cols = ["game_id", "period", "sequence_id", "sequence_event_id", "game_time_sec", "period_time"]
    if "sl_event_id" in df.columns:
        sort_cols.append("sl_event_id")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    expected_sequence_event_id = df.groupby(seq_keys, sort=False).cumcount() + 1
    sequence_event_id_mismatch_rows = int((df["sequence_event_id"] != expected_sequence_event_id).sum())
    if sequence_event_id_mismatch_rows > 0:
        raise ValueError(
            "Phase 3 contract violation: sequence_event_id is not contiguous/ordered within sequence. "
            f"rows={sequence_event_id_mismatch_rows}"
        )

    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    period_cap = (
        df.groupby(["game_id", "period"], sort=False)["period_time"]
        .transform("max")
        .clip(upper=float(cfg.target_timing.period_max_seconds))
    )
    df["period_cap_time"] = period_cap
    df["horizon_end_time"] = np.minimum(
        df["period_time"] + float(cfg.target_timing.default_horizon_seconds),
        df["period_cap_time"],
    )

    base_times = df[["_row_id", "game_id", "period", "period_time"]].dropna(subset=["period_time"]).copy()

    stoppages = df[
        df["event_type"].astype(str).str.lower().isin(["whistle", "goal", "end_of_period"])
    ][["game_id", "period", "period_time"]].dropna(subset=["period_time"]).copy()
    stoppages = stoppages.rename(columns={"period_time": "next_stoppage_time"})
    stoppages = stoppages.sort_values(["game_id", "period", "next_stoppage_time"], kind="mergesort")

    df["next_stoppage_time"] = _assign_next_time_by_group(base_times, stoppages, "next_stoppage_time", len(df))
    df["sec_next_stoppage"] = df["next_stoppage_time"] - df["period_time"]

    goals = df[df["event_type"].astype(str).str.lower() == "goal"][
        ["game_id", "period", "period_time", "team_id"]
    ].dropna(subset=["period_time"]).copy()
    goals = goals.rename(columns={"period_time": "next_goal_time", "team_id": "next_goal_team_id"})
    goals = goals.sort_values(["game_id", "period", "next_goal_time"], kind="mergesort")

    next_goal_time, next_goal_team = _assign_next_goal_by_group(base_times, goals, len(df))
    df["next_goal_time"] = next_goal_time
    df["next_goal_team_id"] = next_goal_team

    in_dynamic_horizon = (
        df["next_goal_time"].notna()
        & (df["next_goal_time"] <= df["horizon_end_time"])
    )

    df["target"] = np.int8(CLASS_NO_GOAL)
    valid_actor_team = df["team_id"].notna()
    poss_goal_mask = in_dynamic_horizon & valid_actor_team & (df["next_goal_team_id"] == df["team_id"])
    def_goal_mask = in_dynamic_horizon & valid_actor_team & (df["next_goal_team_id"] != df["team_id"])

    df.loc[poss_goal_mask, "target"] = np.int8(CLASS_POSS_GOAL)
    df.loc[def_goal_mask, "target"] = np.int8(CLASS_DEF_GOAL)

    overflow_rows = int((df["horizon_end_time"] > df["period_cap_time"]).sum())
    both_goal_classes = int((poss_goal_mask & def_goal_mask).sum())
    invalid_targets = int((~df["target"].isin([CLASS_POSS_GOAL, CLASS_DEF_GOAL, CLASS_NO_GOAL])).sum())

    target_counts = df["target"].value_counts().sort_index()
    class_counts = {
        str(CLASS_POSS_GOAL): int(target_counts.get(CLASS_POSS_GOAL, 0)),
        str(CLASS_DEF_GOAL): int(target_counts.get(CLASS_DEF_GOAL, 0)),
        str(CLASS_NO_GOAL): int(target_counts.get(CLASS_NO_GOAL, 0)),
    }

    if "_row_id" in df.columns:
        df.drop(columns=["_row_id"], inplace=True)

    summary = {
        "abnormal_period_time_rows": abnormal_rows,
        "missing_sequence_event_id_rows": missing_sequence_event_id_rows,
        "sequence_event_id_mismatch_rows": sequence_event_id_mismatch_rows,
        "horizon_seconds": float(cfg.target_timing.default_horizon_seconds),
        "penalty_extension_applied": False,
        "rows_with_horizon": int(df["horizon_end_time"].notna().sum()),
        "rows_with_sec_next_stoppage": int(df["sec_next_stoppage"].notna().sum()),
        "horizon_overflow_rows": overflow_rows,
        "dual_class_rows": both_goal_classes,
        "invalid_target_rows": invalid_targets,
        "target_class_counts": class_counts,
    }
    return df, summary


def _create_target_xg_and_counterfactual(
    df_events: pd.DataFrame,
    cfg: DataPrepConfig,
    vocabs: dict[str, dict[Any, int]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    require_columns(
        df,
        ["game_id", "period", "sequence_id", "period_time", "event_type", "outcome"],
        "phase3_xg_target_input",
    )

    df["period_time"] = pd.to_numeric(df["period_time"], errors="coerce")
    seq_keys = ["game_id", "period", "sequence_id"]
    delta = df.groupby(seq_keys, sort=False)["period_time"].diff()
    abnormal_rows = int((delta < float(cfg.sequence_disorder_threshold_seconds)).sum())

    if abnormal_rows > 0:
        sort_cols = ["game_id", "period", "sequence_id", "period_time"]
        if "sl_event_id" in df.columns:
            sort_cols.append("sl_event_id")
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["target_xg"] = np.int8(0)
    shot_events = df["event_type"].astype(str).str.lower().isin(["shot", "deflection"])

    df["next_event_type"] = df["event_type"].shift(-1)
    df["next_outcome"] = df["outcome"].shift(-1)
    df["next_next_event_type"] = df["event_type"].shift(-2)

    df["next_sequence_id"] = df["sequence_id"].shift(-1)
    df["next_next_sequence_id"] = df["sequence_id"].shift(-2)
    df["same_sequence_next"] = df["sequence_id"] == df["next_sequence_id"]
    df["same_sequence_next_next"] = df["sequence_id"] == df["next_next_sequence_id"]

    direct_scoring = (
        shot_events
        & df["same_sequence_next"]
        & (df["next_event_type"].astype(str).str.lower() == "goal")
    )
    blocked_scoring = (
        shot_events
        & df["same_sequence_next_next"]
        & (df["next_event_type"].astype(str).str.lower() == "block")
        & (df["next_outcome"].astype(str).str.lower() == "failed")
        & (df["next_next_event_type"].astype(str).str.lower() == "goal")
    )

    df.loc[direct_scoring | blocked_scoring, "target_xg"] = np.int8(1)

    df.drop(
        columns=[
            "next_event_type",
            "next_outcome",
            "next_next_event_type",
            "next_sequence_id",
            "next_next_sequence_id",
            "same_sequence_next",
            "same_sequence_next_next",
        ],
        inplace=True,
    )

    df["outcome_xg"] = df["outcome"]
    shot_deflection_mask = df["event_type"].astype(str).str.lower().isin(["shot", "deflection", "defensive_deflection"])
    df.loc[shot_deflection_mask, "outcome_xg"] = np.nan

    mapped_outcome_xg = df["outcome_xg"].map(vocabs["outcome"]).fillna(1)
    df["outcome_xg_id"] = np.where(shot_deflection_mask, 0, mapped_outcome_xg).astype(np.int32)

    summary = {
        "abnormal_period_time_rows": abnormal_rows,
        "shot_deflection_rows": int(shot_events.sum()),
        "target_xg_positive_rows": int(df["target_xg"].sum()),
        "direct_scoring_rows": int(direct_scoring.sum()),
        "blocked_scoring_rows": int(blocked_scoring.sum()),
        "masked_outcome_rows": int(shot_deflection_mask.sum()),
    }
    return df, summary


def _post_whistle_continuity_audit(df_events: pd.DataFrame) -> dict[str, Any]:
    df = df_events
    whistle_mask = df["event_type"].astype(str).str.lower() == "whistle"
    whistle_idx = df.index[whistle_mask].to_numpy()

    if len(whistle_idx) == 0:
        return {
            "whistles_checked": 0,
            "same_sequence_continuations": 0,
            "allowed_post_whistle_penalty": 0,
            "unexpected_same_sequence_continuations": 0,
        }

    next_idx = whistle_idx + 1
    valid = next_idx < len(df)
    whistle_idx = whistle_idx[valid]
    next_idx = next_idx[valid]

    whistle_rows = df.loc[whistle_idx]
    next_rows = df.loc[next_idx]

    same_seq = (
        whistle_rows["game_id"].to_numpy() == next_rows["game_id"].to_numpy()
    ) & (
        whistle_rows["sequence_id"].to_numpy() == next_rows["sequence_id"].to_numpy()
    )

    next_type = next_rows["event_type"].astype(str).str.lower().to_numpy()
    next_has_player = next_rows["player_id"].notna().to_numpy() if "player_id" in next_rows.columns else np.zeros(len(next_rows), dtype=bool)
    allowed = np.isin(next_type, ["penalty", "penaltydrawn"]) & next_has_player
    unexpected = same_seq & (~allowed)

    return {
        "whistles_checked": int(len(whistle_mask[whistle_mask])),
        "same_sequence_continuations": int(same_seq.sum()),
        "allowed_post_whistle_penalty": int((same_seq & allowed).sum()),
        "unexpected_same_sequence_continuations": int(unexpected.sum()),
    }


def _build_feature_definitions(
    vocabs: dict[str, dict[Any, int]],
    unique_text_count: int,
) -> dict[str, Any]:
    return {
        "metadata_columns": [
            "game_id",
            "sequence_id",
            "sl_event_id",
            "sequence_event_id",
            "game_event_id",
            "period",
            "event_type",
            "outcome",
            "player_id",
            "team_id",
            "description_clean",
            "flags_clean",
            "embedding_text_clean",
            "sl_xg_all_shots",
            "target_xg",
            "outcome_xg",
            "outcome_xg_id",
            "horizon_end_time",
            "sec_next_stoppage",
        ],
        "categorical_features": {
            "columns": [f"{feat}_id" for feat in CATEGORICAL_FEATURES],
            "original_features": CATEGORICAL_FEATURES,
            "vocab_sizes": {feat: int(len(vocabs[feat])) for feat in CATEGORICAL_FEATURES},
        },
        "text_embeddings": {
            "columns": ["text_embedding_idx"],
            "embedding_file": "text_embeddings.npy",
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "normalized": True,
            "unique_texts": int(unique_text_count),
            "source_features": TEXT_FEATURES,
        },
        "continuous_event_features": {
            "columns": CONTINUOUS_EVENT_FEATURES,
            "raw_features": CONTINUOUS_EVENT_FEATURES_RAW,
            "possession_relative": CONTINUOUS_POSSESSION_FEATURES,
        },
        "binary_features": {
            "columns": BINARY_EVENT_FEATURES + ["is_eos"],
            "possession_relative": BINARY_EVENT_FEATURES,
            "eos_token": ["is_eos"],
        },
        "tracking_features": {
            "columns": [],
            "actor_features": [],
            "teammate_features": [],
            "opponent_features": [],
            "total_dim": 0,
            "excluded_from_tensor_ready_dataset": True,
            "source_phase2_artifact": "tracking_event_relative.parquet",
        },
        "target_column": "target",
        "target_classes": {
            "0": "Possession Goal",
            "1": "Defending Goal",
            "2": "No Goal",
        },
    }


def _assemble_final_dataset(
    df_events: pd.DataFrame,
    vocabs: dict[str, dict[Any, int]],
    unique_text_count: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    metadata_cols = [
        "game_id",
        "sequence_id",
        "sl_event_id",
        "sequence_event_id",
        "game_event_id",
        "period",
        "event_type",
        "outcome",
        "player_id",
        "team_id",
        "description_clean",
        "flags_clean",
        "embedding_text_clean",
        "sl_xg_all_shots",
        "target_xg",
        "outcome_xg",
        "outcome_xg_id",
        "horizon_end_time",
        "sec_next_stoppage",
    ]

    categorical_encoded = [f"{feat}_id" for feat in CATEGORICAL_FEATURES]
    text_embedding_cols = ["text_embedding_idx"]
    binary_features_final = BINARY_EVENT_FEATURES + ["is_eos"]

    final_cols = (
        metadata_cols
        + categorical_encoded
        + text_embedding_cols
        + CONTINUOUS_EVENT_FEATURES
        + binary_features_final
        + ["target"]
    )

    missing = [c for c in final_cols if c not in df_events.columns]
    if missing:
        raise KeyError(f"Missing columns required for Phase 3 final dataset: {missing}")

    if len(final_cols) != len(set(final_cols)):
        duplicates = sorted({c for c in final_cols if final_cols.count(c) > 1})
        raise AssertionError(f"Phase 3 final column list contains duplicates: {duplicates}")

    df_final = df_events[final_cols].copy()
    assert_no_duplicate_columns(df_final, "phase3_final_dataset")

    tracking_cols_in_final = [
        c for c in df_final.columns if c.startswith("actor_") or c.startswith("tm_") or c.startswith("opp_")
    ]

    feature_definitions = _build_feature_definitions(vocabs, unique_text_count)
    dataset_summary = {
        "rows": int(len(df_final)),
        "columns": int(len(df_final.columns)),
        "categorical_columns": int(len(categorical_encoded)),
        "continuous_columns": int(len(CONTINUOUS_EVENT_FEATURES)),
        "binary_columns": int(len(binary_features_final)),
        "tracking_columns": int(len(tracking_cols_in_final)),
    }
    return df_final, feature_definitions, dataset_summary


def _export_phase3_artifacts(
    df_events: pd.DataFrame,
    df_final: pd.DataFrame,
    vocabs: dict[str, dict[Any, int]],
    feature_definitions: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    events_with_idx_path = output_dir / "events_with_embedding_indices.parquet"
    final_dataset_path = output_dir / "tensor_ready_dataset.parquet"
    vocabs_path = output_dir / "vocabularies.json"
    feature_defs_path = output_dir / "feature_definitions.json"

    write_parquet(events_with_idx_path, df_events)
    write_parquet(final_dataset_path, df_final)

    serializable_vocabs = {
        feature: {str(key): int(value) for key, value in vocab.items()}
        for feature, vocab in vocabs.items()
    }
    write_json(vocabs_path, serializable_vocabs)
    write_json(feature_defs_path, feature_definitions)

    return {
        "events_with_embedding_indices": {
            "path": str(events_with_idx_path),
            "rows": int(len(df_events)),
            "columns": df_events.columns.tolist(),
        },
        "tensor_ready_dataset": {
            "path": str(final_dataset_path),
            "rows": int(len(df_final)),
            "columns": df_final.columns.tolist(),
        },
        "vocabularies": {
            "path": str(vocabs_path),
            "features": sorted(vocabs.keys()),
        },
        "feature_definitions": {
            "path": str(feature_defs_path),
        },
        "text_embeddings": {
            "path": str(output_dir / "text_embeddings.npy"),
        },
        "text_embedding_idx_map": {
            "path": str(output_dir / "text_embedding_idx_map.parquet"),
        },
    }


def run_phase3_tensor_prep(
    events_df: pd.DataFrame,
    games_df: pd.DataFrame,
    config: DataPrepConfig | None,
    output_dir: Path,
    *,
    require_tracking_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or DataPrepConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _prepare_phase3_input_frame(events_df)
    contract_summary = _validate_phase3_input_contract(
        events,
        games_df,
        require_tracking_features=require_tracking_features,
    )
    events, possession_summary = _add_possession_relative_features(events, games_df)
    events, text_summary = _sanitize_embedding_text(events)

    events, unique_text_count, embedding_summary = _encode_sentence_embeddings(events, cfg, out_dir)
    gc.collect()

    vocabs, vocab_summary = _build_categorical_vocabs(events)
    events = _apply_categorical_ids(events, vocabs)
    events, eos_summary = _add_eos_tokens(events)
    events, dedupe_summary = _strict_event_dedupe(events)
    events, xt_summary = _build_actor_relative_targets(events, cfg)
    events, xg_summary = _create_target_xg_and_counterfactual(events, cfg, vocabs)

    continuity_audit = _post_whistle_continuity_audit(events)

    df_final, feature_definitions, dataset_summary = _assemble_final_dataset(events, vocabs, unique_text_count)
    assert_target_domain(df_final)

    artifacts = _export_phase3_artifacts(events, df_final, vocabs, feature_definitions, out_dir)

    phase3_gate = validate_phase3_outputs(
        events,
        df_final,
        embedding_rows=int(embedding_summary["embedding_rows"]),
        default_horizon_seconds=float(cfg.target_timing.default_horizon_seconds),
        post_whistle_summary=continuity_audit,
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "gate_status": phase3_gate["gate_status"],
        "gates": {"phase3": phase3_gate},
        "contract": contract_summary,
        "possession_relative": possession_summary,
        "text_cleaning": text_summary,
        "embeddings": embedding_summary,
        "vocabularies": vocab_summary,
        "eos": eos_summary,
        "dedupe": dedupe_summary,
        "xT_target": xt_summary,
        "xG_target": xg_summary,
        "post_whistle_audit": continuity_audit,
        "dataset": dataset_summary,
        "artifacts": artifacts,
    }

    if phase3_gate["gate_status"] == "fail":
        raise AssertionError("Phase 3 gate failed. Inspect phase3_summary.json for details.")

    gc.collect()
    return df_final, summary


def run_phase3(
    base_dir: Path,
    *,
    run_label: str = "run_current",
    config: DataPrepConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DataPrepConfig()
    resolved_run_label = resolve_run_label(base_dir, run_label)
    paths = DataPrepPaths(base_dir, run_label=resolved_run_label)
    paths.ensure_dirs()

    tracking_rel_path = paths.phase2_tracking_event_relative_output
    tracking_rel_exists = tracking_rel_path.exists()
    tracking_processing_skipped = False
    if paths.phase2_summary_output.exists():
        try:
            with paths.phase2_summary_output.open("r", encoding="utf-8") as f:
                phase2_summary = json.load(f)
            tracking_processing_skipped = bool(phase2_summary.get("tracking_processing_skipped", False))
        except Exception:
            tracking_processing_skipped = False

    required_phase2_artifacts = [paths.phase2_events_output]
    if tracking_rel_exists or (not tracking_processing_skipped):
        required_phase2_artifacts.append(tracking_rel_path)

    require_artifacts_exist(
        required_phase2_artifacts,
        stage_name="Phase 3",
        run_root=paths.runs_root_dir,
    )

    phase2_events = read_parquet(paths.phase2_events_output)
    games = read_parquet(paths.raw_games_path, columns=["game_id", "home_team_id", "away_team_id"])

    if tracking_rel_exists:
        tracking_rel = read_parquet(tracking_rel_path)
        require_columns(tracking_rel, ["game_id", "sl_event_id"], "phase2_tracking_event_relative")
        tracking_cols = [
            c
            for c in tracking_rel.columns
            if c.startswith("actor_") or c.startswith("tm_") or c.startswith("opp_")
        ]
        tracking_merge = tracking_rel[["game_id", "sl_event_id", *tracking_cols]].drop_duplicates(
            subset=["game_id", "sl_event_id"],
            keep="first",
        )
        events = phase2_events.merge(tracking_merge, on=["game_id", "sl_event_id"], how="left")
    else:
        events = phase2_events.copy()

    df_final, prep_summary = run_phase3_tensor_prep(
        events,
        games,
        cfg,
        paths.phase3_dir,
        require_tracking_features=bool(tracking_rel_exists),
    )

    phase3_core_gate = prep_summary.get("gates", {}).get("phase3", {"gate_status": prep_summary.get("gate_status", "pass")})
    key_parity_gate = validate_phase2_phase3_key_parity(phase2_events, df_final)
    phase3_overall_gate = combine_gate_reports(phase3_core_gate, key_parity_gate)

    phase3_summary = {
        "generated_at_utc": utc_now_iso(),
        "phase": "phase3",
        "run_label": resolved_run_label,
        "run_root": str(paths.runs_root_dir),
        "gate_status": phase3_overall_gate["gate_status"],
        "gates": {
            "phase3": phase3_core_gate,
            "phase2_phase3_key_parity": key_parity_gate,
            "phase3_overall": phase3_overall_gate,
        },
        "artifacts": prep_summary["artifacts"],
        "phase3_summary": {
            **prep_summary,
            "phase2_tracking_processing_skipped": bool(tracking_processing_skipped),
            "phase2_tracking_event_relative_included": bool(tracking_rel_exists),
            "key_parity_audit": key_parity_gate.get("audit", {}),
        },
    }

    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase3_events_with_embedding_indices",
        output_path=Path(prep_summary["artifacts"]["events_with_embedding_indices"]["path"]),
        rows=int(prep_summary["artifacts"]["events_with_embedding_indices"]["rows"]),
        columns=list(prep_summary["artifacts"]["events_with_embedding_indices"]["columns"]),
        extra={"gate_status": phase3_summary["gate_status"]},
    )
    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase3_text_embedding_idx_map",
        output_path=paths.phase3_text_embedding_idx_map_output,
        rows=int(prep_summary["embeddings"]["unique_texts"]),
        columns=["text_embedding_idx", "embedding_text_clean"],
        extra={"gate_status": phase3_summary["gate_status"]},
    )
    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase3_tensor_ready_dataset",
        output_path=Path(prep_summary["artifacts"]["tensor_ready_dataset"]["path"]),
        rows=int(prep_summary["artifacts"]["tensor_ready_dataset"]["rows"]),
        columns=list(prep_summary["artifacts"]["tensor_ready_dataset"]["columns"]),
        extra={"gate_status": phase3_summary["gate_status"]},
    )

    write_json(paths.phase3_summary_output, phase3_summary)

    if phase3_summary["gate_status"] == "fail":
        raise AssertionError("Phase 3 gate failed. See phase3_summary.json for details.")

    del phase2_events
    del events
    del games
    if tracking_rel_exists:
        del tracking_rel
    del df_final
    gc.collect()

    return phase3_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 tensor-ready dataset pipeline")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--run-label",
        type=str,
        default="run_current",
        help="Run folder label. Use 'latest' to reuse the newest existing pipeline run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_label = resolve_run_label(args.base_dir, args.run_label)
    if run_label != args.run_label:
        print(f"Resolved run label alias '{args.run_label}' -> '{run_label}'")
    summary = run_phase3(args.base_dir, run_label=run_label)
    print(summary)


if __name__ == "__main__":
    main()
