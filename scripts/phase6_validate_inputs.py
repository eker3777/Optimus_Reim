from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sprint_week_utils import TransformerXTPaths, utc_now_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 6 modeling input artifacts")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument("--sample-rows", type=int, default=2000)
    return parser.parse_args()


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")


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


def _resolve_tracking_path(phase2_dir: Path, phase3_dir: Path) -> Path:
    candidates = [
        phase2_dir / "tracking_absolute_pinned.parquet",
        phase2_dir / "tracking_tensor_pinned.parquet",
        phase3_dir / "tracking_absolute_pinned.parquet",
        phase3_dir / "tracking_tensor_pinned.parquet",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    return existing if existing is not None else candidates[0]


def _check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")
    return required


def run_validation(base_dir: Path, sample_rows: int) -> dict:
    run_dir = _latest_pipeline_run_dir(base_dir)
    phase3_dir = _pipeline_phase_dir(base_dir, "phase3")
    phase2_dir = _pipeline_phase_dir(base_dir, "phase2")

    tensor_path = phase3_dir / "tensor_ready_dataset.parquet"
    embedding_path = phase3_dir / "text_embeddings.npy"
    embedding_map_path = phase3_dir / "text_embedding_idx_map.parquet"
    vocab_path = phase3_dir / "vocabularies.json"
    feature_defs_path = phase3_dir / "feature_definitions.json"
    tracking_path = _resolve_tracking_path(phase2_dir, phase3_dir)
    faceoff_ref_path = phase2_dir / "faceoff_reference.parquet"
    penalty_ref_path = phase2_dir / "penalty_reference.parquet"

    for path in [
        tensor_path,
        embedding_path,
        embedding_map_path,
        vocab_path,
        feature_defs_path,
        tracking_path,
        faceoff_ref_path,
        penalty_ref_path,
    ]:
        _must_exist(path)

    tensor = pd.read_parquet(tensor_path).head(max(1, int(sample_rows)))
    tracking = pd.read_parquet(tracking_path).head(max(1, int(sample_rows)))
    faceoff_ref = pd.read_parquet(faceoff_ref_path)
    penalty_ref = pd.read_parquet(penalty_ref_path)

    events_only_required = [
        "game_id",
        "sl_event_id",
        "target",
        "text_embedding_idx",
        "event_type_id",
        "period_id",
    ]
    _check_required_columns(tensor, events_only_required, "tensor_ready_dataset")

    actor_opp_aliases = {
        "score_differential_actor": "score_differential_poss",
        "n_skaters_actor": "n_skaters_poss",
        "n_skaters_opp": "n_skaters_def",
        "net_empty_actor": "net_empty_poss",
        "net_empty_opp": "net_empty_def",
        "home_team_actor": "home_team_poss",
    }
    missing_actor_opp = [
        f"{new_col} (or legacy {legacy_col})"
        for new_col, legacy_col in actor_opp_aliases.items()
        if (new_col not in tensor.columns and legacy_col not in tensor.columns)
    ]
    if missing_actor_opp:
        raise KeyError(f"tensor_ready_dataset missing required actor/opp context: {missing_actor_opp}")

    has_xy = ({"x", "y"}.issubset(tensor.columns)) or ({"x_adj", "y_adj"}.issubset(tensor.columns))
    if not has_xy:
        raise KeyError("tensor_ready_dataset missing spatial coordinates: expected x/y or x_adj/y_adj")

    if ("period_time" not in tensor.columns) and ("period_time_sec" not in tensor.columns):
        raise KeyError("tensor_ready_dataset missing period_time/period_time_sec")

    warnings: list[str] = []

    event_type_norm = tensor["event_type"].astype(str).str.strip().str.lower() if "event_type" in tensor.columns else pd.Series("", index=tensor.index)
    save_rows = tensor.loc[event_type_norm.eq("save")].copy() if "event_type" in tensor.columns else pd.DataFrame()
    save_row_count = int(len(save_rows))
    save_goalie_id_non_null = 0
    save_source_link_non_null = 0
    save_team_non_null = 0
    save_goalie_coverage = 0.0
    save_source_link_coverage = 0.0
    if save_row_count > 0:
        if "goalie_id" in save_rows.columns:
            save_goalie_id_non_null = int(save_rows["goalie_id"].notna().sum())
        elif "player_id" in save_rows.columns:
            save_goalie_id_non_null = int(save_rows["player_id"].notna().sum())

        if "save_source_sl_event_id" in save_rows.columns:
            save_source_link_non_null = int(pd.to_numeric(save_rows["save_source_sl_event_id"], errors="coerce").notna().sum())

        if "team_id" in save_rows.columns:
            save_team_non_null = int(save_rows["team_id"].notna().sum())

        save_goalie_coverage = float(save_goalie_id_non_null) / float(max(1, save_row_count))
        save_source_link_coverage = float(save_source_link_non_null) / float(max(1, save_row_count))

        if save_goalie_coverage < 0.95:
            warnings.append(
                "save-row goalie identity coverage below 95%; goalie ledger attribution may be incomplete"
            )
        if save_source_link_coverage < 0.95:
            warnings.append(
                "save-row source linkage coverage below 95%; stoppage/goalie override diagnostics may degrade"
            )

    tracking_required = ["game_id", "sl_event_id"]
    for side in ["Home", "Away"]:
        for slot in range(6):
            for metric in [
                "X",
                "Y",
                "Vel_X",
                "Vel_Y",
                "is_present",
                "is_consistent",
            ]:
                tracking_required.append(f"{side}_Track_{slot}_{metric}")
    _check_required_columns(tracking, tracking_required, "tracking_tensor_pinned")

    faceoff_required = [
        "game_id",
        "sl_event_id",
        "kept_sl_event_id",
        "opposing_player_id",
        "opposing_team_id",
    ]
    penalty_required = [
        "game_id",
        "sl_event_id",
        "kept_sl_event_id",
        "penaltydrawn_player_id",
        "penaltydrawn_team_id",
    ]
    _check_required_columns(faceoff_ref, faceoff_required, "phase2_faceoff_reference")
    _check_required_columns(penalty_ref, penalty_required, "phase2_penalty_reference")

    def _coerce_event_id(series: pd.Series) -> pd.Series:
        num = pd.to_numeric(series, errors="coerce")
        num = num.where(np.isclose(num % 1, 0), np.nan)
        return num.astype("Int64")

    tensor_key = tensor[["game_id", "sl_event_id"]].copy()
    tensor_key["game_id"] = tensor_key["game_id"].astype(str)
    tensor_key["sl_event_id"] = _coerce_event_id(tensor_key["sl_event_id"])
    tensor_key = tensor_key.dropna(subset=["sl_event_id"]).drop_duplicates(["game_id", "sl_event_id"])
    tensor_keys = set(tensor_key.itertuples(index=False, name=None))

    faceoff_kept = faceoff_ref[["game_id", "kept_sl_event_id"]].copy()
    faceoff_kept["game_id"] = faceoff_kept["game_id"].astype(str)
    faceoff_kept["kept_sl_event_id"] = _coerce_event_id(faceoff_kept["kept_sl_event_id"])
    faceoff_kept = faceoff_kept.dropna(subset=["kept_sl_event_id"]).drop_duplicates(["game_id", "kept_sl_event_id"])

    penalty_kept = penalty_ref[["game_id", "kept_sl_event_id"]].copy()
    penalty_kept["game_id"] = penalty_kept["game_id"].astype(str)
    penalty_kept["kept_sl_event_id"] = _coerce_event_id(penalty_kept["kept_sl_event_id"])
    penalty_kept = penalty_kept.dropna(subset=["kept_sl_event_id"]).drop_duplicates(["game_id", "kept_sl_event_id"])

    faceoff_matches = int(
        sum((gid, sid) in tensor_keys for gid, sid in faceoff_kept.itertuples(index=False, name=None))
    )
    penalty_matches = int(
        sum((gid, sid) in tensor_keys for gid, sid in penalty_kept.itertuples(index=False, name=None))
    )
    faceoff_match_rate = float(faceoff_matches / max(1, len(faceoff_kept)))
    penalty_match_rate = float(penalty_matches / max(1, len(penalty_kept)))

    if faceoff_match_rate < 0.95:
        warnings.append(
            f"phase2 faceoff_reference kept key coverage below 95% ({faceoff_matches}/{len(faceoff_kept)})"
        )
    if penalty_match_rate < 0.95:
        warnings.append(
            f"phase2 penalty_reference kept key coverage below 95% ({penalty_matches}/{len(penalty_kept)})"
        )

    actor_flag_aliases = [
        (f"{side}_Track_{slot}_is_actor", f"{side}_Track_{slot}_is_primary_actor")
        for side in ["Home", "Away"]
        for slot in range(6)
    ]
    missing_actor_flags = [
        f"{legacy} (or {current})"
        for legacy, current in actor_flag_aliases
        if (legacy not in tracking.columns and current not in tracking.columns)
    ]
    if missing_actor_flags:
        raise KeyError(
            "tracking_tensor_pinned missing actor-flag columns (legacy or current naming): "
            f"{missing_actor_flags}"
        )

    expected_new_cols: list[str] = []
    expected_legacy_actor_cols: list[str] = []
    for side in ["Home", "Away"]:
        for slot in range(6):
            expected_new_cols.append(f"{side}_Track_{slot}_is_possessing_team")
            expected_new_cols.append(f"{side}_Track_{slot}_is_primary_actor")
            expected_new_cols.append(f"{side}_Track_{slot}_slot_vacant")
            expected_legacy_actor_cols.append(f"{side}_Track_{slot}_is_actor")

    missing_new_cols = [c for c in expected_new_cols if c not in tracking.columns]
    legacy_actor_present = [c for c in expected_legacy_actor_cols if c in tracking.columns]
    if missing_new_cols and not legacy_actor_present:
        raise KeyError(
            "tracking_tensor_pinned does not contain new actor/possession/vacancy columns "
            "or legacy actor columns needed for compatibility"
        )

    emb = np.load(embedding_path, mmap_mode="r")
    emb_rows = int(emb.shape[0])
    emb_dim = int(emb.shape[1]) if emb.ndim >= 2 else 0

    text_idx = pd.to_numeric(tensor["text_embedding_idx"], errors="coerce")
    bad_idx = int(((text_idx < 0) | (text_idx >= emb_rows) | text_idx.isna()).sum())
    if bad_idx > 0:
        raise ValueError(f"text_embedding_idx has out-of-range/non-numeric values: {bad_idx}")

    status = "warn" if (missing_new_cols or warnings) else "pass"
    if missing_new_cols:
        warnings.append(
            "tracking tensor uses legacy schema for some columns; rerun Phase 2 absolute tracking "
            "to enable full is_primary_actor/is_possessing_team/slot_vacant coverage"
        )

    return {
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "inputs": {
            "pipeline_run_dir": str(run_dir),
            "phase2_dir": str(phase2_dir),
            "phase3_dir": str(phase3_dir),
            "tensor_ready_dataset": str(tensor_path),
            "tracking_tensor_pinned": str(tracking_path),
            "phase2_faceoff_reference": str(faceoff_ref_path),
            "phase2_penalty_reference": str(penalty_ref_path),
            "text_embeddings": str(embedding_path),
            "text_embedding_idx_map": str(embedding_map_path),
            "vocabularies": str(vocab_path),
            "feature_definitions": str(feature_defs_path),
        },
        "checks": {
            "events_only_required_columns": events_only_required,
            "actor_opp_alias_contract": actor_opp_aliases,
            "has_xy_or_adjusted_xy": bool(has_xy),
            "tracking_required_columns_count": int(len(tracking_required)),
            "tracking_actor_flag_aliases_checked": int(len(actor_flag_aliases)),
            "tracking_new_columns_missing": int(len(missing_new_cols)),
            "tracking_legacy_actor_columns_present": int(len(legacy_actor_present)),
            "phase2_faceoff_reference_rows": int(len(faceoff_ref)),
            "phase2_penalty_reference_rows": int(len(penalty_ref)),
            "phase2_faceoff_kept_key_rows": int(len(faceoff_kept)),
            "phase2_penalty_kept_key_rows": int(len(penalty_kept)),
            "phase2_faceoff_kept_key_matches": int(faceoff_matches),
            "phase2_penalty_kept_key_matches": int(penalty_matches),
            "phase2_faceoff_kept_key_match_rate": float(faceoff_match_rate),
            "phase2_penalty_kept_key_match_rate": float(penalty_match_rate),
            "embedding_rows": emb_rows,
            "embedding_dim": emb_dim,
            "save_rows_in_sample": save_row_count,
            "save_goalie_id_non_null_rows": int(save_goalie_id_non_null),
            "save_goalie_team_non_null_rows": int(save_team_non_null),
            "save_source_link_non_null_rows": int(save_source_link_non_null),
            "save_goalie_id_coverage": float(save_goalie_coverage),
            "save_source_link_coverage": float(save_source_link_coverage),
            "sample_rows_checked": int(len(tensor)),
            "tracking_sample_rows_checked": int(len(tracking)),
            "text_embedding_idx_out_of_range_rows": bad_idx,
        },
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    paths = TransformerXTPaths(args.base_dir, run_label=args.run_label)
    paths.ensure_all()

    summary = run_validation(args.base_dir, sample_rows=args.sample_rows)
    out_path = paths.logs_dir / "phase6_input_validation.json"
    write_json(out_path, summary)

    print(f"Saved: {out_path}")
    print(summary)


if __name__ == "__main__":
    main()
