from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sprint_week_utils import TransformerXTPaths


PENALTY_STATIC_FOR_BONUS = 0.02
PENALTY_STATIC_AGAINST_BONUS = 0.2
FACEOFF_ZONE_EDGE_FT = 25.5
FACEOFF_ZONE_BOUNDARY_EPS_FT = 0.25
FACEOFF_MIN_ZONE_SAMPLE_WARN = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure postprocess for Phase 6 raw OOF predictions")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--variant", type=str, default="events_only")
    parser.add_argument("--pipeline-run-label", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--penalty-static-for", type=float, default=PENALTY_STATIC_FOR_BONUS)
    parser.add_argument("--penalty-static-against", type=float, default=PENALTY_STATIC_AGAINST_BONUS)

    return parser.parse_args()


def _resolve_pipeline_run_dir(base_dir: Path, run_label: str | None) -> Path:
    runs_root = base_dir / "Data" / "Pipeline Runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Pipeline Runs directory not found: {runs_root}")

    if run_label:
        explicit = runs_root / run_label
        if not explicit.exists():
            raise FileNotFoundError(f"Requested pipeline run not found: {explicit}")
        return explicit

    runs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under: {runs_root}")

    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_event_keys(df: pd.DataFrame, key_col: str = "sl_event_id") -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    key_num = pd.to_numeric(out[key_col], errors="coerce").astype("Float64")
    out[key_col] = key_num.round(6)
    return out


def _parse_period_time_to_seconds(s: pd.Series) -> pd.Series:
    base = pd.to_numeric(s, errors="coerce")
    if base.isna().any():
        text = s.astype(str).str.strip()
        mmss = text.str.extract(r"^(\d{1,2}):(\d{2})$")
        parsed = pd.to_numeric(mmss[0], errors="coerce") * 60 + pd.to_numeric(mmss[1], errors="coerce")
        base = base.fillna(parsed)
    return base



def _resolve_sort_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "period",
        "game_event_id",
        "period_time_sec",
        "period_time",
        "sl_event_id",
        "sequence_event_id",
    ]
    return [c for c in candidates if c in df.columns]


def _build_opponent_team_map(df: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    if "team_id" not in df.columns:
        return {}

    team_pairs = (
        df[["game_id", "team_id"]]
        .dropna()
        .astype({"game_id": str, "team_id": str})
        .drop_duplicates()
        .groupby("game_id", sort=False)["team_id"]
        .agg(list)
    )

    mapping: Dict[Tuple[str, str], str] = {}
    for game_id, teams in team_pairs.items():
        if len(teams) != 2:
            continue
        a, b = teams
        mapping[(str(game_id), str(a))] = str(b)
        mapping[(str(game_id), str(b))] = str(a)
    return mapping


def _enrich_with_phase2_events(raw_oof: pd.DataFrame, phase2_events_path: Path) -> pd.DataFrame:
    phase2_events = pd.read_parquet(phase2_events_path)
    phase2_events = _normalize_event_keys(phase2_events, key_col="sl_event_id")

    enrich_cols = [
        "game_id",
        "sl_event_id",
        "event_type",
        "team_id",
        "player_id",
        "period",
        "period_time",
        "period_time_sec",
        "x",
        "x_adj",
        "y_adj",
        "outcome",
        "sequence_id",
        "sequence_event_id",
        "game_event_id",
        "is_goal",
        "goalie_id",
        "opp_goalie_id",
        "is_synthetic_save",
        "save_source_sl_event_id",
        "save_source_event_type",
        "source_linked_goal_sl_event_id",
        "goal_converted_source",
        "goal_linked_source_sl_event_id",
        "is_empty_net_source_attempt",
        "home_goalie_id",
        "away_goalie_id",
        "target",
    ]
    enrich_cols = [c for c in enrich_cols if c in phase2_events.columns]
    phase2_trim = phase2_events[enrich_cols].drop_duplicates(["game_id", "sl_event_id"])

    # Right merge: Phase 2 is the spine. OOF rows that match get their prediction
    # columns overlaid; EOS events (goals, whistles) with no OOF counterpart are
    # included as NaN-prediction rows for downstream xT calculations.
    merged = raw_oof.merge(
        phase2_trim,
        on=["game_id", "sl_event_id"],
        how="right",
        suffixes=("", "_phase2"),
    )

    # For columns that exist in both frames, prefer the OOF value where present
    # and fall back to the Phase 2 value. This preserves model outputs on matched
    # rows while populating context on unmatched EOS rows.
    for col in [
        "event_type",
        "team_id",
        "player_id",
        "period",
        "period_time",
        "period_time_sec",
        "x",
        "x_adj",
        "y_adj",
        "outcome",
        "sequence_id",
        "sequence_event_id",
        "game_event_id",
        "is_goal",
        "target",
    ]:
        phase2_col = f"{col}_phase2"
        if phase2_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[phase2_col]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[phase2_col])

    # Goalie IDs are not suffix-duplicated in the right merge since they only
    # exist in Phase 2, but guard defensively in case schema changes.
    if "goalie_id_phase2" in merged.columns and "goalie_id" not in merged.columns:
        merged["goalie_id"] = merged["goalie_id_phase2"]
    if "opp_goalie_id_phase2" in merged.columns and "opp_goalie_id" not in merged.columns:
        merged["opp_goalie_id"] = merged["opp_goalie_id_phase2"]

    # Normalize period_time_sec: parse from MM:SS string if numeric coercion fails.
    if "period_time_sec" not in merged.columns and "period_time" in merged.columns:
        merged["period_time_sec"] = _parse_period_time_to_seconds(merged["period_time"])
    elif "period_time_sec" in merged.columns:
        merged["period_time_sec"] = pd.to_numeric(merged["period_time_sec"], errors="coerce")

    # Re-sort to maintain chronological integrity across all games.
    sort_cols = [c for c in ["game_id", "period", "period_time_sec", "sl_event_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    return merged

def _normalize_goal_rows_to_scoring_actor(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    out["event_type"] = out.get("event_type", "").astype(str).str.strip().str.lower()

    required_probs = ["P_actor_goal", "P_opp_goal"]
    missing_probs = [c for c in required_probs if c not in out.columns]
    if missing_probs:
        raise RuntimeError(f"Cannot normalize goal rows without probability columns: {missing_probs}")

    if "P_no_goal" not in out.columns:
        out["P_no_goal"] = np.nan

    goal_mask = out["event_type"].eq("goal")
    out.loc[goal_mask, "P_actor_goal"] = 1.0
    out.loc[goal_mask, "P_opp_goal"] = 0.0
    out.loc[goal_mask, "P_no_goal"] = 0.0

    audit = {
        "goal_rows_total": int(goal_mask.sum()),
        "goal_probability_rows_overridden": int(goal_mask.sum()),
    }
    return out, audit


def _export_faceoff_baselines_inspection(df: pd.DataFrame, diagnostics_dir: Path) -> tuple[Path, Dict[str, Any]]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / "faceoff_baselines_inspection.csv"

    out = df.copy()
    out["event_type"] = out.get("event_type", "").astype(str).str.strip().str.lower()
    faceoff_mask = out["event_type"].eq("faceoff")
    baseline_mask = pd.to_numeric(out.get("faceoff_baseline_for", np.nan), errors="coerce").notna()
    rows = out.loc[faceoff_mask & baseline_mask].copy()

    keep_cols = [
        "game_id",
        "sl_event_id",
        "team_id",
        "period",
        "period_time",
        "x",
        "x_adj",
        "faceoff_zone_bucket",
        "faceoff_role_assignment",
        "faceoff_baseline_source",
        "faceoff_baseline_for",
        "faceoff_baseline_against",
        "faceoff_baseline_context",
        "faceoff_baseline_sample_count",
    ]
    keep_cols = [c for c in keep_cols if c in rows.columns]
    rows = rows[keep_cols].copy()
    rows.to_csv(output_path, index=False)

    audit = {
        "faceoff_rows_total": int(faceoff_mask.sum()),
        "faceoff_rows_with_baseline": int((faceoff_mask & baseline_mask).sum()),
        "inspection_csv": str(output_path),
    }
    return output_path, audit


def _inject_eos_whistles_and_apply_faceoff_baselines(
    df: pd.DataFrame,
    *,
    neutral_baseline_for: float,
    neutral_baseline_against: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = out.get("team_id", "").astype("string")
    out["event_type"] = out.get("event_type", "").astype(str).str.strip().str.lower()
    out["is_synthetic_whistle"] = 0

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    sort_cols = _resolve_sort_columns(out)
    if sort_cols:
        out = out.sort_values(group_cols + sort_cols, kind="mergesort").reset_index(drop=True)

    goal_mask = out["event_type"].eq("goal")
    synthetic_count = int(goal_mask.sum())

    if synthetic_count > 0:
        sl_num = pd.to_numeric(out.get("sl_event_id"), errors="coerce")
        sl_fallback = pd.Series(np.arange(len(out), dtype=float), index=out.index)
        goal_rows = out.loc[goal_mask].copy()
        goal_rows["event_type"] = "whistle"
        goal_rows["is_synthetic_whistle"] = 1
        goal_rows["sl_event_id"] = (
            pd.to_numeric(sl_num.loc[goal_mask], errors="coerce")
            .where(pd.to_numeric(sl_num.loc[goal_mask], errors="coerce").notna(), sl_fallback.loc[goal_mask])
            .astype(float)
            + 0.1
        )
        expanded = pd.concat([out, goal_rows], ignore_index=True, sort=False)
    else:
        expanded = out.copy()

    expanded["sl_event_id"] = pd.to_numeric(expanded.get("sl_event_id"), errors="coerce")
    sort_cols2 = _resolve_sort_columns(expanded)
    if sort_cols2:
        expanded = expanded.sort_values(group_cols + sort_cols2, kind="mergesort").reset_index(drop=True)

    grp = expanded.groupby(group_cols, sort=False, dropna=False)
    expanded["next_event_type"] = grp["event_type"].shift(-1)
    expanded["next_event_is_whistle"] = expanded["next_event_type"].astype(str).str.strip().str.lower().eq("whistle")
    expanded["next_event_is_goal"] = expanded["next_event_type"].astype(str).str.strip().str.lower().eq("goal")
    expanded["next_event_is_eos"] = expanded["next_event_type"].astype(str).str.strip().str.lower().isin(["whistle", "goal", "end_of_period"])

    expanded["_row_idx"] = np.arange(len(expanded), dtype=np.int64)
    faceoff_mask = expanded["event_type"].eq("faceoff")
    whistle_mask = expanded["event_type"].eq("whistle")
    expanded["_faceoff_row_idx"] = expanded["_row_idx"].where(faceoff_mask, np.nan)
    # Groupwise bfill gives the immediate future faceoff row index without idxmax edge cases.
    expanded["_next_faceoff_row_idx"] = grp["_faceoff_row_idx"].transform("bfill")

    has_next_faceoff = whistle_mask & expanded["_next_faceoff_row_idx"].notna()
    fallback_mask = whistle_mask.copy()

    if has_next_faceoff.any():
        idx = expanded.loc[has_next_faceoff, "_next_faceoff_row_idx"].astype(int)
        expanded.loc[has_next_faceoff, "team_id"] = expanded.loc[idx, "team_id"].astype("string").values
        expanded.loc[has_next_faceoff, "P_actor_goal"] = pd.to_numeric(
            expanded.loc[idx, "faceoff_baseline_for"], errors="coerce"
        ).fillna(float(neutral_baseline_for)).values
        expanded.loc[has_next_faceoff, "P_opp_goal"] = pd.to_numeric(
            expanded.loc[idx, "faceoff_baseline_against"], errors="coerce"
        ).fillna(float(neutral_baseline_against)).values
        fallback_mask.loc[has_next_faceoff] = False

    expanded.loc[fallback_mask, "P_actor_goal"] = float(neutral_baseline_for)
    expanded.loc[fallback_mask, "P_opp_goal"] = float(neutral_baseline_against)
    recomputed_no_goal = np.clip(
        1.0
        - pd.to_numeric(expanded.get("P_actor_goal", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(expanded.get("P_opp_goal", 0.0), errors="coerce").fillna(0.0),
        0.0,
        1.0,
    )
    if "P_no_goal" not in expanded.columns:
        expanded["P_no_goal"] = np.nan
    expanded.loc[whistle_mask, "P_no_goal"] = pd.Series(recomputed_no_goal, index=expanded.index).loc[whistle_mask].values

    expanded = expanded.drop(columns=["_faceoff_row_idx", "_next_faceoff_row_idx", "_row_idx"], errors="ignore")
    audit = {
        "synthetic_whistles_injected": int(synthetic_count),
        "whistle_rows_total": int(whistle_mask.sum()),
        "whistle_rows_with_faceoff_baseline": int(has_next_faceoff.sum()),
        "whistle_rows_with_neutral_fallback": int((whistle_mask & fallback_mask).sum()),
        "neutral_baseline_for": float(neutral_baseline_for),
        "neutral_baseline_against": float(neutral_baseline_against),
    }
    return expanded, audit


def _compute_universal_actor_relative_deltas(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    required = ["game_id", "team_id", "P_actor_goal", "P_opp_goal"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing required columns for universal deltas: {missing}")

    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = out["team_id"].astype("string")

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    sort_cols = _resolve_sort_columns(out)
    if sort_cols:
        out = out.sort_values(group_cols + sort_cols, kind="mergesort").reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False, dropna=False)
    
    # Forward-fill team_id so whistle/stoppage NaN doesn't break same-team continuous sequences
    filled_teams = grp["team_id"].ffill().astype("string")
    out["_filled_team"] = filled_teams
    prev_team = out.groupby(group_cols, sort=False, dropna=False)["_filled_team"].shift(1) if len(group_cols) > 0 else filled_teams.shift(1)
    out = out.drop(columns=["_filled_team"])
    
    prev_actor = pd.to_numeric(grp["P_actor_goal"].shift(1), errors="coerce")
    prev_opp = pd.to_numeric(grp["P_opp_goal"].shift(1), errors="coerce")
    # string dtype comparisons can yield <NA>; treat unknown alignment as team-flip=False fallback.
    same_team = filled_teams.eq(prev_team).fillna(False).astype(bool)
    first_in_group = grp.cumcount().eq(0)

    prior_for = pd.Series(np.where(same_team, prev_actor, prev_opp), index=out.index)
    prior_against = pd.Series(np.where(same_team, prev_opp, prev_actor), index=out.index)
    prior_for = pd.to_numeric(prior_for, errors="coerce").fillna(0.0).mask(first_in_group, 0.0)
    prior_against = pd.to_numeric(prior_against, errors="coerce").fillna(0.0).mask(first_in_group, 0.0)

    curr_for = pd.to_numeric(out["P_actor_goal"], errors="coerce").fillna(0.0)
    curr_against = pd.to_numeric(out["P_opp_goal"], errors="coerce").fillna(0.0)

    out["previous_row_xT"] = prior_for
    out["Actor_xT_For"] = curr_for - prior_for
    out["Actor_xT_Against"] = curr_against - prior_against
    out["Actor_Net_xT"] = out["Actor_xT_For"] - out["Actor_xT_Against"]
    out["prior_state_reset"] = first_in_group.astype(np.float32)
    out["Adjusted_xT_For"] = out["Actor_xT_For"]
    out["Adjusted_xT_Against"] = out["Actor_xT_Against"]
    out["Adjusted_Net_xT"] = out["Adjusted_xT_For"] - out["Adjusted_xT_Against"]
    out["adjustment_source"] = "base_actor"
    
    # Skaters receive no EV reward for goals (the outcome), only the marginal xT of their shot
    goal_mask = out.get("is_goal", out.get("event_type", "") == "goal") == True
    for col in ["Actor_xT_For", "Actor_xT_Against", "Actor_Net_xT", "Adjusted_xT_For", "Adjusted_xT_Against", "Adjusted_Net_xT"]:
        if col in out.columns:
            out.loc[goal_mask, col] = 0.0

    audit = {
        "rows": int(len(out)),
        "groups": int(out.groupby(group_cols, dropna=False).ngroups),
        "first_in_group_rows": int(first_in_group.sum()),
        "team_flip_rows": int((~same_team & ~first_in_group).sum()),
    }
    return out, audit


def _compute_faceoff_zone_baselines(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()

    required = ["game_id", "team_id", "P_actor_goal", "P_opp_goal"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Raw OOF missing required columns for Pass 1: {missing}")

    out["team_id"] = out["team_id"].astype(str)
    out["event_type"] = out.get("event_type", "").astype(str).str.strip().str.lower()
    out["faceoff_zone_bucket"] = "not_faceoff"
    out["faceoff_role_assignment"] = "not_faceoff"
    out["faceoff_baseline_source"] = "zero_reset_or_inherited"
    out["faceoff_baseline_for"] = np.nan
    out["faceoff_baseline_against"] = np.nan
    out["faceoff_baseline_context"] = pd.NA
    out["faceoff_baseline_sample_count"] = 0

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    sort_cols = _resolve_sort_columns(out)
    if sort_cols:
        out = out.sort_values(group_cols + sort_cols).reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False, dropna=False)
    prev_team = grp["team_id"].shift(1)
    prev_actor = pd.to_numeric(grp["P_actor_goal"].shift(1), errors="coerce")
    prev_opp = pd.to_numeric(grp["P_opp_goal"].shift(1), errors="coerce")

    same_team = out["team_id"] == prev_team
    prev_for_aligned = np.where(same_team, prev_actor, prev_opp)
    prev_against_aligned = np.where(same_team, prev_opp, prev_actor)

    prior_event_type = grp["event_type"].shift(1).astype(str).str.strip().str.lower()
    prior_terminal = prior_event_type.isin({"whistle", "goal", "end_of_period"})
    curr_hard_reset = out["event_type"].isin({"faceoff", "period_start"})

    if "sequence_id" in out.columns:
        sequence_changed = out["sequence_id"] != grp["sequence_id"].shift(1)
    else:
        sequence_changed = pd.Series(False, index=out.index)

    first_in_group = grp.cumcount().eq(0)
    reset_mask = first_in_group | prior_terminal | curr_hard_reset | sequence_changed

    prev_for_aligned = pd.to_numeric(pd.Series(prev_for_aligned, index=out.index), errors="coerce").fillna(0.0)
    prev_against_aligned = pd.to_numeric(pd.Series(prev_against_aligned, index=out.index), errors="coerce").fillna(0.0)

    faceoff_reset_mask = reset_mask & out["event_type"].eq("faceoff")
    faceoff_audit: Dict[str, Any] = {
        "zone_edge_ft": float(FACEOFF_ZONE_EDGE_FT),
        "zone_boundary_eps_ft": float(FACEOFF_ZONE_BOUNDARY_EPS_FT),
        "min_zone_sample_warn_threshold": int(FACEOFF_MIN_ZONE_SAMPLE_WARN),
        "faceoff_rows": int(out["event_type"].eq("faceoff").sum()),
        "faceoff_reset_rows": int(faceoff_reset_mask.sum()),
        "x_adj_required": True,
        "x_tiebreak_required": True,
        "x_boundary_tiebreak_rows": 0,
        "rows_with_dynamic_baseline": 0,
        "rows_with_fallback_zero": 0,
        "rows_oz": 0,
        "rows_dz": 0,
        "rows_nz": 0,
        "rows_unknown_zone": 0,
        "mirrored_neutral_formula_mode": True,
        "fallback_missing_zone_components": False,
        "oz_sample_count": 0,
        "dz_sample_count": 0,
        "nz_sample_count": 0,
        "low_sample_zone_warnings": [],
        "warning_low_sample_zone_baseline": False,
    }

    if faceoff_reset_mask.any():
        for required_coord in ["x_adj", "x"]:
            if required_coord not in out.columns:
                raise RuntimeError(
                    f"Raw OOF missing required faceoff coordinate column '{required_coord}' for zone-aware baseline injection"
                )

        x_adj = pd.to_numeric(out["x_adj"], errors="coerce")
        x_raw = pd.to_numeric(out["x"], errors="coerce")
        abs_x_adj = x_adj.abs()

        zone_bucket = pd.Series("unknown", index=out.index, dtype="object")
        zone_bucket.loc[abs_x_adj <= FACEOFF_ZONE_EDGE_FT] = "nz"
        zone_bucket.loc[x_adj > FACEOFF_ZONE_EDGE_FT] = "oz"
        zone_bucket.loc[x_adj < -FACEOFF_ZONE_EDGE_FT] = "dz"
        out.loc[faceoff_reset_mask, "faceoff_zone_bucket"] = zone_bucket.loc[faceoff_reset_mask].values

        role = pd.Series("unknown", index=out.index, dtype="object")
        role.loc[zone_bucket.eq("oz")] = "offense"
        role.loc[zone_bucket.eq("dz")] = "defense"
        role.loc[zone_bucket.eq("nz")] = "neutral"

        boundary_mask = faceoff_reset_mask & (abs_x_adj - FACEOFF_ZONE_EDGE_FT).abs().le(FACEOFF_ZONE_BOUNDARY_EPS_FT)
        boundary_resolved = boundary_mask & x_raw.notna()
        role.loc[boundary_resolved & x_raw.gt(0)] = "offense"
        role.loc[boundary_resolved & x_raw.lt(0)] = "defense"
        out.loc[faceoff_reset_mask, "faceoff_role_assignment"] = role.loc[faceoff_reset_mask].values

        out["faceoff_baseline_context"] = pd.Series(
            np.where(zone_bucket.eq("oz"), "oz_mirrored_neutral", np.where(zone_bucket.eq("dz"), "dz_mirrored_neutral", "nz_mirrored_neutral")),
            index=out.index,
        )

        role_offense = faceoff_reset_mask & role.eq("offense")
        role_defense = faceoff_reset_mask & role.eq("defense")
        role_neutral = faceoff_reset_mask & role.eq("neutral")

        oz_actor_samples = pd.to_numeric(out.loc[role_offense, "P_actor_goal"], errors="coerce").dropna()
        oz_opp_samples = pd.to_numeric(out.loc[role_offense, "P_opp_goal"], errors="coerce").dropna()
        dz_actor_samples = pd.to_numeric(out.loc[role_defense, "P_actor_goal"], errors="coerce").dropna()
        dz_opp_samples = pd.to_numeric(out.loc[role_defense, "P_opp_goal"], errors="coerce").dropna()
        nz_actor_samples = pd.to_numeric(out.loc[role_neutral, "P_actor_goal"], errors="coerce").dropna()
        nz_opp_samples = pd.to_numeric(out.loc[role_neutral, "P_opp_goal"], errors="coerce").dropna()

        oz_win_for = float(oz_actor_samples.mean()) if len(oz_actor_samples) else np.nan
        oz_win_against = float(oz_opp_samples.mean()) if len(oz_opp_samples) else np.nan
        dz_win_for = float(dz_actor_samples.mean()) if len(dz_actor_samples) else np.nan
        dz_win_against = float(dz_opp_samples.mean()) if len(dz_opp_samples) else np.nan
        nz_win_for = float(nz_actor_samples.mean()) if len(nz_actor_samples) else np.nan
        nz_win_against = float(nz_opp_samples.mean()) if len(nz_opp_samples) else np.nan

        fallback_missing = any(
            np.isnan(v)
            for v in [oz_win_for, oz_win_against, dz_win_for, dz_win_against, nz_win_for, nz_win_against]
        )

        if np.isnan(oz_win_for):
            oz_win_for = 0.0
        if np.isnan(oz_win_against):
            oz_win_against = 0.0
        if np.isnan(dz_win_for):
            dz_win_for = 0.0
        if np.isnan(dz_win_against):
            dz_win_against = 0.0
        if np.isnan(nz_win_for):
            nz_win_for = 0.0
        if np.isnan(nz_win_against):
            nz_win_against = 0.0

        # Exact mirrored-neutral constants (team-relative) from winner-row zone means.
        oz_neutral_for = (oz_win_for + dz_win_against) / 2.0
        oz_neutral_against = (oz_win_against + dz_win_for) / 2.0
        dz_neutral_for = (dz_win_for + oz_win_against) / 2.0
        dz_neutral_against = (dz_win_against + oz_win_for) / 2.0
        nz_neutral_for = (nz_win_for + nz_win_against) / 2.0
        nz_neutral_against = (nz_win_against + nz_win_for) / 2.0

        baseline_for = pd.Series(0.0, index=out.index, dtype=float)
        baseline_against = pd.Series(0.0, index=out.index, dtype=float)
        baseline_for.loc[role_offense] = oz_neutral_for
        baseline_against.loc[role_offense] = oz_neutral_against
        baseline_for.loc[role_defense] = dz_neutral_for
        baseline_against.loc[role_defense] = dz_neutral_against
        baseline_for.loc[role_neutral] = nz_neutral_for
        baseline_against.loc[role_neutral] = nz_neutral_against

        zone_sample_count = pd.Series(0, index=out.index, dtype=int)
        zone_sample_count.loc[role_offense] = int(len(oz_actor_samples))
        zone_sample_count.loc[role_defense] = int(len(dz_actor_samples))
        zone_sample_count.loc[role_neutral] = int(len(nz_actor_samples))

        known_role_mask = faceoff_reset_mask & role.isin(["offense", "defense", "neutral"])
        out.loc[known_role_mask, "faceoff_baseline_source"] = "winner_relative_zone_prior"
        out.loc[faceoff_reset_mask & (~known_role_mask), "faceoff_baseline_source"] = "fallback_zero_missing_zone"
        out.loc[faceoff_reset_mask, "faceoff_baseline_for"] = baseline_for.loc[faceoff_reset_mask].values
        out.loc[faceoff_reset_mask, "faceoff_baseline_against"] = baseline_against.loc[faceoff_reset_mask].values
        out.loc[faceoff_reset_mask, "faceoff_baseline_sample_count"] = zone_sample_count.loc[faceoff_reset_mask].values

        prev_for_aligned = prev_for_aligned.mask(known_role_mask, baseline_for)
        prev_against_aligned = prev_against_aligned.mask(known_role_mask, baseline_against)

        faceoff_audit.update(
            {
                "x_boundary_tiebreak_rows": int(boundary_resolved.sum()),
                "rows_with_dynamic_baseline": int(known_role_mask.sum()),
                "rows_with_fallback_zero": int((faceoff_reset_mask & (~known_role_mask)).sum()),
                "rows_oz": int((faceoff_reset_mask & zone_bucket.eq("oz")).sum()),
                "rows_dz": int((faceoff_reset_mask & zone_bucket.eq("dz")).sum()),
                "rows_nz": int((faceoff_reset_mask & zone_bucket.eq("nz")).sum()),
                "rows_unknown_zone": int((faceoff_reset_mask & zone_bucket.eq("unknown")).sum()),
                "fallback_missing_zone_components": bool(fallback_missing),
                "oz_sample_count": int(len(oz_actor_samples)),
                "dz_sample_count": int(len(dz_actor_samples)),
                "nz_sample_count": int(len(nz_actor_samples)),
                "oz_win_for": float(oz_win_for),
                "oz_win_against": float(oz_win_against),
                "dz_win_for": float(dz_win_for),
                "dz_win_against": float(dz_win_against),
                "nz_win_for": float(nz_win_for),
                "nz_win_against": float(nz_win_against),
                "oz_neutral_for": float(oz_neutral_for),
                "oz_neutral_against": float(oz_neutral_against),
                "dz_neutral_for": float(dz_neutral_for),
                "dz_neutral_against": float(dz_neutral_against),
                "nz_neutral_for": float(nz_neutral_for),
                "nz_neutral_against": float(nz_neutral_against),
            }
        )

        low_sample_zones: list[str] = []
        if bool(role_offense.any()) and len(oz_actor_samples) < FACEOFF_MIN_ZONE_SAMPLE_WARN:
            low_sample_zones.append("oz")
        if bool(role_defense.any()) and len(dz_actor_samples) < FACEOFF_MIN_ZONE_SAMPLE_WARN:
            low_sample_zones.append("dz")
        if bool(role_neutral.any()) and len(nz_actor_samples) < FACEOFF_MIN_ZONE_SAMPLE_WARN:
            low_sample_zones.append("nz")

        faceoff_audit["low_sample_zone_warnings"] = low_sample_zones
        faceoff_audit["warning_low_sample_zone_baseline"] = bool(low_sample_zones)

    return out, faceoff_audit


def _compute_goal_predecessor_warning_audit(df: pd.DataFrame) -> Dict[str, Any]:
    out = df.copy()
    evt = out.get("event_type", "").astype(str).str.strip().str.lower()
    out["_event_type_norm"] = evt

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    sort_cols = _resolve_sort_columns(out)
    if sort_cols:
        out = out.sort_values(group_cols + sort_cols).reset_index(drop=True)

    prev_event_type = out.groupby(group_cols, sort=False, dropna=False)["_event_type_norm"].shift(1)
    canonical_goal_mask = out["_event_type_norm"].eq("goal")
    
    if "tensor_is_goal_row" in out.columns:
        goal_mask = out["tensor_is_goal_row"].fillna(False).astype(bool)
        population_source = "tensor_goal_rows"
    else:
        goal_mask = canonical_goal_mask
        population_source = "canonical_goal_rows_fallback"

    goal_prev = prev_event_type.loc[goal_mask].fillna("<missing>").astype(str).str.strip().str.lower()
    expected_set = {
        "shot",
        "deflection",
        "defensive_deflection",
        "block",
        "pressure",
        "high_pressure",
    }

    def _is_expected(label: str) -> bool:
        if label in expected_set:
            return True
        return "pressure" in label

    expected_mask = goal_prev.map(_is_expected).fillna(False).astype(bool)
    counts = goal_prev.value_counts(dropna=False)
    unexpected_counts = counts.loc[[idx for idx in counts.index if not _is_expected(str(idx))]]

    return {
        "population_source": population_source,
        "goal_rows_total": int(goal_mask.sum()),
        "canonical_goal_rows_total": int(canonical_goal_mask.sum()),
        "goal_rows_with_prev_event": int(goal_prev.ne("<missing>").sum()),
        "expected_predecessor_rows": int(expected_mask.sum()),
        "unexpected_predecessor_rows": int((~expected_mask).sum()),
        "unexpected_predecessor_rate": float((~expected_mask).mean()) if len(goal_prev) else 0.0,
        "warning_unexpected_predecessors_present": bool((~expected_mask).any()),
        "predecessor_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
        "unexpected_predecessor_counts": {str(k): int(v) for k, v in unexpected_counts.to_dict().items()},
        "expected_predecessor_set": sorted(expected_set),
        "policy": "warn_only",
    }


def _resolve_empty_net_mask(df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    candidates = [
        "is_net_empty",
        "empty_net",
        "net_empty",
        "is_empty_net",
        "is_empty_net_goal",
    ]
    mask = pd.Series(False, index=df.index, dtype=bool)
    used_cols: list[str] = []

    for col in candidates:
        if col not in df.columns:
            continue

        raw = df[col]
        used_cols.append(col)
        if pd.api.types.is_bool_dtype(raw):
            part = raw.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(raw):
            part = pd.to_numeric(raw, errors="coerce").fillna(0).astype(int).eq(1)
        else:
            part = raw.astype("string").str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

        mask = mask | part.fillna(False).astype(bool)

    return mask.astype(bool), used_cols


def _build_goalie_ledger(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = out.get("team_id", "").astype(str)
    evt = out.get("event_type", "").astype(str).str.strip().str.lower()
    out["_event_type_norm"] = evt

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    sort_cols = _resolve_sort_columns(out)
    if sort_cols:
        out = out.sort_values(group_cols + sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False, dropna=False)
    out["_prev_team_id"] = grp["team_id"].shift(1).astype("string")
    out["_prev_event_type"] = grp["_event_type_norm"].shift(1).astype("string")
    out["_prev_P_actor_goal"] = pd.to_numeric(grp["P_actor_goal"].shift(1), errors="coerce")
    out["_prev_P_opp_goal"] = pd.to_numeric(grp["P_opp_goal"].shift(1), errors="coerce")
    out["_next_team_id"] = grp["team_id"].shift(-1).astype("string")
    out["_next_P_actor_goal"] = pd.to_numeric(grp["P_actor_goal"].shift(-1), errors="coerce")
    out["_next_P_opp_goal"] = pd.to_numeric(grp["P_opp_goal"].shift(-1), errors="coerce")

    if "next_event_type" in out.columns:
        next_evt = out["next_event_type"].astype("string").str.strip().str.lower()
    else:
        next_evt = grp["_event_type_norm"].shift(-1).astype("string")

    if "next_event_is_whistle" in out.columns:
        next_event_is_whistle = out["next_event_is_whistle"].fillna(False).astype(bool)
    else:
        next_event_is_whistle = next_evt.eq("whistle").fillna(False).astype(bool)

    if "next_event_is_goal" in out.columns:
        next_event_is_goal = out["next_event_is_goal"].fillna(False).astype(bool)
    else:
        next_event_is_goal = next_evt.eq("goal").fillna(False).astype(bool)

    if "next_event_is_eos" in out.columns:
        next_event_is_eos = out["next_event_is_eos"].fillna(False).astype(bool)
    else:
        next_event_is_eos = next_evt.isin(["whistle", "goal", "end_of_period"]).fillna(False).astype(bool)

    empty_net_mask, empty_net_cols = _resolve_empty_net_mask(out)
    empty_net_cols_str = ",".join(empty_net_cols) if empty_net_cols else ""

    if "goalie_id" in out.columns:
        save_goalie_id_series = out["goalie_id"].copy()
        if "opp_goalie_id" in out.columns:
            save_goalie_id_series = save_goalie_id_series.where(save_goalie_id_series.notna(), out["opp_goalie_id"])
    elif "opp_goalie_id" in out.columns:
        save_goalie_id_series = out["opp_goalie_id"].copy()
    else:
        save_goalie_id_series = pd.Series(pd.NA, index=out.index, dtype="object")

    # Goal-penalty attribution is strict: only opposing goalie ids are eligible.
    if "opp_goalie_id" in out.columns:
        goal_penalty_goalie_id_series = out["opp_goalie_id"].copy()
    else:
        goal_penalty_goalie_id_series = pd.Series(pd.NA, index=out.index, dtype="object")

    opp_map = _build_opponent_team_map(out)
    if opp_map:
        opp_map_series = pd.Series(opp_map, dtype="object")
        opp_keys = pd.MultiIndex.from_arrays(
            [out["game_id"].astype(str), out["team_id"].astype(str)]
        )
        goalie_team_series_default = pd.Series(
            opp_map_series.reindex(opp_keys).to_numpy(),
            index=out.index,
            dtype="string",
        )
    else:
        goalie_team_series_default = pd.Series(pd.NA, index=out.index, dtype="string")

    # Save component rows: include all save rows with provenance labels.
    if "is_synthetic_save" in out.columns:
        synthetic_save_flag = pd.to_numeric(out["is_synthetic_save"], errors="coerce").fillna(0).astype(int).eq(1)
    else:
        synthetic_save_flag = pd.Series(False, index=out.index)

    if "save_source_sl_event_id" in out.columns:
        save_source_link_present = pd.to_numeric(out["save_source_sl_event_id"], errors="coerce").notna()
    else:
        save_source_link_present = pd.Series(False, index=out.index)

    save_mask = evt.eq("save")
    save_provenance_bucket = pd.Series("neither", index=out.index, dtype="object")
    save_provenance_bucket.loc[save_mask & synthetic_save_flag & save_source_link_present] = "synthetic_and_source_linked"
    save_provenance_bucket.loc[save_mask & synthetic_save_flag & (~save_source_link_present)] = "synthetic_only"
    save_provenance_bucket.loc[save_mask & (~synthetic_save_flag) & save_source_link_present] = "source_linked_only"

    goalie_team_series_save = goalie_team_series_default.copy()
    synthetic_save_with_team = save_mask & synthetic_save_flag & out["team_id"].notna() & out["team_id"].astype(str).str.strip().ne("")
    goalie_team_series_save.loc[synthetic_save_with_team] = out.loc[synthetic_save_with_team, "team_id"].astype(str)

    restart_event_type = pd.Series(pd.NA, index=out.index, dtype="object")
    restart_sl_event_id = pd.Series(np.nan, index=out.index, dtype="float64")
    restart_team_id = pd.Series(pd.NA, index=out.index, dtype="object")
    restart_actor_goal = pd.Series(np.nan, index=out.index, dtype="float64")
    restart_opp_goal = pd.Series(np.nan, index=out.index, dtype="float64")
    restart_faceoff_baseline_for = pd.Series(np.nan, index=out.index, dtype="float64")
    restart_faceoff_baseline_against = pd.Series(np.nan, index=out.index, dtype="float64")

    out["_row_idx"] = np.arange(len(out), dtype=np.int64)
    out["_non_whistle_row_idx"] = out["_row_idx"].where(out["_event_type_norm"].ne("whistle"), np.nan)
    out["_next_non_whistle_row_idx"] = out.groupby(group_cols, sort=False, dropna=False)["_non_whistle_row_idx"].shift(-1)
    out["_next_non_whistle_row_idx"] = out.groupby(group_cols, sort=False, dropna=False)["_next_non_whistle_row_idx"].transform("bfill")

    next_non_whistle_row_idx = pd.to_numeric(out["_next_non_whistle_row_idx"], errors="coerce")
    valid_next = next_non_whistle_row_idx.notna()
    if valid_next.any():
        src_idx = next_non_whistle_row_idx.index[valid_next]
        dst_idx = next_non_whistle_row_idx.loc[valid_next].astype(int).to_numpy()

        restart_event_type.loc[src_idx] = out["_event_type_norm"].iloc[dst_idx].astype(str).values
        restart_sl_event_id.loc[src_idx] = pd.to_numeric(out["sl_event_id"].iloc[dst_idx], errors="coerce").values
        restart_team_id.loc[src_idx] = out["team_id"].iloc[dst_idx].astype(str).values
        restart_actor_goal.loc[src_idx] = pd.to_numeric(out.get("P_actor_goal", 0.0).iloc[dst_idx], errors="coerce").values
        restart_opp_goal.loc[src_idx] = pd.to_numeric(out.get("P_opp_goal", 0.0).iloc[dst_idx], errors="coerce").values
        restart_faceoff_baseline_for.loc[src_idx] = pd.to_numeric(out.get("faceoff_baseline_for", np.nan).iloc[dst_idx], errors="coerce").values
        restart_faceoff_baseline_against.loc[src_idx] = pd.to_numeric(out.get("faceoff_baseline_against", np.nan).iloc[dst_idx], errors="coerce").values

    out = out.drop(columns=["_row_idx", "_non_whistle_row_idx", "_next_non_whistle_row_idx"], errors="ignore")

    known_next = next_evt.notna() & next_evt.ne("")
    is_save_with_stoppage = save_mask & next_event_is_eos
    is_save_play_continued = save_mask & known_next & (~next_event_is_goal) & (~is_save_with_stoppage)
    save_with_whistle_then_restart = save_mask & next_event_is_whistle & restart_event_type.notna()
    save_with_whistle_then_faceoff = save_with_whistle_then_restart & restart_event_type.astype("string").str.strip().str.lower().eq("faceoff")

    save_team = out["team_id"].fillna("").astype(str)
    restart_team = restart_team_id.fillna("").astype(str)
    same_team_restart = save_team == restart_team

    restart_threat_against_save_team = pd.Series(
        np.where(same_team_restart, restart_opp_goal, restart_actor_goal), 
        index=out.index
    )
    
    restart_baseline_against_save_team = pd.Series(
        np.where(same_team_restart, restart_faceoff_baseline_against, restart_faceoff_baseline_for), 
        index=out.index
    )
    
    use_faceoff_baseline = save_with_whistle_then_faceoff & restart_baseline_against_save_team.notna()
    use_restart_probs = save_with_whistle_then_restart & (~use_faceoff_baseline) & restart_threat_against_save_team.notna()

    restart_baseline = pd.Series(np.nan, index=out.index)
    restart_baseline = restart_baseline.mask(use_faceoff_baseline, restart_baseline_against_save_team)
    restart_baseline = restart_baseline.mask(use_restart_probs, restart_threat_against_save_team)

    post_save_threat_against_save_team = pd.to_numeric(out["P_opp_goal"], errors="coerce").fillna(0.0)

    # Freeze bonus: Threat drops from post_save_state (curr_against) to restart_baseline
    freeze_delta = np.clip(post_save_threat_against_save_team - restart_baseline, 0.0, None)
    
    missing_restart_context = save_with_whistle_then_restart & (~use_faceoff_baseline) & (~use_restart_probs)
    freeze_fallback_mask = is_save_with_stoppage & missing_restart_context
    freeze_delta = pd.Series(freeze_delta, index=out.index)
    freeze_delta = freeze_delta.where(is_save_with_stoppage & (~missing_restart_context), 0.0)

    # Inherit natural save delta from universal pass
    native_save_xT_For = pd.to_numeric(out.get("Adjusted_xT_For", 0.0), errors="coerce").fillna(0.0)
    native_save_xT_Against = pd.to_numeric(out.get("Adjusted_xT_Against", 0.0), errors="coerce").fillna(0.0)
    extra_freeze_credit_xT_Against = -freeze_delta 

    xT_For_save = native_save_xT_For
    xT_Against_save = native_save_xT_Against + extra_freeze_credit_xT_Against
    net_xT_save = xT_For_save - xT_Against_save
    
    shot_value_xt = pd.Series(0.0, index=out.index)

    save_reason = pd.Series("save_unknown_next_event", index=out.index, dtype="object")
    save_reason.loc[is_save_play_continued] = "save_play_continued"
    save_reason.loc[is_save_with_stoppage] = "save_with_stoppage_bonus"
    save_reason.loc[save_with_whistle_then_faceoff] = "save_with_stoppage_faceoff_baseline_bonus"
    save_reason.loc[save_mask & next_event_is_goal] = "save_attempt_goal_allowed"

    save_rows = pd.DataFrame(
        {
            "goalie_credit_row_type": "save_component",
            "game_id": out.loc[save_mask, "game_id"].values,
            "sl_event_id": out.loc[save_mask, "sl_event_id"].values,
            "linked_goal_sl_event_id": pd.NA,
            "goalie_id": save_goalie_id_series.loc[save_mask].values,
            "goalie_team_id": goalie_team_series_save.loc[save_mask].values,
            "event_type": out.loc[save_mask, "_event_type_norm"].values,
            "next_event_type": next_evt.loc[save_mask].values,
            "restart_event_type": restart_event_type.loc[save_mask].values,
            "restart_sl_event_id": restart_sl_event_id.loc[save_mask].values,
            "save_is_synthetic": synthetic_save_flag.loc[save_mask].values,
            "save_has_source_link": save_source_link_present.loc[save_mask].values,
            "save_provenance_bucket": save_provenance_bucket.loc[save_mask].values,
            "next_event_is_whistle": next_event_is_whistle.loc[save_mask].values,
            "next_event_is_goal": next_event_is_goal.loc[save_mask].values,
            "next_event_is_eos": next_event_is_eos.loc[save_mask].values,
            "shot_value_xt": shot_value_xt.loc[save_mask].values,
            "native_save_xT_For": native_save_xT_For.loc[save_mask].values,
            "native_save_xT_Against": native_save_xT_Against.loc[save_mask].values,
            "extra_freeze_credit_xT_Against": extra_freeze_credit_xT_Against.loc[save_mask].values,
            "goal_penalty_xT_Against": 0.0,
            "xT_For": xT_For_save.loc[save_mask].values,
            "xT_Against": xT_Against_save.loc[save_mask].values,
            "Net_xT": net_xT_save.loc[save_mask].values,
            "goalie_credit_reason": save_reason.loc[save_mask].values,
            "goalie_empty_net_exempt": empty_net_mask.loc[save_mask].values,
        }
    )

    # Goal-penalty rows: tensor-goal population is recovered from linked goal ids and immediate pre-goal rows.
    sl_event_id_num = pd.to_numeric(out.get("sl_event_id"), errors="coerce")
    next_sl_event_id_num = pd.to_numeric(out.get("next_sl_event_id"), errors="coerce")

    canonical_goal = evt.eq("goal")

    tensor_goal_key_frames: list[pd.DataFrame] = []
    tensor_goal_mask = canonical_goal
    tensor_goal_key_frames.append(
        pd.DataFrame(
            {
                "game_id": out.loc[tensor_goal_mask, "game_id"].astype(str).values,
                "linked_goal_sl_event_id": sl_event_id_num.loc[tensor_goal_mask].values,
            }
        )
    )
    if "next_event_is_goal" in out.columns:
        pregoal_next_mask = out["next_event_is_goal"].fillna(False).astype(bool)
        tensor_goal_key_frames.append(
            pd.DataFrame(
                {
                    "game_id": out.loc[pregoal_next_mask, "game_id"].astype(str).values,
                    "linked_goal_sl_event_id": next_sl_event_id_num.loc[pregoal_next_mask].values,
                }
            )
        )

    tensor_goal_key_duplicates_dropped = 0
    if tensor_goal_key_frames:
        tensor_goal_keys = pd.concat(tensor_goal_key_frames, ignore_index=True)
        tensor_goal_keys = tensor_goal_keys.dropna(subset=["linked_goal_sl_event_id"])
        tensor_goal_keys_pre_dedup = int(len(tensor_goal_keys))
        tensor_goal_keys = tensor_goal_keys.drop_duplicates(["game_id", "linked_goal_sl_event_id"], keep="last")
        tensor_goal_key_duplicates_dropped = int(max(0, tensor_goal_keys_pre_dedup - len(tensor_goal_keys)))
    else:
        tensor_goal_keys = pd.DataFrame(columns=["game_id", "linked_goal_sl_event_id"])

    linked_goal_ids = pd.Series(pd.NA, index=out.index, dtype="Float64")
    if "source_linked_goal_sl_event_id" in out.columns:
        linked_goal_ids = pd.to_numeric(out["source_linked_goal_sl_event_id"], errors="coerce").astype("Float64")

    shot_like_source = evt.isin(["shot", "deflection", "defensive_deflection"])
    linked_source_mask = shot_like_source & linked_goal_ids.notna()
    linked_source_missing_opp_goalie = int((linked_source_mask & goal_penalty_goalie_id_series.isna()).sum())

    source_link = pd.DataFrame(
        {
            "game_id": out["game_id"].astype(str),
            "linked_goal_sl_event_id": linked_goal_ids,
            "linked_source_sl_event_id": sl_event_id_num,
            "scoring_team_id": out["team_id"].astype(str),
            "linked_opp_goalie_id": goal_penalty_goalie_id_series,
            "linked_goalie_team_id": goalie_team_series_default,
            "linked_goal_empty_net": empty_net_mask,
            "source_actor_goal": pd.to_numeric(out.get("P_actor_goal", 0.0), errors="coerce").fillna(0.0),
            "source_opp_goal": pd.to_numeric(out.get("P_opp_goal", 0.0), errors="coerce").fillna(0.0),
            "source_event_type": out["_event_type_norm"],
        }
    )
    source_link = source_link.loc[linked_source_mask].copy()
    if not tensor_goal_keys.empty and not source_link.empty:
        source_link = source_link.merge(
            tensor_goal_keys,
            on=["game_id", "linked_goal_sl_event_id"],
            how="inner",
        )
    else:
        source_link = source_link.iloc[0:0].copy()
    source_link_duplicates_dropped = 0
    if not source_link.empty:
        source_link_pre_dedup = int(len(source_link))
        source_link = source_link.dropna(subset=["linked_goal_sl_event_id"]).drop_duplicates(
            ["game_id", "linked_goal_sl_event_id"], keep="last"
        )
        source_link_duplicates_dropped = int(max(0, source_link_pre_dedup - len(source_link)))

    pregoal_mask = next_event_is_goal & next_sl_event_id_num.notna()

    pregoal_link = pd.DataFrame(
        {
            "game_id": out["game_id"].astype(str),
            "linked_goal_sl_event_id": next_sl_event_id_num,
            "pregoal_event_type": out["_event_type_norm"],
            "pregoal_team_id": out["team_id"].astype(str),
            "pregoal_actor_goal": pd.to_numeric(out.get("P_actor_goal", 0.0), errors="coerce").fillna(0.0),
            "pregoal_opp_goal": pd.to_numeric(out.get("P_opp_goal", 0.0), errors="coerce").fillna(0.0),
        }
    )
    pregoal_link = pregoal_link.loc[pregoal_mask].copy()
    if not tensor_goal_keys.empty and not pregoal_link.empty:
        pregoal_link = pregoal_link.merge(
            tensor_goal_keys,
            on=["game_id", "linked_goal_sl_event_id"],
            how="inner",
        )
    else:
        pregoal_link = pregoal_link.iloc[0:0].copy()
    pregoal_link_duplicates_dropped = 0
    if not pregoal_link.empty:
        pregoal_link_pre_dedup = int(len(pregoal_link))
        pregoal_link = pregoal_link.dropna(subset=["linked_goal_sl_event_id"]).drop_duplicates(
            ["game_id", "linked_goal_sl_event_id"], keep="last"
        )
        pregoal_link_duplicates_dropped = int(max(0, pregoal_link_pre_dedup - len(pregoal_link)))

    goal_map = source_link.merge(
        pregoal_link,
        on=["game_id", "linked_goal_sl_event_id"],
        how="left",
    )

    if goal_map.empty:
        goal_rows = pd.DataFrame(columns=save_rows.columns.tolist() + ["linked_source_sl_event_id"])
        resolved_goalies = 0
        linked_goal_count = 0
    else:
        linked_goal_count = int(len(goal_map[["game_id", "linked_goal_sl_event_id"]].drop_duplicates()))
        goal_map = goal_map.loc[goal_map["linked_opp_goalie_id"].notna()].copy()
        resolved_goalies = int(len(goal_map[["game_id", "linked_goal_sl_event_id"]].drop_duplicates()))

        # Use the max of Actor/Opp threat, ensuring robust expected value prior to goal outcome
        pregoal_max = np.maximum(
            pd.to_numeric(goal_map["pregoal_actor_goal"], errors="coerce").fillna(0),
            pd.to_numeric(goal_map["pregoal_opp_goal"], errors="coerce").fillna(0)
        )
        source_max = np.maximum(
            pd.to_numeric(goal_map["source_actor_goal"], errors="coerce").fillna(0),
            pd.to_numeric(goal_map["source_opp_goal"], errors="coerce").fillna(0)
        )
        
        pregoal_threat = pd.Series(pregoal_max, index=goal_map.index).replace(0, np.nan)
        source_fallback = pd.Series(source_max, index=goal_map.index).replace(0, np.nan)
        prior_scoring_team_threat = pregoal_threat.fillna(source_fallback).fillna(0.0).clip(0.0, 1.0)

        goal_penalty_xT_Against = (1.0 - prior_scoring_team_threat).clip(0.0, 1.0)
        goal_penalty_xT_Against = goal_penalty_xT_Against.where(~goal_map["linked_goal_empty_net"].fillna(False), 0.0)

        goal_reason = pd.Series("goal_penalty_dynamic", index=goal_map.index, dtype="object")
        goal_reason.loc[goal_map["linked_goal_empty_net"].fillna(False)] = "goal_allowed_empty_net_exempt"
        pregoal_label = goal_map["pregoal_event_type"].fillna("unknown").astype(str)
        goal_reason.loc[~goal_map["linked_goal_empty_net"].fillna(False)] = "goal_penalty_post_" + pregoal_label

        xT_For_goal = pd.Series(0.0, index=goal_map.index)
        xT_Against_goal = goal_penalty_xT_Against
        net_xT_goal = xT_For_goal - xT_Against_goal

        goal_rows = pd.DataFrame(
            {
                "goalie_credit_row_type": "goal_penalty",
                "game_id": goal_map["game_id"].values,
                "sl_event_id": goal_map["linked_goal_sl_event_id"].values,
                "linked_goal_sl_event_id": goal_map["linked_goal_sl_event_id"].values,
                "linked_source_sl_event_id": goal_map["linked_source_sl_event_id"].values,
                "goalie_id": goal_map["linked_opp_goalie_id"].values,
                "goalie_team_id": goal_map["linked_goalie_team_id"].values,
                "event_type": "goal",
                "prior_event_type_for_goal_penalty": goal_map["pregoal_event_type"].values,
                "prior_event_team_id_for_goal_penalty": goal_map["pregoal_team_id"].values,
                "prior_threat_for_goal_penalty": prior_scoring_team_threat.values,
                "next_event_type": pd.NA,
                "next_event_is_whistle": False,
                "next_event_is_goal": False,
                "next_event_is_eos": True,
                "shot_value_xt": 0.0,
                "native_save_xT_For": 0.0,
                "native_save_xT_Against": 0.0,
                "extra_freeze_credit_xT_Against": 0.0,
                "goal_penalty_xT_Against": goal_penalty_xT_Against.values,
                "xT_For": xT_For_goal.values,
                "xT_Against": xT_Against_goal.values,
                "Net_xT": net_xT_goal.values,
                "goalie_credit_reason": goal_reason.values,
                "goalie_empty_net_exempt": goal_map["linked_goal_empty_net"].fillna(False).values,
            }
        )

    ledger = pd.concat([save_rows, goal_rows], ignore_index=True, sort=False)

    if "outcome" in out.columns:
        outcome_lookup = out[["game_id", "sl_event_id", "outcome"]].copy()
        outcome_lookup["sl_event_id"] = pd.to_numeric(outcome_lookup["sl_event_id"], errors="coerce").astype("Float64").round(6)
        ledger = ledger.merge(outcome_lookup.drop_duplicates(["game_id", "sl_event_id"]), on=["game_id", "sl_event_id"], how="left")

    if "tensor_event_present" in out.columns:
        tensor_lookup = out[["game_id", "sl_event_id", "tensor_event_present"]].copy()
        tensor_lookup["sl_event_id"] = pd.to_numeric(tensor_lookup["sl_event_id"], errors="coerce").astype("Float64").round(6)
        tensor_lookup["tensor_event_present"] = tensor_lookup["tensor_event_present"].fillna(False).astype(bool)
        ledger = ledger.merge(tensor_lookup.drop_duplicates(["game_id", "sl_event_id"]), on=["game_id", "sl_event_id"], how="left")

    ledger["game_id"] = ledger["game_id"].astype(str)
    ledger["sl_event_id"] = pd.to_numeric(ledger["sl_event_id"], errors="coerce").astype("Float64").round(6)
    ledger["linked_goal_sl_event_id"] = pd.to_numeric(ledger["linked_goal_sl_event_id"], errors="coerce").astype("Float64").round(6)
    if "linked_source_sl_event_id" in ledger.columns:
        ledger["linked_source_sl_event_id"] = pd.to_numeric(ledger["linked_source_sl_event_id"], errors="coerce").astype("Float64").round(6)

    goal_penalty_empty_net_exempt_rows = int(goal_rows.get("goalie_empty_net_exempt", pd.Series(dtype=bool)).fillna(False).sum())
    goalie_audit: Dict[str, Any] = {
        "goal_penalty_population_source": "tensor_goal_rows_only",
        "goal_penalty_strict_opp_goalie_required": True,
        "canonical_goal_rows_total": int(canonical_goal.sum()),
        "tensor_goal_rows_total": int(len(tensor_goal_keys)),
        "tensor_goal_key_duplicates_dropped": int(tensor_goal_key_duplicates_dropped),
        "source_link_duplicates_dropped": int(source_link_duplicates_dropped),
        "pregoal_link_duplicates_dropped": int(pregoal_link_duplicates_dropped),
        "tensor_goal_rows_with_linked_source": int(linked_goal_count),
        "tensor_goal_rows_with_resolved_goalie": int(resolved_goalies),
        "goal_penalty_rows_emitted": int(len(goal_rows)),
        "goal_penalty_unresolved_linkage_rows": int(max(0, len(tensor_goal_keys) - linked_goal_count)),
        "goal_penalty_unresolved_goalie_rows": int(max(0, linked_goal_count - resolved_goalies)),
        "goal_penalty_linked_source_rows_missing_opp_goalie": int(linked_source_missing_opp_goalie),
        "goal_penalty_empty_net_exempt_rows": goal_penalty_empty_net_exempt_rows,
        "save_rows_total": int(save_mask.sum()),
        "save_rows_synthetic_only": int((save_mask & synthetic_save_flag & (~save_source_link_present)).sum()),
        "save_rows_source_linked_only": int((save_mask & (~synthetic_save_flag) & save_source_link_present).sum()),
        "save_rows_synthetic_and_source_linked": int((save_mask & synthetic_save_flag & save_source_link_present).sum()),
        "save_rows_neither_synthetic_nor_source_linked": int((save_mask & (~synthetic_save_flag) & (~save_source_link_present)).sum()),
        "save_rows_whistle_then_restart": int(save_with_whistle_then_restart.sum()),
        "save_rows_whistle_then_faceoff": int(save_with_whistle_then_faceoff.sum()),
        "save_rows_faceoff_baseline_used": int((save_mask & use_faceoff_baseline).sum()),
        "save_rows_missing_goalie_id": int((save_mask & save_goalie_id_series.isna()).sum()),
        "save_rows_missing_goalie_team_id": int((save_mask & goalie_team_series_save.isna()).sum()),
        "freeze_fallback_rows": int(freeze_fallback_mask.sum()),
    }

    return ledger.reset_index(drop=True), goalie_audit


def _audit_sidecar_conservation(
    base_df: pd.DataFrame,
    faceoff_rows: pd.DataFrame,
    penalty_drawer_rows: pd.DataFrame,
) -> Dict[str, Any]:
    tol = 1e-9

    out: Dict[str, Any] = {
        "faceoff_rows": int(0 if faceoff_rows is None else len(faceoff_rows)),
        "penalty_drawer_rows": int(0 if penalty_drawer_rows is None else len(penalty_drawer_rows)),
        "faceoff_groups_checked": 0,
        "faceoff_groups_mismatch": 0,
        "faceoff_max_abs_residual": 0.0,
        "penalty_groups_checked": 0,
        "penalty_groups_mismatch": 0,
        "penalty_max_abs_residual": 0.0,
        "tolerance": tol,
    }

    if faceoff_rows is not None and not faceoff_rows.empty:
        base_faceoff = base_df[["game_id", "sl_event_id", "Actor_Net_xT"]].copy()
        base_faceoff = _normalize_event_keys(base_faceoff, key_col="sl_event_id")
        base_faceoff = base_faceoff.rename(columns={"sl_event_id": "linked_kept_sl_event_id", "Actor_Net_xT": "kept_value"})
        base_faceoff = base_faceoff.drop_duplicates(["game_id", "linked_kept_sl_event_id"])

        face = faceoff_rows[["game_id", "linked_kept_sl_event_id", "Adjusted_Net_xT"]].copy()
        face = _normalize_event_keys(face, key_col="linked_kept_sl_event_id")
        face_sum = face.groupby(["game_id", "linked_kept_sl_event_id"], dropna=False, as_index=False)["Adjusted_Net_xT"].sum()
        face_sum = face_sum.rename(columns={"Adjusted_Net_xT": "inverse_sum"})

        aligned = face_sum.merge(base_faceoff, on=["game_id", "linked_kept_sl_event_id"], how="left")
        aligned = aligned.loc[aligned["kept_value"].notna()].copy()
        if not aligned.empty:
            residual = pd.to_numeric(aligned["kept_value"], errors="coerce").fillna(0.0) + pd.to_numeric(aligned["inverse_sum"], errors="coerce").fillna(0.0)
            out["faceoff_groups_checked"] = int(len(aligned))
            out["faceoff_groups_mismatch"] = int((residual.abs() > tol).sum())
            out["faceoff_max_abs_residual"] = float(residual.abs().max())

    if penalty_drawer_rows is not None and not penalty_drawer_rows.empty:
        penalty_mask = base_df.get("event_type", "").astype(str).str.strip().str.lower().eq("penalty")
        base_pen = base_df.loc[penalty_mask, ["game_id", "sl_event_id", "Adjusted_Net_xT"]].copy()
        base_pen = _normalize_event_keys(base_pen, key_col="sl_event_id")
        base_pen = base_pen.rename(columns={"sl_event_id": "linked_kept_sl_event_id", "Adjusted_Net_xT": "kept_value"})
        base_pen = base_pen.drop_duplicates(["game_id", "linked_kept_sl_event_id"])

        inv = penalty_drawer_rows[["game_id", "linked_kept_sl_event_id", "Adjusted_Net_xT"]].copy()
        inv = _normalize_event_keys(inv, key_col="linked_kept_sl_event_id")
        inv_sum = inv.groupby(["game_id", "linked_kept_sl_event_id"], dropna=False, as_index=False)["Adjusted_Net_xT"].sum()
        inv_sum = inv_sum.rename(columns={"Adjusted_Net_xT": "inverse_sum"})

        aligned = inv_sum.merge(base_pen, on=["game_id", "linked_kept_sl_event_id"], how="left")
        aligned = aligned.loc[aligned["kept_value"].notna()].copy()
        if not aligned.empty:
            residual = pd.to_numeric(aligned["kept_value"], errors="coerce").fillna(0.0) + pd.to_numeric(aligned["inverse_sum"], errors="coerce").fillna(0.0)
            out["penalty_groups_checked"] = int(len(aligned))
            out["penalty_groups_mismatch"] = int((residual.abs() > tol).sum())
            out["penalty_max_abs_residual"] = float(residual.abs().max())

    out["warning_sidecar_conservation_mismatch"] = bool(
        out["faceoff_groups_mismatch"] > 0 or out["penalty_groups_mismatch"] > 0
    )
    return out


def _build_faceoff_inverse_rows(base_df: pd.DataFrame, faceoff_ref: pd.DataFrame) -> pd.DataFrame:
    if faceoff_ref.empty:
        return pd.DataFrame()

    scored = base_df[["game_id", "sl_event_id", "Actor_Net_xT"]].copy()
    scored = _normalize_event_keys(scored, key_col="sl_event_id")
    scored = scored.rename(columns={"sl_event_id": "kept_sl_event_id", "Actor_Net_xT": "kept_actor_net_xt"})
    scored = scored.drop_duplicates(["game_id", "kept_sl_event_id"])

    ref = _normalize_event_keys(faceoff_ref, key_col="sl_event_id")
    ref = _normalize_event_keys(ref, key_col="kept_sl_event_id")

    merged = ref.merge(scored, on=["game_id", "kept_sl_event_id"], how="left")
    merged = merged.loc[merged["kept_actor_net_xt"].notna()].copy()

    inverse = -pd.to_numeric(merged["kept_actor_net_xt"], errors="coerce").fillna(0.0)
    rows = pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "sl_event_id": merged["sl_event_id"],
            "linked_kept_sl_event_id": merged["kept_sl_event_id"],
            "player_id": merged.get("opposing_player_id"),
            "team_id": merged.get("opposing_team_id"),
            "event_type": "faceoff",
            "Actor_xT_For": 0.0,
            "Actor_xT_Against": 0.0,
            "Actor_Net_xT": 0.0,
            "Adjusted_xT_For": np.where(inverse >= 0.0, inverse, 0.0),
            "Adjusted_xT_Against": np.where(inverse < 0.0, -inverse, 0.0),
            "Adjusted_Net_xT": inverse,
            "adjustment_source": "faceoff_inverse",
            "faceoff_abs_x": pd.to_numeric(merged.get("faceoff_abs_x"), errors="coerce"),
        }
    )

    if "faceoff_abs_x" not in rows.columns or rows["faceoff_abs_x"].isna().all():
        rows["faceoff_abs_x"] = np.nan

    abs_x = pd.to_numeric(rows["faceoff_abs_x"], errors="coerce").abs()
    rows["faceoff_area_strict"] = np.where(
        abs_x.le(5.0),
        "center_ice",
        np.where(abs_x.le(25.0), "neutral_zone", "oz_dz"),
    )

    return rows


def _ensure_penalty_taker_adjusted(
    base_df: pd.DataFrame,
    for_bonus: float,
    against_bonus: float,
) -> tuple[pd.DataFrame, int]:
    out = base_df.copy()
    evt = out.get("event_type", "").astype(str).str.strip().str.lower()
    penalty_mask = evt.eq("penalty")

    if "adjustment_source" in out.columns:
        src = out["adjustment_source"].astype(str).str.strip().str.lower()
        already_adjusted = src.eq("penalty_static_taker")
    else:
        already_adjusted = pd.Series(False, index=out.index)

    needs_adjustment = penalty_mask & (~already_adjusted)

    if needs_adjustment.any():
        out.loc[needs_adjustment, "Adjusted_xT_For"] = (
            pd.to_numeric(out.loc[needs_adjustment, "Actor_xT_For"], errors="coerce").fillna(0.0)
            + float(for_bonus)
        )
        out.loc[needs_adjustment, "Adjusted_xT_Against"] = (
            pd.to_numeric(out.loc[needs_adjustment, "Actor_xT_Against"], errors="coerce").fillna(0.0)
            + float(against_bonus)
        )
        out.loc[needs_adjustment, "Adjusted_Net_xT"] = (
            pd.to_numeric(out.loc[needs_adjustment, "Adjusted_xT_For"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out.loc[needs_adjustment, "Adjusted_xT_Against"], errors="coerce").fillna(0.0)
        )
        out.loc[needs_adjustment, "adjustment_source"] = "penalty_static_taker"

    return out, int(needs_adjustment.sum())


def _build_penalty_drawer_inverse_rows(base_df: pd.DataFrame, penalty_ref: pd.DataFrame) -> pd.DataFrame:
    if penalty_ref.empty:
        return pd.DataFrame()

    out = base_df.copy()
    evt = out.get("event_type", "").astype(str).str.strip().str.lower()
    penalty_mask = evt.eq("penalty")

    # Use already-adjusted penalty taker values and assign the converse to drawer rows.
    taker = out.loc[penalty_mask, ["game_id", "sl_event_id", "Adjusted_Net_xT"]].copy()
    taker = _normalize_event_keys(taker, key_col="sl_event_id")
    taker = taker.rename(columns={"sl_event_id": "kept_sl_event_id", "Adjusted_Net_xT": "taker_adjusted_net_xt"})
    taker = taker.drop_duplicates(["game_id", "kept_sl_event_id"])

    if taker.empty:
        return pd.DataFrame()

    ref = _normalize_event_keys(penalty_ref, key_col="sl_event_id")
    ref = _normalize_event_keys(ref, key_col="kept_sl_event_id")

    merged = ref.merge(taker, on=["game_id", "kept_sl_event_id"], how="left")
    merged = merged.loc[merged["taker_adjusted_net_xt"].notna()].copy()

    inverse = -pd.to_numeric(merged["taker_adjusted_net_xt"], errors="coerce").fillna(0.0)
    drawer_rows = pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "sl_event_id": merged["sl_event_id"],
            "linked_kept_sl_event_id": merged["kept_sl_event_id"],
            "player_id": merged.get("penaltydrawn_player_id"),
            "team_id": merged.get("penaltydrawn_team_id"),
            "event_type": "penaltydrawn",
            "Actor_xT_For": 0.0,
            "Actor_xT_Against": 0.0,
            "Actor_Net_xT": 0.0,
            "Adjusted_xT_For": np.where(inverse >= 0.0, inverse, 0.0),
            "Adjusted_xT_Against": np.where(inverse < 0.0, -inverse, 0.0),
            "Adjusted_Net_xT": inverse,
            "adjustment_source": "penalty_drawer_inverse",
        }
    )

    return drawer_rows


def _build_event_level_player_ledger(adjusted_df: pd.DataFrame) -> pd.DataFrame:
    ledger = adjusted_df.copy()

    ledger["game_id"] = ledger["game_id"].astype(str)
    ledger["sl_event_id"] = pd.to_numeric(ledger["sl_event_id"], errors="coerce").astype("Float64").round(6)
    ledger["credit_for"] = pd.to_numeric(ledger.get("Adjusted_xT_For", 0.0), errors="coerce").fillna(0.0)
    ledger["credit_against"] = pd.to_numeric(ledger.get("Adjusted_xT_Against", 0.0), errors="coerce").fillna(0.0)
    ledger["credit_xt"] = pd.to_numeric(ledger.get("Adjusted_Net_xT", 0.0), errors="coerce").fillna(0.0)

    if "player_id" in ledger.columns:
        player_id = ledger["player_id"].astype("string").str.strip()
        valid_player = player_id.notna() & (~player_id.isin(["", "<NA>", "nan", "None"]))
        ledger = ledger.loc[valid_player].copy()

    keep_cols = [
        "game_id",
        "sl_event_id",
        "linked_kept_sl_event_id",
        "team_id",
        "player_id",
        "event_type",
        "period",
        "sequence_id",
        "model_variant",
        "variant_name",
        "adjustment_source",
        "outcome",
        "credit_for",
        "credit_against",
        "credit_xt",
        "Adjusted_xT_For",
        "Adjusted_xT_Against",
        "Adjusted_Net_xT",
    ]
    keep_cols = [c for c in keep_cols if c in ledger.columns]

    return ledger[keep_cols].reset_index(drop=True)


def _build_goalie_player_ledger(goalie_ledger: pd.DataFrame) -> pd.DataFrame:
    if goalie_ledger is None or goalie_ledger.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "sl_event_id",
                "linked_kept_sl_event_id",
                "team_id",
                "player_id",
                "event_type",
                "period",
                "sequence_id",
                "model_variant",
                "variant_name",
                "adjustment_source",
                "outcome",
                "credit_for",
                "credit_against",
                "credit_xt",
                "Adjusted_xT_For",
                "Adjusted_xT_Against",
                "Adjusted_Net_xT",
            ]
        )

    ledger = pd.DataFrame(
        {
            "game_id": goalie_ledger.get("game_id", pd.Series(dtype="object")).astype(str),
            "sl_event_id": pd.to_numeric(goalie_ledger.get("sl_event_id"), errors="coerce").astype("Float64").round(6),
            "linked_kept_sl_event_id": pd.to_numeric(goalie_ledger.get("linked_goal_sl_event_id"), errors="coerce").astype("Float64").round(6),
            "team_id": goalie_ledger.get("goalie_team_id", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "player_id": goalie_ledger.get("goalie_id", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "event_type": goalie_ledger.get("event_type", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "period": goalie_ledger.get("period", pd.Series(pd.NA, index=goalie_ledger.index)),
            "sequence_id": goalie_ledger.get("sequence_id", pd.Series(pd.NA, index=goalie_ledger.index)),
            "model_variant": goalie_ledger.get("model_variant", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "variant_name": goalie_ledger.get("variant_name", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "adjustment_source": "goalie_credit",
            "outcome": goalie_ledger.get("goalie_credit_reason", pd.Series(pd.NA, index=goalie_ledger.index)).astype("string"),
            "credit_for": pd.to_numeric(goalie_ledger.get("xT_For", 0.0), errors="coerce").fillna(0.0),
            "credit_against": pd.to_numeric(goalie_ledger.get("xT_Against", 0.0), errors="coerce").fillna(0.0),
            "credit_xt": pd.to_numeric(goalie_ledger.get("Net_xT", 0.0), errors="coerce").fillna(0.0),
            "Adjusted_xT_For": pd.to_numeric(goalie_ledger.get("xT_For", 0.0), errors="coerce").fillna(0.0),
            "Adjusted_xT_Against": pd.to_numeric(goalie_ledger.get("xT_Against", 0.0), errors="coerce").fillna(0.0),
            "Adjusted_Net_xT": pd.to_numeric(goalie_ledger.get("Net_xT", 0.0), errors="coerce").fillna(0.0),
        }
    )

    player_id = ledger["player_id"].astype("string").str.strip()
    valid_player = player_id.notna() & (~player_id.isin(["", "<NA>", "nan", "None"]))
    ledger = ledger.loc[valid_player].copy()

    return ledger.reset_index(drop=True)


def _build_consolidated_player_goalie_summary(
    player_ledger: pd.DataFrame,
    goalie_ledger: pd.DataFrame,
) -> pd.DataFrame:
    player_frame = player_ledger.copy() if player_ledger is not None else pd.DataFrame()
    goalie_frame = goalie_ledger.copy() if goalie_ledger is not None else pd.DataFrame()

    if not player_frame.empty:
        if "Net_xT" not in player_frame.columns:
            if "credit_xt" in player_frame.columns:
                player_frame["Net_xT"] = pd.to_numeric(player_frame["credit_xt"], errors="coerce")
            elif "Adjusted_Net_xT" in player_frame.columns:
                player_frame["Net_xT"] = pd.to_numeric(player_frame["Adjusted_Net_xT"], errors="coerce")
            else:
                player_frame["Net_xT"] = 0.0

    if not goalie_frame.empty and "player_id" not in goalie_frame.columns and "goalie_id" in goalie_frame.columns:
        goalie_frame["player_id"] = goalie_frame["goalie_id"]

    keep = ["player_id", "team_id", "event_type", "Net_xT"]
    player_use = player_frame[[c for c in keep if c in player_frame.columns]].copy() if not player_frame.empty else pd.DataFrame(columns=keep)
    goalie_use = goalie_frame[[c for c in keep if c in goalie_frame.columns]].copy() if not goalie_frame.empty else pd.DataFrame(columns=keep)

    combined = pd.concat([player_use, goalie_use], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=["player_id", "team_id", "Total_Net_xT"])

    combined["player_id"] = combined.get("player_id", pd.Series(pd.NA, index=combined.index)).astype("string")
    combined["team_id"] = combined.get("team_id", pd.Series(pd.NA, index=combined.index)).astype("string")
    combined["event_type"] = combined.get("event_type", pd.Series("unknown", index=combined.index)).astype(str).str.strip().str.lower()
    combined["Net_xT"] = pd.to_numeric(combined.get("Net_xT", 0.0), errors="coerce").fillna(0.0)

    valid = combined["player_id"].notna() & combined["team_id"].notna()
    combined = combined.loc[valid].copy()
    if combined.empty:
        return pd.DataFrame(columns=["player_id", "team_id", "Total_Net_xT"])

    grouped = (
        combined.groupby(["player_id", "team_id", "event_type"], dropna=False)
        .agg(Events=("event_type", "size"), Net_xT=("Net_xT", "sum"))
        .reset_index()
    )

    events_pivot = grouped.pivot_table(
        index=["player_id", "team_id"],
        columns="event_type",
        values="Events",
        aggfunc="sum",
        fill_value=0,
    )
    net_pivot = grouped.pivot_table(
        index=["player_id", "team_id"],
        columns="event_type",
        values="Net_xT",
        aggfunc="sum",
        fill_value=0.0,
    )

    events_pivot.columns = [f"{col}_Events" for col in events_pivot.columns]
    net_pivot.columns = [f"{col}_Net_xT" for col in net_pivot.columns]

    summary = pd.concat([events_pivot, net_pivot], axis=1).reset_index()
    net_cols = [c for c in summary.columns if c.endswith("_Net_xT")]
    summary["Total_Net_xT"] = summary[net_cols].sum(axis=1) if net_cols else 0.0
    return summary.sort_values(["team_id", "player_id"], kind="mergesort").reset_index(drop=True)


def _build_adjusted_output(
    base_df: pd.DataFrame,
    goalie_ledger: pd.DataFrame,
    faceoff_rows: pd.DataFrame,
    penalty_drawer_rows: pd.DataFrame,
) -> pd.DataFrame:
    keep_cols = list(base_df.columns)

    extra_rows = []
    for extra in [faceoff_rows, penalty_drawer_rows]:
        if extra is None or extra.empty:
            continue
        extra_rows.append(extra.reindex(columns=keep_cols))

    if extra_rows:
        adjusted = pd.concat([base_df] + extra_rows, ignore_index=True, sort=False)
    else:
        adjusted = base_df.copy()

    adjusted["game_id"] = adjusted["game_id"].astype(str)
    adjusted["sl_event_id"] = pd.to_numeric(adjusted["sl_event_id"], errors="coerce").astype("Float64").round(6)

    return adjusted


def _audit_adjusted_output_contract(raw_oof: pd.DataFrame, adjusted: pd.DataFrame) -> Dict[str, Any]:
    raw_cols = list(raw_oof.columns)
    adjusted_cols = set(adjusted.columns)
    missing_raw_cols = [c for c in raw_cols if c not in adjusted_cols]
    if missing_raw_cols:
        raise RuntimeError(
            'Adjusted OOF contract failed: adjusted output dropped raw OOF columns. '
            f'Missing {len(missing_raw_cols)} columns (sample): {missing_raw_cols[:10]}'
        )

    if raw_oof.empty:
        return {
            "raw_rows": 0,
            "adjusted_rows": int(len(adjusted)),
            "raw_columns": int(len(raw_cols)),
            "adjusted_columns": int(len(adjusted.columns)),
            "missing_raw_columns_in_adjusted": 0,
            "added_rows_in_adjusted": int(max(0, len(adjusted))),
            "key_columns": ["game_id", "sl_event_id"],
            "probability_mismatch_counts": {},
        }

    key_cols = ["game_id", "sl_event_id"]
    for optional_key in ["model_variant", "variant_name"]:
        if optional_key in raw_oof.columns and optional_key in adjusted.columns:
            key_cols.append(optional_key)

    pred_cols = [c for c in ["P_actor_goal", "P_opp_goal", "P_no_goal"] if c in raw_oof.columns and c in adjusted.columns]
    if len(pred_cols) != 3:
        raise RuntimeError(
            'Adjusted OOF contract failed: required probability columns missing from raw/adjusted outputs. '
            f'Found: {pred_cols}'
        )

    raw_norm = raw_oof[key_cols + pred_cols].copy()
    adjusted_keep_cols = key_cols + pred_cols + (["adjustment_source"] if "adjustment_source" in adjusted.columns else [])
    adjusted_norm = adjusted[adjusted_keep_cols].copy()
    for df in [raw_norm, adjusted_norm]:
        df["game_id"] = df["game_id"].astype(str)
        df["sl_event_id"] = pd.to_numeric(df["sl_event_id"], errors="coerce").astype("Float64").round(6)

    raw_dups = int(raw_norm.duplicated(key_cols, keep=False).sum())

    if "adjustment_source" in adjusted_norm.columns:
        src = adjusted_norm["adjustment_source"].astype(str).str.strip().str.lower()
        adjusted_base = adjusted_norm.loc[~src.isin(["faceoff_inverse", "penalty_drawer_inverse"])].copy()
    else:
        adjusted_base = adjusted_norm.copy()

    adjusted_dups = int(adjusted_base.duplicated(key_cols, keep=False).sum())

    raw_counts = raw_norm.groupby(key_cols, dropna=False).size().rename("raw_key_count").reset_index()
    adjusted_counts = adjusted_base.groupby(key_cols, dropna=False).size().rename("adjusted_key_count").reset_index()
    key_coverage = raw_counts.merge(adjusted_counts, on=key_cols, how="left", validate="one_to_one")

    missing_adjusted_rows = int(key_coverage["adjusted_key_count"].isna().sum())
    if missing_adjusted_rows > 0:
        raise RuntimeError(
            'Adjusted OOF contract failed: raw key groups are missing in adjusted output after key alignment. '
            f'Missing groups: {missing_adjusted_rows:,} using keys {key_cols}.'
        )

    adjusted_lt_raw = int((key_coverage["adjusted_key_count"].fillna(0) < key_coverage["raw_key_count"]).sum())
    if adjusted_lt_raw > 0:
        raise RuntimeError(
            'Adjusted OOF contract failed: adjusted output has fewer rows than raw within one or more key groups. '
            f'Groups failing cardinality check: {adjusted_lt_raw:,} using keys {key_cols}.'
        )

    adjusted_event_type = None
    if "event_type" in adjusted.columns:
        adjusted_event_type = adjusted[key_cols + ["event_type"]].copy()
        adjusted_event_type["event_type"] = adjusted_event_type["event_type"].astype(str).str.strip().str.lower()
        adjusted_event_type = adjusted_event_type.drop_duplicates(key_cols, keep="first")

    raw_view = raw_norm[key_cols + pred_cols].groupby(key_cols, dropna=False, as_index=False).mean(numeric_only=True)
    adjusted_view = adjusted_base[key_cols + pred_cols].groupby(key_cols, dropna=False, as_index=False).mean(numeric_only=True)
    aligned = raw_view.merge(
        adjusted_view,
        on=key_cols,
        how="left",
        suffixes=("_raw", "_adj"),
        validate="one_to_one",
    )
    if adjusted_event_type is not None:
        aligned = aligned.merge(adjusted_event_type, on=key_cols, how="left")
    else:
        aligned["event_type"] = ""

    missing_adjusted_prob_groups = int(aligned[f"{pred_cols[0]}_adj"].isna().sum())
    if missing_adjusted_prob_groups > 0:
        raise RuntimeError(
            'Adjusted OOF contract failed: grouped raw key rows are missing probability values in adjusted output after key alignment. '
            f'Missing grouped rows: {missing_adjusted_prob_groups:,} using keys {key_cols}.'
        )

    mismatch_counts: Dict[str, int] = {}
    mutable_goal_rows = aligned["event_type"].astype(str).str.strip().str.lower().eq("goal")
    for col in pred_cols:
        raw_vals = pd.to_numeric(aligned[f"{col}_raw"], errors="coerce").to_numpy(dtype=float)
        adj_vals = pd.to_numeric(aligned[f"{col}_adj"], errors="coerce").to_numpy(dtype=float)
        mismatch_mask = ~np.isclose(raw_vals, adj_vals, rtol=0.0, atol=1e-12, equal_nan=True)
        mismatch = int((mismatch_mask & (~mutable_goal_rows.to_numpy(dtype=bool))).sum())
        mismatch_counts[col] = mismatch
        if mismatch > 0:
            raise RuntimeError(
                'Adjusted OOF contract failed: probability columns changed during postprocess for raw rows. '
                f'Column {col} mismatch rows: {mismatch:,}.'
            )

    return {
        "raw_rows": int(len(raw_norm)),
        "adjusted_rows": int(len(adjusted_norm)),
        "adjusted_base_rows": int(len(adjusted_base)),
        "raw_columns": int(len(raw_cols)),
        "adjusted_columns": int(len(adjusted.columns)),
        "missing_raw_columns_in_adjusted": 0,
        "added_rows_in_adjusted": int(max(0, len(adjusted_norm) - len(raw_norm))),
        "key_columns": key_cols,
        "raw_duplicate_key_rows": raw_dups,
        "adjusted_duplicate_key_rows": adjusted_dups,
        "missing_adjusted_key_groups": missing_adjusted_rows,
        "adjusted_key_groups_lt_raw": adjusted_lt_raw,
        "goal_rows_exempted_from_prob_immutability": int(mutable_goal_rows.sum()),
        "probability_mismatch_counts": mismatch_counts,
    }


def main() -> None:
    args = parse_args()
    variant = str(args.variant).strip().lower()
    if variant != "events_only":
        raise ValueError(f"Unsupported variant for this script: {args.variant}. Supported: events_only")

    base_dir = args.base_dir.resolve()

    paths = TransformerXTPaths(base_dir=base_dir, run_label=args.run_label)
    paths.ensure_all()

    pipeline_run_dir = _resolve_pipeline_run_dir(base_dir, args.pipeline_run_label)
    phase2_dir = pipeline_run_dir / "phase2"
    phase3_dir = pipeline_run_dir / "phase3"

    raw_oof_path = paths.run_results_dir / "raw_oof_predictions.parquet"
    if not raw_oof_path.exists():
        raise FileNotFoundError(
            f"Required raw OOF artifact not found: {raw_oof_path}. "
            "Run training first to generate raw_oof_predictions.parquet."
        )

    phase2_events_path = phase2_dir / "events_phase2_enriched.parquet"
    faceoff_ref_path = phase2_dir / "faceoff_reference.parquet"
    penalty_ref_path = phase2_dir / "penalty_reference.parquet"
    tensor_ready_path = phase3_dir / "tensor_ready_dataset.parquet"

    for required in [phase2_events_path, faceoff_ref_path, penalty_ref_path, tensor_ready_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required Phase 2 reference not found: {required}")

    raw_oof = pd.read_parquet(raw_oof_path)
    raw_oof = _normalize_event_keys(raw_oof, key_col="sl_event_id")
    if args.sample_rows and int(args.sample_rows) > 0:
        raw_oof = raw_oof.head(int(args.sample_rows)).copy()

    enriched = _enrich_with_phase2_events(raw_oof, phase2_events_path)
    enriched, goal_actor_normalization_audit = _normalize_goal_rows_to_scoring_actor(enriched)
    baseline_ready, faceoff_baseline_audit = _compute_faceoff_zone_baselines(enriched)
    eos_ready, eos_reinjection_audit = _inject_eos_whistles_and_apply_faceoff_baselines(
        baseline_ready,
        neutral_baseline_for=float(faceoff_baseline_audit.get("nz_neutral_for", 0.0) or 0.0),
        neutral_baseline_against=float(faceoff_baseline_audit.get("nz_neutral_against", 0.0) or 0.0),
    )
    universal_ready, universal_delta_audit = _compute_universal_actor_relative_deltas(eos_ready)
    faceoff_baselines_csv_path, faceoff_baseline_export_audit = _export_faceoff_baselines_inspection(
        universal_ready,
        paths.inspections_dir,
    )
    goal_predecessor_warning_audit = _compute_goal_predecessor_warning_audit(enriched)
    base_adjusted = universal_ready

    faceoff_ref = pd.read_parquet(faceoff_ref_path)
    penalty_ref = pd.read_parquet(penalty_ref_path)

    pass2_taker, penalty_rows_newly_adjusted = _ensure_penalty_taker_adjusted(
        base_adjusted,
        for_bonus=float(args.penalty_static_for),
        against_bonus=float(args.penalty_static_against),
    )
    penalty_drawer_rows = _build_penalty_drawer_inverse_rows(pass2_taker, penalty_ref)
    faceoff_rows = _build_faceoff_inverse_rows(pass2_taker, faceoff_ref)
    goalie_ledger, goalie_goal_penalty_audit = _build_goalie_ledger(pass2_taker)
    sidecar_conservation_audit = _audit_sidecar_conservation(
        base_df=pass2_taker,
        faceoff_rows=faceoff_rows,
        penalty_drawer_rows=penalty_drawer_rows,
    )

    adjusted = _build_adjusted_output(
        base_df=pass2_taker,
        goalie_ledger=goalie_ledger,
        faceoff_rows=faceoff_rows,
        penalty_drawer_rows=penalty_drawer_rows,
    )
    contract_audit = _audit_adjusted_output_contract(raw_oof=raw_oof, adjusted=adjusted)

    player_ledger = _build_event_level_player_ledger(adjusted)
    consolidated_summary = _build_consolidated_player_goalie_summary(player_ledger, goalie_ledger)

    player_ledger_path = paths.run_results_dir / "player_ledger.parquet"
    goalie_path = paths.run_results_dir / "goalie_ledger.parquet"
    consolidated_summary_path = paths.run_results_dir / "consolidated_player_goalie_summary.parquet"
    goalie_ledger.to_parquet(goalie_path, index=False)
    player_ledger.to_parquet(player_ledger_path, index=False)
    consolidated_summary.to_parquet(consolidated_summary_path, index=False)

    summary_path = paths.logs_dir / "phase6_postprocess_summary.json"
    goalie_row_type_counts: Dict[str, int] = {}
    if "goalie_credit_row_type" in goalie_ledger.columns:
        goalie_row_type_counts = {
            str(k): int(v)
            for k, v in goalie_ledger["goalie_credit_row_type"].value_counts(dropna=False).to_dict().items()
        }

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "run_label": args.run_label,
        "variant": variant,
        "pipeline_run_dir": str(pipeline_run_dir),
        "raw_oof_path": str(raw_oof_path),
        "faceoff_baselines_inspection_csv": str(faceoff_baselines_csv_path),
        "player_ledger_path": str(player_ledger_path),
        "goalie_ledger_path": str(goalie_path),
        "consolidated_player_goalie_summary_path": str(consolidated_summary_path),
        "summary_path": str(summary_path),
        "rows": {
            "raw_oof": int(len(raw_oof)),
            "adjusted": int(len(adjusted)),
            "player_ledger": int(len(player_ledger)),
            "goalie_ledger": int(len(goalie_ledger)),
            "consolidated_player_goalie_summary": int(len(consolidated_summary)),
            "tensor_goal_rows": int(goalie_goal_penalty_audit.get("tensor_goal_rows_total", 0)),
            "tensor_goal_rows_with_linked_source": int(goalie_goal_penalty_audit.get("tensor_goal_rows_with_linked_source", 0)),
            "tensor_goal_rows_with_resolved_goalie": int(goalie_goal_penalty_audit.get("tensor_goal_rows_with_resolved_goalie", 0)),
            "faceoff_inverse_rows": int(len(faceoff_rows)),
            "penalty_drawer_inverse_rows": int(len(penalty_drawer_rows)),
            "penalty_taker_rows_newly_adjusted": int(penalty_rows_newly_adjusted),
            "goalie_row_type_counts": goalie_row_type_counts,
        },
        "goal_actor_normalization_audit": goal_actor_normalization_audit,
        "eos_reinjection_audit": eos_reinjection_audit,
        "universal_delta_audit": universal_delta_audit,
        "faceoff_baseline_export_audit": faceoff_baseline_export_audit,
        "faceoff_baseline_audit": faceoff_baseline_audit,
        "goal_predecessor_warning_audit": goal_predecessor_warning_audit,
        "goalie_goal_penalty_audit": goalie_goal_penalty_audit,
        "sidecar_conservation_audit": sidecar_conservation_audit,
        "raw_to_adjusted_contract_audit": contract_audit,
        "penalty_policy": {
            "static_for_bonus": float(args.penalty_static_for),
            "static_against_bonus": float(args.penalty_static_against),
        },
        "faceoff_baseline_policy": {
            "enabled": True,
            "application_scope": "faceoff reset rows only (winner-kept pre-sidecar)",
            "primary_coordinate": "x_adj",
            "boundary_tiebreak_coordinate": "x",
            "zone_bucket_rule": "OZ: x_adj > 25.5; DZ: x_adj < -25.5; NZ: -25.5 <= x_adj <= 25.5",
            "boundary_epsilon_ft": float(FACEOFF_ZONE_BOUNDARY_EPS_FT),
            "formula_mode": "explicit_zone_means_mirrored_neutral",
            "dynamic_mapping": {
                "offense_role": "baseline_for=oz_neutral_for, baseline_against=oz_neutral_against",
                "defense_role": "baseline_for=dz_neutral_for, baseline_against=dz_neutral_against",
                "neutral_role": "baseline_for=nz_neutral_for, baseline_against=nz_neutral_against",
            },
            "mirrored_neutral_formulas": {
                "oz_neutral_for": "(oz_win_for + dz_win_against) / 2",
                "oz_neutral_against": "(oz_win_against + dz_win_for) / 2",
                "dz_neutral_for": "(dz_win_for + oz_win_against) / 2",
                "dz_neutral_against": "(dz_win_against + oz_win_for) / 2",
                "nz_neutral_for": "(nz_win_for + nz_win_against) / 2",
                "nz_neutral_against": "(nz_win_against + nz_win_for) / 2",
            },
        },
        "goalie_policy": {
            "save_source_event_types": ["shot", "deflection"],
            "goal_penalty_source_event_types": ["tensor goal rows (shot-linked only)"],
            "reason_labels": [
                "save_play_continued",
                "save_with_stoppage_bonus",
                "save_with_stoppage_faceoff_baseline_bonus",
                "save_attempt_goal_allowed",
                "goal_penalty_dynamic",
                "goal_penalty_post_<prior_event_type>",
                "goal_allowed_empty_net_exempt",
                "save_unknown_next_event",
            ],
            "classification_rule": "all save rows are included with provenance labels; save->whistle->faceoff uses faceoff restart baseline when available; goal penalties are tensor-goal-linked rows only",
            "native_save_formula": "native_save_xT_Against = -shot_value_xt",
            "freeze_credit_formula": "extra_freeze_credit_xT_Against = -max(0, threat_after_native_save - restart_baseline_threat) when immediate next event is EOS; threat_after_native_save = max(0, pre_save_attacking_threat - shot_value_xt), and restart baseline uses faceoff zone baseline when available for save->whistle->faceoff",
            "goal_penalty_formula": "goal_penalty_xT_Against = max(0, 1.0 - prior_scoring_team_threat) on separate goal_penalty rows",
            "ledger_formula": "Net_xT = xT_For - xT_Against",
            "component_columns": [
                "native_save_xT_For",
                "native_save_xT_Against",
                "extra_freeze_credit_xT_Against",
                "goal_penalty_xT_Against",
                "xT_For",
                "xT_Against",
                "Net_xT",
            ],
        },
    }

    if goalie_goal_penalty_audit.get("tensor_goal_rows_total", 0) > 0 and goalie_goal_penalty_audit.get("goal_penalty_rows_emitted", 0) == 0:
        print(
            "Warning: tensor goal rows were detected but no goalie goal_penalty rows were emitted. "
            f"audit={goalie_goal_penalty_audit}"
        )

    if goal_predecessor_warning_audit["warning_unexpected_predecessors_present"]:
        print(
            "Warning: unexpected immediate predecessor event types were found before goal rows. "
            f"Unexpected rows={goal_predecessor_warning_audit['unexpected_predecessor_rows']}, "
            f"rate={goal_predecessor_warning_audit['unexpected_predecessor_rate']:.2%}."
        )

    if faceoff_baseline_audit.get("warning_low_sample_zone_baseline", False):
        print(
            "Warning: low-sample faceoff zone baselines detected. "
            f"zones={faceoff_baseline_audit.get('low_sample_zone_warnings', [])}, "
            f"threshold={faceoff_baseline_audit.get('min_zone_sample_warn_threshold', FACEOFF_MIN_ZONE_SAMPLE_WARN)}"
        )

    if goalie_goal_penalty_audit.get("freeze_fallback_rows", 0) > 0:
        print(
            "Warning: goalie freeze-credit fallback to pre-save threat was used for rows with missing post-save context. "
            f"freeze_fallback_rows={goalie_goal_penalty_audit.get('freeze_fallback_rows', 0)}"
        )

    if sidecar_conservation_audit.get("warning_sidecar_conservation_mismatch", False):
        print(
            "Warning: sidecar inverse conservation mismatches detected. "
            f"faceoff_groups_mismatch={sidecar_conservation_audit.get('faceoff_groups_mismatch', 0)}, "
            f"penalty_groups_mismatch={sidecar_conservation_audit.get('penalty_groups_mismatch', 0)}"
        )

    dedup_warning_total = int(
        goalie_goal_penalty_audit.get("tensor_goal_key_duplicates_dropped", 0)
        + goalie_goal_penalty_audit.get("source_link_duplicates_dropped", 0)
        + goalie_goal_penalty_audit.get("pregoal_link_duplicates_dropped", 0)
    )
    if dedup_warning_total > 0:
        print(
            "Warning: duplicate goal-link rows were collapsed during postprocess linkage recovery. "
            f"tensor_goal_key_duplicates_dropped={goalie_goal_penalty_audit.get('tensor_goal_key_duplicates_dropped', 0)}, "
            f"source_link_duplicates_dropped={goalie_goal_penalty_audit.get('source_link_duplicates_dropped', 0)}, "
            f"pregoal_link_duplicates_dropped={goalie_goal_penalty_audit.get('pregoal_link_duplicates_dropped', 0)}"
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Postprocess complete. Player ledger: {player_ledger_path}")
    print(f"Goalie ledger: {goalie_path}")
    print(f"Consolidated player-goalie summary: {consolidated_summary_path}")
    print(f"Faceoff baseline inspection CSV: {faceoff_baselines_csv_path}")
    print(f"Raw-to-adjusted contract audit: {contract_audit}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
