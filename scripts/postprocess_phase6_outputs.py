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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess Phase 6 raw OOF predictions")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--variant", type=str, default="events_only")
    parser.add_argument("--pipeline-run-label", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--penalty-static-for", type=float, default=PENALTY_STATIC_FOR_BONUS)
    parser.add_argument("--penalty-static-against", type=float, default=PENALTY_STATIC_AGAINST_BONUS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_event_keys(df: pd.DataFrame, key_col: str = "sl_event_id") -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out[key_col] = pd.to_numeric(out[key_col], errors="coerce").astype("Float64").round(6)
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
    candidates = ["period", "game_event_id", "period_time_sec", "period_time", "sl_event_id", "sequence_event_id"]
    return [c for c in candidates if c in df.columns]


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


# ---------------------------------------------------------------------------
# Step 1: Enrich with Phase 2 events + inject synthetic whistles
# ---------------------------------------------------------------------------

def _enrich_with_phase2_events(raw_oof: pd.DataFrame, tensor_ready_path: Path, phase2_events_path: Path) -> pd.DataFrame:
    """
    Enrich tensor_ready_dataset (authoritative event spine) with OOF predictions
    and Phase 2 enrichment columns.

    Start with tensor_ready_dataset (all events including goals, whistles, saves).
    Left-join OOF predictions (bring only NEW columns: P_actor_goal, P_opp_goal, etc).
    Left-join Phase 2 enrichment (bring only NEW columns: goalie_id, opp_goalie_id, etc).
    This ensures complete event coverage (goals/whistles never lost) while adding
    model predictions and enrichment metadata.

    After join, inject a synthetic whistle row immediately after each goal
    to preserve the reset/baseline state for faceoff delta computation downstream.
    The synthetic whistle looks ahead to the next faceoff row within the same game
    to determine the acting team and zone-appropriate baseline probabilities.
    If no subsequent faceoff exists, the whistle row gets NaN probabilities
    (neutral fallback is applied later in the baseline step).
    """
    # Load tensor_ready_dataset as authoritative event spine.
    tensor_ready = pd.read_parquet(tensor_ready_path)
    tensor_ready = _normalize_event_keys(tensor_ready, key_col="sl_event_id")

    # LEFT JOIN OOF predictions (only new prediction columns, not event metadata).
    # Prediction columns to bring from raw_oof.
    oof_pred_cols = [
        "game_id", "sl_event_id",
        "P_actor_goal", "P_opp_goal", "P_no_goal",
    ]
    oof_pred_cols = [c for c in oof_pred_cols if c in raw_oof.columns]
    if not raw_oof.empty and oof_pred_cols:
        raw_oof_pred = raw_oof[oof_pred_cols].drop_duplicates(["game_id", "sl_event_id"])
        merged = tensor_ready.merge(
            raw_oof_pred,
            on=["game_id", "sl_event_id"],
            how="left",
            suffixes=("", "_oof"),
        )
    else:
        merged = tensor_ready.copy()

    # LEFT JOIN Phase 2 enrichment (only new enrichment columns, not event metadata).
    phase2_events = pd.read_parquet(phase2_events_path)
    phase2_events = _normalize_event_keys(phase2_events, key_col="sl_event_id")

    phase2_enrich_cols = [
        "game_id", "sl_event_id",
        "goalie_id", "opp_goalie_id", "is_synthetic_save",
        "save_source_sl_event_id", "save_source_event_type",
        "source_linked_goal_sl_event_id", "goal_converted_source",
        "goal_linked_source_sl_event_id", "is_empty_net_source_attempt",
        "is_home_net_empty", "is_away_net_empty",
        "home_goalie_id", "away_goalie_id", "target",
    ]
    phase2_enrich_cols = [c for c in phase2_enrich_cols if c in phase2_events.columns]
    if phase2_enrich_cols and not phase2_events.empty:
        phase2_trim = phase2_events[phase2_enrich_cols].drop_duplicates(["game_id", "sl_event_id"])
        merged = merged.merge(
            phase2_trim,
            on=["game_id", "sl_event_id"],
            how="left",
            suffixes=("", "_phase2"),
        )

        # Coalesce enrichment columns (prefer tensor_ready, fall back to Phase 2 where NaN).
        for col in [
            "goalie_id", "opp_goalie_id", "is_synthetic_save",
            "save_source_sl_event_id", "save_source_event_type",
            "source_linked_goal_sl_event_id", "goal_converted_source",
            "goal_linked_source_sl_event_id", "is_empty_net_source_attempt",
            "is_home_net_empty", "is_away_net_empty",
            "home_goalie_id", "away_goalie_id", "target",
        ]:
            phase2_col = f"{col}_phase2"
            if phase2_col in merged.columns:
                if col not in merged.columns:
                    merged[col] = merged[phase2_col]
                else:
                    merged[col] = merged[col].where(merged[col].notna(), merged[phase2_col])

        # Drop all _phase2 suffix columns.
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_phase2")], errors="ignore")

    if "period_time_sec" not in merged.columns and "period_time" in merged.columns:
        merged["period_time_sec"] = _parse_period_time_to_seconds(merged["period_time"])
    elif "period_time_sec" in merged.columns:
        merged["period_time_sec"] = pd.to_numeric(merged["period_time_sec"], errors="coerce")

    merged["event_type"] = merged["event_type"].astype(str).str.strip().str.lower()

    # Sort chronologically before synthetic whistle injection.
    sort_cols = [c for c in ["game_id", "period", "period_time_sec", "sl_event_id"] if c in merged.columns]
    merged = merged.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    # --- Inject synthetic whistles after each goal ---
    # For each goal row, look ahead within the game to find the next faceoff.
    # The synthetic whistle inherits the faceoff's team_id, x_adj, and period context
    # so that the baseline injection step can stamp the correct zone probabilities.
    # sl_event_id is set to goal_sl_event_id + 0.1 to slot it immediately after.

    goal_mask = merged["event_type"].eq("goal")
    if goal_mask.any():
        merged["_row_idx"] = np.arange(len(merged))

        # Build a forward faceoff lookup: for each row, what is the next faceoff
        # row index within the same game?
        faceoff_mask = merged["event_type"].eq("faceoff")
        merged["_faceoff_row_idx"] = merged["_row_idx"].where(faceoff_mask, np.nan)
        merged["_next_faceoff_row_idx"] = (
            merged.groupby("game_id", sort=False)["_faceoff_row_idx"]
            .transform(lambda s: s.bfill())
        )

        goal_rows = merged.loc[goal_mask].copy()
        synthetic = goal_rows.copy()
        synthetic["event_type"] = "whistle"
        synthetic["is_synthetic_whistle"] = 1
        synthetic["sl_event_id"] = (
            pd.to_numeric(goal_rows["sl_event_id"], errors="coerce").fillna(0.0) + 0.1
        ).astype("Float64").round(6)

        # Look up next faceoff context for team/location on the whistle row.
        next_fo_idx = pd.to_numeric(goal_rows["_next_faceoff_row_idx"], errors="coerce")
        has_next_fo = next_fo_idx.notna()
        if has_next_fo.any():
            fo_idx_int = next_fo_idx.loc[has_next_fo].astype(int).to_numpy()
            synthetic.loc[has_next_fo, "team_id"] = merged["team_id"].iloc[fo_idx_int].values
            synthetic.loc[has_next_fo, "x_adj"] = merged["x_adj"].iloc[fo_idx_int].values
            synthetic.loc[has_next_fo, "x"] = merged["x"].iloc[fo_idx_int].values

        # Rows without a subsequent faceoff get NaN probs; neutral fallback applied later.
        for prob_col in ["P_actor_goal", "P_opp_goal", "P_no_goal"]:
            if prob_col in synthetic.columns:
                synthetic[prob_col] = np.nan

        merged = pd.concat([merged, synthetic], ignore_index=True, sort=False)
        merged = merged.drop(columns=["_row_idx", "_faceoff_row_idx", "_next_faceoff_row_idx"], errors="ignore")

        # Re-sort so synthetic whistles slot immediately after their goal rows.
        sort_cols2 = [c for c in ["game_id", "period", "period_time_sec", "sl_event_id"] if c in merged.columns]
        merged = merged.sort_values(by=sort_cols2, kind="mergesort").reset_index(drop=True)

    if "is_synthetic_whistle" not in merged.columns:
        merged["is_synthetic_whistle"] = 0
    else:
        merged["is_synthetic_whistle"] = merged["is_synthetic_whistle"].fillna(0).astype(int)

    return merged


# ---------------------------------------------------------------------------
# Step 2: Normalize goal row probabilities
# ---------------------------------------------------------------------------

def _normalize_goal_rows_to_scoring_actor(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Goal rows arrive from Phase 2 with no model probabilities (they were excluded
    from training). Set their probabilities to the certain outcome state:
    P_actor_goal=1.0, P_opp_goal=0.0. team_id is already populated from Phase 2.

    This is used only for goalie penalty calculation. It does not affect skater
    deltas because goal rows are zeroed in the universal delta pass.
    """
    out = df.copy()
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()

    for col in ["P_actor_goal", "P_opp_goal"]:
        if col not in out.columns:
            raise RuntimeError(f"Cannot normalize goal rows: missing column {col}")

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


# ---------------------------------------------------------------------------
# Step 3: Compute faceoff zone baselines and stamp onto whistle rows
# ---------------------------------------------------------------------------

def _compute_faceoff_zone_baselines(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute OOF-wide mean P_actor_goal / P_opp_goal for faceoff rows, bucketed
    by zone (OZ / NZ / DZ) using x_adj relative to the acting team (winner).

    Zone boundaries:
        OZ:  x_adj >  25.5  (winner is in attacking zone)
        DZ:  x_adj < -25.5  (winner is in defensive zone)
        NZ: -25.5 <= x_adj <= 25.5

    Boundary rows within FACEOFF_ZONE_BOUNDARY_EPS_FT of 25.5 use raw x as
    tiebreaker.

    After computing zone means, stamp the appropriate baseline P_actor_goal /
    P_opp_goal onto every whistle row by looking up the zone of the next faceoff
    in the game. 

    The whistle baseline represents the PRE-FACEOFF state. Since we do not know 
    who will win the draw, the baseline is a 50/50 blend of the winner's and 
    loser's historical outcomes (e.g., an OZ draw is an average of an OZ win 
    and a DZ loss). A faceoff winner's delta in the universal pass is then their 
    actual model output minus this baseline — crediting them for exceeding 
    (or penalizing for falling below) the neutral zone-average threat.
    """
    out = df.copy()
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()

    faceoff_mask = out["event_type"].eq("faceoff")

    if not faceoff_mask.any():
        raise RuntimeError("No faceoff rows found — cannot compute zone baselines.")

    if "x_adj" not in out.columns or "x" not in out.columns:
        raise RuntimeError("Columns x_adj and x are required for faceoff zone baseline computation.")

    x_adj = pd.to_numeric(out["x_adj"], errors="coerce")
    x_raw = pd.to_numeric(out["x"], errors="coerce")
    abs_x_adj = x_adj.abs()

    zone = pd.Series("nz", index=out.index, dtype="object")
    zone.loc[x_adj > FACEOFF_ZONE_EDGE_FT] = "oz"
    zone.loc[x_adj < -FACEOFF_ZONE_EDGE_FT] = "dz"

    # Boundary tiebreak: rows within epsilon of the zone edge use raw x.
    boundary = faceoff_mask & (abs_x_adj - FACEOFF_ZONE_EDGE_FT).abs().le(FACEOFF_ZONE_BOUNDARY_EPS_FT)
    zone.loc[boundary & x_raw.gt(0)] = "oz"
    zone.loc[boundary & x_raw.lt(0)] = "dz"

    out["faceoff_zone"] = np.where(faceoff_mask, zone, pd.NA)

    p_actor = pd.to_numeric(out["P_actor_goal"], errors="coerce")
    p_opp = pd.to_numeric(out["P_opp_goal"], errors="coerce")

    def _zone_means(z: str) -> tuple[float, float, int]:
        mask = faceoff_mask & zone.eq(z)
        samples = int(mask.sum())
        return (
            float(p_actor.loc[mask].mean()) if samples > 0 else 0.0,
            float(p_opp.loc[mask].mean()) if samples > 0 else 0.0,
            samples,
        )

    # 1. Get the raw post-win means
    oz_for, oz_against, oz_n = _zone_means("oz")
    dz_for, dz_against, dz_n = _zone_means("dz")
    nz_for, nz_against, nz_n = _zone_means("nz")

    # 2. Blend them into pre-faceoff "Mirrored Neutral" baselines (assuming 50% win rate)
    # If Team A is in the OZ, they have a 50% chance of getting oz_for, and a 50% chance 
    # of losing it (meaning Team B gets dz_for, which manifests as dz_against for Team A).
    oz_neutral_for = (oz_for + dz_against) / 2.0
    oz_neutral_against = (oz_against + dz_for) / 2.0

    dz_neutral_for = (dz_for + oz_against) / 2.0
    dz_neutral_against = (dz_against + oz_for) / 2.0

    nz_neutral_for = (nz_for + nz_against) / 2.0
    nz_neutral_against = (nz_against + nz_for) / 2.0

    # 3. Apply the blended baselines
    baselines = {
        "oz": (oz_neutral_for, oz_neutral_against),
        "dz": (dz_neutral_for, dz_neutral_against),
        "nz": (nz_neutral_for, nz_neutral_against),
    }

    # Stamp baselines onto whistle rows by looking up the next faceoff zone.
    whistle_mask = out["event_type"].eq("whistle")

    out["_row_idx"] = np.arange(len(out))
    out["_faceoff_zone_at_idx"] = np.where(faceoff_mask, zone, np.nan).astype(object)
    
    # bfill within game gives each whistle its next faceoff's zone.
    out["_next_faceoff_zone"] = (
        out.groupby("game_id", sort=False)["_faceoff_zone_at_idx"]
        .transform(lambda s: s.bfill())
    )

    # Initialise whistle baseline columns.
    out["whistle_baseline_for"] = np.nan
    out["whistle_baseline_against"] = np.nan

    for z, (b_for, b_against) in baselines.items():
        zmask = whistle_mask & out["_next_faceoff_zone"].eq(z)
        out.loc[zmask, "whistle_baseline_for"] = b_for
        out.loc[zmask, "whistle_baseline_against"] = b_against

    # Fallback: whistles with no subsequent faceoff get NZ neutral.
    fallback = whistle_mask & out["whistle_baseline_for"].isna()
    out.loc[fallback, "whistle_baseline_for"] = nz_neutral_for
    out.loc[fallback, "whistle_baseline_against"] = nz_neutral_against

    # Stamp baseline probabilities onto whistle rows so the universal delta
    # pass treats them as any other row.
    out.loc[whistle_mask, "P_actor_goal"] = out.loc[whistle_mask, "whistle_baseline_for"]
    out.loc[whistle_mask, "P_opp_goal"] = out.loc[whistle_mask, "whistle_baseline_against"]
    if "P_no_goal" in out.columns:
        out.loc[whistle_mask, "P_no_goal"] = np.clip(
            1.0
            - out.loc[whistle_mask, "whistle_baseline_for"]
            - out.loc[whistle_mask, "whistle_baseline_against"],
            0.0, 1.0,
        )

    out = out.drop(columns=["_row_idx", "_faceoff_zone_at_idx", "_next_faceoff_zone"], errors="ignore")

    # Sort by game_id, game_event_id to ensure correct event sequencing for bfill().
    sort_cols = [c for c in ["game_id", "game_event_id"] if c in out.columns]
    if len(sort_cols) == 2:
        out = out.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    low_sample_warnings = []
    for z, n in [("oz", oz_n), ("dz", dz_n), ("nz", nz_n)]:
        if n < FACEOFF_MIN_ZONE_SAMPLE_WARN:
            low_sample_warnings.append({"zone": z, "sample_count": n})

    audit: Dict[str, Any] = {
        "faceoff_rows_total": int(faceoff_mask.sum()),
        "oz_sample_count": oz_n,
        "dz_sample_count": dz_n,
        "nz_sample_count": nz_n,
        "oz_win_for": oz_for,
        "oz_win_against": oz_against,
        "dz_win_for": dz_for,
        "dz_win_against": dz_against,
        "nz_win_for": nz_for,
        "nz_win_against": nz_against,
        "oz_neutral_baseline_for": oz_neutral_for,
        "oz_neutral_baseline_against": oz_neutral_against,
        "dz_neutral_baseline_for": dz_neutral_for,
        "dz_neutral_baseline_against": dz_neutral_against,
        "nz_neutral_baseline_for": nz_neutral_for,
        "nz_neutral_baseline_against": nz_neutral_against,
        "whistle_rows_stamped": int(whistle_mask.sum()),
        "whistle_rows_fallback_nz": int(fallback.sum()),
        "low_sample_zone_warnings": low_sample_warnings,
        "warning_low_sample_zone_baseline": len(low_sample_warnings) > 0,
    }
    return out, audit


# ---------------------------------------------------------------------------
# Step 4: Universal actor-relative delta computation
# ---------------------------------------------------------------------------

def _compute_universal_actor_relative_deltas(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For every event except goals and whistles, compute the marginal change in
    scoring probability from the actor's perspective relative to the previous row.

    When the acting team changes between consecutive rows, the prior row's
    P_actor and P_opp are flipped so the delta is always computed from the
    current actor's point of view.

    Goals and whistles receive zero delta. Whistles have already had their
    P_actor/P_opp set to the zone baseline in Step 3, so they serve as the
    prior state for the next event (the faceoff) without contributing a delta
    themselves.

    Output columns:
        Actor_xT_For    — increase in own scoring threat
        Actor_xT_Against — increase in opponent scoring threat (positive = bad)
        Actor_Net_xT    — For - Against
        Adjusted_xT_For, Adjusted_xT_Against, Adjusted_Net_xT — copies at this
            stage; penalties will be adjusted in Step 5.
    """
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = out.get("team_id", pd.Series("", index=out.index)).astype("string")
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()

    for col in ["P_actor_goal", "P_opp_goal"]:
        if col not in out.columns:
            raise RuntimeError(f"Missing required column for delta computation: {col}")

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    # Sort by game_id, game_event_id for correct event sequence within each game.
    sort_cols = [c for c in ["game_id", "game_event_id"] if c in out.columns]
    if len(sort_cols) == 2:
        out = out.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False, dropna=False)

    # Forward-fill team_id so stoppage rows (whistle NaN team) don't break
    # same-team continuity detection across sequences.
    filled_team = grp["team_id"].ffill().astype("string")
    prev_team = grp["team_id"].ffill().shift(1).astype("string")

    prev_actor = pd.to_numeric(grp["P_actor_goal"].shift(1), errors="coerce")
    prev_opp = pd.to_numeric(grp["P_opp_goal"].shift(1), errors="coerce")

    same_team = filled_team.eq(prev_team).fillna(False)
    first_in_group = grp.cumcount().eq(0)

    # Align prior state to current actor's perspective.
    prior_for = pd.Series(
        np.where(same_team, prev_actor, prev_opp), index=out.index
    )
    prior_against = pd.Series(
        np.where(same_team, prev_opp, prev_actor), index=out.index
    )
    prior_for = pd.to_numeric(prior_for, errors="coerce").fillna(0.0).mask(first_in_group, 0.0)
    prior_against = pd.to_numeric(prior_against, errors="coerce").fillna(0.0).mask(first_in_group, 0.0)

    curr_for = pd.to_numeric(out["P_actor_goal"], errors="coerce").fillna(0.0)
    curr_against = pd.to_numeric(out["P_opp_goal"], errors="coerce").fillna(0.0)

    delta_for = curr_for - prior_for
    delta_against = curr_against - prior_against

    # Goals and whistles get zero delta — they are boundary/state rows only.
    zero_mask = out["event_type"].isin(["goal", "whistle"])
    delta_for = delta_for.where(~zero_mask, 0.0)
    delta_against = delta_against.where(~zero_mask, 0.0)

    out["Actor_xT_For"] = delta_for
    out["Actor_xT_Against"] = delta_against
    out["Actor_Net_xT"] = delta_for - delta_against
    out["prior_state_reset"] = first_in_group.astype(np.float32)

    # Adjusted columns start as copies; penalties are modified in Step 5.
    out["Adjusted_xT_For"] = delta_for
    out["Adjusted_xT_Against"] = delta_against
    out["Adjusted_Net_xT"] = delta_for - delta_against
    out["adjustment_source"] = "base_actor"

    audit: Dict[str, Any] = {
        "rows_total": int(len(out)),
        "rows_zero_delta_goal_whistle": int(zero_mask.sum()),
        "first_in_group_rows": int(first_in_group.sum()),
        "team_flip_rows": int((~same_team & ~first_in_group).sum()),
    }
    return out, audit


# ---------------------------------------------------------------------------
# Step 5: Penalty adjustments and sidecar inverse rows
# ---------------------------------------------------------------------------

def _ensure_penalty_taker_adjusted(
    df: pd.DataFrame,
    for_bonus: float,
    against_bonus: float,
) -> tuple[pd.DataFrame, int]:
    """Add static bonus values to penalty taker rows."""
    out = df.copy()
    evt = out["event_type"].astype(str).str.strip().str.lower()
    penalty_mask = evt.eq("penalty")

    already = (
        out["adjustment_source"].astype(str).str.strip().str.lower().eq("penalty_static_taker")
        if "adjustment_source" in out.columns
        else pd.Series(False, index=out.index)
    )
    needs = penalty_mask & (~already)

    if needs.any():
        out.loc[needs, "Adjusted_xT_For"] = (
            pd.to_numeric(out.loc[needs, "Actor_xT_For"], errors="coerce").fillna(0.0) + for_bonus
        )
        out.loc[needs, "Adjusted_xT_Against"] = (
            pd.to_numeric(out.loc[needs, "Actor_xT_Against"], errors="coerce").fillna(0.0) + against_bonus
        )
        out.loc[needs, "Adjusted_Net_xT"] = (
            out.loc[needs, "Adjusted_xT_For"] - out.loc[needs, "Adjusted_xT_Against"]
        )
        out.loc[needs, "adjustment_source"] = "penalty_static_taker"

    return out, int(needs.sum())


def _build_faceoff_inverse_rows(base_df: pd.DataFrame, faceoff_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror faceoff winner deltas to the losing player/team.
    The loser's Adjusted_Net_xT is the negative of the winner's.
    """
    if faceoff_ref.empty:
        return pd.DataFrame()

    scored = (
        base_df[["game_id", "sl_event_id", "Adjusted_Net_xT"]]
        .copy()
        .pipe(_normalize_event_keys, key_col="sl_event_id")
        .rename(columns={"sl_event_id": "kept_sl_event_id", "Adjusted_Net_xT": "winner_net_xt"})
        .drop_duplicates(["game_id", "kept_sl_event_id"])
    )

    ref = (
        _normalize_event_keys(faceoff_ref, key_col="sl_event_id")
        .pipe(_normalize_event_keys, key_col="kept_sl_event_id")
    )

    merged = ref.merge(scored, on=["game_id", "kept_sl_event_id"], how="left")
    merged = merged.loc[merged["winner_net_xt"].notna()].copy()

    inverse = -pd.to_numeric(merged["winner_net_xt"], errors="coerce").fillna(0.0)

    return pd.DataFrame({
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
    })


def _build_penalty_drawer_inverse_rows(base_df: pd.DataFrame, penalty_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Assign the inverse of the penalty taker's post-static Adjusted_Net_xT
    to the player who drew the penalty.
    """
    if penalty_ref.empty:
        return pd.DataFrame()

    evt = base_df["event_type"].astype(str).str.strip().str.lower()
    taker = (
        base_df.loc[evt.eq("penalty"), ["game_id", "sl_event_id", "Adjusted_Net_xT"]]
        .copy()
        .pipe(_normalize_event_keys, key_col="sl_event_id")
        .rename(columns={"sl_event_id": "kept_sl_event_id", "Adjusted_Net_xT": "taker_net_xt"})
        .drop_duplicates(["game_id", "kept_sl_event_id"])
    )

    if taker.empty:
        return pd.DataFrame()

    ref = (
        _normalize_event_keys(penalty_ref, key_col="sl_event_id")
        .pipe(_normalize_event_keys, key_col="kept_sl_event_id")
    )

    merged = ref.merge(taker, on=["game_id", "kept_sl_event_id"], how="left")
    merged = merged.loc[merged["taker_net_xt"].notna()].copy()

    inverse = -pd.to_numeric(merged["taker_net_xt"], errors="coerce").fillna(0.0)

    return pd.DataFrame({
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
    })


# ---------------------------------------------------------------------------
# Step 6: Goalie ledger
# ---------------------------------------------------------------------------

def _build_goalie_ledger(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build goalie credit rows from save events and goal events.

    For each save row:
        xT_For      = Actor_xT_For from universal pass (marginal own-team scoring
                      threat increase from the save)
        xT_Against  = Actor_xT_Against from universal pass (native save credit,
                      negative = good) PLUS freeze credit if the next event is a
                      whistle.

        Freeze credit = -(whistle_baseline_against - P_opp_goal_on_save_row)
            clipped so it cannot make xT_Against less negative (i.e. only adds
            additional credit, never penalises).
            whistle_baseline_against is taken from the whistle row that immediately
            follows the save (stamped in Step 3).

        Net_xT = xT_For - xT_Against  (positive = good goalie performance)

    For each goal allowed (one penalty row per goal):
        Attacking team threat before the goal = max(P_actor_goal, P_opp_goal)
        on the row immediately preceding the goal (pre-normalization state).
        xT_Against = 1.0 - pre_goal_attacking_threat  (positive = bad)
        xT_For     = 0.0
        Net_xT     = -xT_Against  (negative = bad outcome)
    """
    out = df.copy()
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()

    group_cols = ["game_id"]
    for opt in ["model_variant", "variant_name"]:
        if opt in out.columns:
            group_cols.append(opt)

    # Sort by game_id, game_event_id to ensure saves are correctly paired with following whistles.
    sort_cols = [c for c in ["game_id", "game_event_id"] if c in out.columns]
    if len(sort_cols) == 2:
        out = out.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    grp = out.groupby(group_cols, sort=False, dropna=False)

    save_mask = out["event_type"].eq("save")
    goal_mask = out["event_type"].eq("goal")

    # --- Goalie ID resolution ---
    # For saves: the goalie is the one making the save (defending team's goalie).
    # Prefer goalie_id; fall back to opp_goalie_id.
    if "goalie_id" in out.columns:
        save_goalie = out["goalie_id"].copy()
        if "opp_goalie_id" in out.columns:
            save_goalie = save_goalie.where(save_goalie.notna(), out["opp_goalie_id"])
    elif "opp_goalie_id" in out.columns:
        save_goalie = out["opp_goalie_id"].copy()
    else:
        save_goalie = pd.Series(pd.NA, index=out.index, dtype="object")

    # For goal penalties: the goalie who allowed the goal is the opposing goalie.
    goal_goalie = (
        out["opp_goalie_id"].copy()
        if "opp_goalie_id" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="object")
    )

    # Opponent team map for resolving goalie team.
    opp_map = _build_opponent_team_map(out)
    if opp_map:
        opp_series = pd.Series(opp_map, dtype="object")
        opp_keys = pd.MultiIndex.from_arrays(
            [out["game_id"].astype(str), out["team_id"].astype(str)]
        )
        goalie_team = pd.Series(
            opp_series.reindex(opp_keys).to_numpy(), index=out.index, dtype="string"
        )
    else:
        goalie_team = pd.Series(pd.NA, index=out.index, dtype="string")

    # --- Next-event context for freeze credit ---
    # After Step 3 the whistle rows immediately following saves have
    # whistle_baseline_against stamped on them. Shift by -1 within each game
    # to get the next row's event_type and baseline values onto the save row.
    next_event_type = grp["event_type"].shift(-1).astype("string")
    next_whistle_baseline_against = (
        pd.to_numeric(grp["whistle_baseline_against"].shift(-1), errors="coerce")
        if "whistle_baseline_against" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    next_is_whistle = next_event_type.str.strip().str.lower().eq("whistle").fillna(False)

    # --- Save rows ---
    p_opp_on_save = pd.to_numeric(out["P_opp_goal"], errors="coerce").fillna(0.0)

    # Freeze credit: goalie reduced threat further from post-save opponent
    # level down to the restart baseline by freezing the puck.
    # Credit is negative (reduces xT_Against) and clipped so it cannot
    # accidentally increase xT_Against.
    freeze_credit = pd.Series(0.0, index=out.index)
    has_whistle_baseline = next_is_whistle & next_whistle_baseline_against.notna()
    raw_freeze = next_whistle_baseline_against - p_opp_on_save
    # Only credit where baseline is actually lower than current opp threat.
    freeze_credit = freeze_credit.where(
        ~(save_mask & has_whistle_baseline),
        -np.clip(-raw_freeze, 0.0, None),  # negative clip: credit cannot be positive
    )

    xT_For_save = pd.to_numeric(out["Actor_xT_For"], errors="coerce").fillna(0.0)
    xT_Against_save = (
        pd.to_numeric(out["Actor_xT_Against"], errors="coerce").fillna(0.0)
        + freeze_credit
    )
    net_xT_save = xT_For_save - xT_Against_save

    save_reason = pd.Series("save_no_next_event", index=out.index, dtype="object")
    save_reason.loc[save_mask & next_is_whistle & has_whistle_baseline] = "save_with_freeze_credit"
    save_reason.loc[save_mask & next_is_whistle & (~has_whistle_baseline)] = "save_with_whistle_no_baseline"
    save_reason.loc[save_mask & (~next_is_whistle)] = "save_play_continued"

    save_rows = pd.DataFrame({
        "goalie_credit_row_type": "save",
        "game_id": out.loc[save_mask, "game_id"].values,
        "sl_event_id": out.loc[save_mask, "sl_event_id"].values,
        "goalie_id": save_goalie.loc[save_mask].values,
        "goalie_team_id": goalie_team.loc[save_mask].values,
        "event_type": "save",
        "next_event_type": next_event_type.loc[save_mask].values,
        "next_whistle_baseline_against": next_whistle_baseline_against.loc[save_mask].values,
        "freeze_credit": freeze_credit.loc[save_mask].values,
        "xT_For": xT_For_save.loc[save_mask].values,
        "xT_Against": xT_Against_save.loc[save_mask].values,
        "Net_xT": net_xT_save.loc[save_mask].values,
        "goalie_credit_reason": save_reason.loc[save_mask].values,
    })

    # --- Goal penalty rows ---
    # Pre-goal attacking threat: look at the row immediately before the goal.
    prev_p_actor = pd.to_numeric(grp["P_actor_goal"].shift(1), errors="coerce").fillna(0.0)
    prev_p_opp = pd.to_numeric(grp["P_opp_goal"].shift(1), errors="coerce").fillna(0.0)
    # Attacking team always has the higher probability.
    pre_goal_threat = np.maximum(prev_p_actor.to_numpy(), prev_p_opp.to_numpy())
    pre_goal_threat = pd.Series(pre_goal_threat, index=out.index)

    goal_penalty_xT_Against = (1.0 - pre_goal_threat).clip(lower=0.0)

    # Empty net exemption: do not penalise the goalie for goals scored against
    # an empty net. Check if the opposing net was empty when the goal was scored.
    # is_home_net_empty / is_away_net_empty indicate which net was empty.
    # Both being True means both nets were empty (some odd scenario); treat as empty net.
    empty_net_mask = pd.Series(False, index=out.index)
    
    if "is_home_net_empty" in out.columns and "is_away_net_empty" in out.columns:
        home_empty = pd.to_numeric(out["is_home_net_empty"], errors="coerce").fillna(0).astype(bool)
        away_empty = pd.to_numeric(out["is_away_net_empty"], errors="coerce").fillna(0).astype(bool)
        # A goal was empty net if either net was empty (credit to scoring team's effectiveness on empty net)
        empty_net_mask = home_empty | away_empty
    
    goal_penalty_xT_Against = goal_penalty_xT_Against.where(~empty_net_mask, 0.0)

    goal_net_xT = -goal_penalty_xT_Against

    goal_rows = pd.DataFrame({
        "goalie_credit_row_type": "goal_penalty",
        "game_id": out.loc[goal_mask, "game_id"].values,
        "sl_event_id": out.loc[goal_mask, "sl_event_id"].values,
        "goalie_id": goal_goalie.loc[goal_mask].values,
        "goalie_team_id": goalie_team.loc[goal_mask].values,
        "event_type": "goal",
        "next_event_type": next_event_type.loc[goal_mask].values,
        "next_whistle_baseline_against": np.nan,
        "freeze_credit": 0.0,
        "xT_For": 0.0,
        "xT_Against": goal_penalty_xT_Against.loc[goal_mask].values,
        "Net_xT": goal_net_xT.loc[goal_mask].values,
        "goalie_credit_reason": "goal_allowed",
    })

    ledger = pd.concat([save_rows, goal_rows], ignore_index=True, sort=False)

    # Drop rows where goalie is unidentifiable.
    valid_goalie = (
        ledger["goalie_id"].astype("string").str.strip()
        .isin(["", "<NA>", "nan", "None", pd.NA])
        .eq(False)
        & ledger["goalie_id"].notna()
    )
    ledger = ledger.loc[valid_goalie].reset_index(drop=True)

    audit: Dict[str, Any] = {
        "save_rows": int(save_mask.sum()),
        "goal_penalty_rows": int(goal_mask.sum()),
        "goal_penalty_rows_empty_net_exempted": int((goal_mask & empty_net_mask).sum()),
        "goalie_ledger_rows": int(len(ledger)),
        "save_rows_with_freeze_credit": int((save_mask & has_whistle_baseline).sum()),
        "save_rows_whistle_no_baseline": int(
            (save_mask & next_is_whistle & (~has_whistle_baseline)).sum()
        ),
        "goal_rows_missing_goalie": int(
            goal_mask.sum() - (goal_goalie.loc[goal_mask].notna().sum())
        ),
    }
    return ledger, audit


# ---------------------------------------------------------------------------
# Step 7: Player ledger
# ---------------------------------------------------------------------------

def _build_event_level_player_ledger(adjusted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Skater event-level credit ledger. Includes all event types except saves
    (handled in goalie ledger), whistles, and goals. Faceoff winners, faceoff
    loser sidecars, penalty takers, and penalty drawer sidecars are all included.

    Uses Adjusted_Net_xT uniformly. For faceoffs this equals Actor_Net_xT since
    no static adjustment is applied. For penalties it reflects the post-static value.
    Sidecar rows carry their inverse values directly.
    """
    ledger = adjusted_df.copy()
    ledger["event_type"] = ledger["event_type"].astype(str).str.strip().str.lower()

    exclude = {"save", "whistle", "goal"}
    ledger = ledger.loc[~ledger["event_type"].isin(exclude)].copy()

    ledger["sl_event_id"] = pd.to_numeric(ledger["sl_event_id"], errors="coerce").astype("Float64").round(6)
    ledger["credit_for"] = pd.to_numeric(ledger.get("Adjusted_xT_For", 0.0), errors="coerce").fillna(0.0)
    ledger["credit_against"] = pd.to_numeric(ledger.get("Adjusted_xT_Against", 0.0), errors="coerce").fillna(0.0)
    ledger["credit_xt"] = pd.to_numeric(ledger.get("Adjusted_Net_xT", 0.0), errors="coerce").fillna(0.0)

    if "player_id" in ledger.columns:
        player_id = ledger["player_id"].astype("string").str.strip()
        ledger = ledger.loc[
            player_id.notna() & (~player_id.isin(["", "<NA>", "nan", "None"]))
        ].copy()

    keep_cols = [
        "game_id", "sl_event_id", "linked_kept_sl_event_id", "team_id", "player_id",
        "event_type", "period", "sequence_id", "adjustment_source", "outcome",
        "credit_for", "credit_against", "credit_xt",
        "Adjusted_xT_For", "Adjusted_xT_Against", "Adjusted_Net_xT",
    ]
    keep_cols = [c for c in keep_cols if c in ledger.columns]
    return ledger[keep_cols].reset_index(drop=True)


def _build_consolidated_player_goalie_summary(
    player_ledger: pd.DataFrame,
    goalie_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot both ledgers into a per-player summary with event-type breakdowns
    and a Total_Net_xT column.
    """
    def _prep_frame(frame: pd.DataFrame, net_col: str, id_col: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["player_id", "team_id", "event_type", "Net_xT"])
        f = frame.copy()
        if id_col != "player_id" and id_col in f.columns:
            f["player_id"] = f[id_col]
        f["Net_xT"] = pd.to_numeric(f.get(net_col, 0.0), errors="coerce").fillna(0.0)
        
        # Normalize team_id column: handle both "team_id" and "goalie_team_id"
        if "team_id" not in f.columns and "goalie_team_id" in f.columns:
            f["team_id"] = f["goalie_team_id"]
        
        # Ensure all required columns exist
        required_cols = ["player_id", "team_id", "event_type", "Net_xT"]
        missing_cols = [c for c in required_cols if c not in f.columns]
        if missing_cols:
            raise KeyError(
                f"Missing required columns in {id_col} ledger: {missing_cols}. "
                f"Available columns: {sorted(f.columns.tolist())}"
            )
        return f[required_cols].copy()

    player_use = _prep_frame(player_ledger, "credit_xt", "player_id")
    goalie_use = _prep_frame(goalie_ledger, "Net_xT", "goalie_id")

    combined = pd.concat([player_use, goalie_use], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=["player_id", "team_id", "Total_Net_xT"])

    combined["player_id"] = combined["player_id"].astype("string")
    combined["team_id"] = combined["team_id"].astype("string")
    combined["event_type"] = combined["event_type"].astype(str).str.strip().str.lower()
    combined["Net_xT"] = pd.to_numeric(combined["Net_xT"], errors="coerce").fillna(0.0)

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
        index=["player_id", "team_id"], columns="event_type",
        values="Events", aggfunc="sum", fill_value=0,
    )
    net_pivot = grouped.pivot_table(
        index=["player_id", "team_id"], columns="event_type",
        values="Net_xT", aggfunc="sum", fill_value=0.0,
    )
    events_pivot.columns = [f"{c}_Events" for c in events_pivot.columns]
    net_pivot.columns = [f"{c}_Net_xT" for c in net_pivot.columns]

    summary = pd.concat([events_pivot, net_pivot], axis=1).reset_index()
    net_cols = [c for c in summary.columns if c.endswith("_Net_xT")]
    summary["Total_Net_xT"] = summary[net_cols].sum(axis=1) if net_cols else 0.0
    return summary.sort_values(["team_id", "player_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _export_faceoff_baselines_inspection(df: pd.DataFrame, diagnostics_dir: Path) -> tuple[Path, Dict[str, Any]]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / "faceoff_baselines_inspection.csv"

    evt = df["event_type"].astype(str).str.strip().str.lower()
    
    # NEW: Include BOTH whistles and faceoffs to see the sequential baseline hand-off
    inspect_mask = evt.isin(["whistle", "faceoff"])
    rows = df.loc[inspect_mask].copy()

    # Added 'event_type' to keep_cols so you can easily distinguish the rows in the CSV
    keep_cols = [
        "game_id", "sl_event_id", "event_type", "team_id", "period", "period_time",
        "x", "x_adj", "faceoff_zone", "whistle_baseline_for", "whistle_baseline_against",
        "P_actor_goal", "P_opp_goal", "Actor_Net_xT",
    ]
    keep_cols = [c for c in keep_cols if c in rows.columns]
    
    rows[keep_cols].to_csv(output_path, index=False)

    return output_path, {
        "rows_exported": int(inspect_mask.sum()),
        "faceoff_rows_exported": int(evt.eq("faceoff").sum()),
        "whistle_rows_exported": int(evt.eq("whistle").sum()),
        "inspection_csv": str(output_path),
    }


def _audit_sidecar_conservation(
    base_df: pd.DataFrame,
    faceoff_rows: pd.DataFrame,
    penalty_drawer_rows: pd.DataFrame,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "faceoff_groups_checked": 0,
        "faceoff_groups_mismatch": 0,
        "penalty_groups_checked": 0,
        "penalty_groups_mismatch": 0,
        "warning_sidecar_conservation_mismatch": False,
    }

    for label, base_evt, sidecar_df, link_col in [
        ("faceoff", "faceoff", faceoff_rows, "linked_kept_sl_event_id"),
        ("penalty", "penalty", penalty_drawer_rows, "linked_kept_sl_event_id"),
    ]:
        if sidecar_df is None or sidecar_df.empty:
            continue
        evt = base_df["event_type"].astype(str).str.strip().str.lower()
        base_scored = (
            base_df.loc[evt.eq(base_evt), ["game_id", "sl_event_id", "Adjusted_Net_xT"]]
            .copy()
            .pipe(_normalize_event_keys, key_col="sl_event_id")
        )
        sidecar_norm = _normalize_event_keys(sidecar_df.copy(), key_col=link_col)

        merged = base_scored.merge(
            sidecar_norm[["game_id", link_col, "Adjusted_Net_xT"]].rename(
                columns={link_col: "sl_event_id", "Adjusted_Net_xT": "sidecar_net"}
            ),
            on=["game_id", "sl_event_id"],
            how="inner",
        )
        if merged.empty:
            continue
        residual = pd.to_numeric(merged["Adjusted_Net_xT"], errors="coerce") + pd.to_numeric(merged["sidecar_net"], errors="coerce")
        n_checked = int(len(merged))
        n_mismatch = int((residual.abs() > tol).sum())
        out[f"{label}_groups_checked"] = n_checked
        out[f"{label}_groups_mismatch"] = n_mismatch

    out["warning_sidecar_conservation_mismatch"] = (
        out["faceoff_groups_mismatch"] > 0 or out["penalty_groups_mismatch"] > 0
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    variant = str(args.variant).strip().lower()
    if variant != "events_only":
        raise ValueError(f"Unsupported variant: {args.variant}. Supported: events_only")

    base_dir = args.base_dir.resolve()
    paths = TransformerXTPaths(base_dir=base_dir, run_label=args.run_label)
    paths.ensure_all()

    pipeline_run_dir = _resolve_pipeline_run_dir(base_dir, args.pipeline_run_label)
    phase2_dir = pipeline_run_dir / "phase2"
    phase3_dir = pipeline_run_dir / "phase3"

    raw_oof_path = paths.run_results_dir / "raw_oof_predictions.parquet"
    if not raw_oof_path.exists():
        raise FileNotFoundError(
            f"Raw OOF artifact not found: {raw_oof_path}. Run training first."
        )

    tensor_ready_path = phase3_dir / "tensor_ready_dataset.parquet"
    phase2_events_path = phase2_dir / "events_phase2_enriched.parquet"
    faceoff_ref_path = phase2_dir / "faceoff_reference.parquet"
    penalty_ref_path = phase2_dir / "penalty_reference.parquet"

    for required in [tensor_ready_path, phase2_events_path, faceoff_ref_path, penalty_ref_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required reference not found: {required}")

    # Load and optionally sample raw OOF.
    raw_oof = pd.read_parquet(raw_oof_path)
    raw_oof = _normalize_event_keys(raw_oof, key_col="sl_event_id")
    if args.sample_rows and int(args.sample_rows) > 0:
        raw_oof = raw_oof.head(int(args.sample_rows)).copy()

    # Step 1: Enrich + inject synthetic whistles.
    enriched = _enrich_with_phase2_events(raw_oof, tensor_ready_path, phase2_events_path)

    # Step 2: Normalize goal probabilities (for goalie penalty only).
    enriched, goal_normalization_audit = _normalize_goal_rows_to_scoring_actor(enriched)

    # Step 3: Compute faceoff zone baselines and stamp onto whistle rows.
    baseline_ready, faceoff_baseline_audit = _compute_faceoff_zone_baselines(enriched)

    # Step 4: Universal actor-relative delta computation.
    universal_ready, universal_delta_audit = _compute_universal_actor_relative_deltas(baseline_ready)

    # Step 5: Penalty adjustments and sidecar rows.
    faceoff_ref = pd.read_parquet(faceoff_ref_path)
    penalty_ref = pd.read_parquet(penalty_ref_path)

    pass2_taker, penalty_rows_adjusted = _ensure_penalty_taker_adjusted(
        universal_ready,
        for_bonus=float(args.penalty_static_for),
        against_bonus=float(args.penalty_static_against),
    )
    faceoff_rows = _build_faceoff_inverse_rows(pass2_taker, faceoff_ref)
    penalty_drawer_rows = _build_penalty_drawer_inverse_rows(pass2_taker, penalty_ref)

    # Step 6: Goalie ledger.
    goalie_ledger, goalie_audit = _build_goalie_ledger(pass2_taker)

    # Sidecar conservation check.
    sidecar_audit = _audit_sidecar_conservation(
        base_df=pass2_taker,
        faceoff_rows=faceoff_rows,
        penalty_drawer_rows=penalty_drawer_rows,
    )

    # Assemble adjusted output (base + sidecars).
    keep_cols = list(pass2_taker.columns)
    extra_frames = [
        f.reindex(columns=keep_cols)
        for f in [faceoff_rows, penalty_drawer_rows]
        if f is not None and not f.empty
    ]
    adjusted = pd.concat([pass2_taker] + extra_frames, ignore_index=True, sort=False)
    adjusted["game_id"] = adjusted["game_id"].astype(str)
    adjusted["sl_event_id"] = pd.to_numeric(adjusted["sl_event_id"], errors="coerce").astype("Float64").round(6)

    # Step 7: Player ledger (built from fully adjusted frame).
    player_ledger = _build_event_level_player_ledger(adjusted)

    # Consolidated summary.
    consolidated_summary = _build_consolidated_player_goalie_summary(player_ledger, goalie_ledger)

    # Diagnostics export.
    faceoff_baselines_csv_path, faceoff_export_audit = _export_faceoff_baselines_inspection(
        universal_ready, paths.inspections_dir
    )

    # Write outputs.
    player_ledger_path = paths.run_results_dir / "player_ledger.parquet"
    goalie_path = paths.run_results_dir / "goalie_ledger.parquet"
    consolidated_summary_path = paths.run_results_dir / "consolidated_player_goalie_summary.parquet"
    player_ledger.to_parquet(player_ledger_path, index=False)
    goalie_ledger.to_parquet(goalie_path, index=False)
    consolidated_summary.to_parquet(consolidated_summary_path, index=False)

    summary_path = paths.logs_dir / "phase6_postprocess_summary.json"
    summary = {
        "generated_at_utc": _utc_now_iso(),
        "run_label": args.run_label,
        "variant": variant,
        "pipeline_run_dir": str(pipeline_run_dir),
        "raw_oof_path": str(raw_oof_path),
        "rows": {
            "raw_oof": int(len(raw_oof)),
            "enriched": int(len(enriched)),
            "adjusted": int(len(adjusted)),
            "player_ledger": int(len(player_ledger)),
            "goalie_ledger": int(len(goalie_ledger)),
            "consolidated_summary": int(len(consolidated_summary)),
            "faceoff_inverse_rows": int(len(faceoff_rows)) if not faceoff_rows.empty else 0,
            "penalty_drawer_rows": int(len(penalty_drawer_rows)) if not penalty_drawer_rows.empty else 0,
            "penalty_taker_rows_adjusted": int(penalty_rows_adjusted),
        },
        "goal_normalization_audit": goal_normalization_audit,
        "faceoff_baseline_audit": faceoff_baseline_audit,
        "faceoff_baseline_export_audit": faceoff_export_audit,
        "universal_delta_audit": universal_delta_audit,
        "goalie_audit": goalie_audit,
        "sidecar_conservation_audit": sidecar_audit,
        "penalty_policy": {
            "static_for_bonus": float(args.penalty_static_for),
            "static_against_bonus": float(args.penalty_static_against),
        },
    }

    if sidecar_audit.get("warning_sidecar_conservation_mismatch"):
        print(
            f"Warning: sidecar conservation mismatches — "
            f"faceoff={sidecar_audit['faceoff_groups_mismatch']}, "
            f"penalty={sidecar_audit['penalty_groups_mismatch']}"
        )

    if faceoff_baseline_audit.get("warning_low_sample_zone_baseline"):
        print(
            f"Warning: low-sample faceoff zone baselines — "
            f"{faceoff_baseline_audit['low_sample_zone_warnings']}"
        )

    if goalie_audit.get("goal_rows_missing_goalie", 0) > 0:
        print(
            f"Warning: {goalie_audit['goal_rows_missing_goalie']} goal rows "
            f"could not be attributed to a goalie."
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Postprocess complete.")
    print(f"  Player ledger:   {player_ledger_path}")
    print(f"  Goalie ledger:   {goalie_path}")
    print(f"  Summary:         {summary_path}")
    print(f"  Faceoff inspect: {faceoff_baselines_csv_path}")


if __name__ == "__main__":
    main()
