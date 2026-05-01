from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import read_parquet, write_parquet
    from .phase2_event_pipeline import _infer_goalie_ids
    from .validation import assert_key_uniqueness, require_columns
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import read_parquet, write_parquet
    from phase2_event_pipeline import _infer_goalie_ids
    from validation import assert_key_uniqueness, require_columns


def _detect_tracking_coord_system(df: pd.DataFrame) -> str:
    player_str = df["player_id_str"] if "player_id_str" in df.columns else df["player_id"].astype(str)
    event_player_str = (
        df["event_player_id_str"] if "event_player_id_str" in df.columns else df["event_player_id"].astype(str)
    )
    actor_rows = df.loc[player_str == event_player_str].copy()
    if actor_rows.empty:
        return "raw"

    tracking_x = pd.to_numeric(actor_rows["tracking_x"], errors="coerce")
    tracking_y = pd.to_numeric(actor_rows["tracking_y"], errors="coerce")
    x_raw = pd.to_numeric(actor_rows["x"], errors="coerce")
    y_raw = pd.to_numeric(actor_rows["y"], errors="coerce")
    x_adj = pd.to_numeric(actor_rows["x_adj"], errors="coerce")
    y_adj = pd.to_numeric(actor_rows["y_adj"], errors="coerce")

    raw_diff = (
        (tracking_x - x_raw).abs().mean()
        + (tracking_y - y_raw).abs().mean()
    )
    adj_diff = (
        (tracking_x - x_adj).abs().mean()
        + (tracking_y - y_adj).abs().mean()
    )
    return "raw" if raw_diff <= adj_diff else "adjusted"


def _build_role_wide(
    tracking: pd.DataFrame,
    *,
    role_mask: pd.Series,
    role_prefix: str,
    n_slots: int,
    keys: list[str],
) -> pd.DataFrame:
    role_df = tracking.loc[role_mask, [*keys, "distance", "player_id", "rel_x", "rel_y", "vx_std", "vy_std"]].copy()
    if role_df.empty:
        return pd.DataFrame(columns=keys)

    role_df = role_df.sort_values([*keys, "distance", "player_id"], kind="mergesort")
    role_df["slot_index"] = role_df.groupby(keys, sort=False).cumcount() + 1
    role_df = role_df.loc[role_df["slot_index"] <= n_slots]
    if role_df.empty:
        return pd.DataFrame(columns=keys)

    wide = role_df.set_index([*keys, "slot_index"])[["rel_x", "rel_y", "vx_std", "vy_std", "distance", "player_id"]].unstack(
        "slot_index"
    )
    metric_map = {"vx_std": "vx_rel", "vy_std": "vy_rel"}
    wide.columns = [f"{role_prefix}_{int(slot)}_{metric_map.get(metric, metric)}" for metric, slot in wide.columns]
    return wide.reset_index()


def run_phase2_tracking_event_relative(
    base_dir: Path,
    *,
    run_label: str,
    config: DataPrepConfig | None = None,
    events_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or DataPrepConfig()
    paths = DataPrepPaths(base_dir, run_label=run_label)

    events = events_df.copy() if events_df is not None else read_parquet(paths.phase2_events_output)
    require_columns(
        events,
        ["game_id", "sl_event_id", "game_stint", "team_id", "player_id", "x", "y", "x_adj", "y_adj"],
        "phase2_events",
    )

    tracking = read_parquet(paths.raw_tracking_path)
    require_columns(tracking, ["game_id", "sl_event_id", "team_id", "player_id", "tracking_x", "tracking_y"], "tracking")

    events = events.copy()
    tracking = tracking.copy()

    event_keys = events[["game_id", "sl_event_id"]].drop_duplicates()
    tracking = tracking.merge(event_keys, on=["game_id", "sl_event_id"], how="inner")

    tracking["player_id_str"] = tracking["player_id"].astype(str)
    tracking = tracking.loc[tracking["player_id"].notna() & tracking["player_id_str"].ne("None")].copy()
    tracking = tracking.drop_duplicates(subset=["game_id", "sl_event_id", "team_id", "player_id"], keep="first")

    players = read_parquet(paths.raw_players_path) if paths.raw_players_path.exists() else None
    goalie_ids = _infer_goalie_ids(players)
    if goalie_ids:
        tracking = tracking.loc[~tracking["player_id_str"].isin(goalie_ids)].copy()

    context_cols = [
        "game_id",
        "sl_event_id",
        "game_stint",
        "team_id",
        "player_id",
        "x",
        "y",
        "x_adj",
        "y_adj",
        "home_team_id",
        "away_team_id",
        "is_boundary_event",
    ]
    context_cols = [c for c in context_cols if c in events.columns]
    context = events[context_cols].drop_duplicates(subset=["game_id", "sl_event_id"]).copy()
    context = context.rename(columns={"team_id": "acting_team_id", "player_id": "event_player_id"})

    optional_keys = [c for c in ["game_stint", "sequence_id", "game_event_id", "sequence_event_id"] if c in events.columns]
    extra_optional_keys = [c for c in optional_keys if c not in context.columns]
    event_optional = events[["game_id", "sl_event_id", *extra_optional_keys]].drop_duplicates(
        subset=["game_id", "sl_event_id"]
    )
    event_optional_full = events[["game_id", "sl_event_id", *optional_keys]].drop_duplicates(
        subset=["game_id", "sl_event_id"]
    )

    tracking = tracking.merge(context, on=["game_id", "sl_event_id"], how="left")
    if extra_optional_keys:
        tracking = tracking.merge(event_optional, on=["game_id", "sl_event_id"], how="left")

    if cfg.enable_tracking_ghost_filter and paths.raw_stints_path.exists():
        stints = read_parquet(paths.raw_stints_path)
        if all(c in stints.columns for c in ["game_id", "game_stint", "team_id", "player_id"]):
            roster = stints[["game_id", "game_stint", "team_id", "player_id"]].drop_duplicates()
            tracking = tracking.merge(
                roster,
                on=["game_id", "game_stint", "team_id", "player_id"],
                how="inner",
            )

    x_num = pd.to_numeric(tracking["x"], errors="coerce")
    y_num = pd.to_numeric(tracking["y"], errors="coerce")
    x_adj_num = pd.to_numeric(tracking["x_adj"], errors="coerce")
    y_adj_num = pd.to_numeric(tracking["y_adj"], errors="coerce")
    tracking_x_num = pd.to_numeric(tracking["tracking_x"], errors="coerce")
    tracking_y_num = pd.to_numeric(tracking["tracking_y"], errors="coerce")
    vx_num = pd.to_numeric(tracking.get("tracking_vel_x", 0.0), errors="coerce").fillna(0.0)
    vy_num = pd.to_numeric(tracking.get("tracking_vel_y", 0.0), errors="coerce").fillna(0.0)

    tracking["flip_required"] = (x_num != x_adj_num) | (y_num != y_adj_num)

    tracking["event_player_id_str"] = tracking["event_player_id"].astype(str)
    tracking["team_id_str"] = tracking["team_id"].astype(str)
    tracking["acting_team_id_str"] = tracking["acting_team_id"].astype(str)

    coord_system = _detect_tracking_coord_system(tracking)
    if coord_system == "raw":
        tracking["x_std"] = np.where(tracking["flip_required"], -tracking_x_num, tracking_x_num)
        tracking["y_std"] = np.where(tracking["flip_required"], -tracking_y_num, tracking_y_num)
        tracking["vx_std"] = np.where(tracking["flip_required"], -vx_num, vx_num)
        tracking["vy_std"] = np.where(tracking["flip_required"], -vy_num, vy_num)
    else:
        tracking["x_std"] = tracking_x_num
        tracking["y_std"] = tracking_y_num
        tracking["vx_std"] = vx_num
        tracking["vy_std"] = vy_num

    tracking["is_actor"] = tracking["player_id_str"] == tracking["event_player_id_str"]
    tracking["is_teammate"] = (tracking["team_id_str"] == tracking["acting_team_id_str"]) & (~tracking["is_actor"])
    tracking["is_opponent"] = (
        tracking["team_id"].notna()
        & tracking["acting_team_id"].notna()
        & (tracking["team_id_str"] != tracking["acting_team_id_str"])
    )

    tracking["rel_x"] = pd.to_numeric(tracking["x_std"], errors="coerce") - x_adj_num
    tracking["rel_y"] = pd.to_numeric(tracking["y_std"], errors="coerce") - y_adj_num
    tracking["distance"] = np.sqrt(tracking["rel_x"] ** 2 + tracking["rel_y"] ** 2)

    keys = ["game_id", "sl_event_id"]

    out = event_optional_full[["game_id", "sl_event_id", *optional_keys]].drop_duplicates(subset=keys).copy()
    out["tracking_coord_system"] = coord_system

    actor = (
        tracking.loc[tracking["is_actor"], [*keys, "player_id", "rel_x", "rel_y", "vx_std", "vy_std", "distance"]]
        .sort_values([*keys, "distance", "player_id"], kind="mergesort")
        .groupby(keys, as_index=False, sort=False)
        .first()
        .rename(
            columns={
                "rel_x": "actor_rel_x",
                "rel_y": "actor_rel_y",
                "vx_std": "actor_vx_rel",
                "vy_std": "actor_vy_rel",
                "distance": "actor_distance",
                "player_id": "actor_player_id",
            }
        )
    )
    if not actor.empty:
        actor["actor_is_present"] = 1
        actor["actor_is_missing"] = 0
        actor["actor_is_imputed"] = 0
        out = out.merge(actor, on=keys, how="left")

    tm_wide = _build_role_wide(
        tracking,
        role_mask=tracking["is_teammate"],
        role_prefix="tm",
        n_slots=cfg.slot_schema.teammate_slots,
        keys=keys,
    )
    if not tm_wide.empty:
        out = out.merge(tm_wide, on=keys, how="left")

    opp_wide = _build_role_wide(
        tracking,
        role_mask=tracking["is_opponent"],
        role_prefix="opp",
        n_slots=cfg.slot_schema.opponent_slots,
        keys=keys,
    )
    if not opp_wide.empty:
        out = out.merge(opp_wide, on=keys, how="left")

    out["actor_is_present"] = pd.to_numeric(out.get("actor_is_present", 0), errors="coerce").fillna(0).astype(int)
    out["actor_is_missing"] = np.where(out["actor_is_present"].eq(1), 0, 1).astype(int)
    out["actor_is_imputed"] = np.where(out["actor_is_present"].eq(1), 0, 1).astype(int)
    out["actor_rel_x"] = pd.to_numeric(out.get("actor_rel_x"), errors="coerce").fillna(cfg.phase2_zero_fill_value)
    out["actor_rel_y"] = pd.to_numeric(out.get("actor_rel_y"), errors="coerce").fillna(cfg.phase2_zero_fill_value)
    out["actor_vx_rel"] = pd.to_numeric(out.get("actor_vx_rel"), errors="coerce").fillna(cfg.phase2_zero_fill_value)
    out["actor_vy_rel"] = pd.to_numeric(out.get("actor_vy_rel"), errors="coerce").fillna(cfg.phase2_zero_fill_value)
    out["actor_distance"] = pd.to_numeric(out.get("actor_distance"), errors="coerce").fillna(cfg.phase2_zero_fill_value)

    for role_prefix, n_slots in (("tm", cfg.slot_schema.teammate_slots), ("opp", cfg.slot_schema.opponent_slots)):
        for i in range(1, n_slots + 1):
            player_col = f"{role_prefix}_{i}_player_id"
            rel_x_col = f"{role_prefix}_{i}_rel_x"
            rel_y_col = f"{role_prefix}_{i}_rel_y"
            vx_col = f"{role_prefix}_{i}_vx_rel"
            vy_col = f"{role_prefix}_{i}_vy_rel"
            dist_col = f"{role_prefix}_{i}_distance"

            if player_col not in out.columns:
                out[player_col] = np.nan
            if rel_x_col not in out.columns:
                out[rel_x_col] = cfg.phase2_zero_fill_value
            if rel_y_col not in out.columns:
                out[rel_y_col] = cfg.phase2_zero_fill_value
            if vx_col not in out.columns:
                out[vx_col] = cfg.phase2_zero_fill_value
            if vy_col not in out.columns:
                out[vy_col] = cfg.phase2_zero_fill_value
            if dist_col not in out.columns:
                out[dist_col] = cfg.phase2_zero_fill_value

            present_col = f"{role_prefix}_{i}_is_present"
            missing_col = f"{role_prefix}_{i}_is_missing"
            imputed_col = f"{role_prefix}_{i}_is_imputed"

            is_present = out[player_col].notna().astype(int)
            out[present_col] = is_present
            out[missing_col] = (1 - is_present).astype(int)
            out[imputed_col] = (1 - is_present).astype(int)

            out[rel_x_col] = pd.to_numeric(out[rel_x_col], errors="coerce").fillna(cfg.phase2_zero_fill_value)
            out[rel_y_col] = pd.to_numeric(out[rel_y_col], errors="coerce").fillna(cfg.phase2_zero_fill_value)
            out[vx_col] = pd.to_numeric(out[vx_col], errors="coerce").fillna(cfg.phase2_zero_fill_value)
            out[vy_col] = pd.to_numeric(out[vy_col], errors="coerce").fillna(cfg.phase2_zero_fill_value)
            out[dist_col] = pd.to_numeric(out[dist_col], errors="coerce").fillna(cfg.phase2_zero_fill_value)

    assert_key_uniqueness(out, ["game_id", "sl_event_id"], "phase2_tracking_event_relative")

    # Ensure all events exist in output (zero-padded rows when no tracking rows survived filtering).
    # `out` is built from full event keys, so no backfill append is needed.

    write_parquet(paths.phase2_tracking_event_relative_output, out)

    summary = {
        "rows": int(len(out)),
        "events_total": int(events[["game_id", "sl_event_id"]].drop_duplicates().shape[0]),
        "coord_system": coord_system,
        "output": str(paths.phase2_tracking_event_relative_output),
        "actor_imputed_rows": int(out.get("actor_is_imputed", pd.Series(dtype=int)).sum()),
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 event-relative tracking features")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = run_phase2_tracking_event_relative(args.base_dir, run_label=args.run_label)
    print(summary)


if __name__ == "__main__":
    main()
