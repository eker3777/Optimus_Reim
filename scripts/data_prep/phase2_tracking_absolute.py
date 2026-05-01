from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

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


def _parse_home_start_sign(series: pd.Series) -> pd.Series:
    mapping = {"neg_x": -1.0, "pos_x": 1.0, "-1": -1.0, "1": 1.0}
    raw = series.astype(str).str.lower().str.strip()
    sign = raw.map(mapping)
    sign = sign.where(sign.isin([-1.0, 1.0]), -1.0)
    return sign.astype(float)


def add_directional_anchors(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    game_cols = ["game_id", "home_team_id", "away_team_id"]
    if "home_start_net" in games.columns:
        game_cols.append("home_start_net")

    game_ref = games[game_cols].drop_duplicates(subset=["game_id"])
    df = df.merge(game_ref, on="game_id", how="left", suffixes=("", "_games"))

    # Coalesce duplicate team columns created by prior upstream merges.
    for col in ["home_team_id", "away_team_id", "home_start_net"]:
        alt_col = f"{col}_games"
        if alt_col not in df.columns:
            continue
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), df[alt_col])
            df = df.drop(columns=[alt_col])
        else:
            df = df.rename(columns={alt_col: col})

    if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
        raise KeyError("directional anchor merge did not produce home/away team identifiers")

    if "home_start_net" in df.columns:
        home_start_sign = _parse_home_start_sign(df["home_start_net"])
    else:
        home_start_sign = pd.Series(-1.0, index=df.index, dtype=float)

    period_num = pd.to_numeric(df.get("period"), errors="coerce").fillna(1.0)
    is_even_period = (period_num % 2 == 0)

    home_attack_base = -home_start_sign
    home_attack = np.where(is_even_period, -home_attack_base, home_attack_base)

    team = df.get("team_id", pd.Series(np.nan, index=df.index)).astype(str)
    home_team = df.get("home_team_id", pd.Series(np.nan, index=df.index)).astype(str)
    away_team = df.get("away_team_id", pd.Series(np.nan, index=df.index)).astype(str)

    attack_direction_x = np.where(
        team.eq(home_team),
        home_attack,
        np.where(team.eq(away_team), -home_attack, 1.0),
    )
    attack_direction_x = np.where(attack_direction_x >= 0, 1.0, -1.0)

    df["home_start_net_sign"] = home_start_sign.astype(float)
    df["attack_direction_x"] = attack_direction_x.astype(float)
    df["target_net_x"] = (89.0 * df["attack_direction_x"]).astype(float)
    return df


def _order_newcomers_by_puck_proximity(
    newcomers: list[str],
    tracking_xy_lookup: Mapping[str, tuple[float, float]],
    puck_x: float,
    puck_y: float,
) -> list[str]:
    if not newcomers:
        return []
    if not tracking_xy_lookup:
        return sorted(newcomers)

    def sort_key(player_id: str) -> tuple[float, str]:
        xy = tracking_xy_lookup.get(player_id)
        if xy is None:
            return (float("inf"), player_id)
        px, py = xy
        if pd.isna(px) or pd.isna(py):
            return (float("inf"), player_id)
        dx = px - puck_x
        dy = py - puck_y
        return (dx * dx + dy * dy, player_id)

    return sorted(newcomers, key=sort_key)


def compute_slot_assignments(
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    stints: pd.DataFrame,
    goalie_ids: set[str],
    n_slots: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not all(c in stints.columns for c in ["game_id", "game_stint", "team_id", "player_id"]):
        return pd.DataFrame(columns=["game_id", "game_stint", "team_id", "player_id", "slot_index"]), pd.DataFrame()

    roster = stints[["game_id", "game_stint", "team_id", "player_id"]].dropna(subset=["team_id", "player_id"]).copy()
    roster["player_id"] = roster["player_id"].astype(str)
    if goalie_ids:
        roster = roster.loc[~roster["player_id"].isin(goalie_ids)].copy()
    roster = roster.drop_duplicates()

    event_sort_col = "game_event_id" if "game_event_id" in events.columns else "sl_event_id"
    first_events = (
        events.sort_values(["game_id", "game_stint", event_sort_col], kind="mergesort")
        .groupby(["game_id", "game_stint"], as_index=False)
        .first()[["game_id", "game_stint", "sl_event_id", "x", "y"]]
        .rename(columns={"sl_event_id": "first_sl_event_id", "x": "puck_x", "y": "puck_y"})
    )

    tracking_first = tracking.merge(
        first_events[["game_id", "game_stint", "first_sl_event_id"]],
        on=["game_id", "game_stint"],
        how="inner",
    )
    tracking_first = tracking_first.loc[
        pd.to_numeric(tracking_first["sl_event_id"], errors="coerce")
        == pd.to_numeric(tracking_first["first_sl_event_id"], errors="coerce")
    ].copy()
    tracking_first["player_id"] = tracking_first["player_id"].astype(str)
    tracking_first["tracking_x_num"] = pd.to_numeric(tracking_first["tracking_x"], errors="coerce")
    tracking_first["tracking_y_num"] = pd.to_numeric(tracking_first["tracking_y"], errors="coerce")

    first_events["puck_x"] = pd.to_numeric(first_events["puck_x"], errors="coerce")
    first_events["puck_y"] = pd.to_numeric(first_events["puck_y"], errors="coerce")

    first_events_lookup = {
        (row.game_id, row.game_stint): (
            float(row.puck_x) if pd.notna(row.puck_x) else 0.0,
            float(row.puck_y) if pd.notna(row.puck_y) else 0.0,
        )
        for row in first_events.itertuples(index=False)
    }
    tracking_xy_by_key: dict[tuple[Any, Any, Any], dict[str, tuple[float, float]]] = {}
    for key, grp in tracking_first.groupby(["game_id", "game_stint", "team_id"], sort=False):
        dedup = grp[["player_id", "tracking_x_num", "tracking_y_num"]].drop_duplicates(
            subset=["player_id"], keep="first"
        )
        tracking_xy_by_key[key] = {
            row.player_id: (
                float(row.tracking_x_num) if pd.notna(row.tracking_x_num) else np.nan,
                float(row.tracking_y_num) if pd.notna(row.tracking_y_num) else np.nan,
            )
            for row in dedup.itertuples(index=False)
        }

    slot_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []

    for (game_id, team_id), team_roster in roster.groupby(["game_id", "team_id"], sort=False):
        unique_stints = pd.Series(team_roster["game_stint"].dropna().unique())
        if unique_stints.empty:
            continue
        stint_order = pd.DataFrame({"game_stint": unique_stints})
        stint_order["_stint_num"] = pd.to_numeric(stint_order["game_stint"], errors="coerce")
        stint_order["_stint_str"] = stint_order["game_stint"].astype(str)
        stints_list = (
            stint_order.sort_values(["_stint_num", "_stint_str"], kind="mergesort")["game_stint"].tolist()
        )
        active_by_stint = {
            stint: set(players.astype(str).tolist())
            for stint, players in team_roster.groupby("game_stint", sort=False)["player_id"]
        }

        slots: list[str | None] = [None] * n_slots
        prev_active: set[str] = set()

        for game_stint in stints_list:
            active = active_by_stint.get(game_stint, set())

            for i, pid in enumerate(slots):
                if pid is not None and pid not in active:
                    slots[i] = None

            incumbents = {pid for pid in slots if pid is not None}
            newcomers = [pid for pid in sorted(active) if pid not in incumbents]
            available = [i for i, pid in enumerate(slots) if pid is None]

            puck_x, puck_y = first_events_lookup.get((game_id, game_stint), (0.0, 0.0))

            tracking_xy = tracking_xy_by_key.get((game_id, game_stint, team_id), {})
            ordered_new = _order_newcomers_by_puck_proximity(newcomers, tracking_xy, puck_x, puck_y)

            for pid, idx in zip(ordered_new, available):
                slots[idx] = pid

            entered = sorted(active - prev_active)
            exited = sorted(prev_active - active)
            persisted = sorted(active & prev_active)

            stint_slot_rows = [
                {
                    "game_id": game_id,
                    "game_stint": game_stint,
                    "team_id": team_id,
                    "player_id": pid,
                    "slot_index": int(slot_index),
                    "is_consistent": int(pid in persisted),
                    "is_new_to_stint": int(pid in entered),
                }
                for slot_index, pid in enumerate(slots)
                if pid is not None
            ]
            slot_rows.extend(stint_slot_rows)

            change_rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "game_stint": game_stint,
                    "num_entered": int(len(entered)),
                    "num_exited": int(len(exited)),
                    "num_persisted": int(len(persisted)),
                    "entered_players": ",".join(entered),
                    "exited_players": ",".join(exited),
                    "persisted_players": ",".join(persisted),
                }
            )
            prev_active = active

    slot_map = pd.DataFrame(slot_rows)
    stint_changes = pd.DataFrame(change_rows)
    return slot_map, stint_changes


def materialize_absolute_tensor(
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    slot_map: pd.DataFrame,
    n_slots: int,
) -> pd.DataFrame:
    base_cols = [
        "game_id",
        "sl_event_id",
        "game_stint",
        "team_id",
        "player_id",
        "home_team_id",
        "away_team_id",
        "target_net_x",
        "attack_direction_x",
    ]
    base_cols = [c for c in base_cols if c in events.columns]

    event_base = events[base_cols].drop_duplicates(subset=["game_id", "sl_event_id"]).copy()
    event_base = event_base.rename(columns={"team_id": "event_team_id", "player_id": "event_player_id"})

    # Defensive fallback for legacy/misaligned upstream merges.
    missing_team_cols = [c for c in ["home_team_id", "away_team_id"] if c not in event_base.columns]
    if missing_team_cols:
        fallback_cols = ["game_id", "sl_event_id", *[c for c in missing_team_cols if c in events.columns]]
        if len(fallback_cols) > 2:
            fallback = events[fallback_cols].drop_duplicates(subset=["game_id", "sl_event_id"])
            event_base = event_base.merge(fallback, on=["game_id", "sl_event_id"], how="left")

    missing_after = [c for c in ["home_team_id", "away_team_id"] if c not in event_base.columns]
    if missing_after:
        raise KeyError(f"absolute tensor materialization missing team columns: {missing_after}")

    sort_col = "game_event_id" if "game_event_id" in events.columns else "sl_event_id"
    event_base = event_base.merge(
        events[["game_id", "sl_event_id", sort_col]].drop_duplicates(subset=["game_id", "sl_event_id"]),
        on=["game_id", "sl_event_id"],
        how="left",
    )
    event_base = event_base.sort_values(["game_id", sort_col], kind="mergesort").reset_index(drop=True)
    event_base["prev_stint"] = event_base.groupby("game_id", sort=False)["game_stint"].shift(1)
    event_base["is_stint_change"] = (
        event_base["game_stint"].notna() & (event_base["game_stint"] != event_base["prev_stint"])
    ).astype(int)

    side_base_cols = [
        "game_id",
        "sl_event_id",
        "game_stint",
        "event_team_id",
        "event_player_id",
        "home_team_id",
        "away_team_id",
    ]
    side_base_cols = [c for c in side_base_cols if c in event_base.columns]

    home_rows = event_base[side_base_cols].copy()
    home_rows["side"] = "Home"
    home_rows["team_id"] = home_rows["home_team_id"]

    away_rows = event_base[side_base_cols].copy()
    away_rows["side"] = "Away"
    away_rows["team_id"] = away_rows["away_team_id"]

    event_team = pd.concat([home_rows, away_rows], ignore_index=True)

    if not slot_map.empty:
        mapped = event_team.merge(
            slot_map,
            on=["game_id", "game_stint", "team_id"],
            how="left",
        )
        mapped = mapped.loc[mapped["slot_index"].notna()].copy()
        if not mapped.empty:
            mapped["slot_index"] = pd.to_numeric(mapped["slot_index"], errors="coerce").astype(int)
    else:
        mapped = event_team.head(0).copy()
        mapped["slot_index"] = pd.Series(dtype=int)
        mapped["player_id"] = pd.Series(dtype=object)
        mapped["is_consistent"] = pd.Series(dtype=int)
        mapped["is_new_to_stint"] = pd.Series(dtype=int)

    tracking_keep = [
        "game_id",
        "sl_event_id",
        "team_id",
        "player_id",
        "tracking_x",
        "tracking_y",
        "tracking_vel_x",
        "tracking_vel_y",
    ]
    for c in ["tracking_vel_x", "tracking_vel_y"]:
        if c not in tracking.columns:
            tracking[c] = 0.0

    tracking_use = tracking[tracking_keep].drop_duplicates(
        subset=["game_id", "sl_event_id", "team_id", "player_id"], keep="first"
    )

    if not mapped.empty:
        mapped = mapped.merge(
            tracking_use,
            on=["game_id", "sl_event_id", "team_id", "player_id"],
            how="left",
        )

        mapped["slot_vacant"] = mapped["player_id"].isna().astype(int)
        mapped["is_present"] = (
            mapped["player_id"].notna() & mapped["tracking_x"].notna() & mapped["tracking_y"].notna()
        ).astype(int)
        mapped["is_primary_actor"] = (mapped["player_id"].astype(str) == mapped["event_player_id"].astype(str)).astype(int)
        mapped["is_possessing_team"] = (mapped["team_id"].astype(str) == mapped["event_team_id"].astype(str)).astype(int)
        mapped["is_tracking_imputed"] = (mapped["player_id"].notna() & mapped["is_present"].eq(0)).astype(int)

        mapped["is_consistent"] = pd.to_numeric(mapped.get("is_consistent", 0), errors="coerce").fillna(0).astype(int)
        mapped["is_new_to_stint"] = pd.to_numeric(mapped.get("is_new_to_stint", 0), errors="coerce").fillna(0).astype(int)

        mapped["X"] = pd.to_numeric(mapped["tracking_x"], errors="coerce").fillna(0.0)
        mapped["Y"] = pd.to_numeric(mapped["tracking_y"], errors="coerce").fillna(0.0)
        mapped["Vel_X"] = pd.to_numeric(mapped["tracking_vel_x"], errors="coerce").fillna(0.0)
        mapped["Vel_Y"] = pd.to_numeric(mapped["tracking_vel_y"], errors="coerce").fillna(0.0)

    metrics = [
        "X",
        "Y",
        "Vel_X",
        "Vel_Y",
        "is_present",
        "is_primary_actor",
        "is_possessing_team",
        "is_consistent",
        "is_new_to_stint",
        "is_tracking_imputed",
        "slot_vacant",
    ]

    index_cols = ["game_id", "sl_event_id"]
    if not mapped.empty:
        wide_long = mapped.set_index([*index_cols, "side", "slot_index"])[metrics]
        wide = wide_long.unstack(["side", "slot_index"])
        full_cols = pd.MultiIndex.from_product(
            [metrics, ["Home", "Away"], list(range(n_slots))],
            names=wide.columns.names,
        )
        wide = wide.reindex(columns=full_cols)
        wide.columns = [f"{side}_Track_{int(slot)}_{metric}" for metric, side, slot in wide.columns]
        wide = wide.reset_index()
    else:
        wide = event_base[index_cols].copy()

    expected_cols: list[str] = []
    for side in ["Home", "Away"]:
        for slot in range(n_slots):
            for metric in metrics:
                expected_cols.append(f"{side}_Track_{slot}_{metric}")

    for col in expected_cols:
        if col not in wide.columns:
            if col.endswith("_slot_vacant"):
                wide[col] = 1
            else:
                wide[col] = 0.0
        elif col.endswith("_slot_vacant"):
            wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(1).astype(int)
        else:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)
    wide = wide[["game_id", "sl_event_id", *expected_cols]]

    anchor_cols = ["game_id", "sl_event_id", "game_stint", "target_net_x", "attack_direction_x", "is_stint_change"]
    anchor = event_base[[c for c in anchor_cols if c in event_base.columns]].drop_duplicates(subset=["game_id", "sl_event_id"])
    out = anchor.merge(wide, on=["game_id", "sl_event_id"], how="left")

    # Keep possession context explicit for all six slots, including the extra-skater vacancy slot.
    possession_ref = event_base[
        ["game_id", "sl_event_id", "event_team_id", "home_team_id", "away_team_id"]
    ].drop_duplicates(subset=["game_id", "sl_event_id"])
    out = out.merge(possession_ref, on=["game_id", "sl_event_id"], how="left")

    home_possessing = (out["event_team_id"].astype(str) == out["home_team_id"].astype(str)).astype(int)
    away_possessing = (out["event_team_id"].astype(str) == out["away_team_id"].astype(str)).astype(int)
    for slot in range(n_slots):
        out[f"Home_Track_{slot}_is_possessing_team"] = home_possessing
        out[f"Away_Track_{slot}_is_possessing_team"] = away_possessing

    out = out.drop(columns=["event_team_id", "home_team_id", "away_team_id"])

    int_cols = [
        c
        for c in out.columns
        if c.endswith("_is_present")
        or c.endswith("_is_primary_actor")
        or c.endswith("_is_possessing_team")
        or c.endswith("_is_consistent")
    ]
    int_cols += [
        c
        for c in out.columns
        if c.endswith("_is_new_to_stint") or c.endswith("_is_tracking_imputed") or c.endswith("_slot_vacant")
    ]
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    float_cols = [c for c in out.columns if c.endswith("_X") or c.endswith("_Y") or c.endswith("_Vel_X") or c.endswith("_Vel_Y")]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    assert_key_uniqueness(out, ["game_id", "sl_event_id"], "phase2_tracking_absolute")
    return out


def run_phase2_tracking_absolute(
    base_dir: Path,
    *,
    run_label: str,
    config: DataPrepConfig | None = None,
    events_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or DataPrepConfig()
    paths = DataPrepPaths(base_dir, run_label=run_label)

    events = events_df.copy() if events_df is not None else read_parquet(paths.phase2_events_output)
    games = read_parquet(paths.raw_games_path)
    tracking = read_parquet(paths.raw_tracking_path)
    stints = read_parquet(paths.raw_stints_path)
    players = read_parquet(paths.raw_players_path) if paths.raw_players_path.exists() else None

    require_columns(events, ["game_id", "sl_event_id", "game_stint", "team_id", "player_id", "x", "y"], "phase2_events")
    require_columns(games, ["game_id", "home_team_id", "away_team_id"], "games")
    require_columns(tracking, ["game_id", "sl_event_id", "team_id", "player_id", "tracking_x", "tracking_y"], "tracking")

    events = add_directional_anchors(events, games)
    require_columns(events, ["home_team_id", "away_team_id", "target_net_x", "attack_direction_x"], "events_with_anchors")

    tracking = tracking.loc[tracking["player_id"].notna() & tracking["player_id"].astype(str).ne("None")].copy()
    tracking = tracking.drop_duplicates(subset=["game_id", "sl_event_id", "team_id", "player_id"], keep="first")

    event_stint = events[["game_id", "sl_event_id", "game_stint"]].drop_duplicates(subset=["game_id", "sl_event_id"])
    tracking = tracking.merge(event_stint, on=["game_id", "sl_event_id"], how="inner")

    goalie_ids = _infer_goalie_ids(players)
    if goalie_ids:
        # Goalies are excluded from skater slots; slot index 5 remains extra-attacker capacity.
        tracking = tracking.loc[~tracking["player_id"].astype(str).isin(goalie_ids)].copy()

    if cfg.enable_tracking_ghost_filter and all(c in stints.columns for c in ["game_id", "game_stint", "team_id", "player_id"]):
        roster = stints[["game_id", "game_stint", "team_id", "player_id"]].drop_duplicates().copy()
        roster["player_id"] = roster["player_id"].astype(str)
        if goalie_ids:
            roster = roster.loc[~roster["player_id"].isin(goalie_ids)].copy()
        tracking = tracking.merge(roster, on=["game_id", "game_stint", "team_id", "player_id"], how="inner")

    slot_map, stint_changes = compute_slot_assignments(
        events,
        tracking,
        stints,
        goalie_ids=goalie_ids,
        n_slots=cfg.absolute_tracking_slots_per_side,
    )
    absolute = materialize_absolute_tensor(
        events,
        tracking,
        slot_map,
        n_slots=cfg.absolute_tracking_slots_per_side,
    )

    write_parquet(paths.phase2_tracking_absolute_output, absolute)
    write_parquet(paths.phase2_tracking_slot_mapping_output, slot_map)
    write_parquet(paths.phase2_tracking_stint_changes_output, stint_changes)

    summary = {
        "rows": int(len(absolute)),
        "slot_map_rows": int(len(slot_map)),
        "stint_changes_rows": int(len(stint_changes)),
        "output": str(paths.phase2_tracking_absolute_output),
        "slot_map_output": str(paths.phase2_tracking_slot_mapping_output),
        "stint_changes_output": str(paths.phase2_tracking_stint_changes_output),
    }
    return absolute, slot_map, stint_changes, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 absolute tracking tensor (canonical)")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, _, summary = run_phase2_tracking_absolute(args.base_dir, run_label=args.run_label)
    print(summary)


if __name__ == "__main__":
    main()
