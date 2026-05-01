from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import append_manifest_record, read_parquet, utc_now_iso, write_csv, write_json, write_parquet
    from .validation import (
        assert_no_duplicate_columns,
        collect_phase1_secondary_warnings,
        phase1_boundary_violation_counts,
        require_columns,
        sequence_disorder_count,
    )
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import append_manifest_record, read_parquet, utc_now_iso, write_csv, write_json, write_parquet
    from validation import (
        assert_no_duplicate_columns,
        collect_phase1_secondary_warnings,
        phase1_boundary_violation_counts,
        require_columns,
        sequence_disorder_count,
    )


TERMINATOR_EVENTS = {"whistle", "goal", "end_of_period"}
PENALTY_EVENTS = {"penalty", "penaltydrawn"}
GOAL_SOURCE_EVENTS = {"shot", "deflection", "defensive_deflection"}
SHOOTOUT_CONTEXT_EVENTS = {
    "shootout",
    "teamwithozonright",
    "solpr",
    "socarry",
    "soshot",
    "sogoal",
    "sopuckprotection",
    "socheck",
}
PENALTY_SHOT_EVENTS = {"penaltyshot", "penalty_shot", "pslpr", "pscarry", "psshot"}


def apply_regular_time_boundary(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    period = pd.to_numeric(work["period"], errors="coerce")
    period_time = pd.to_numeric(work["period_time"], errors="coerce")
    # Legacy parity: period 4 at and after 300s is removed.
    keep_mask = (period <= 4) & ((period < 4) | (period_time < 300))
    return work.loc[keep_mask.fillna(False)].copy()


def load_raw_inputs(base_dir: Path, events_path: Path | None = None) -> pd.DataFrame:
    paths = DataPrepPaths(base_dir)
    source = events_path or (paths.raw_data_dir / "events.parquet")
    events = read_parquet(source)
    require_columns(
        events,
        [
            "game_id",
            "period",
            "sequence_id",
            "sl_event_id",
            "event_type",
            "period_time",
            "player_id",
            "team_id",
            "detail",
            "x_adj",
            "y_adj",
        ],
        "events",
    )
    return events


def extract_and_remove_shootout_penalty_shot_noise(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = events.copy()
    detail = work["detail"].fillna("").astype(str).str.lower()
    event_type = work["event_type"].fillna("").astype(str).str.lower()

    is_shootout = (
        detail.str.contains("shootout", regex=False)
        | work["period"].astype(str).eq("5")
        | event_type.isin(SHOOTOUT_CONTEXT_EVENTS)
    )
    is_penalty_shot = (
        detail.str.contains("penalty shot", regex=False)
        | event_type.isin(PENALTY_SHOT_EVENTS)
    )
    noise_mask = is_shootout | is_penalty_shot

    # Keep all noise goals for score reconciliation, even when player/team ids are missing.
    extra_goals = work.loc[
        noise_mask & event_type.eq("goal"),
        ["game_id", "team_id", "player_id", "period", "sl_event_id", "event_type", "detail"],
    ].copy()

    cleaned = work.loc[~noise_mask].copy()
    return cleaned, extra_goals


def remove_post_shootout_tail_events(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy().reset_index(drop=True)
    event_type = work["event_type"].fillna("").astype(str).str.lower()

    trigger_rows = work.loc[event_type.eq("shootout"), ["game_id"]].copy()
    if trigger_rows.empty:
        return work

    work["_row_idx"] = work.index
    first_trigger_idx = (
        work.loc[event_type.eq("shootout"), ["game_id", "_row_idx"]]
        .groupby("game_id", sort=False)["_row_idx"]
        .min()
    )

    trigger_map = work["game_id"].map(first_trigger_idx)
    drop_tail = trigger_map.notna() & work["_row_idx"].gt(trigger_map)
    out = work.loc[~drop_tail].copy()
    return out.drop(columns=["_row_idx"], errors="ignore")


def remove_penaltyshot_windows(events: pd.DataFrame) -> pd.DataFrame:
    work = events.sort_values(["game_id", "period", "sequence_id", "sl_event_id"], kind="mergesort").reset_index(drop=True)
    event_type = work["event_type"].fillna("").astype(str).str.lower()

    starts = work.index[event_type.isin({"penaltyshot", "penalty_shot"})].tolist()
    if not starts:
        return work

    to_drop: set[int] = set()
    n = len(work)
    for idx in starts:
        start_seq = work.at[idx, "sequence_id"]
        game_id = work.at[idx, "game_id"]

        j = idx
        while j < n:
            row = work.iloc[j]
            if row["game_id"] != game_id:
                break
            if str(row.get("event_type", "")).lower() == "faceoff" and row["sequence_id"] != start_seq:
                break
            to_drop.add(j)
            j += 1

    if not to_drop:
        return work

    keep_mask = ~work.index.isin(to_drop)
    return work.loc[keep_mask].copy()


def remove_explicit_shootout_context_events(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    drop_mask = event_type.isin(SHOOTOUT_CONTEXT_EVENTS)
    return work.loc[~drop_mask].copy()


def remove_penalty_bookkeeping_without_player(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    drop_mask = event_type.isin(PENALTY_EVENTS) & work["player_id"].isna()
    return work.loc[~drop_mask].copy()


def remove_faceoff_rows_without_player(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    drop_mask = event_type.eq("faceoff") & work["player_id"].isna()
    return work.loc[~drop_mask].copy()


def remove_duplicate_goals_keep_earliest(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["_orig_index"] = work.index
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    goal_mask = event_type.eq("goal")

    group_keys = ["game_id", "period", "sequence_id", "team_id"]
    candidates = work.loc[goal_mask].copy()
    if candidates.empty:
        return work

    candidates["_dup_rank"] = candidates.groupby(group_keys, sort=False).cumcount()
    dup_groups = candidates.groupby(group_keys, sort=False)["_dup_rank"].transform("max") >= 1
    dupes = candidates.loc[dup_groups].copy()

    if dupes.empty:
        return work

    work = work.reset_index(drop=True)
    work["_row_idx"] = work.index
    work["_event_type_norm"] = work["event_type"].fillna("").astype(str).str.lower()
    work["_sl_event_id_num"] = pd.to_numeric(work["sl_event_id"], errors="coerce")
    work["_period_time_num"] = pd.to_numeric(work["period_time"], errors="coerce")

    row_idx_from_orig = work.set_index("_orig_index")["_row_idx"]

    seq_keys = ["game_id", "period", "sequence_id"]
    seq_order = work.sort_values(seq_keys + ["_period_time_num", "_sl_event_id_num", "_row_idx"], kind="mergesort", na_position="last").copy()
    seq_order["_seq_ord"] = seq_order.groupby(seq_keys, sort=False).cumcount()

    seq_ord_lookup = seq_order.set_index("_row_idx")["_seq_ord"]
    seq_work = seq_order[["game_id", "period", "sequence_id", "team_id", "_event_type_norm", "_seq_ord", "_row_idx"]].copy()

    drop_ids: set[object] = set()
    for _, grp in dupes.groupby(group_keys, sort=False):
        ranked = grp.copy()
        ranked["_row_idx"] = ranked["_orig_index"].map(row_idx_from_orig)
        ranked = ranked.dropna(subset=["_row_idx"]).copy()
        ranked["_row_idx"] = ranked["_row_idx"].astype(int)
        ranked = ranked.merge(
            work[["_row_idx", "sl_event_id", "_period_time_num", "_sl_event_id_num"]],
            on=["_row_idx", "sl_event_id"],
            how="left",
        )
        ranked["_goal_seq_ord"] = ranked["_row_idx"].map(seq_ord_lookup)

        key_vals = {
            "game_id": grp.iloc[0]["game_id"],
            "period": grp.iloc[0]["period"],
            "sequence_id": grp.iloc[0]["sequence_id"],
            "team_id": grp.iloc[0]["team_id"],
        }
        seq_subset = seq_work.loc[
            (seq_work["game_id"] == key_vals["game_id"])
            & (seq_work["period"] == key_vals["period"])
            & (seq_work["sequence_id"] == key_vals["sequence_id"])
            & (seq_work["team_id"] == key_vals["team_id"])
            & (seq_work["_event_type_norm"].isin(GOAL_SOURCE_EVENTS))
        ][["_seq_ord"]].copy()

        if seq_subset.empty:
            ranked["_source_gap"] = np.inf
        else:
            source_ord = seq_subset["_seq_ord"].sort_values(kind="mergesort").to_numpy(dtype=np.int64)

            def _nearest_preceding_gap(goal_ord: float) -> float:
                if pd.isna(goal_ord):
                    return np.inf
                idx = np.searchsorted(source_ord, int(goal_ord), side="left") - 1
                if idx < 0:
                    return np.inf
                return float(int(goal_ord) - source_ord[idx])

            ranked["_source_gap"] = ranked["_goal_seq_ord"].apply(_nearest_preceding_gap)

        ranked = ranked.sort_values(
            ["_source_gap", "_period_time_num", "_sl_event_id_num", "_row_idx"],
            kind="mergesort",
            na_position="last",
        )

        grp_drop = ranked.iloc[1:]
        drop_ids.update(grp_drop["sl_event_id"].tolist())

    out = work.loc[~work["sl_event_id"].isin(drop_ids)].copy()
    out = out.drop(columns=["_orig_index", "_row_idx", "_event_type_norm", "_sl_event_id_num", "_period_time_num"], errors="ignore")
    return out


def remove_false_start_sequences(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()

    group_keys = ["game_id", "sequence_id"]
    has_faceoff = event_type.eq("faceoff").groupby([work[k] for k in group_keys], sort=False).transform("any")
    has_real_play = (~event_type.isin({"faceoff", "whistle"})).groupby([work[k] for k in group_keys], sort=False).transform("any")
    drop_mask = has_faceoff & (~has_real_play)
    return work.loc[~drop_mask].copy()


def remove_post_terminator_residuals(events: pd.DataFrame) -> pd.DataFrame:
    work = events.sort_values(["game_id", "sequence_id", "sl_event_id"], kind="mergesort").copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()

    work["_seq_idx"] = work.groupby(["game_id", "sequence_id"], sort=False).cumcount()
    is_terminator = event_type.isin(TERMINATOR_EVENTS)

    first_terminator = (
        work.loc[is_terminator, ["game_id", "sequence_id", "_seq_idx", "event_type"]]
        .sort_values(["game_id", "sequence_id", "_seq_idx"], kind="mergesort")
        .groupby(["game_id", "sequence_id"], sort=False)
        .first()
        .reset_index()
        .rename(columns={"_seq_idx": "_first_term_idx", "event_type": "_first_term_event_type"})
    )
    work = work.merge(first_terminator, on=["game_id", "sequence_id"], how="left")

    after_terminator = work["_first_term_idx"].notna() & (work["_seq_idx"] > work["_first_term_idx"])
    first_term_type = work["_first_term_event_type"].fillna("").astype(str).str.lower()
    is_post_whistle = after_terminator & first_term_type.eq("whistle")
    keep_penalty_exception = (
        is_post_whistle
        & event_type.isin(PENALTY_EVENTS)
        & work["player_id"].notna()
    )
    work["is_post_whistle_penalty"] = keep_penalty_exception.astype(int)

    drop_mask = after_terminator & (~keep_penalty_exception)
    keep = work.loc[~drop_mask].drop(columns=["_seq_idx", "_first_term_idx", "_first_term_event_type"])
    return keep


def merge_pass_destinations(events: pd.DataFrame, max_delta_events: int = 3) -> pd.DataFrame:
    # 1. Sort chronologically and create a strict index for safe backward merging
    work = events.sort_values(["game_id", "period", "sequence_id", "sl_event_id"], kind="mergesort").reset_index(drop=True)
    work["_row_idx"] = work.index
    
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    work["_type_expects_destination_mapping"] = event_type.eq("pass")

    # 2. Isolate passes and destination events (both failed locations and receptions)
    passes = work.loc[event_type.eq("pass"), ["game_id", "period", "sequence_id", "sl_event_id", "_row_idx"]].copy()
    if passes.empty:
        return _fill_blank_destinations(work)
    
    passes = passes.rename(columns={"sl_event_id": "pass_event_id"})

    # Target both event types to map backwards
    dest_target_mask = event_type.isin(["failedpasslocation", "reception"])
    dest_events = work.loc[
        dest_target_mask,
        ["game_id", "period", "sequence_id", "sl_event_id", "event_type", "x_adj", "y_adj", "_row_idx"],
    ].copy()
    
    if dest_events.empty:
        return _fill_blank_destinations(work)

    # 3. Match the Destination Event backwards to the nearest Pass
    mapped = pd.merge_asof(
        dest_events,
        passes,
        on="_row_idx",
        by=["game_id", "period", "sequence_id"],
        direction="backward",
        tolerance=max_delta_events,
    )

    mapped = mapped.dropna(subset=["pass_event_id"]).copy()
    if mapped.empty:
        return _fill_blank_destinations(work)

    # Deduplicate in case a pass has both (e.g., failed location + weird reception log)
    mapped = mapped.drop_duplicates(subset=["pass_event_id"], keep="first")
    mapped = mapped.rename(
        columns={
            "x_adj": "mapped_dest_x",
            "y_adj": "mapped_dest_y",
            "event_type": "_dest_source_event_type",
            "sl_event_id": "_dest_source_sl_event_id",
        }
    )
    mapped["_dest_is_mapped"] = True

    # 4. ONLY join the mapped coordinates back onto the specific Pass rows
    work = work.merge(
        mapped[
            [
                "pass_event_id",
                "mapped_dest_x",
                "mapped_dest_y",
                "_dest_is_mapped",
                "_dest_source_event_type",
                "_dest_source_sl_event_id",
            ]
        ],
        left_on="sl_event_id", 
        right_on="pass_event_id", 
        how="left"
    )

    work["_dest_is_mapped"] = work.get("_dest_is_mapped", False).fillna(False).astype(bool)
    if "_dest_source_event_type" not in work.columns:
        work["_dest_source_event_type"] = pd.NA
    if "_dest_source_sl_event_id" not in work.columns:
        work["_dest_source_sl_event_id"] = np.nan

    # 5. Patch destination columns using explicit contract:
    #    - pass rows: mapped destination when available, else own x_adj/y_adj
    #    - non-pass rows: always zero-padded destination (0, 0)
    merge_stats: dict[str, int] = {
        "pass_rows_total": int(event_type.eq("pass").sum()),
        "pass_rows_mapped": 0,
        "pass_rows_self_fallback": 0,
        "non_pass_rows_total": int((~event_type.eq("pass")).sum()),
        "non_pass_dest_x_zero_overwrites": 0,
        "non_pass_dest_y_zero_overwrites": 0,
    }
    for coord, mapped_col in [("dest_x_adj", "mapped_dest_x"), ("dest_y_adj", "mapped_dest_y")]:
        if coord not in work.columns:
            work[coord] = np.nan

        origin_col = coord.replace("dest_", "")  # "dest_x_adj" -> "x_adj"
        pass_mask = event_type.eq("pass")
        non_pass_mask = ~pass_mask

        coord_num = pd.to_numeric(work[coord], errors="coerce")
        origin_num = pd.to_numeric(work[origin_col], errors="coerce")

        # Count non-pass rows that currently violate zero-destination before forced overwrite.
        mismatch_mask = non_pass_mask & (~np.isclose(
            coord_num.to_numpy(dtype=float),
            np.zeros(len(work), dtype=float),
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ))
        mismatch_count = int(mismatch_mask.sum())
        if coord == "dest_x_adj":
            merge_stats["non_pass_dest_x_zero_overwrites"] = mismatch_count
        else:
            merge_stats["non_pass_dest_y_zero_overwrites"] = mismatch_count

        mapped_series = pd.to_numeric(work[mapped_col], errors="coerce")

        # Pass rows: use mapped destination when available, otherwise own coordinates.
        mapped_pass_mask = pass_mask & mapped_series.notna()
        fallback_pass_mask = pass_mask & (~mapped_series.notna())
        work.loc[mapped_pass_mask, coord] = mapped_series.loc[mapped_pass_mask]
        work.loc[fallback_pass_mask, coord] = origin_num.loc[fallback_pass_mask]

        # Non-pass rows: always zero-padded coordinates.
        work.loc[non_pass_mask, coord] = 0.0

    merge_stats["pass_rows_mapped"] = int((event_type.eq("pass") & pd.to_numeric(work["_dest_source_sl_event_id"], errors="coerce").notna()).sum())
    merge_stats["pass_rows_self_fallback"] = int(merge_stats["pass_rows_total"] - merge_stats["pass_rows_mapped"])

    # 6. Clean up temp columns and drop only failed-pass-location helper rows.
    # Reception rows are retained in the canonical event stream.
    work = work.drop(columns=["pass_event_id", "mapped_dest_x", "mapped_dest_y", "_row_idx"])
    work = work.loc[~event_type.eq("failedpasslocation")].copy()
    work.attrs["merge_pass_destinations_stats"] = merge_stats

    return work

def _fill_blank_destinations(work: pd.DataFrame) -> pd.DataFrame:
    """Helper function to guarantee no blanks if no passes/mappings are found."""
    work = work.drop(columns=["_row_idx"], errors="ignore")
    event_type = work["event_type"].fillna("").astype(str).str.lower() if "event_type" in work.columns else pd.Series("", index=work.index)
    if "_dest_is_mapped" not in work.columns:
        work["_dest_is_mapped"] = False
    work["_dest_is_mapped"] = work["_dest_is_mapped"].fillna(False).astype(bool)
    if "_dest_source_event_type" not in work.columns:
        work["_dest_source_event_type"] = pd.NA
    if "_dest_source_sl_event_id" not in work.columns:
        work["_dest_source_sl_event_id"] = np.nan
    if "_type_expects_destination_mapping" not in work.columns:
        work["_type_expects_destination_mapping"] = event_type.eq("pass")
    work["_type_expects_destination_mapping"] = work["_type_expects_destination_mapping"].fillna(False).astype(bool)

    for coord in ["dest_x_adj", "dest_y_adj"]:
        if coord not in work.columns:
            work[coord] = np.nan
        origin_col = coord.replace("dest_", "")
        pass_mask = work["_type_expects_destination_mapping"].astype(bool)
        work.loc[pass_mask, coord] = work.loc[pass_mask, coord].fillna(work.loc[pass_mask, origin_col])
        work.loc[~pass_mask, coord] = 0.0
    return work


def check_non_pass_destination_zero_padding(
    events: pd.DataFrame,
    *,
    sample_size: int = 250,
    random_state: int = 42,
) -> dict[str, int | float | bool]:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    non_pass = work.loc[~event_type.eq("pass")].copy()

    if non_pass.empty:
        return {
            "sample_size_requested": int(sample_size),
            "sample_rows_checked": 0,
            "mismatch_rows": 0,
            "zero_padding_ok": True,
        }

    n = min(int(sample_size), int(len(non_pass)))
    sample = non_pass.sample(n=n, random_state=int(random_state), replace=False)

    x_zero = np.isclose(
        pd.to_numeric(sample["dest_x_adj"], errors="coerce").to_numpy(dtype=float),
        np.zeros(n, dtype=float),
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    y_zero = np.isclose(
        pd.to_numeric(sample["dest_y_adj"], errors="coerce").to_numpy(dtype=float),
        np.zeros(n, dtype=float),
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    mismatch_mask = ~(x_zero & y_zero)
    mismatch_rows = int(mismatch_mask.sum())

    if mismatch_rows > 0:
        mismatch_sample = sample.loc[mismatch_mask, [
            c for c in [
                "game_id",
                "sl_event_id",
                "period",
                "sequence_id",
                "event_type",
                "x_adj",
                "y_adj",
                "dest_x_adj",
                "dest_y_adj",
                "_dest_is_mapped",
                "_dest_source_event_type",
                "_dest_source_sl_event_id",
            ]
            if c in sample.columns
        ]].head(5)

        raise AssertionError(
            "Non-pass destination zero-padding check failed: sampled non-pass rows have dest_x_adj/dest_y_adj "
            f"different from 0.0/0.0. mismatch_rows={mismatch_rows}/{n}; "
            f"sample={mismatch_sample.to_dict('records')}"
        )

    return {
        "sample_size_requested": int(sample_size),
        "sample_rows_checked": int(n),
        "mismatch_rows": mismatch_rows,
        "mismatch_rate": float(mismatch_rows) / float(max(1, n)),
        "zero_padding_ok": True,
    }



def remove_assist_events(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    return work.loc[~event_type.eq("assist")].copy()


def add_eos_flags(events: pd.DataFrame) -> pd.DataFrame:
    work = events.sort_values(["game_id", "sequence_id", "sl_event_id"], kind="mergesort").copy()
    work["is_last_in_sequence"] = (
        work.groupby(["game_id", "sequence_id"], sort=False).cumcount(ascending=False) == 0
    ).astype(int)

    event_type = work["event_type"].fillna("").astype(str).str.lower()
    work["is_eos"] = (work["is_last_in_sequence"].eq(1) & event_type.isin(TERMINATOR_EVENTS)).astype(int)
    return work


def reorder_penalty_lpr_before_goal(events: pd.DataFrame, max_scan_depth: int = 5) -> pd.DataFrame:
    mover_events = {"shot", "deflection", "block", "blockedshot"}

    def reorder_group(group: pd.DataFrame) -> pd.DataFrame:
        ordered = group.sort_values("sl_event_id", kind="mergesort")
        rows = ordered.to_dict("records")
        i = 0
        while i < len(rows) - 1:
            cur_type = str(rows[i].get("event_type", "")).lower()
            nxt_type = str(rows[i + 1].get("event_type", "")).lower()
            if cur_type in {"penalty", "lpr"} and nxt_type == "goal":
                anchor = None
                for k in range(max(0, i - max_scan_depth), i):
                    if str(rows[k].get("event_type", "")).lower() in mover_events:
                        anchor = k
                if anchor is not None and anchor + 1 < i:
                    row = rows.pop(i)
                    rows.insert(anchor + 1, row)
                    i = max(anchor + 1, 0)
                    continue
            i += 1
        return pd.DataFrame(rows, columns=ordered.columns)

    pieces = []
    for _, grp in events.groupby(["game_id", "sequence_id"], sort=False):
        pieces.append(reorder_group(grp))

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["game_id", "sequence_id", "sl_event_id"], kind="mergesort").reset_index(drop=True)


def run_phase1_cleaning(
    base_dir: Path,
    *,
    config: DataPrepConfig | None = None,
    run_label: str = "run_current",
    events_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = config or DataPrepConfig()
    paths = DataPrepPaths(base_dir, run_label=run_label)
    paths.ensure_dirs()

    events = load_raw_inputs(base_dir, events_path=events_path)
    starting_rows = len(events)
    transform_stats: list[dict[str, int | str]] = []

    def record_transform(name: str, before_rows: int, after_rows: int) -> None:
        transform_stats.append(
            {
                "name": name,
                "rows_before": int(before_rows),
                "rows_after": int(after_rows),
                "rows_removed": int(before_rows - after_rows),
            }
        )

    before = len(events)
    events, extra_goals = extract_and_remove_shootout_penalty_shot_noise(events)
    record_transform("extract_and_remove_shootout_penalty_shot_noise", before, len(events))

    before = len(events)
    events = apply_regular_time_boundary(events)
    record_transform("apply_regular_time_boundary", before, len(events))

    before = len(events)
    events = remove_post_shootout_tail_events(events)
    record_transform("remove_post_shootout_tail_events", before, len(events))

    before = len(events)
    events = remove_penaltyshot_windows(events)
    record_transform("remove_penaltyshot_windows", before, len(events))

    before = len(events)
    events = remove_explicit_shootout_context_events(events)
    record_transform("remove_explicit_shootout_context_events", before, len(events))

    before = len(events)
    events = remove_penalty_bookkeeping_without_player(events)
    record_transform("remove_penalty_bookkeeping_without_player", before, len(events))

    before = len(events)
    events = remove_faceoff_rows_without_player(events)
    record_transform("remove_faceoff_rows_without_player", before, len(events))

    before = len(events)
    events = remove_duplicate_goals_keep_earliest(events)
    record_transform("remove_duplicate_goals_keep_earliest", before, len(events))

    before = len(events)
    events = remove_false_start_sequences(events)
    record_transform("remove_false_start_sequences", before, len(events))

    before = len(events)
    events = remove_post_terminator_residuals(events)
    record_transform("remove_post_terminator_residuals", before, len(events))

    before = len(events)
    # This now maps both failed locations and receptions, AND fills all blank destinations
    events = merge_pass_destinations(events, max_delta_events=cfg.pass_merge_max_delta_events)
    record_transform("merge_pass_destinations", before, len(events))
    merge_pass_destinations_stats = dict(events.attrs.get("merge_pass_destinations_stats", {}))

    non_pass_destination_zero_padding = check_non_pass_destination_zero_padding(events)

    before = len(events)
    events = remove_assist_events(events)
    record_transform("remove_assist_events", before, len(events))

    before = len(events)
    events = reorder_penalty_lpr_before_goal(events)
    record_transform("reorder_penalty_lpr_before_goal", before, len(events))

    before = len(events)
    events = add_eos_flags(events)
    record_transform("add_eos_flags", before, len(events))

    assert_no_duplicate_columns(events, "phase1_events")

    disorder = sequence_disorder_count(
        events,
        threshold=cfg.sequence_disorder_threshold_seconds,
    )
    warnings = collect_phase1_secondary_warnings(
        events,
        disorder_rows=disorder,
        required_event_types=cfg.required_phase1_event_types,
        sequence_disorder_warn_threshold_rows=cfg.sequence_disorder_warn_threshold_rows,
    )
    gate_status = "warn" if warnings else "pass"

    event_type = events["event_type"].fillna("").astype(str).str.lower()
    boundary_violations = phase1_boundary_violation_counts(events)

    out_path = output_path or paths.phase1_events_output
    write_parquet(out_path, events)
    write_csv(paths.phase1_extra_goals_output, extra_goals)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "input_rows": int(starting_rows),
        "output_rows": int(len(events)),
        "extra_goal_rows": int(len(extra_goals)),
        "sequence_disorder_rows": int(disorder),
        "warnings": warnings,
        "gate_status": gate_status,
        "transform_stats": transform_stats,
        "assists_remaining": int(event_type.eq("assist").sum()),
        "failedpasslocation_remaining": int(event_type.eq("failedpasslocation").sum()),
        "reception_remaining": int(event_type.eq("reception").sum()),
        "post_whistle_penalties_flagged": int(events.get("is_post_whistle_penalty", pd.Series(dtype=int)).sum()),
        "boundary_violations_remaining": boundary_violations,
        "merge_pass_destinations_stats": merge_pass_destinations_stats,
        "non_pass_destination_zero_padding": non_pass_destination_zero_padding,
        "run_label": run_label,
        "run_root": str(paths.runs_root_dir),
        "output": str(out_path),
        "extra_goals_output": str(paths.phase1_extra_goals_output),
    }
    write_json(paths.prep_logs_dir / "phase1_cleaning_summary.json", summary)
    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase1_cleaning",
        output_path=out_path,
        rows=len(events),
        columns=events.columns.tolist(),
        extra={
            "gate_status": gate_status,
            "warnings_count": int(len(warnings)),
        },
    )

    return events, extra_goals, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 data cleaning pipeline")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, summary = run_phase1_cleaning(
        args.base_dir,
        run_label=args.run_label,
        events_path=args.events_path,
        output_path=args.output,
    )
    print(summary)


if __name__ == "__main__":
    main()
