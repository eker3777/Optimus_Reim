from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .config import DataPrepPaths
    from .io_utils import append_manifest_record, read_csv, read_parquet, utc_now_iso, write_csv, write_json
    from .validation import require_columns
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepPaths
    from io_utils import append_manifest_record, read_csv, read_parquet, utc_now_iso, write_csv, write_json
    from validation import require_columns


def build_goal_counts(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    event_type = work["event_type"].fillna("").astype(str).str.lower()
    goals = work.loc[event_type.eq("goal"), ["game_id", "team_id"]].copy()
    if goals.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "cleaned_event_goals"])

    out = goals.groupby(["game_id", "team_id"], as_index=False).size()
    out = out.rename(columns={"size": "cleaned_event_goals"})
    return out


def build_extra_goal_counts(extra_goals: pd.DataFrame) -> pd.DataFrame:
    if extra_goals.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "extra_goals"])

    keyed = extra_goals.loc[extra_goals["team_id"].notna(), ["game_id", "team_id"]].copy()
    if keyed.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "extra_goals"])

    out = keyed.groupby(["game_id", "team_id"], as_index=False).size()
    out = out.rename(columns={"size": "extra_goals"})
    return out


def build_unassigned_extra_goal_counts(extra_goals: pd.DataFrame) -> pd.DataFrame:
    if extra_goals.empty:
        return pd.DataFrame(columns=["game_id", "unassigned_extra_goals"])

    unassigned = extra_goals.loc[extra_goals["team_id"].isna(), ["game_id"]].copy()
    if unassigned.empty:
        return pd.DataFrame(columns=["game_id", "unassigned_extra_goals"])

    out = unassigned.groupby(["game_id"], as_index=False).size()
    out = out.rename(columns={"size": "unassigned_extra_goals"})
    return out


def verify_phase1_scores(
    base_dir: Path,
    events_path: Path | None = None,
    extra_goals_path: Path | None = None,
    run_label: str = "run_current",
    hard_fail_on_mismatch: bool = True,
) -> dict:
    paths = DataPrepPaths(base_dir, run_label=run_label)
    events_source = events_path or paths.phase1_events_output
    extra_source = extra_goals_path or paths.phase1_extra_goals_output
    games_source = paths.raw_data_dir / "games.parquet"

    events = read_parquet(events_source)
    games = read_parquet(games_source)

    require_columns(events, ["game_id", "team_id", "event_type"], "phase1_events")
    require_columns(games, ["game_id", "home_team_id", "away_team_id", "home_score", "away_score"], "games")

    goal_counts = build_goal_counts(events)

    if Path(extra_source).exists():
        extra_goals = read_csv(extra_source)
        require_columns(extra_goals, ["game_id", "team_id"], "extra_goals")
    else:
        extra_goals = pd.DataFrame(columns=["game_id", "team_id", "extra_goals"])

    extra_counts = build_extra_goal_counts(extra_goals)
    unassigned_extra_counts = build_unassigned_extra_goal_counts(extra_goals)

    official_home = games[["game_id", "home_team_id", "home_score"]].rename(
        columns={"home_team_id": "team_id", "home_score": "official_goals"}
    )
    official_away = games[["game_id", "away_team_id", "away_score"]].rename(
        columns={"away_team_id": "team_id", "away_score": "official_goals"}
    )
    official = pd.concat([official_home, official_away], ignore_index=True)

    compare = official.merge(goal_counts, on=["game_id", "team_id"], how="left")
    compare = compare.merge(extra_counts, on=["game_id", "team_id"], how="left")
    compare = compare.merge(unassigned_extra_counts, on=["game_id"], how="left")

    compare["cleaned_event_goals"] = compare["cleaned_event_goals"].fillna(0).astype(int)
    compare["extra_goals"] = compare["extra_goals"].fillna(0).astype(int)
    compare["unassigned_extra_goals"] = compare["unassigned_extra_goals"].fillna(0).astype(int)

    # Allocate unassigned (teamless) noise goals to the most likely team by score deficit.
    compare["base_calculated_goals"] = compare["cleaned_event_goals"] + compare["extra_goals"]
    compare["goal_deficit"] = compare["official_goals"] - compare["base_calculated_goals"]
    compare["allocated_unassigned_extra_goals"] = 0

    def allocate_unassigned(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        unassigned = int(out["unassigned_extra_goals"].iloc[0])
        if unassigned <= 0:
            return out

        while unassigned > 0:
            deficits = out["goal_deficit"].astype(int)
            max_deficit = int(deficits.max())
            if max_deficit <= 0:
                break
            idx = deficits.idxmax()
            out.loc[idx, "allocated_unassigned_extra_goals"] += 1
            out.loc[idx, "goal_deficit"] -= 1
            unassigned -= 1
        return out

    compare = (
        compare.groupby("game_id", group_keys=False, sort=False)
        .apply(allocate_unassigned)
        .reset_index(drop=True)
    )

    compare["calculated_goals"] = (
        compare["base_calculated_goals"] + compare["allocated_unassigned_extra_goals"]
    )
    compare["score_match"] = compare["calculated_goals"].eq(compare["official_goals"]) 

    mismatches = compare.loc[~compare["score_match"]].copy()

    report = {
        "generated_at_utc": utc_now_iso(),
        "rows_checked": int(len(compare)),
        "matches": int(compare["score_match"].sum()),
        "mismatches": int((~compare["score_match"]).sum()),
        "games_with_mismatch": int(mismatches["game_id"].nunique()) if not mismatches.empty else 0,
        "all_match": bool(compare["score_match"].all()),
        "run_label": run_label,
        "run_root": str(paths.runs_root_dir),
        "events_source": str(events_source),
        "extra_goals_source": str(extra_source),
        "hard_fail_on_mismatch": bool(hard_fail_on_mismatch),
    }

    paths.ensure_dirs()
    mismatch_path = paths.prep_logs_dir / "phase1_score_mismatches.csv"
    write_csv(mismatch_path, mismatches)
    report["mismatch_path"] = str(mismatch_path)
    if "game_id" in mismatches.columns:
        report["mismatch_game_ids"] = sorted([str(v) for v in mismatches["game_id"].dropna().unique().tolist()])
    else:
        report["mismatch_game_ids"] = []
    report["gate_status"] = "pass" if report["all_match"] else "fail"
    write_json(paths.phase1_score_report_output, report)

    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase1_score_verification",
        output_path=paths.phase1_score_report_output,
        rows=len(compare),
        columns=compare.columns.tolist(),
        extra={
            "gate_status": report["gate_status"],
            "mismatches": report["mismatches"],
        },
    )

    if hard_fail_on_mismatch and not report["all_match"]:
        raise AssertionError(
            "Phase 1 score verification failed: "
            f"{report['mismatches']} team-level mismatches across {report['games_with_mismatch']} games. "
            f"See {mismatch_path}."
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase 1 scores against games.parquet")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--extra-goals-path", type=Path, default=None)
    parser.add_argument(
        "--no-hard-fail",
        action="store_true",
        help="Do not raise on score mismatches (still writes report and mismatch CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_phase1_scores(
        args.base_dir,
        events_path=args.events_path,
        extra_goals_path=args.extra_goals_path,
        run_label=args.run_label,
        hard_fail_on_mismatch=not args.no_hard_fail,
    )
    print(report)


if __name__ == "__main__":
    main()
