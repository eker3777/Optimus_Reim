from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sprint_week_utils import SprintPaths, required_columns, utc_now_iso, write_csv
except ImportError:
    from scripts.sprint_week_utils import SprintPaths, required_columns, utc_now_iso, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate penalty macro values using offset-filtered penalties and 20s/120s goal windows"
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=120.0)
    parser.add_argument(
        "--offset-event-gap",
        type=int,
        default=4,
        help="Event-id gap threshold for opposite-team penalty offset filtering",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def classify_penalty_type(detail: str) -> str:
    s = (detail or "").strip().lower()
    if not s:
        return "other"
    if "double" in s and "minor" in s:
        return "double_minor"
    if "major" in s:
        return "major"
    if "misconduct" in s or "game misconduct" in s:
        return "misconduct"
    if "match" in s:
        return "match"
    if "minor" in s:
        return "minor"
    return "other"


def resolve_default_events_path(base_dir: Path) -> Path:
    runs_root = base_dir / "Data" / "Pipeline Runs"
    if runs_root.exists():
        runs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in runs:
            candidate = run_dir / "phase3" / "tensor_ready_dataset.parquet"
            if candidate.exists():
                return candidate
    return base_dir / "HALO Hackathon Data" / "events.parquet"


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["event_type_clean"] = events["event_type"].astype(str).str.lower().str.strip()
    events["period"] = pd.to_numeric(events["period"], errors="coerce")
    events["period_time"] = pd.to_numeric(events["period_time"], errors="coerce")
    events["sl_event_id"] = pd.to_numeric(events["sl_event_id"], errors="coerce")
    
    # CRITICAL FIX: Force string to prevent team_id type mismatches (str vs float) when equating poss/def goals
    events["team_id"] = events["team_id"].astype(str).str.strip()

    # Continuous absolute game time (20-minute periods).
    events["abs_time"] = (events["period"] - 1.0) * 1200.0 + events["period_time"]
    events = events.sort_values(["game_id", "sl_event_id", "period", "period_time"], kind="mergesort").reset_index(drop=True)
    return events


def flag_offsetting_penalties(penalties: pd.DataFrame, offset_event_gap: int) -> pd.DataFrame:
    out = penalties.copy()
    out = out.sort_values(["game_id", "sl_event_id"], kind="mergesort").reset_index(drop=True)

    # Simplified forward/backward looking matching simple logic
    out["next_event_id"] = out.groupby("game_id", sort=False)["sl_event_id"].shift(-1)
    out["next_team_id"] = out.groupby("game_id", sort=False)["team_id"].shift(-1)
    out["prev_event_id"] = out.groupby("game_id", sort=False)["sl_event_id"].shift(1)
    out["prev_team_id"] = out.groupby("game_id", sort=False)["team_id"].shift(1)

    offset_next = (
        (out["next_event_id"] - out["sl_event_id"] <= offset_event_gap) 
        & (out["next_team_id"].notna()) 
        & (out["next_team_id"] != out["team_id"])
    )
    offset_prev = (
        (out["sl_event_id"] - out["prev_event_id"] <= offset_event_gap) 
        & (out["prev_team_id"].notna()) 
        & (out["prev_team_id"] != out["team_id"])
    )

    out["is_offsetting"] = (offset_next | offset_prev).fillna(False)
    return out


def summarize_penalties(
    events: pd.DataFrame,
    window_sec: float,
    offset_event_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = prepare_events(events)

    penalties = events[events["event_type_clean"] == "penalty"].copy()
    if "detail" in penalties.columns:
        penalty_detail = penalties["detail"].astype(str)
    else:
        penalty_detail = pd.Series("", index=penalties.index)
    penalties["penalty_type"] = penalty_detail.map(classify_penalty_type)
    penalties = flag_offsetting_penalties(penalties, offset_event_gap=offset_event_gap)

    total_raw_penalties = int(len(penalties))
    valid_penalties = penalties.loc[~penalties["is_offsetting"]].copy()
    total_valid = int(len(valid_penalties))

    valid_penalties["penalty_row_id"] = np.arange(len(valid_penalties), dtype=np.int64)

    # ====================================================================
    # METHOD A: Using the existing target column (20-second horizon)
    # Target 0 = Poss Goal, Target 1 = Def Goal
    # ====================================================================
    target_col = None
    for candidate in ["target", "target_xt"]:
        if candidate in valid_penalties.columns:
            target_col = candidate
            break

    if target_col is not None:
        target_vals = pd.to_numeric(valid_penalties[target_col], errors="coerce")
        valid_penalties["poss_goal_20"] = target_vals.eq(0)
        valid_penalties["def_goal_20"] = target_vals.eq(1)
    else:
        valid_penalties["poss_goal_20"] = False
        valid_penalties["def_goal_20"] = False

    # ====================================================================
    # METHOD B: Manual 120-Second Look-Ahead (True Powerplay Window)
    # ====================================================================
    goals = events.loc[events["event_type_clean"] == "goal", ["game_id", "abs_time", "team_id"]].copy()
    goals = goals.rename(columns={"abs_time": "abs_time_goal", "team_id": "team_id_goal"})

    pen_base = valid_penalties[["penalty_row_id", "game_id", "abs_time", "team_id", "sl_event_id"]].copy()
    pen_base = pen_base.rename(columns={"abs_time": "abs_time_pen", "team_id": "team_id_pen"})

    merged = pen_base.merge(goals, on="game_id", how="left")
    
    # Filter for goals that occur AFTER the penalty but within continuous 120 seconds
    valid_goal_window = merged.loc[
        merged["abs_time_goal"].ge(merged["abs_time_pen"])
        & merged["abs_time_goal"].le(merged["abs_time_pen"] + float(window_sec))
    ].copy()
    
    # Sort by time so we capture the FIRST goal scored after the penalty
    valid_goal_window = valid_goal_window.sort_values(["penalty_row_id", "abs_time_goal"], kind="mergesort")
    first_goals = valid_goal_window.drop_duplicates(subset=["penalty_row_id"], keep="first").copy()
    
    # Check if the team that took the penalty scored (Poss_Goal) or opponent scored (Def_Goal)
    first_goals["poss_goal_120"] = first_goals["team_id_pen"] == first_goals["team_id_goal"]
    first_goals["def_goal_120"] = ~first_goals["poss_goal_120"]

    first_goal_flags = first_goals[["penalty_row_id", "poss_goal_120", "def_goal_120"]].copy()
    valid_penalties = valid_penalties.merge(first_goal_flags, on="penalty_row_id", how="left")
    valid_penalties["poss_goal_120"] = valid_penalties["poss_goal_120"].fillna(False)
    valid_penalties["def_goal_120"] = valid_penalties["def_goal_120"].fillna(False)
    valid_penalties["goal_in_120"] = valid_penalties["poss_goal_120"] | valid_penalties["def_goal_120"]

    valid_penalties["macro_outcome_20"] = valid_penalties["def_goal_20"].astype(float) - valid_penalties["poss_goal_20"].astype(float)
    valid_penalties["macro_outcome_120"] = valid_penalties["def_goal_120"].astype(float) - valid_penalties["poss_goal_120"].astype(float)

    # Aggregating values
    group = (
        valid_penalties.groupby("penalty_type", as_index=False)
        .agg(
            sample_count=("penalty_row_id", "size"),
            poss_goal_20_count=("poss_goal_20", "sum"),
            def_goal_20_count=("def_goal_20", "sum"),
            poss_goal_120_count=("poss_goal_120", "sum"),
            def_goal_120_count=("def_goal_120", "sum"),
            goal_in_120_count=("goal_in_120", "sum"),
            macro_outcome_20_mean=("macro_outcome_20", "mean"),
            macro_outcome_20_std=("macro_outcome_20", "std"),
            macro_outcome_120_mean=("macro_outcome_120", "mean"),
            macro_outcome_120_std=("macro_outcome_120", "std"),
        )
        .copy()
    )

    for col in ["macro_outcome_20_std", "macro_outcome_120_std"]:
        group[col] = group[col].fillna(0.0)

    n = group["sample_count"].clip(lower=1).astype(float)
    group["poss_goal_20_rate"] = group["poss_goal_20_count"] / n
    group["def_goal_20_rate"] = group["def_goal_20_count"] / n
    group["poss_goal_120_rate"] = group["poss_goal_120_count"] / n
    group["def_goal_120_rate"] = group["def_goal_120_count"] / n
    group["goal_in_120_rate"] = group["goal_in_120_count"] / n

    group["penalty_macro_value_20"] = group["def_goal_20_rate"] - group["poss_goal_20_rate"]
    group["penalty_macro_value_120"] = group["def_goal_120_rate"] - group["poss_goal_120_rate"]
    group["penalty_macro_value"] = group["penalty_macro_value_120"]

    group["sem_macro_120"] = group["macro_outcome_120_std"] / np.sqrt(n)
    group["ci95_lower_120"] = group["penalty_macro_value_120"] - 1.96 * group["sem_macro_120"]
    group["ci95_upper_120"] = group["penalty_macro_value_120"] + 1.96 * group["sem_macro_120"]

    if total_valid > 0:
        all_row = {
            "penalty_type": "__all__",
            "sample_count": int(total_valid),
            "poss_goal_20_count": int(valid_penalties["poss_goal_20"].sum()),
            "def_goal_20_count": int(valid_penalties["def_goal_20"].sum()),
            "poss_goal_120_count": int(valid_penalties["poss_goal_120"].sum()),
            "def_goal_120_count": int(valid_penalties["def_goal_120"].sum()),
            "goal_in_120_count": int(valid_penalties["goal_in_120"].sum()),
            "macro_outcome_20_mean": float(valid_penalties["macro_outcome_20"].mean()),
            "macro_outcome_20_std": float(valid_penalties["macro_outcome_20"].std(ddof=1)) if total_valid > 1 else 0.0,
            "macro_outcome_120_mean": float(valid_penalties["macro_outcome_120"].mean()),
            "macro_outcome_120_std": float(valid_penalties["macro_outcome_120"].std(ddof=1)) if total_valid > 1 else 0.0,
        }
        all_row["poss_goal_20_rate"] = all_row["poss_goal_20_count"] / float(total_valid)
        all_row["def_goal_20_rate"] = all_row["def_goal_20_count"] / float(total_valid)
        all_row["poss_goal_120_rate"] = all_row["poss_goal_120_count"] / float(total_valid)
        all_row["def_goal_120_rate"] = all_row["def_goal_120_count"] / float(total_valid)
        all_row["goal_in_120_rate"] = all_row["goal_in_120_count"] / float(total_valid)
        all_row["penalty_macro_value_20"] = all_row["def_goal_20_rate"] - all_row["poss_goal_20_rate"]
        all_row["penalty_macro_value_120"] = all_row["def_goal_120_rate"] - all_row["poss_goal_120_rate"]
        all_row["penalty_macro_value"] = all_row["penalty_macro_value_120"]
        all_row["sem_macro_120"] = all_row["macro_outcome_120_std"] / np.sqrt(float(total_valid)) if total_valid > 0 else np.nan
        all_row["ci95_lower_120"] = all_row["penalty_macro_value_120"] - 1.96 * all_row["sem_macro_120"]
        all_row["ci95_upper_120"] = all_row["penalty_macro_value_120"] + 1.96 * all_row["sem_macro_120"]
        group = pd.concat([group, pd.DataFrame([all_row])], ignore_index=True)

    group = group.sort_values(["penalty_type"], kind="mergesort").reset_index(drop=True)

    meta = {
        "generated_at_utc": utc_now_iso(),
        "window_sec": float(window_sec),
        "offset_event_gap": int(offset_event_gap),
        "target_column_used": target_col,
        "total_raw_penalties": total_raw_penalties,
        "offsetting_penalties_excluded": int(total_raw_penalties - total_valid),
        "valid_penalties": total_valid,
        "goals_within_window": int(valid_penalties["goal_in_120"].sum()) if total_valid else 0,
    }

    return group, valid_penalties, meta


def write_report(path: Path, summary: pd.DataFrame, samples: pd.DataFrame, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Penalty Macro Estimation Report")
    lines.append("")
    lines.append(f"Generated at (UTC): {utc_now_iso()}")
    lines.append(f"Window length (seconds): {float(meta['window_sec'])}")
    lines.append(f"Offset event gap: {int(meta['offset_event_gap'])}")
    lines.append(f"Target column used for 20s method: {meta.get('target_column_used')}")
    lines.append(f"Raw penalties: {int(meta['total_raw_penalties']):,}")
    lines.append(f"Offsetting penalties excluded: {int(meta['offsetting_penalties_excluded']):,}")
    lines.append(f"Valid penalties analyzed: {int(meta['valid_penalties']):,}")
    lines.append(f"Penalties with first goal in 120s: {int(meta['goals_within_window']):,}")
    lines.append("")
    lines.append("## Method")
    lines.append("Penalties are filtered for offsetting calls using nearest opposite-team penalties within event-id gap threshold.")
    lines.append("20s method uses target labels: class 0=penalized team scores, class 1=opponent scores.")
    lines.append("120s method uses first-goal look-ahead in continuous absolute time.")
    lines.append("Penalty macro value is defined as `def_goal_rate - poss_goal_rate` (penalized-team expected cost).")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")

    if summary.empty:
        lines.append("No penalty events were available for estimation.")
    else:
        table_cols = [
            "penalty_type",
            "sample_count",
            "poss_goal_20_rate",
            "def_goal_20_rate",
            "poss_goal_120_rate",
            "def_goal_120_rate",
            "penalty_macro_value",
            "ci95_lower_120",
            "ci95_upper_120",
        ]
        view = summary[table_cols].copy()
        for col in [
            "poss_goal_20_rate",
            "def_goal_20_rate",
            "poss_goal_120_rate",
            "def_goal_120_rate",
            "penalty_macro_value",
            "ci95_lower_120",
            "ci95_upper_120",
        ]:
            view[col] = view[col].map(lambda x: f"{x:.6f}")

        header = "| " + " | ".join(table_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(table_cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for row in view.itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    paths = SprintPaths(base_dir)
    paths.ensure_all()

    events_path = args.events_path or resolve_default_events_path(base_dir)
    output_csv = args.output_csv or (paths.results_sprint_week / "penalty_macro_values_by_type.csv")
    output_report = args.output_report or (paths.results_sprint_week / "penalty_macro_estimation_report.md")

    events = pd.read_parquet(events_path)
    required_columns(events, ["game_id", "period", "period_time", "sl_event_id", "event_type", "team_id"], "events")
    if "detail" not in events.columns:
        events["detail"] = ""

    summary, samples, meta = summarize_penalties(
        events,
        window_sec=args.window_sec,
        offset_event_gap=args.offset_event_gap,
    )
    write_csv(output_csv, summary)
    write_report(output_report, summary, samples, meta)
    write_csv(paths.logs_dir / "penalty_macro_valid_penalties.csv", samples)

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_report}")
    print(f"Events source: {events_path}")
    print(f"Raw penalties: {int(meta['total_raw_penalties']):,}")
    print(f"Offsetting penalties excluded: {int(meta['offsetting_penalties_excluded']):,}")
    print(f"Valid penalties analyzed: {int(meta['valid_penalties']):,}")


if __name__ == "__main__":
    main()