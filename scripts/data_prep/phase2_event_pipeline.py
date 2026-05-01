from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import read_parquet, write_csv, write_parquet
    from .validation import assert_no_duplicate_columns, require_columns
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import read_parquet, write_csv, write_parquet
    from validation import assert_no_duplicate_columns, require_columns


LEAK_REGEX = re.compile(
    r"\bgoal\b|\bgoals\b|\bon\s*net\b|\bonnet\b|\bwith\s*goal\b|\bwith\s*shot\s*on\s*net(?:\s*whistle)?\b",
    flags=re.IGNORECASE,
)
BLOCKED_COMPACT_TOKENS = {
    "withgoal",
    "withshotonnet",
    "withshotonnetwhistle",
}
DROP_FLAG_TOKENS = frozenset(
    {
        "gwg",
        "highdangermissedshotrecovery",
        "fivehole",
        "deflected",
        "blocked",
        "missed",
        "miss pass",
        "miss shot",
        "successful",
        "fanned",
        "firstafterpp",
        "whistlebeforeexit",
        "withcyclepass",
        "withfailedcyclepass",
        "withplayafter",
        "withpressure",
        "withrebound",
        "withslotshot",
    }
)
FORBIDDEN_DESCRIPTION_TERMS = [
    "successful",
    "miss pass",
    "miss shot",
    "blocked",
    "missed",
]
SAVE_SOURCE_EVENT_TYPES = {"shot", "deflection"}
DEFENSIVE_DEFLECTION_LOOKBACK_EVENTS = 3
DEFENSIVE_DEFLECTION_RESET_EVENTS = {
    "faceoff",
    "whistle",
    "stoppage",
    "goal",
    "end_of_period",
    "period_end",
    "period_start",
    "start_of_period",
}
ON_NET_REGEX = re.compile(
    r"\bon\s*net\b|\bonnet\b|\bwith\s*shot\s*on\s*net\b|\bwithshotonnet\b|\bwithshotonnetwhistle\b",
    flags=re.IGNORECASE,
)
GOAL_SOURCE_REGEX = re.compile(r"\bwith\s*goal\b|\bwithgoal\b", flags=re.IGNORECASE)


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("+", " ").replace("-", " ")
    return " ".join(text.split()).lower()


def _safe_text_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace("+", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    return cleaned.fillna("")


def _remove_leak_terms(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    cleaned = LEAK_REGEX.sub(" ", str(value).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else np.nan


@lru_cache(maxsize=200000)
def _normalize_flag_token(token: str) -> str:
    text = str(token).lower()
    text = text.replace("+", " ").replace("-", " ")
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text = re.sub(r"^(?:shot|deflection)\s*_?\s*flags?\s*[:_\-\s]*", "", text)
    text = re.sub(r"[;|/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,._:-")
    return text


@lru_cache(maxsize=200000)
def _split_clean_flags_cached(value_text: str) -> tuple[str, ...]:
    text = value_text.lower().replace("+", " ").replace("-", " ")
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text = re.sub(r"\s*[,;|/]\s*", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ()

    tokens = [_normalize_flag_token(tok.strip()) for tok in text.split(",") if tok.strip()]
    tokens = [tok for tok in tokens if tok]
    return tuple(dict.fromkeys(tokens))


def _split_clean_flags(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return list(_split_clean_flags_cached(str(value)))


@lru_cache(maxsize=200000)
def _keep_non_leak_flag_token(token: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(token).lower())
    if compact in BLOCKED_COMPACT_TOKENS:
        return False
    return LEAK_REGEX.search(str(token)) is None


@lru_cache(maxsize=200000)
def _filter_non_leak_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(tok for tok in tokens if _keep_non_leak_flag_token(tok))


@lru_cache(maxsize=200000)
def _finalize_flag_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        clean_tok = tok.lower().strip()
        if "*" in clean_tok:
            continue
        if clean_tok in DROP_FLAG_TOKENS:
            continue
        if clean_tok in seen:
            continue
        seen.add(clean_tok)
        out.append(clean_tok)
    return tuple(out)


def drop_phase1_leak_rows(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df_events.copy()

    detail = _safe_text_series(df["detail"]) if "detail" in df.columns else pd.Series("", index=df.index)
    description = _safe_text_series(df["description"]) if "description" in df.columns else pd.Series("", index=df.index)
    event_type = _safe_text_series(df["event_type"]) if "event_type" in df.columns else pd.Series("", index=df.index)
    period = pd.to_numeric(df.get("period"), errors="coerce")

    is_shootout = detail.str.contains("shootout", regex=False) | description.str.contains("shootout", regex=False)
    is_shootout = is_shootout | period.eq(5)
    is_penalty_shot = detail.str.contains("penalty shot", regex=False) | description.str.contains("penalty shot", regex=False)
    is_penalty_shot = is_penalty_shot | event_type.eq("penalty_shot")
    is_home_shoots_right = detail.str.contains("home team shoots right", regex=False) | description.str.contains(
        "home team shoots right", regex=False
    )
    is_away_shoots_right = detail.str.contains("away team shoots right", regex=False) | description.str.contains(
        "away team shoots right", regex=False
    )

    drop_mask = is_shootout | is_penalty_shot | is_home_shoots_right | is_away_shoots_right
    out = df.loc[~drop_mask].copy()

    summary = {
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
        "dropped_rows_total": int(drop_mask.sum()),
        "dropped_shootout_rows": int(is_shootout.sum()),
        "dropped_penalty_shot_rows": int(is_penalty_shot.sum()),
        "dropped_home_team_shoots_right_rows": int(is_home_shoots_right.sum()),
        "dropped_away_team_shoots_right_rows": int(is_away_shoots_right.sum()),
    }
    return out, summary


def clean_event_text(df_events: pd.DataFrame, diagnostics_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    df["description_clean"] = _safe_text_series(df["description"]) if "description" in df.columns else np.nan
    df["flags_clean"] = _safe_text_series(df["flags"]) if "flags" in df.columns else np.nan

    abbrev_patterns = [
        (r"\b(f/off|face\s*off|faceoff)\b", "faceoff"),
        (r"\bchk\b", "check"),
        (r"\brec\b", "reception"),
        (r"\bdef\b", "defence"),
        (r"\bblue hold\b", "blueline hold"),
        (r"\b(e\s*w|e/w)\b", "east/west"),
    ]
    zone_patterns = [
        (r"\b(dz|d[- ]?zone|dzone|defensive zone)\b", "defensive zone"),
        (r"\b(nz|n[- ]?zone|nzone|neutral zone)\b", "neutral zone"),
        (r"\b(oz|o[- ]?zone|ozone|offensive zone)\b", "offensive zone"),
        (r"\b(dzpass|d[- ]?zone\s*pass|defensive zone\s*pass)\b", "defensive zone pass"),
        (r"\b(nzpass|n[- ]?zone\s*pass|neutral zone\s*pass)\b", "neutral zone pass"),
        (r"\b(ozpass|o[- ]?zone\s*pass|offensive zone\s*pass)\b", "offensive zone pass"),
        (r"\b(dzdeke|d[- ]?zone\s*deke|defensive zone\s*deke)\b", "defensive zone deke"),
        (r"\b(nzdeke|n[- ]?zone\s*deke|neutral zone\s*deke)\b", "neutral zone deke"),
        (r"\b(ozdeke|o[- ]?zone\s*deke|offensive zone\s*deke)\b", "offensive zone deke"),
    ]
    replacements = {
        r"\blpr\b": "loose puck recovery",
        r"\bcont\b": "contested",
        r"\breb\b": "rebound",
        r"\bopdump\b": "opposing dump",
        r"\bhipresopdump\b": "high pressure opposing dump",
        r"\bnofore\b": "no forecheck",
        r"\bhi press\b": "high pressure",
        r"offboards": "off boards",
        r"\bd2d\b": "defenceman to defenceman pass",
        r"\bbc\b": "board cycle",
        r"\bsc\b": "slot cycle",
        r"\bci\b": "chip in",
        r"\bdi\b": "dump in",
        "nzpuckprotection": "neutral zone puck protection",
        "ozpuckprotection": "offensive zone puck protection",
        "dzpuckprotection": "defensive zone puck protection",
        "lprreb": "loose puck recovery rebound",
        "failed pass trajectory location": "failed pass to",
        "offensive zone entry pass reception": "offensive zone entry pass successful",
        "defensive deflection; original description: outside shot for onnet": "defensive deflection",
        "defensive deflection; original description: slot shot for onnet": "defensive deflection",
        r"\bnz\b": "neutral zone",
        r"\boz\b": "offensive zone",
        r"\bdz\b": "defensive zone",
    }

    for pattern, replacement in abbrev_patterns:
        df["description_clean"] = df["description_clean"].str.replace(pattern, replacement, regex=True)
    for pattern, replacement in zone_patterns:
        df["description_clean"] = df["description_clean"].str.replace(pattern, replacement, regex=True)
    for old, new in replacements.items():
        df["description_clean"] = df["description_clean"].str.replace(old, new, regex=True, case=False)

    generic_removals = [
        r";\s*original\s+description\s*:.*$",
        r"\bsuccessful\b",
        r"\bmiss\s+pass\b",
        r"\bmiss\s+shot\b",
        r"\bblocked\b",
        r"\bmissed\b",
    ]
    for pattern in generic_removals:
        df["description_clean"] = df["description_clean"].str.replace(pattern, " ", regex=True, case=False)

    description_clean = (
        df["description_clean"]
        .astype("string")
        .str.replace(LEAK_REGEX, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["description_clean"] = description_clean.replace("", pd.NA)

    flags_series = df["flags_clean"].fillna("").astype(str)
    flags_tokens = (
        flags_series.map(_split_clean_flags_cached)
        .map(_filter_non_leak_tokens)
        .map(_finalize_flag_tokens)
    )
    df["flags_clean"] = flags_tokens.map(lambda toks: ", ".join(toks) if toks else np.nan)

    df["embedding_text_clean"] = (df["description_clean"].fillna("") + " ; " + df["flags_clean"].fillna(""))
    df["embedding_text_clean"] = df["embedding_text_clean"].str.strip(" ;")
    df.loc[df["embedding_text_clean"] == "", "embedding_text_clean"] = np.nan

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    unique_flags = sorted(set(tok for toks in flags_tokens for tok in toks))
    write_csv(diagnostics_dir / "unique_flags_clean_tokens.csv", pd.DataFrame({"flag_token_clean": unique_flags}))

    unique_desc = (
        df["description_clean"].dropna().astype(str).drop_duplicates().sort_values().rename("description_clean")
    )
    unique_desc.to_csv(diagnostics_dir / "description_unique_clean.csv", index=False)

    description_clean_text = df["description_clean"].fillna("").astype(str)
    flags_clean_text = df["flags_clean"].fillna("").astype(str)

    summary = {
        "unique_description_clean": int(df["description_clean"].nunique(dropna=True)),
        "unique_flags_clean_tokens": int(len(unique_flags)),
        "leak_hits_description_clean": int(description_clean_text.str.contains(LEAK_REGEX, na=False).sum()),
        "leak_hits_flags_clean": int(flags_clean_text.str.contains(LEAK_REGEX, na=False).sum()),
        "forbidden_description_hits": {
            token: int(description_clean_text.str.contains(re.escape(token), regex=True, na=False).sum())
            for token in FORBIDDEN_DESCRIPTION_TERMS
        },
    }
    return df, summary


def _infer_goalie_ids(players: pd.DataFrame | None) -> set[str]:
    if players is None or players.empty or "player_id" not in players.columns:
        return set()

    pos_cols = [c for c in players.columns if "position" in c.lower()]
    if not pos_cols:
        return set()

    pos_col = pos_cols[0]
    pos = players[pos_col].astype(str).str.upper()
    mask = pos.str.contains(r"(?:^|\b)G(?:\b|$)|GOAL", regex=True, na=False)
    ids = players.loc[mask, "player_id"].dropna().astype(str).unique().tolist()
    return set(ids)


def merge_stints_and_goalies(df_events: pd.DataFrame, paths: DataPrepPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    stints = read_parquet(paths.raw_stints_path)
    games = read_parquet(paths.raw_games_path, columns=["game_id", "home_team_id", "away_team_id"])
    players = read_parquet(paths.raw_players_path) if paths.raw_players_path.exists() else None

    require_columns(stints, ["game_id", "game_stint"], "stints")
    require_columns(games, ["game_id", "home_team_id", "away_team_id"], "games")

    df = df.merge(games, on="game_id", how="left")

    keep_cols = [
        "game_id",
        "period",
        "period_time_start",
        "period_time_end",
        "game_stint",
        "n_home_skaters",
        "n_away_skaters",
        "is_home_net_empty",
        "is_away_net_empty",
        "home_score",
        "away_score",
    ]
    cols = [c for c in keep_cols if c in stints.columns]
    stints_base = stints[cols].copy()

    dedupe_keys = [c for c in ["game_id", "period", "game_stint"] if c in stints_base.columns]
    if dedupe_keys:
        stints_base = stints_base.drop_duplicates(subset=dedupe_keys, keep="first")

    merge_keys = [c for c in ["game_id", "period", "game_stint"] if c in stints_base.columns and c in df.columns]
    if merge_keys:
        df = df.merge(stints_base, on=merge_keys, how="left")

    if "home_score" in df.columns and "away_score" in df.columns:
        df["score_differential_home"] = pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(
            df["away_score"], errors="coerce"
        )
        df["score_differential_away"] = pd.to_numeric(df["away_score"], errors="coerce") - pd.to_numeric(
            df["home_score"], errors="coerce"
        )

    goalie_ids = _infer_goalie_ids(players)
    df["home_goalie_id"] = np.nan
    df["away_goalie_id"] = np.nan

    if goalie_ids and all(c in stints.columns for c in ["game_id", "game_stint", "team_id", "player_id"]):
        roster = stints[["game_id", "game_stint", "team_id", "player_id"]].copy()
        roster = roster.loc[roster["player_id"].astype(str).isin(goalie_ids)].copy()
        roster = roster.dropna(subset=["team_id", "player_id"])
        roster = roster.drop_duplicates(subset=["game_id", "game_stint", "team_id", "player_id"])

        home_lookup = df[["game_id", "game_stint", "home_team_id"]].drop_duplicates().rename(
            columns={"home_team_id": "team_id"}
        )
        away_lookup = df[["game_id", "game_stint", "away_team_id"]].drop_duplicates().rename(
            columns={"away_team_id": "team_id"}
        )

        home_goalie = (
            home_lookup.merge(roster, on=["game_id", "game_stint", "team_id"], how="left")
            .dropna(subset=["player_id"])
            .drop_duplicates(subset=["game_id", "game_stint"], keep="first")
            .rename(columns={"player_id": "home_goalie_id"})[["game_id", "game_stint", "home_goalie_id"]]
        )
        away_goalie = (
            away_lookup.merge(roster, on=["game_id", "game_stint", "team_id"], how="left")
            .dropna(subset=["player_id"])
            .drop_duplicates(subset=["game_id", "game_stint"], keep="first")
            .rename(columns={"player_id": "away_goalie_id"})[["game_id", "game_stint", "away_goalie_id"]]
        )

        df = df.drop(columns=["home_goalie_id", "away_goalie_id"])
        df = df.merge(home_goalie, on=["game_id", "game_stint"], how="left")
        df = df.merge(away_goalie, on=["game_id", "game_stint"], how="left")

    summary = {
        "rows": int(len(df)),
        "missing_stints_rows": int(df["game_stint"].isna().sum()) if "game_stint" in df.columns else int(len(df)),
        "home_goalie_rows": int(df["home_goalie_id"].notna().sum()),
        "away_goalie_rows": int(df["away_goalie_id"].notna().sum()),
    }
    return df, summary


def add_opp_goalie_id(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df_events.copy()

    for col in ["team_id", "home_team_id", "away_team_id", "home_goalie_id", "away_goalie_id"]:
        if col not in out.columns:
            out[col] = np.nan

    team = out["team_id"]
    home_team = out["home_team_id"]
    away_team = out["away_team_id"]

    out["opp_goalie_id"] = np.where(
        team.eq(home_team),
        out["away_goalie_id"],
        np.where(team.eq(away_team), out["home_goalie_id"], np.nan),
    )

    summary = {
        "rows_total": int(len(out)),
        "rows_with_opp_goalie_id": int(out["opp_goalie_id"].notna().sum()),
        "rows_missing_opp_goalie_id": int(out["opp_goalie_id"].isna().sum()),
    }
    return out, summary


def _ensure_is_goal_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    out = df.copy()

    if "is_goal" in out.columns:
        is_goal_num = pd.to_numeric(out["is_goal"], errors="coerce").fillna(0)
        out["is_goal"] = is_goal_num.astype(np.int8)
        return out, is_goal_num.eq(1), "input_column"

    event_type_norm = _safe_text_series(out["event_type"]) if "event_type" in out.columns else pd.Series("", index=out.index)
    derived_is_goal = event_type_norm.eq("goal")

    if all(c in out.columns for c in ["game_id", "sequence_id", "team_id"]):
        same_game = out["game_id"].eq(out["game_id"].shift(-1))
        same_sequence = out["sequence_id"].eq(out["sequence_id"].shift(-1))
        same_team = out["team_id"].eq(out["team_id"].shift(-1))
        next_is_goal = event_type_norm.shift(-1, fill_value="").eq("goal")

        scoring_shot_mask = (
            event_type_norm.isin(SAVE_SOURCE_EVENT_TYPES)
            & next_is_goal
            & same_game
            & same_sequence
            & same_team
        )
        derived_is_goal = derived_is_goal | scoring_shot_mask

    idx = out.index
    flags = _safe_text_series(out["flags"]) if "flags" in out.columns else pd.Series("", index=idx)
    detail = _safe_text_series(out["detail"]) if "detail" in out.columns else pd.Series("", index=idx)
    desc = _safe_text_series(out["description"]) if "description" in out.columns else pd.Series("", index=idx)
    goal_tagged_source = event_type_norm.isin(SAVE_SOURCE_EVENT_TYPES) & (
        flags.str.contains(GOAL_SOURCE_REGEX, na=False)
        | detail.str.contains(GOAL_SOURCE_REGEX, na=False)
        | desc.str.contains(GOAL_SOURCE_REGEX, na=False)
    )
    derived_is_goal = derived_is_goal | goal_tagged_source

    out["is_goal"] = derived_is_goal.astype(np.int8)
    return out, derived_is_goal, "derived_from_sequence_and_text"


def _on_net_source_mask(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    flags = _safe_text_series(df["flags"]) if "flags" in df.columns else pd.Series("", index=idx)
    detail = _safe_text_series(df["detail"]) if "detail" in df.columns else pd.Series("", index=idx)
    desc = _safe_text_series(df["description"]) if "description" in df.columns else pd.Series("", index=idx)

    return (
        flags.str.contains(ON_NET_REGEX, na=False)
        | detail.str.contains(ON_NET_REGEX, na=False)
        | desc.str.contains(ON_NET_REGEX, na=False)
    )


def _link_goals_to_latest_source_attempt(
    df: pd.DataFrame,
    *,
    event_type_norm: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()

    if "goal_linked_source_sl_event_id" not in out.columns:
        out["goal_linked_source_sl_event_id"] = np.nan
    if "source_linked_goal_sl_event_id" not in out.columns:
        out["source_linked_goal_sl_event_id"] = np.nan
    out["goal_converted_source"] = np.int8(0)

    if not all(c in out.columns for c in ["game_id", "sequence_id"]):
        goal_rows = int(event_type_norm.eq("goal").sum())
        return out, {
            "goal_rows_canonical": goal_rows,
            "goals_linked_to_source": 0,
            "goals_without_shot_source": goal_rows,
            "goal_converted_attempts": 0,
            "linkage_skipped_missing_keys": 1,
        }

    sort_cols = [c for c in ["game_id", "sequence_id", "period", "period_time", "sl_event_id"] if c in out.columns]
    ordered = out.copy()
    ordered["_orig_idx"] = ordered.index

    if sort_cols:
        ordered["_sort_sl_event_id"] = pd.to_numeric(ordered.get("sl_event_id"), errors="coerce")
        ordered = ordered.sort_values(
            by=sort_cols + ["_sort_sl_event_id", "_orig_idx"],
            kind="mergesort",
            na_position="last",
        )

    ordered_event_type = event_type_norm.loc[ordered["_orig_idx"]]
    grouped = ordered.groupby(["game_id", "sequence_id"], sort=False, dropna=False).groups

    goals_linked_to_source = 0
    goals_without_shot_source = 0
    goal_rows_canonical = int(event_type_norm.eq("goal").sum())

    source_linked_goal_sl_event_id = pd.to_numeric(out.get("source_linked_goal_sl_event_id"), errors="coerce")
    goal_linked_source_sl_event_id = pd.to_numeric(out.get("goal_linked_source_sl_event_id"), errors="coerce")
    goal_converted_source = pd.Series(np.int8(0), index=out.index)

    goal_row_indices: list[int] = []
    goal_linked_source_values: list[float] = []
    source_row_indices: list[int] = []
    source_linked_goal_values: list[float] = []

    for group_index in grouped.values():
        idx_list = ordered.loc[group_index, "_orig_idx"].tolist()

        # Stack captures source attempts seen so far in sequence order.
        source_stack: list[tuple[int, float]] = []

        for idx in idx_list:
            event_type = ordered_event_type.loc[idx]

            if event_type in SAVE_SOURCE_EVENT_TYPES:
                source_sl_event_id = pd.to_numeric(out.at[idx, "sl_event_id"], errors="coerce") if "sl_event_id" in out.columns else np.nan
                source_stack.append((int(idx), source_sl_event_id))
                continue

            if event_type != "goal":
                continue

            if not source_stack:
                goals_without_shot_source += 1
                continue

            source_idx, source_sl_event_id = source_stack.pop()
            goal_sl_event_id = pd.to_numeric(out.at[idx, "sl_event_id"], errors="coerce") if "sl_event_id" in out.columns else np.nan

            goal_row_indices.append(int(idx))
            goal_linked_source_values.append(source_sl_event_id)
            source_row_indices.append(source_idx)
            source_linked_goal_values.append(goal_sl_event_id)

            goals_linked_to_source += 1

    if goal_row_indices:
        goal_linked_source_sl_event_id.loc[goal_row_indices] = pd.Series(goal_linked_source_values, index=goal_row_indices).values
    if source_row_indices:
        source_linked_goal_sl_event_id.loc[source_row_indices] = pd.Series(source_linked_goal_values, index=source_row_indices).values
        goal_converted_source.loc[source_row_indices] = np.int8(1)

    out["goal_linked_source_sl_event_id"] = goal_linked_source_sl_event_id
    out["source_linked_goal_sl_event_id"] = source_linked_goal_sl_event_id
    out["goal_converted_source"] = goal_converted_source.astype(np.int8)

    return out, {
        "goal_rows_canonical": int(goal_rows_canonical),
        "goals_linked_to_source": int(goals_linked_to_source),
        "goals_without_shot_source": int(goals_without_shot_source),
        "goal_converted_attempts": int(goal_converted_source.sum()),
        "linkage_skipped_missing_keys": 0,
    }


def inject_synthetic_successful_saves(df_events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df_events.copy()

    if "is_synthetic_save" not in work.columns:
        work["is_synthetic_save"] = 0
    if "save_outcome" not in work.columns:
        work["save_outcome"] = pd.NA
    if "save_source_sl_event_id" not in work.columns:
        work["save_source_sl_event_id"] = np.nan
    if "save_source_event_type" not in work.columns:
        work["save_source_event_type"] = pd.NA

    work, is_goal, is_goal_mode = _ensure_is_goal_column(work)

    event_type_norm = _safe_text_series(work["event_type"]) if "event_type" in work.columns else pd.Series("", index=work.index)
    work, linkage_summary = _link_goals_to_latest_source_attempt(work, event_type_norm=event_type_norm)

    is_source_event = event_type_norm.isin(SAVE_SOURCE_EVENT_TYPES)
    is_on_net = _on_net_source_mask(work)
    goal_converted_source = pd.to_numeric(work.get("goal_converted_source"), errors="coerce").fillna(0).eq(1)

    if "opp_goalie_id" in work.columns:
        opp_goalie = work["opp_goalie_id"]
        opp_goalie_missing = opp_goalie.isna() | opp_goalie.astype("string").str.strip().fillna("").eq("")
    else:
        opp_goalie_missing = pd.Series(True, index=work.index)

    is_empty_net_source_attempt = is_source_event & opp_goalie_missing
    work["is_empty_net_source_attempt"] = is_empty_net_source_attempt.astype(np.int8)

    successful_shot_deflection_mask = is_source_event & is_on_net
    empty_net_non_goal_on_net_attempts = successful_shot_deflection_mask & is_empty_net_source_attempt & (~goal_converted_source)

    # Exclude goal-converted and empty-net source attempts from synthetic save injection.
    eligible_mask = successful_shot_deflection_mask & (~goal_converted_source) & (~is_empty_net_source_attempt)

    source = work.loc[eligible_mask].copy()
    if source.empty:
        return work, {
            "goal_rows_canonical": int(linkage_summary["goal_rows_canonical"]),
            "goals_linked_to_source": int(linkage_summary["goals_linked_to_source"]),
            "goals_without_shot_source": int(linkage_summary["goals_without_shot_source"]),
            "goal_converted_attempts": int(linkage_summary["goal_converted_attempts"]),
            "linkage_skipped_missing_keys": int(linkage_summary["linkage_skipped_missing_keys"]),
            "successful_shot_deflection_count": int(successful_shot_deflection_mask.sum()),
            "empty_net_source_attempts": int(is_empty_net_source_attempt.sum()),
            "empty_net_goal_converted_attempts": int((is_empty_net_source_attempt & goal_converted_source).sum()),
            "empty_net_non_goal_on_net_attempts": int(empty_net_non_goal_on_net_attempts.sum()),
            "actual_goal_count": int(linkage_summary["goal_rows_canonical"]),
            "expected_injected_saves": 0,
            "actual_injected_saves": 0,
            "injected_save_count_delta": 0,
            "eligible_non_goal_on_net_shots": 0,
            "synthetic_saves_injected": 0,
            "injected_after_failed_block_or_pressure": 0,
            "non_numeric_source_sl_event_id_skipped": 0,
            "synthetic_id_collisions": 0,
        }

    # For shot/deflection -> failed block/pressure patterns, anchor the synthetic save
    # after the immediate failed defensive event while keeping source linkage on the shot.
    order_cols = [c for c in ["game_id", "sequence_id", "period", "period_time"] if c in work.columns]
    ordered = work.copy()
    ordered["_row_idx"] = ordered.index
    ordered["_sort_sl_event_id"] = pd.to_numeric(ordered.get("sl_event_id"), errors="coerce")
    ordered = ordered.sort_values(order_cols + ["_sort_sl_event_id", "_row_idx"], kind="mergesort", na_position="last")

    if "sequence_id" in ordered.columns:
        shift_group = ordered.groupby(["game_id", "sequence_id"], sort=False, dropna=False)
    else:
        shift_group = ordered.groupby(["game_id"], sort=False, dropna=False)

    if "event_type" in ordered.columns:
        ordered["_next_event_type"] = _safe_text_series(shift_group["event_type"].shift(-1))
    else:
        ordered["_next_event_type"] = ""

    if "outcome" in ordered.columns:
        ordered["_next_outcome"] = _safe_text_series(shift_group["outcome"].shift(-1))
    else:
        ordered["_next_outcome"] = ""

    ordered["_next_period_time"] = pd.to_numeric(shift_group["period_time"].shift(-1), errors="coerce") if "period_time" in ordered.columns else np.nan
    next_context = ordered.set_index("_row_idx")[["_next_event_type", "_next_outcome", "_next_period_time"]]

    source = source.join(next_context, how="left")
    source_next_failed_defensive = (
        source["_next_event_type"].isin({"block", "pressure"})
        & source["_next_outcome"].eq("failed")
        & source["_next_period_time"].notna()
    )

    source_sl_num = pd.to_numeric(source["sl_event_id"], errors="coerce")
    valid_id_mask = source_sl_num.notna()
    skipped = int((~valid_id_mask).sum())

    source = source.loc[valid_id_mask].copy()
    source_next_failed_defensive = source_next_failed_defensive.loc[valid_id_mask]
    source_sl_num = source_sl_num.loc[valid_id_mask]
    if source.empty:
        expected_injected_saves = int(eligible_mask.sum()) - int(skipped)
        return work, {
            "goal_rows_canonical": int(linkage_summary["goal_rows_canonical"]),
            "goals_linked_to_source": int(linkage_summary["goals_linked_to_source"]),
            "goals_without_shot_source": int(linkage_summary["goals_without_shot_source"]),
            "goal_converted_attempts": int(linkage_summary["goal_converted_attempts"]),
            "linkage_skipped_missing_keys": int(linkage_summary["linkage_skipped_missing_keys"]),
            "successful_shot_deflection_count": int(successful_shot_deflection_mask.sum()),
            "empty_net_source_attempts": int(is_empty_net_source_attempt.sum()),
            "empty_net_goal_converted_attempts": int((is_empty_net_source_attempt & goal_converted_source).sum()),
            "empty_net_non_goal_on_net_attempts": int(empty_net_non_goal_on_net_attempts.sum()),
            "actual_goal_count": int(linkage_summary["goal_rows_canonical"]),
            "expected_injected_saves": int(expected_injected_saves),
            "actual_injected_saves": 0,
            "injected_save_count_delta": int(expected_injected_saves),
            "eligible_non_goal_on_net_shots": int(eligible_mask.sum()),
            "synthetic_saves_injected": 0,
            "injected_after_failed_block_or_pressure": 0,
            "non_numeric_source_sl_event_id_skipped": skipped,
            "synthetic_id_collisions": 0,
        }

    saves = source.copy()

    team = source["team_id"] if "team_id" in source.columns else pd.Series(np.nan, index=source.index)
    home_team = source["home_team_id"] if "home_team_id" in source.columns else pd.Series(np.nan, index=source.index)
    away_team = source["away_team_id"] if "away_team_id" in source.columns else pd.Series(np.nan, index=source.index)
    defending_team = np.where(team.eq(home_team), away_team, np.where(team.eq(away_team), home_team, np.nan))

    source_period_time = pd.to_numeric(source["period_time"], errors="coerce")
    save_anchor_period_time = source_period_time.where(~source_next_failed_defensive, source["_next_period_time"])

    saves["sl_event_id"] = source_sl_num + 0.5
    saves["period_time"] = save_anchor_period_time + 0.1
    saves["event_type"] = "save"
    saves["event_type_clean"] = "save"
    saves["is_goal"] = 0
    saves["is_synthetic_save"] = 1
    saves["save_outcome"] = "successful"
    saves["outcome"] = "successful"
    saves["save_source_sl_event_id"] = source_sl_num.astype(float)
    saves["save_source_event_type"] = source["event_type"].astype(str).str.lower().str.strip()
    saves["save_insertion_anchor_event_type"] = np.where(
        source_next_failed_defensive,
        source["_next_event_type"],
        source["event_type"].astype(str).str.lower().str.strip(),
    )
    saves["team_id"] = defending_team
    saves["player_id"] = source.get("opp_goalie_id", pd.Series(np.nan, index=source.index))
    saves["goalie_id"] = source.get("opp_goalie_id", pd.Series(np.nan, index=source.index))
    saves["x_adj"] = -89.0
    saves["y_adj"] = 0.0

    combined = pd.concat([work, saves], ignore_index=True, sort=False)
    combined["_sort_sl_event_id"] = pd.to_numeric(combined["sl_event_id"], errors="coerce")

    sort_keys = [c for c in ["game_id", "period", "sequence_id", "period_time"] if c in combined.columns]
    sort_keys.append("_sort_sl_event_id")
    combined = combined.sort_values(sort_keys, kind="mergesort").drop(columns=["_sort_sl_event_id"]).reset_index(drop=True)

    if "sequence_id" in combined.columns:
        combined["sequence_event_id"] = combined.groupby("sequence_id").cumcount() + 1
    else:
        combined["sequence_event_id"] = 1

    if "game_id" in combined.columns:
        combined["game_event_id"] = combined.groupby("game_id").cumcount() + 1

    if "game_id" in combined.columns:
        dup_mask = combined.duplicated(subset=["game_id", "sl_event_id"], keep=False)
    else:
        dup_mask = combined.duplicated(subset=["sl_event_id"], keep=False)

    synthetic_collision_rows = combined.loc[dup_mask & combined["is_synthetic_save"].eq(1)]
    collision_count = int(len(synthetic_collision_rows))
    if collision_count > 0:
        raise ValueError(f"Synthetic sl_event_id collision detected for {collision_count} injected save rows")

    summary = {
        "is_goal_mode": is_goal_mode,
        "is_goal_positive_rows": int(pd.to_numeric(work["is_goal"], errors="coerce").fillna(0).eq(1).sum()),
        "goal_rows_canonical": int(linkage_summary["goal_rows_canonical"]),
        "goals_linked_to_source": int(linkage_summary["goals_linked_to_source"]),
        "goals_without_shot_source": int(linkage_summary["goals_without_shot_source"]),
        "goal_converted_attempts": int(linkage_summary["goal_converted_attempts"]),
        "linkage_skipped_missing_keys": int(linkage_summary["linkage_skipped_missing_keys"]),
        "successful_shot_deflection_count": int(successful_shot_deflection_mask.sum()),
        "empty_net_source_attempts": int(is_empty_net_source_attempt.sum()),
        "empty_net_goal_converted_attempts": int((is_empty_net_source_attempt & goal_converted_source).sum()),
        "empty_net_non_goal_on_net_attempts": int(empty_net_non_goal_on_net_attempts.sum()),
        "actual_goal_count": int(linkage_summary["goal_rows_canonical"]),
        "expected_injected_saves": int(eligible_mask.sum() - skipped),
        "actual_injected_saves": int(len(saves)),
        "injected_save_count_delta": int((eligible_mask.sum() - skipped) - len(saves)),
        "eligible_non_goal_on_net_shots": int(eligible_mask.sum()),
        "synthetic_saves_injected": int(len(saves)),
        "injected_after_failed_block_or_pressure": int(source_next_failed_defensive.sum()),
        "non_numeric_source_sl_event_id_skipped": skipped,
        "synthetic_id_collisions": collision_count,
    }
    return combined, summary


def reclassify_defensive_deflections(
    df_events: pd.DataFrame,
    *,
    lookback_events: int = DEFENSIVE_DEFLECTION_LOOKBACK_EVENTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df_events.copy()

    required = ["game_id", "sequence_id", "event_type", "team_id"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        return work, {
            "lookback_window_events": int(lookback_events),
            "deflection_rows_scanned": 0,
            "defensive_deflections_reclassified": 0,
            "lookback_shot_hits": 0,
            "lookback_reset_stops": 0,
            "lookback_no_prior_shot": 0,
            "lookback_same_team_shot": 0,
            "lookback_missing_team_values": 0,
            "skipped_missing_required_columns": 1,
        }

    event_type_norm = _safe_text_series(work["event_type"])
    reclass_mask = pd.Series(False, index=work.index)

    deflection_rows_scanned = 0
    lookback_shot_hits = 0
    lookback_reset_stops = 0
    lookback_no_prior_shot = 0
    lookback_same_team_shot = 0
    lookback_missing_team_values = 0

    grouped = work.groupby(["game_id", "sequence_id"], sort=False, dropna=False).groups
    for group_index in grouped.values():
        idx_list = list(group_index)
        group_types = event_type_norm.loc[idx_list].tolist()
        group_teams = work.loc[idx_list, "team_id"].tolist()
        deflection_positions = [pos for pos, et in enumerate(group_types) if et == "deflection"]

        for pos in deflection_positions:

            deflection_rows_scanned += 1
            found_shot = False
            stopped_on_reset = False

            for offset in range(1, int(lookback_events) + 1):
                prev_pos = pos - offset
                if prev_pos < 0:
                    break

                prev_type = str(group_types[prev_pos]).strip().lower()
                if prev_type in DEFENSIVE_DEFLECTION_RESET_EVENTS:
                    lookback_reset_stops += 1
                    stopped_on_reset = True
                    break

                if prev_type == "shot":
                    lookback_shot_hits += 1
                    found_shot = True
                    shot_team = group_teams[prev_pos]
                    deflection_team = group_teams[pos]

                    if pd.isna(shot_team) or pd.isna(deflection_team):
                        lookback_missing_team_values += 1
                    elif shot_team != deflection_team:
                        reclass_mask.at[idx_list[pos]] = True
                    else:
                        lookback_same_team_shot += 1
                    break

            if (not found_shot) and (not stopped_on_reset):
                lookback_no_prior_shot += 1

    if reclass_mask.any():
        work.loc[reclass_mask, "event_type"] = "defensive_deflection"
        if "event_type_clean" in work.columns:
            work.loc[reclass_mask, "event_type_clean"] = "defensive_deflection"

    summary = {
        "lookback_window_events": int(lookback_events),
        "deflection_rows_scanned": int(deflection_rows_scanned),
        "defensive_deflections_reclassified": int(reclass_mask.sum()),
        "lookback_shot_hits": int(lookback_shot_hits),
        "lookback_reset_stops": int(lookback_reset_stops),
        "lookback_no_prior_shot": int(lookback_no_prior_shot),
        "lookback_same_team_shot": int(lookback_same_team_shot),
        "lookback_missing_team_values": int(lookback_missing_team_values),
        "skipped_missing_required_columns": 0,
    }
    return work, summary


def _extract_penalty_drawer_sidecar(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    work = df_events.copy()
    event_type = _safe_text_series(work["event_type"]) if "event_type" in work.columns else pd.Series("", index=work.index)

    penalty_mask = event_type.eq("penalty")
    penaltydrawn_mask = event_type.eq("penaltydrawn")

    sidecar_cols = [
        "game_id",
        "sl_event_id",
        "kept_sl_event_id",
        "sequence_id",
        "period",
        "period_time",
        "penaltydrawn_player_id",
        "penaltydrawn_team_id",
    ]

    penalties = work.loc[penalty_mask].copy()
    penaltydrawn = work.loc[penaltydrawn_mask].copy()
    if penaltydrawn.empty:
        return work, pd.DataFrame(columns=sidecar_cols), {
            "penaltydrawn_rows": 0,
            "mapped_rows": 0,
            "unmapped_rows": 0,
            "rows_removed_from_main": 0,
        }

    penaltydrawn = penaltydrawn.rename(
        columns={
            "player_id": "penaltydrawn_player_id",
            "team_id": "penaltydrawn_team_id",
        }
    )
    penaltydrawn["_pd_row_id"] = np.arange(len(penaltydrawn), dtype=np.int64)
    penaltydrawn["_pd_event_num"] = pd.to_numeric(penaltydrawn["sl_event_id"], errors="coerce")

    penalties = penalties.copy()
    penalties["kept_sl_event_id"] = pd.to_numeric(penalties["sl_event_id"], errors="coerce")

    mapped = penaltydrawn[["_pd_row_id"]].copy()
    mapped["kept_sl_event_id"] = np.nan

    by_keys = ["game_id"]
    if "sequence_id" in penaltydrawn.columns and "sequence_id" in penalties.columns:
        by_keys.append("sequence_id")

    lhs = penaltydrawn.loc[penaltydrawn["_pd_event_num"].notna()].copy()
    rhs = penalties.loc[penalties["kept_sl_event_id"].notna()].copy()
    if not lhs.empty and not rhs.empty:
        nearest_parts: list[pd.DataFrame] = []
        for key_vals, lhs_grp in lhs.groupby(by_keys, sort=False, dropna=False):
            key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)

            rhs_grp = rhs
            for key_col, key_val in zip(by_keys, key_tuple):
                if pd.isna(key_val):
                    rhs_grp = rhs_grp.loc[rhs_grp[key_col].isna()]
                else:
                    rhs_grp = rhs_grp.loc[rhs_grp[key_col] == key_val]

            if rhs_grp.empty:
                continue

            lhs_sorted = lhs_grp[["_pd_row_id", "_pd_event_num"]].sort_values("_pd_event_num", kind="mergesort")
            rhs_sorted = rhs_grp[["kept_sl_event_id"]].sort_values("kept_sl_event_id", kind="mergesort")

            nearest_grp = pd.merge_asof(
                lhs_sorted,
                rhs_sorted,
                left_on="_pd_event_num",
                right_on="kept_sl_event_id",
                direction="nearest",
            )[["_pd_row_id", "kept_sl_event_id"]]
            nearest_parts.append(nearest_grp)

        if nearest_parts:
            nearest = pd.concat(nearest_parts, ignore_index=True)
            mapped = mapped.drop(columns=["kept_sl_event_id"]).merge(nearest, on="_pd_row_id", how="left")

    penaltydrawn = penaltydrawn.merge(mapped, on="_pd_row_id", how="left", suffixes=("", "_mapped"))
    if "kept_sl_event_id_mapped" in penaltydrawn.columns:
        penaltydrawn["kept_sl_event_id"] = penaltydrawn["kept_sl_event_id_mapped"]
        penaltydrawn = penaltydrawn.drop(columns=["kept_sl_event_id_mapped"])

    sidecar = penaltydrawn.reindex(columns=sidecar_cols)

    kept = work.loc[~penaltydrawn_mask].copy()
    summary = {
        "penaltydrawn_rows": int(len(sidecar)),
        "mapped_rows": int(sidecar["kept_sl_event_id"].notna().sum()),
        "unmapped_rows": int(sidecar["kept_sl_event_id"].isna().sum()),
        "rows_removed_from_main": int(penaltydrawn_mask.sum()),
    }
    return kept, sidecar, summary


def _extract_faceoff_loser_sidecar(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    work = df_events.copy()
    event_type = _safe_text_series(work["event_type"]) if "event_type" in work.columns else pd.Series("", index=work.index)
    faceoff_mask = event_type.eq("faceoff")

    sidecar_cols = [
        "game_id",
        "sl_event_id",
        "kept_sl_event_id",
        "sequence_id",
        "period",
        "period_time",
        "opposing_player_id",
        "opposing_team_id",
    ]

    faceoffs = work.loc[faceoff_mask].copy()
    if faceoffs.empty:
        return work, pd.DataFrame(columns=sidecar_cols), {
            "faceoff_rows": 0,
            "winner_rows_kept": 0,
            "loser_rows_sidecar": 0,
        }

    group_keys = [c for c in ["game_id", "period", "sequence_id"] if c in faceoffs.columns]
    if not group_keys:
        group_keys = ["game_id"]

    outcome = _safe_text_series(faceoffs["outcome"]) if "outcome" in faceoffs.columns else pd.Series("", index=faceoffs.index)
    faceoffs["_is_successful"] = outcome.eq("successful").astype(np.int8)
    faceoffs["_has_player"] = faceoffs["player_id"].notna().astype(np.int8) if "player_id" in faceoffs.columns else 0
    faceoffs["_sl_event_num"] = pd.to_numeric(faceoffs["sl_event_id"], errors="coerce")

    faceoffs = faceoffs.sort_values(
        by=group_keys + ["_is_successful", "_has_player", "_sl_event_num"],
        ascending=[True] * len(group_keys) + [False, False, True],
        kind="mergesort",
        na_position="last",
    )
    faceoffs["_faceoff_rank"] = faceoffs.groupby(group_keys, sort=False).cumcount()

    winners = faceoffs.loc[faceoffs["_faceoff_rank"].eq(0), group_keys + ["sl_event_id"]].rename(
        columns={"sl_event_id": "kept_sl_event_id"}
    )
    losers = faceoffs.loc[faceoffs["_faceoff_rank"].gt(0)].copy()
    losers = losers.merge(winners, on=group_keys, how="left")
    losers = losers.rename(columns={"player_id": "opposing_player_id", "team_id": "opposing_team_id"})

    sidecar = losers.reindex(columns=sidecar_cols)

    loser_keys = losers[["game_id", "sl_event_id"]].copy() if all(
        c in losers.columns for c in ["game_id", "sl_event_id"]
    ) else pd.DataFrame(columns=["game_id", "sl_event_id"])
    if not loser_keys.empty and all(c in work.columns for c in ["game_id", "sl_event_id"]):
        loser_keys = loser_keys.drop_duplicates()
        work = work.merge(loser_keys.assign(_drop_faceoff_loser=1), on=["game_id", "sl_event_id"], how="left")
        keep_mask = work["_drop_faceoff_loser"].fillna(0).eq(0)
        kept = work.loc[keep_mask].drop(columns=["_drop_faceoff_loser"])
    else:
        kept = work

    summary = {
        "faceoff_rows": int(len(faceoffs)),
        "winner_rows_kept": int(len(winners)),
        "loser_rows_sidecar": int(len(sidecar)),
    }
    return kept, sidecar, summary


def standardize_sidecar_contracts(
    df_events: pd.DataFrame,
    paths: DataPrepPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with_faceoff_main, faceoff_reference, faceoff_summary = _extract_faceoff_loser_sidecar(df_events)
    with_penalty_main, penalty_reference, penalty_summary = _extract_penalty_drawer_sidecar(with_faceoff_main)

    write_parquet(paths.phase2_faceoff_reference_output, faceoff_reference)
    write_parquet(paths.phase2_penalty_reference_output, penalty_reference)

    summary = {
        "faceoff_reference_rows": int(len(faceoff_reference)),
        "penalty_reference_rows": int(len(penalty_reference)),
        "faceoff_reference_output": str(paths.phase2_faceoff_reference_output),
        "penalty_reference_output": str(paths.phase2_penalty_reference_output),
        "faceoff": faceoff_summary,
        "penalty": penalty_summary,
    }
    return with_penalty_main, summary


def enforce_event_order(df_events: pd.DataFrame, cfg: DataPrepConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()
    if "period_time" in df.columns:
        df["period_time"] = pd.to_numeric(df["period_time"], errors="coerce")

    event_type_norm = df["event_type"].astype(str).str.lower().str.strip()
    df["is_end_of_period_event"] = event_type_norm.eq("end_of_period").astype(np.int8)
    if cfg.normalize_end_of_period_to_whistle:
        df.loc[event_type_norm.eq("end_of_period"), "event_type"] = "whistle"

    df["_ingest_order"] = np.arange(len(df), dtype=np.int64)
    seq_keys = [c for c in ["game_id", "period", "sequence_id"] if c in df.columns]
    if not seq_keys:
        seq_keys = ["game_id"]

    if "period_time" in df.columns:
        delta = df.groupby(seq_keys, sort=False)["period_time"].diff()
        abnormal_mask = delta < cfg.sequence_disorder_threshold_seconds
    else:
        abnormal_mask = pd.Series(False, index=df.index)

    n_abnormal = int(abnormal_mask.sum())

    if n_abnormal > 0:
        sort_keys = [c for c in ["game_id", "period", "sequence_id", "period_time", "sl_event_id"] if c in df.columns]
        df = df.sort_values(by=sort_keys, kind="mergesort").reset_index(drop=True)
        order_mode = "corrective_temporal_sort"
    else:
        df = df.sort_values(by=["_ingest_order"], kind="mergesort").reset_index(drop=True)
        order_mode = "preserve_ingest_order"

    if all(c in df.columns for c in ["game_id", "sequence_id"]):
        df["sequence_event_id"] = df.groupby(["game_id", "sequence_id"], sort=False).cumcount() + 1
    else:
        df["sequence_event_id"] = 1

    if "game_id" in df.columns:
        df["game_event_id"] = df.groupby("game_id", sort=False).cumcount() + 1

    event_type_final = df["event_type"].astype(str).str.lower().str.strip()
    df["is_boundary_event"] = event_type_final.isin({"whistle", "goal"}).astype(np.int8)

    df["game_time_sec"] = np.nan
    if "period" in df.columns and "period_time" in df.columns:
        df["_period_elapsed_sec"] = df["period_time"]
        for (_, _), idx in df.groupby(["game_id", "period"]).groups.items():
            vals = df.loc[idx, "period_time"].to_numpy(dtype=float)
            if len(vals) <= 1:
                continue
            deltas = np.diff(vals)
            counts_down = (deltas < 0).sum() > (deltas > 0).sum()
            if counts_down:
                df.loc[idx, "_period_elapsed_sec"] = 1200.0 - vals

        period_idx = pd.to_numeric(df["period"], errors="coerce").fillna(1.0)
        df["game_time_sec"] = (period_idx - 1.0) * 1200.0 + pd.to_numeric(df["_period_elapsed_sec"], errors="coerce")

    drop_cols = [c for c in ["_ingest_order", "_period_elapsed_sec"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    summary = {
        "ordering_mode": order_mode,
        "abnormal_period_time_rows": n_abnormal,
        "boundary_events": int(df["is_boundary_event"].sum()),
        "end_of_period_normalized_rows": int(df["is_end_of_period_event"].sum()),
    }
    return df, summary


def add_event_spatial_features(
    df_events: pd.DataFrame,
    cfg: DataPrepConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_events.copy()

    x_adj = pd.to_numeric(df.get("x_adj"), errors="coerce")
    y_adj = pd.to_numeric(df.get("y_adj"), errors="coerce")
    x_abs = pd.to_numeric(df.get("x"), errors="coerce")
    y_abs = pd.to_numeric(df.get("y"), errors="coerce")

    # 1. Use pure absolute coordinates for transition physics. 
    # Mixing x_adj into x_abs without un-flipping it will cause massive distance spikes.
    x_transition = x_abs
    y_transition = y_abs

    # 2. Net-relative features use Adjusted coordinates (assuming attacking net is at X=89)
    distance_to_net_raw = np.sqrt((89.0 - x_adj) ** 2 + (0.0 - y_adj) ** 2)
    df["distance_to_net_event"] = distance_to_net_raw
    df["angle_to_net_event"] = np.arctan2(y_adj, 89.0 - x_adj)

    # 3. Transition features must group by PERIOD to prevent linking events across intermissions
    if "game_id" in df.columns and "period" in df.columns:
        # Group by Game AND Period
        prev_x = x_transition.groupby([df["game_id"], df["period"]], sort=False).shift(1)
        prev_y = y_transition.groupby([df["game_id"], df["period"]], sort=False).shift(1)
        prev_t = df.groupby([df["game_id"], df["period"]], sort=False)["game_time_sec"].shift(1)
    else:
        # Fallback if period data is completely missing
        prev_x = pd.Series(np.nan, index=df.index)
        prev_y = pd.Series(np.nan, index=df.index)
        prev_t = pd.Series(np.nan, index=df.index)

    dx = x_transition - pd.to_numeric(prev_x, errors="coerce")
    dy = y_transition - pd.to_numeric(prev_y, errors="coerce")
    dist = np.sqrt(dx**2 + dy**2)
    dt = pd.to_numeric(df.get("game_time_sec"), errors="coerce") - pd.to_numeric(prev_t, errors="coerce")

    # Canonical temporal delta for downstream models.
    time_since_last = pd.to_numeric(dt, errors="coerce").clip(lower=0.0)
    df["time_since_last_event"] = time_since_last

    df["distance_from_last_event"] = dist
    speed_raw = pd.Series(np.where(dt > 0, dist / dt, np.nan), index=df.index)
    df["angle_from_last_event"] = np.arctan2(dy, dx)

    event_type_norm = (
        df.get("event_type", pd.Series("", index=df.index))
        .astype(str)
        .str.lower()
        .str.strip()
    )
    is_save_row = event_type_norm.eq("save")
    # Synthetic save rows are injected with small time offsets; suppress transition speed on save events.
    speed_raw = speed_raw.mask(is_save_row, np.nan)
    df["speed_from_last_event"] = speed_raw

    # 4. Clipping outliers based on configuration
    max_dist_to_net = float(cfg.phase2_max_distance_to_net_feet)
    max_dist_from_last = float(cfg.phase2_max_distance_from_last_event_feet)
    max_speed_from_last = float(cfg.phase2_max_speed_from_last_event_ft_per_sec)

    df["distance_to_net_event"] = pd.to_numeric(df["distance_to_net_event"], errors="coerce").clip(
        lower=0.0,
        upper=max_dist_to_net,
    )
    # Canonical alias consumed by training.
    df["distance_to_net"] = pd.to_numeric(df["distance_to_net_event"], errors="coerce")
    df["distance_from_last_event"] = pd.to_numeric(df["distance_from_last_event"], errors="coerce").clip(
        lower=0.0,
        upper=max_dist_from_last,
    )
    df["speed_from_last_event"] = pd.to_numeric(df["speed_from_last_event"], errors="coerce").clip(
        lower=0.0,
        upper=max_speed_from_last,
    )

    # 5. Summarize capping for diagnostics
    dist_to_net_capped_rows = int((distance_to_net_raw > max_dist_to_net).sum())
    dist_from_last_capped_rows = int((dist > max_dist_from_last).sum())
    speed_from_last_capped_rows = int((pd.to_numeric(speed_raw, errors="coerce") > max_speed_from_last).sum())
    speed_missing_due_to_save_rows = int(is_save_row.sum())

    summary = {
        "distance_to_net_non_null": int(df["distance_to_net_event"].notna().sum()),
        "distance_to_net_alias_non_null": int(df["distance_to_net"].notna().sum()),
        "time_since_last_event_non_null": int(df["time_since_last_event"].notna().sum()),
        "transition_distance_non_null": int(df["distance_from_last_event"].notna().sum()),
        "transition_speed_non_null": int(df["speed_from_last_event"].notna().sum()),
        "transition_speed_save_rows_suppressed": speed_missing_due_to_save_rows,
        "transition_coordinate_source": "pure_absolute_xy",
        "distance_to_net_max_feet": max_dist_to_net,
        "distance_from_last_event_max_feet": max_dist_from_last,
        "speed_from_last_event_max_ft_per_sec": max_speed_from_last,
        "distance_to_net_capped_rows": dist_to_net_capped_rows,
        "distance_from_last_event_capped_rows": dist_from_last_capped_rows,
        "speed_from_last_event_capped_rows": speed_from_last_capped_rows,
    }
    return df, summary


def add_shot_goalie_angle_change(
    df_events: pd.DataFrame,
    *,
    reception_pass_lookback_events: int = 3,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df_events.copy()

    required_cols = ["event_type", "x_adj", "y_adj"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        df["goalie_angle_change"] = np.nan
        return df, {
            "shot_like_rows": 0,
            "shot_like_rows_with_prior_location": 0,
            "shot_like_rows_missing_prior_location": 0,
            "shot_like_rows_reception_prior": 0,
            "shot_like_rows_reception_pass_substitutions": 0,
            "shot_like_rows_reception_pass_not_found": 0,
            "skipped_missing_required_columns": 1,
        }

    group_cols = [c for c in ["game_id", "period", "sequence_id"] if c in df.columns]
    if not group_cols:
        group_cols = ["game_id"] if "game_id" in df.columns else []

    if not group_cols:
        df["goalie_angle_change"] = np.nan
        return df, {
            "shot_like_rows": 0,
            "shot_like_rows_with_prior_location": 0,
            "shot_like_rows_missing_prior_location": 0,
            "shot_like_rows_reception_prior": 0,
            "shot_like_rows_reception_pass_substitutions": 0,
            "shot_like_rows_reception_pass_not_found": 0,
            "skipped_missing_required_columns": 1,
        }

    ordered = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    order_cols = [c for c in ["game_event_id", "sequence_event_id", "period_time", "sl_event_id", "_orig_idx"] if c in ordered.columns]
    ordered = ordered.sort_values(group_cols + order_cols, kind="mergesort", na_position="last").reset_index(drop=True)

    event_type_norm = ordered["event_type"].astype(str).str.lower().str.strip()
    shot_like = event_type_norm.isin({"shot", "deflection", "defensive_deflection"})

    grp = ordered.groupby(group_cols, sort=False, dropna=False)
    ordered["_prev_event_type"] = grp["event_type"].shift(1).astype("string").str.lower().str.strip().fillna("")
    ordered["_prior_x"] = pd.to_numeric(grp["x_adj"].shift(1), errors="coerce")
    ordered["_prior_y"] = pd.to_numeric(grp["y_adj"].shift(1), errors="coerce")

    reception_prior_mask = shot_like & ordered["_prev_event_type"].eq("reception")
    reception_count = int(reception_prior_mask.sum())
    pass_substitutions = 0
    pass_not_found = 0

    dest_x = pd.to_numeric(ordered.get("dest_x_adj"), errors="coerce") if "dest_x_adj" in ordered.columns else pd.Series(np.nan, index=ordered.index)
    dest_y = pd.to_numeric(ordered.get("dest_y_adj"), errors="coerce") if "dest_y_adj" in ordered.columns else pd.Series(np.nan, index=ordered.index)
    x_adj = pd.to_numeric(ordered["x_adj"], errors="coerce")
    y_adj = pd.to_numeric(ordered["y_adj"], errors="coerce")

    if reception_count > 0:
        grouped_indices = ordered.groupby(group_cols, sort=False, dropna=False).indices
        event_type_arr = event_type_norm.to_numpy(dtype=object)
        prior_x_arr = ordered["_prior_x"].to_numpy(dtype=float, copy=True)
        prior_y_arr = ordered["_prior_y"].to_numpy(dtype=float, copy=True)
        dest_x_arr = dest_x.to_numpy(dtype=float, copy=True)
        dest_y_arr = dest_y.to_numpy(dtype=float, copy=True)
        x_adj_arr = x_adj.to_numpy(dtype=float, copy=True)
        y_adj_arr = y_adj.to_numpy(dtype=float, copy=True)
        reception_arr = reception_prior_mask.to_numpy(dtype=bool)

        for positions in grouped_indices.values():
            pos_arr = np.asarray(positions, dtype=np.int64)
            for local_pos, global_pos in enumerate(pos_arr):
                if not reception_arr[global_pos]:
                    continue

                found_pass = False
                for back in range(2, int(reception_pass_lookback_events) + 2):
                    cand_local = local_pos - back
                    if cand_local < 0:
                        break
                    cand_pos = int(pos_arr[cand_local])
                    if str(event_type_arr[cand_pos]).strip().lower() != "pass":
                        continue

                    px = dest_x_arr[cand_pos]
                    py = dest_y_arr[cand_pos]
                    if pd.isna(px) or pd.isna(py):
                        px = x_adj_arr[cand_pos]
                        py = y_adj_arr[cand_pos]

                    prior_x_arr[global_pos] = px
                    prior_y_arr[global_pos] = py
                    pass_substitutions += 1
                    found_pass = True
                    break

                if not found_pass:
                    pass_not_found += 1

        ordered["_prior_x"] = prior_x_arr
        ordered["_prior_y"] = prior_y_arr

    cur_angle = np.arctan2(y_adj, 89.0 - x_adj)
    prior_angle = np.arctan2(pd.to_numeric(ordered["_prior_y"], errors="coerce"), 89.0 - pd.to_numeric(ordered["_prior_x"], errors="coerce"))
    wrapped_delta = np.arctan2(np.sin(cur_angle - prior_angle), np.cos(cur_angle - prior_angle))

    valid_shot_prior = shot_like & ordered["_prior_x"].notna() & ordered["_prior_y"].notna() & x_adj.notna() & y_adj.notna()
    ordered["goalie_angle_change"] = np.nan
    ordered.loc[valid_shot_prior, "goalie_angle_change"] = np.abs(pd.to_numeric(wrapped_delta[valid_shot_prior], errors="coerce"))

    out = df.copy()
    out["goalie_angle_change"] = np.nan
    out.loc[ordered["_orig_idx"].to_numpy(), "goalie_angle_change"] = pd.to_numeric(ordered["goalie_angle_change"], errors="coerce").to_numpy()

    shot_like_rows = int(shot_like.sum())
    shot_like_with_prior = int(valid_shot_prior.sum())

    summary = {
        "shot_like_rows": shot_like_rows,
        "shot_like_rows_with_prior_location": shot_like_with_prior,
        "shot_like_rows_missing_prior_location": int(max(shot_like_rows - shot_like_with_prior, 0)),
        "shot_like_rows_reception_prior": reception_count,
        "shot_like_rows_reception_pass_substitutions": int(pass_substitutions),
        "shot_like_rows_reception_pass_not_found": int(pass_not_found),
        "skipped_missing_required_columns": 0,
    }
    return out, summary


def run_phase2_event_pipeline(
    base_dir: Path,
    *,
    run_label: str,
    config: DataPrepConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or DataPrepConfig()
    paths = DataPrepPaths(base_dir, run_label=run_label)
    paths.ensure_dirs()

    events = read_parquet(paths.phase1_events_output)
    require_columns(
        events,
        ["game_id", "sl_event_id", "event_type", "period", "period_time", "game_stint", "x_adj", "y_adj"],
        "phase1_events",
    )

    start_rows = len(events)
    events, leak_guard_summary = drop_phase1_leak_rows(events)
    events, stints_summary = merge_stints_and_goalies(events, paths)
    events, opp_goalie_summary = add_opp_goalie_id(events)
    events, save_injection_summary = inject_synthetic_successful_saves(events)
    events, sidecar_summary = standardize_sidecar_contracts(events, paths)
    events, defensive_deflection_summary = reclassify_defensive_deflections(events)
    events, text_summary = clean_event_text(events, paths.phase2_diagnostics_dir)
    events, order_summary = enforce_event_order(events, cfg)
    events, feature_summary = add_event_spatial_features(events, cfg)
    events, goalie_angle_summary = add_shot_goalie_angle_change(events, reception_pass_lookback_events=3)

    assert_no_duplicate_columns(events, "phase2_events")
    write_parquet(paths.phase2_events_output, events)

    summary = {
        "input_rows": int(start_rows),
        "output_rows": int(len(events)),
        "output": str(paths.phase2_events_output),
        "phase1_leak_guard": leak_guard_summary,
        "stints_merge": stints_summary,
        "opp_goalie": opp_goalie_summary,
        "synthetic_save_injection": save_injection_summary,
        "sidecars": sidecar_summary,
        "defensive_deflection_reclassification": defensive_deflection_summary,
        "text_cleaning": text_summary,
        "event_order": order_summary,
        "spatial_features": feature_summary,
        "goalie_angle_features": goalie_angle_summary,
    }
    return events, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 event-only enrichment pipeline")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-label", type=str, default="run_current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = run_phase2_event_pipeline(args.base_dir, run_label=args.run_label)
    print(summary)


if __name__ == "__main__":
    main()
