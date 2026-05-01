from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing required columns: {missing}")


def assert_key_uniqueness(df: pd.DataFrame, keys: list[str], df_name: str) -> None:
    dupes = int(df.duplicated(keys).sum())
    if dupes:
        raise AssertionError(f"{df_name} has duplicate key rows for {keys}: {dupes}")


def sequence_disorder_count(
    df: pd.DataFrame,
    *,
    game_col: str = "game_id",
    seq_col: str = "sequence_id",
    time_col: str = "period_time",
    threshold: float = -1.0,
) -> int:
    if df.empty:
        return 0

    work = df[[game_col, seq_col, time_col]].copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    delta = work.groupby([game_col, seq_col], sort=False)[time_col].diff()
    return int((delta < threshold).sum())


def assert_target_domain(df: pd.DataFrame) -> None:
    if "target" in df.columns:
        bad_target = ~df["target"].isin([0, 1, 2])
        if bad_target.any():
            raise AssertionError(f"target contains invalid labels: {sorted(df.loc[bad_target, 'target'].dropna().unique().tolist())}")

    if "target_xg" in df.columns:
        bad_xg = ~df["target_xg"].isin([0, 1])
        if bad_xg.any():
            raise AssertionError(
                f"target_xg contains invalid labels: {sorted(df.loc[bad_xg, 'target_xg'].dropna().unique().tolist())}"
            )


def assert_no_duplicate_columns(df: pd.DataFrame, df_name: str) -> None:
    if df.columns.duplicated().any():
        dupes = sorted({c for c in df.columns[df.columns.duplicated()].tolist()})
        raise AssertionError(f"{df_name} has duplicate columns: {dupes}")


def finite_metric(value: float, metric_name: str) -> None:
    if not np.isfinite(value):
        raise AssertionError(f"{metric_name} must be finite but got {value}")


def event_type_set(df: pd.DataFrame, event_col: str = "event_type") -> set[str]:
    if event_col not in df.columns:
        return set()
    vals = df[event_col].dropna().astype(str).str.lower().str.strip()
    return set(vals[vals != ""].unique().tolist())


def missing_required_event_types(
    df: pd.DataFrame,
    required_event_types: Iterable[str],
    *,
    event_col: str = "event_type",
) -> list[str]:
    present = event_type_set(df, event_col=event_col)
    required = {str(v).lower().strip() for v in required_event_types if str(v).strip()}
    return sorted(required - present)


def phase1_boundary_violation_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {
        "period_gt_4": 0,
        "period4_time_gt_300": 0,
    }
    if "period" not in df.columns or "period_time" not in df.columns:
        return counts

    period = pd.to_numeric(df["period"], errors="coerce")
    period_time = pd.to_numeric(df["period_time"], errors="coerce")

    counts["period_gt_4"] = int((period > 4).fillna(False).sum())
    counts["period4_time_gt_300"] = int(((period == 4) & (period_time > 300)).fillna(False).sum())
    return counts


def collect_phase1_secondary_warnings(
    df: pd.DataFrame,
    *,
    disorder_rows: int,
    required_event_types: Iterable[str],
    sequence_disorder_warn_threshold_rows: int = 0,
) -> list[str]:
    warnings: list[str] = []

    if disorder_rows > sequence_disorder_warn_threshold_rows:
        warnings.append(
            "sequence_disorder_rows exceeds warning threshold: "
            f"{disorder_rows} > {sequence_disorder_warn_threshold_rows}"
        )

    missing_types = missing_required_event_types(df, required_event_types)
    if missing_types:
        warnings.append(f"missing required event types: {missing_types}")

    present = event_type_set(df)
    if "assist" in present:
        warnings.append("assist events remain after Phase 1 leakage guard")
    if "failedpasslocation" in present:
        warnings.append("failedpasslocation events remain after merge step")

    boundary_counts = phase1_boundary_violation_counts(df)
    if boundary_counts["period_gt_4"] > 0:
        warnings.append(f"period > 4 rows remain: {boundary_counts['period_gt_4']}")
    if boundary_counts["period4_time_gt_300"] > 0:
        warnings.append(f"period 4 with period_time > 300 rows remain: {boundary_counts['period4_time_gt_300']}")

    if "is_post_whistle_penalty" not in df.columns:
        warnings.append("is_post_whistle_penalty flag column is missing")

    return warnings


def _gate_payload(checks: dict[str, bool], warnings: list[str], errors: list[str]) -> dict[str, object]:
    gate_status = "fail" if errors else ("warn" if warnings else "pass")
    return {
        "gate_status": gate_status,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def validate_phase2_events(df: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_cols = [
        "game_id",
        "sl_event_id",
        "event_type",
        "game_stint",
        "description_clean",
        "flags_clean",
        "embedding_text_clean",
        "is_boundary_event",
        "is_end_of_period_event",
        "distance_to_net_event",
        "angle_to_net_event",
        "distance_from_last_event",
        "speed_from_last_event",
        "angle_from_last_event",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    checks["required_columns"] = len(missing) == 0
    if missing:
        errors.append(f"phase2 events missing required columns: {missing}")

    try:
        assert_key_uniqueness(df, ["game_id", "sl_event_id"], "phase2_events")
        checks["unique_event_key"] = True
    except AssertionError as exc:
        checks["unique_event_key"] = False
        errors.append(str(exc))

    if "event_type" in df.columns and "is_end_of_period_event" in df.columns:
        eop_flagged = int(pd.to_numeric(df["is_end_of_period_event"], errors="coerce").fillna(0).sum())
        warnings.extend([] if eop_flagged > 0 else ["no end-of-period rows were flagged before normalization"])

    if "home_goalie_id" in df.columns and "away_goalie_id" in df.columns:
        home_cov = float(df["home_goalie_id"].notna().mean()) if len(df) else 0.0
        away_cov = float(df["away_goalie_id"].notna().mean()) if len(df) else 0.0
        if home_cov < 0.05:
            warnings.append(f"home_goalie_id coverage is low: {home_cov:.3f}")
        if away_cov < 0.05:
            warnings.append(f"away_goalie_id coverage is low: {away_cov:.3f}")

    if all(c in df.columns for c in ["is_synthetic_save", "x_adj", "y_adj"]):
        synth_mask = pd.to_numeric(df["is_synthetic_save"], errors="coerce").fillna(0).eq(1)
        synth_count = int(synth_mask.sum())
        checks["synthetic_save_rows_present"] = synth_count > 0
        if synth_count > 0:
            x_ok = pd.to_numeric(df.loc[synth_mask, "x_adj"], errors="coerce").eq(-89.0).all()
            y_ok = pd.to_numeric(df.loc[synth_mask, "y_adj"], errors="coerce").eq(0.0).all()
            checks["synthetic_save_x_net_snapped"] = bool(x_ok)
            checks["synthetic_save_y_net_snapped"] = bool(y_ok)
            if not x_ok:
                errors.append("synthetic save rows have non net-snapped x_adj values (expected -89.0)")
            if not y_ok:
                errors.append("synthetic save rows have non net-snapped y_adj values (expected 0.0)")
    else:
        checks["synthetic_save_rows_present"] = False

    if "description_clean" in df.columns:
        forbidden_terms = ["successful", "miss pass", "miss shot", "blocked", "missed"]
        forbidden_hits = {
            term: int(df["description_clean"].astype(str).str.contains(term, regex=False, na=False).sum())
            for term in forbidden_terms
        }
        checks["forbidden_description_terms_removed"] = all(v == 0 for v in forbidden_hits.values())
        if not checks["forbidden_description_terms_removed"]:
            errors.append(f"forbidden terms remain in description_clean: {forbidden_hits}")

    detail = df["detail"].astype(str).str.lower() if "detail" in df.columns else pd.Series("", index=df.index)
    description = df["description"].astype(str).str.lower() if "description" in df.columns else pd.Series("", index=df.index)
    event_type = df["event_type"].astype(str).str.lower() if "event_type" in df.columns else pd.Series("", index=df.index)
    period = pd.to_numeric(df["period"], errors="coerce") if "period" in df.columns else pd.Series(np.nan, index=df.index)

    leak_masks = {
        "shootout": detail.str.contains("shootout", regex=False)
        | description.str.contains("shootout", regex=False)
        | period.eq(5),
        "penalty_shot": detail.str.contains("penalty shot", regex=False)
        | description.str.contains("penalty shot", regex=False)
        | event_type.eq("penalty_shot"),
        "home_team_shoots_right": detail.str.contains("home team shoots right", regex=False)
        | description.str.contains("home team shoots right", regex=False),
        "away_team_shoots_right": detail.str.contains("away team shoots right", regex=False)
        | description.str.contains("away team shoots right", regex=False),
    }
    leak_counts = {name: int(mask.sum()) for name, mask in leak_masks.items()}
    checks["phase1_leak_rows_removed"] = all(v == 0 for v in leak_counts.values())
    if not checks["phase1_leak_rows_removed"]:
        errors.append(f"phase1 leak patterns remain in phase2 events: {leak_counts}")

    return _gate_payload(checks, warnings, errors)


def validate_phase2_tracking_event_relative(df: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_cols = [
        "game_id",
        "sl_event_id",
        "actor_rel_x",
        "actor_rel_y",
        "actor_is_present",
        "actor_is_imputed",
        "tm_1_rel_x",
        "tm_1_is_present",
        "opp_1_rel_x",
        "opp_1_is_present",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    checks["required_columns"] = len(missing) == 0
    if missing:
        errors.append(f"phase2 event-relative tracking missing required columns: {missing}")

    try:
        assert_key_uniqueness(df, ["game_id", "sl_event_id"], "phase2_tracking_event_relative")
        checks["unique_event_key"] = True
    except AssertionError as exc:
        checks["unique_event_key"] = False
        errors.append(str(exc))

    if "actor_is_imputed" in df.columns:
        imputed_rate = float(pd.to_numeric(df["actor_is_imputed"], errors="coerce").fillna(0).mean()) if len(df) else 0.0
        if imputed_rate > 0.6:
            warnings.append(f"actor imputation rate is high: {imputed_rate:.3f}")

    return _gate_payload(checks, warnings, errors)


def validate_phase2_tracking_absolute(df: pd.DataFrame, slots_per_side: int = 6) -> dict[str, object]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_anchor = ["game_id", "sl_event_id", "game_stint", "target_net_x", "attack_direction_x", "is_stint_change"]
    missing_anchor = [c for c in required_anchor if c not in df.columns]
    checks["required_anchor_columns"] = len(missing_anchor) == 0
    if missing_anchor:
        errors.append(f"phase2 absolute tracking missing anchor columns: {missing_anchor}")

    expected_metrics = [
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
    expected_cols = [f"{side}_Track_{slot}_{metric}" for side in ["Home", "Away"] for slot in range(slots_per_side) for metric in expected_metrics]
    missing_wide = [c for c in expected_cols if c not in df.columns]
    checks["wide_slot_schema"] = len(missing_wide) == 0
    if missing_wide:
        errors.append(f"phase2 absolute tracking missing wide columns count={len(missing_wide)}")

    try:
        assert_key_uniqueness(df, ["game_id", "sl_event_id"], "phase2_tracking_absolute")
        checks["unique_event_key"] = True
    except AssertionError as exc:
        checks["unique_event_key"] = False
        errors.append(str(exc))

    x_cols = [c for c in df.columns if c.endswith("_X")]
    y_cols = [c for c in df.columns if c.endswith("_Y")]
    if x_cols:
        x_vals = df[x_cols].to_numpy(dtype=float)
        if (x_vals < -120).any() or (x_vals > 120).any():
            warnings.append("absolute tracking X coordinates exceed expected rink bounds")
    if y_cols:
        y_vals = df[y_cols].to_numpy(dtype=float)
        if (y_vals < -80).any() or (y_vals > 80).any():
            warnings.append("absolute tracking Y coordinates exceed expected rink bounds")

    return _gate_payload(checks, warnings, errors)


def validate_phase3_outputs(
    df_events_with_idx: pd.DataFrame,
    df_final: pd.DataFrame,
    *,
    embedding_rows: int,
    default_horizon_seconds: float = 20.0,
    post_whistle_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_events_cols = ["game_id", "sl_event_id", "embedding_text_clean", "text_embedding_idx"]
    missing_events = [c for c in required_events_cols if c not in df_events_with_idx.columns]
    checks["events_with_idx_required_columns"] = len(missing_events) == 0
    if missing_events:
        errors.append(f"phase3 events_with_embedding_indices missing required columns: {missing_events}")

    required_final_cols = [
        "game_id",
        "sl_event_id",
        "period_time",
        "horizon_end_time",
        "event_type",
        "text_embedding_idx",
        "event_type_id",
        "outcome_id",
        "outcome_xg_id",
        "period_id",
        "target",
        "target_xg",
    ]
    missing_final = [c for c in required_final_cols if c not in df_final.columns]
    checks["final_dataset_required_columns"] = len(missing_final) == 0
    if missing_final:
        errors.append(f"phase3 tensor_ready_dataset missing required columns: {missing_final}")

    try:
        assert_key_uniqueness(df_events_with_idx, ["game_id", "sl_event_id"], "phase3_events_with_idx")
        checks["events_with_idx_unique_key"] = True
    except AssertionError as exc:
        checks["events_with_idx_unique_key"] = False
        errors.append(str(exc))

    try:
        assert_key_uniqueness(df_final, ["game_id", "sl_event_id"], "phase3_final_dataset")
        checks["final_dataset_unique_key"] = True
    except AssertionError as exc:
        checks["final_dataset_unique_key"] = False
        errors.append(str(exc))

    try:
        assert_no_duplicate_columns(df_events_with_idx, "phase3_events_with_idx")
        checks["events_with_idx_no_duplicate_columns"] = True
    except AssertionError as exc:
        checks["events_with_idx_no_duplicate_columns"] = False
        errors.append(str(exc))

    try:
        assert_no_duplicate_columns(df_final, "phase3_final_dataset")
        checks["final_dataset_no_duplicate_columns"] = True
    except AssertionError as exc:
        checks["final_dataset_no_duplicate_columns"] = False
        errors.append(str(exc))

    try:
        assert_target_domain(df_final)
        checks["target_domains_valid"] = True
    except AssertionError as exc:
        checks["target_domains_valid"] = False
        errors.append(str(exc))

    checks["embedding_rows_positive"] = int(embedding_rows) > 0
    if not checks["embedding_rows_positive"]:
        errors.append(f"embedding_rows must be positive but received {embedding_rows}")

    def _check_embedding_index(series: pd.Series, scope_name: str) -> None:
        idx = pd.to_numeric(series, errors="coerce")
        non_numeric = int(idx.isna().sum())
        negative = int((idx < 0).fillna(False).sum())
        above = int((idx >= int(embedding_rows)).fillna(False).sum()) if int(embedding_rows) > 0 else len(idx)

        ok_non_numeric = non_numeric == 0
        ok_negative = negative == 0
        ok_above = above == 0

        checks[f"{scope_name}_embedding_index_numeric"] = ok_non_numeric
        checks[f"{scope_name}_embedding_index_non_negative"] = ok_negative
        checks[f"{scope_name}_embedding_index_in_bounds"] = ok_above

        if not ok_non_numeric:
            errors.append(f"{scope_name} has non-numeric text_embedding_idx rows: {non_numeric}")
        if not ok_negative:
            errors.append(f"{scope_name} has negative text_embedding_idx rows: {negative}")
        if not ok_above:
            errors.append(
                f"{scope_name} has text_embedding_idx rows outside embedding matrix bounds: {above} "
                f"(embedding_rows={embedding_rows})"
            )

    if "text_embedding_idx" in df_events_with_idx.columns:
        _check_embedding_index(df_events_with_idx["text_embedding_idx"], "events_with_idx")

    if "text_embedding_idx" in df_final.columns:
        _check_embedding_index(df_final["text_embedding_idx"], "final_dataset")

    if "period_time" in df_final.columns and "horizon_end_time" in df_final.columns:
        period_time = pd.to_numeric(df_final["period_time"], errors="coerce")
        horizon_end = pd.to_numeric(df_final["horizon_end_time"], errors="coerce")
        horizon_span = horizon_end - period_time

        overflow_mask = horizon_span > (float(default_horizon_seconds) + 1e-6)
        overflow_rows = int(overflow_mask.fillna(False).sum())
        checks["uniform_horizon_window"] = overflow_rows == 0
        if overflow_rows > 0:
            errors.append(
                f"phase3 horizon exceeds {default_horizon_seconds:.3f}s on {overflow_rows} rows "
                "(penalty extension must be disabled)"
            )

        if "event_type" in df_final.columns:
            penalty_mask = df_final["event_type"].astype(str).str.lower().isin(["penalty", "penaltydrawn"])
            penalty_overflow = int((overflow_mask & penalty_mask).fillna(False).sum())
            checks["penalty_no_extended_horizon"] = penalty_overflow == 0
            if penalty_overflow > 0:
                errors.append(
                    f"penalty or penaltydrawn rows exceeded {default_horizon_seconds:.3f}s horizon: {penalty_overflow}"
                )

    if "event_type" in df_final.columns and "outcome_xg_id" in df_final.columns:
        shot_mask = df_final["event_type"].astype(str).str.lower().isin(["shot", "deflection", "defensive_deflection"])
        leaked_ids = int(df_final.loc[shot_mask, "outcome_xg_id"].ne(0).sum())
        checks["counterfactual_outcome_masking"] = leaked_ids == 0
        if leaked_ids > 0:
            errors.append(
                f"outcome_xg_id must be 0 for shot/deflection/defensive_deflection rows, found {leaked_ids} violations"
            )

    if post_whistle_summary:
        unexpected = int(post_whistle_summary.get("unexpected_same_sequence_continuations", 0))
        checks["post_whistle_audit_executed"] = True
        if unexpected > 0:
            warnings.append(
                f"post-whistle continuity audit found unexpected same-sequence continuations: {unexpected}"
            )
    else:
        checks["post_whistle_audit_executed"] = False
        warnings.append("post-whistle continuity summary not provided to phase3 validator")

    return _gate_payload(checks, warnings, errors)


def combine_gate_reports(*reports: dict[str, object]) -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, bool] = {}

    for idx, report in enumerate(reports, start=1):
        gate = str(report.get("gate_status", "pass")).lower()
        report_checks = report.get("checks", {}) if isinstance(report.get("checks", {}), dict) else {}
        for key, value in report_checks.items():
            checks[f"report{idx}:{key}"] = bool(value)
        warnings.extend([str(w) for w in report.get("warnings", [])])
        errors.extend([str(e) for e in report.get("errors", [])])
        if gate == "fail" and not report.get("errors"):
            errors.append(f"report{idx} returned fail without explicit errors")

    return _gate_payload(checks, warnings, errors)


def validate_phase2_phase3_key_parity(
    phase2_events: pd.DataFrame,
    phase3_final: pd.DataFrame,
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    required_keys = ["game_id", "sl_event_id"]
    missing_phase2 = [c for c in required_keys if c not in phase2_events.columns]
    missing_phase3 = [c for c in required_keys if c not in phase3_final.columns]
    checks["phase2_key_columns_present"] = len(missing_phase2) == 0
    checks["phase3_key_columns_present"] = len(missing_phase3) == 0

    if missing_phase2:
        errors.append(f"phase2 key parity input missing required columns: {missing_phase2}")
    if missing_phase3:
        errors.append(f"phase3 key parity input missing required columns: {missing_phase3}")

    if errors:
        payload = _gate_payload(checks, warnings, errors)
        payload["audit"] = {
            "total_phase2_keys": 0,
            "total_phase3_keys": 0,
            "matched_keys": 0,
            "phase2_only_keys": 0,
            "phase3_only_keys": 0,
            "per_game_breakdown": [],
        }
        return payload

    phase2_keys = phase2_events[required_keys].dropna(subset=required_keys).drop_duplicates().copy()
    phase3_keys = phase3_final[required_keys].dropna(subset=required_keys).drop_duplicates().copy()

    merged = phase2_keys.merge(
        phase3_keys,
        on=required_keys,
        how="outer",
        indicator=True,
    )

    matched_keys = int((merged["_merge"] == "both").sum())
    phase2_only_keys = int((merged["_merge"] == "left_only").sum())
    phase3_only_keys = int((merged["_merge"] == "right_only").sum())

    checks["strict_global_key_parity"] = (phase2_only_keys == 0 and phase3_only_keys == 0)
    if not checks["strict_global_key_parity"]:
        errors.append(
            "phase2-to-phase3 key parity mismatch: "
            f"phase2_only={phase2_only_keys}, phase3_only={phase3_only_keys}"
        )

    p2_game = phase2_keys.groupby("game_id", dropna=False).size().rename("phase2_keys")
    p3_game = phase3_keys.groupby("game_id", dropna=False).size().rename("phase3_keys")

    p2_only_game = (
        merged.loc[merged["_merge"] == "left_only"]
        .groupby("game_id", dropna=False)
        .size()
        .rename("phase2_only_keys")
    )
    p3_only_game = (
        merged.loc[merged["_merge"] == "right_only"]
        .groupby("game_id", dropna=False)
        .size()
        .rename("phase3_only_keys")
    )
    both_game = (
        merged.loc[merged["_merge"] == "both"]
        .groupby("game_id", dropna=False)
        .size()
        .rename("matched_keys")
    )

    per_game = (
        pd.concat([p2_game, p3_game, both_game, p2_only_game, p3_only_game], axis=1)
        .fillna(0)
        .reset_index()
    )
    for col in ["phase2_keys", "phase3_keys", "matched_keys", "phase2_only_keys", "phase3_only_keys"]:
        per_game[col] = per_game[col].astype(int)

    per_game_breakdown = per_game.to_dict(orient="records")

    payload = _gate_payload(checks, warnings, errors)
    payload["audit"] = {
        "total_phase2_keys": int(len(phase2_keys)),
        "total_phase3_keys": int(len(phase3_keys)),
        "matched_keys": matched_keys,
        "phase2_only_keys": phase2_only_keys,
        "phase3_only_keys": phase3_only_keys,
        "per_game_breakdown": per_game_breakdown,
    }
    return payload
