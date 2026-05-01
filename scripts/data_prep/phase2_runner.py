from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .config import DataPrepConfig, DataPrepPaths
    from .io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json
    from .phase2_event_pipeline import run_phase2_event_pipeline
    from .phase2_tracking_absolute import run_phase2_tracking_absolute
    from .phase2_tracking_event_relative import run_phase2_tracking_event_relative
    from .validation import (
        combine_gate_reports,
        validate_phase2_events,
        validate_phase2_tracking_absolute,
        validate_phase2_tracking_event_relative,
    )
    from .run_resolver import require_artifacts_exist, resolve_run_label
except ImportError:  # Allows running as a direct script path.
    from config import DataPrepConfig, DataPrepPaths
    from io_utils import append_manifest_record, read_parquet, utc_now_iso, write_json
    from phase2_event_pipeline import run_phase2_event_pipeline
    from phase2_tracking_absolute import run_phase2_tracking_absolute
    from phase2_tracking_event_relative import run_phase2_tracking_event_relative
    from validation import (
        combine_gate_reports,
        validate_phase2_events,
        validate_phase2_tracking_absolute,
        validate_phase2_tracking_event_relative,
    )
    from run_resolver import require_artifacts_exist, resolve_run_label


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_phase1_gate(paths: DataPrepPaths) -> None:
    clean_summary = _load_json(paths.prep_logs_dir / "phase1_cleaning_summary.json")
    score_report = _load_json(paths.phase1_score_report_output)

    clean_gate = str(clean_summary.get("gate_status", "")).lower()
    score_gate = str(score_report.get("gate_status", "")).lower()
    all_match = bool(score_report.get("all_match", False))

    if clean_gate not in {"pass", "warn"}:
        raise RuntimeError("Phase 1 cleaning gate is not pass/warn. Run Stage 1 first.")
    if score_gate != "pass" or not all_match:
        raise RuntimeError("Phase 1 score gate is not pass/all_match. Resolve score mismatches before Phase 2.")


def _assert_phase1_artifacts(paths: DataPrepPaths) -> None:
    require_artifacts_exist(
        [
            paths.phase1_events_output,
            paths.phase1_extra_goals_output,
            paths.prep_logs_dir / "phase1_cleaning_summary.json",
            paths.phase1_score_report_output,
        ],
        stage_name="Phase 2",
        run_root=paths.runs_root_dir,
    )


def run_phase2(
    base_dir: Path,
    *,
    run_label: str = "run_current",
    config: DataPrepConfig | None = None,
    skip_tracking_processing: bool = False,
) -> dict:
    cfg = config or DataPrepConfig()
    resolved_run_label = resolve_run_label(base_dir, run_label)
    paths = DataPrepPaths(base_dir, run_label=resolved_run_label)
    paths.ensure_dirs()

    _assert_phase1_artifacts(paths)
    _assert_phase1_gate(paths)

    events, event_summary = run_phase2_event_pipeline(base_dir, run_label=resolved_run_label, config=cfg)

    if bool(skip_tracking_processing):
        tracking_rel = pd.DataFrame()
        tracking_abs = pd.DataFrame()
        slot_map = pd.DataFrame()
        stint_changes = pd.DataFrame()
        rel_summary = {
            "skipped": True,
            "reason": "skip_tracking_processing enabled",
            "rows": 0,
        }
        abs_summary = {
            "skipped": True,
            "reason": "skip_tracking_processing enabled",
            "rows": 0,
        }
    else:
        tracking_rel, rel_summary = run_phase2_tracking_event_relative(
            base_dir,
            run_label=resolved_run_label,
            config=cfg,
            events_df=events,
        )
        tracking_abs, slot_map, stint_changes, abs_summary = run_phase2_tracking_absolute(
            base_dir,
            run_label=resolved_run_label,
            config=cfg,
            events_df=events,
        )

    event_gate = validate_phase2_events(events)
    if bool(skip_tracking_processing):
        rel_gate = {
            "gate_status": "pass",
            "checks": {"tracking_event_relative_skipped": True},
            "warnings": ["tracking_event_relative processing skipped by request"],
            "errors": [],
        }
        abs_gate = {
            "gate_status": "pass",
            "checks": {"tracking_absolute_skipped": True},
            "warnings": ["tracking_absolute processing skipped by request"],
            "errors": [],
        }
    else:
        rel_gate = validate_phase2_tracking_event_relative(tracking_rel)
        abs_gate = validate_phase2_tracking_absolute(tracking_abs, slots_per_side=cfg.absolute_tracking_slots_per_side)
    phase2_gate = combine_gate_reports(event_gate, rel_gate, abs_gate)

    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase2_events",
        output_path=paths.phase2_events_output,
        rows=len(events),
        columns=events.columns.tolist(),
        extra={"gate_status": event_gate["gate_status"]},
    )
    if not bool(skip_tracking_processing):
        append_manifest_record(
            paths.prep_logs_dir / "data_prep_manifest.json",
            name="phase2_tracking_event_relative",
            output_path=paths.phase2_tracking_event_relative_output,
            rows=len(tracking_rel),
            columns=tracking_rel.columns.tolist(),
            extra={"gate_status": rel_gate["gate_status"]},
        )
        append_manifest_record(
            paths.prep_logs_dir / "data_prep_manifest.json",
            name="phase2_tracking_absolute",
            output_path=paths.phase2_tracking_absolute_output,
            rows=len(tracking_abs),
            columns=tracking_abs.columns.tolist(),
            extra={"gate_status": abs_gate["gate_status"]},
        )
        append_manifest_record(
            paths.prep_logs_dir / "data_prep_manifest.json",
            name="phase2_tracking_slot_mapping",
            output_path=paths.phase2_tracking_slot_mapping_output,
            rows=len(slot_map),
            columns=slot_map.columns.tolist(),
            extra={"gate_status": abs_gate["gate_status"]},
        )
        append_manifest_record(
            paths.prep_logs_dir / "data_prep_manifest.json",
            name="phase2_tracking_stint_changes",
            output_path=paths.phase2_tracking_stint_changes_output,
            rows=len(stint_changes),
            columns=stint_changes.columns.tolist(),
            extra={"gate_status": abs_gate["gate_status"]},
        )

    faceoff_reference = read_parquet(paths.phase2_faceoff_reference_output)
    penalty_reference = read_parquet(paths.phase2_penalty_reference_output)

    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase2_faceoff_reference",
        output_path=paths.phase2_faceoff_reference_output,
        rows=len(faceoff_reference),
        columns=faceoff_reference.columns.tolist(),
        extra={"gate_status": event_gate["gate_status"]},
    )
    append_manifest_record(
        paths.prep_logs_dir / "data_prep_manifest.json",
        name="phase2_penalty_reference",
        output_path=paths.phase2_penalty_reference_output,
        rows=len(penalty_reference),
        columns=penalty_reference.columns.tolist(),
        extra={"gate_status": event_gate["gate_status"]},
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "phase": "phase2",
        "run_label": resolved_run_label,
        "run_root": str(paths.runs_root_dir),
        "tracking_processing_skipped": bool(skip_tracking_processing),
        "gate_status": phase2_gate["gate_status"],
        "gates": {
            "events": event_gate,
            "tracking_event_relative": rel_gate,
            "tracking_absolute": abs_gate,
            "phase2_overall": phase2_gate,
        },
        "artifacts": {
            "events_phase2_enriched": str(paths.phase2_events_output),
            "faceoff_reference": str(paths.phase2_faceoff_reference_output),
            "penalty_reference": str(paths.phase2_penalty_reference_output),
            "tracking_event_relative": None if bool(skip_tracking_processing) else str(paths.phase2_tracking_event_relative_output),
            "tracking_absolute_pinned": None if bool(skip_tracking_processing) else str(paths.phase2_tracking_absolute_output),
            "tracking_slot_mapping": None if bool(skip_tracking_processing) else str(paths.phase2_tracking_slot_mapping_output),
            "tracking_stint_changes": None if bool(skip_tracking_processing) else str(paths.phase2_tracking_stint_changes_output),
        },
        "events_summary": event_summary,
        "tracking_event_relative_summary": rel_summary,
        "tracking_absolute_summary": abs_summary,
    }
    write_json(paths.phase2_summary_output, summary)

    if summary["gate_status"] == "fail":
        raise AssertionError("Phase 2 gate failed. See phase2_summary.json for details.")

    return summary
