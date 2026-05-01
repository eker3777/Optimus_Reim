from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .phase1_cleaning import run_phase1_cleaning
    from .phase2_runner import run_phase2 as run_phase2_impl
    from .phase3_tensor_prep import run_phase3 as run_phase3_impl
    from .phase1_score_verification import verify_phase1_scores
    from .run_resolver import resolve_run_label
except ImportError:  # Allows running as a direct script path.
    from phase1_cleaning import run_phase1_cleaning
    from phase2_runner import run_phase2 as run_phase2_impl
    from phase3_tensor_prep import run_phase3 as run_phase3_impl
    from phase1_score_verification import verify_phase1_scores
    from run_resolver import resolve_run_label


def run_phase1(base_dir: Path, run_label: str = "run_current") -> dict:
    _, _, phase1_summary = run_phase1_cleaning(base_dir, run_label=run_label)
    score_report = verify_phase1_scores(base_dir, run_label=run_label)
    return {
        "phase": "phase1",
        "run_label": run_label,
        "phase1_summary": phase1_summary,
        "score_verification": score_report,
    }


def run_phase2(base_dir: Path, run_label: str = "run_current", *, skip_tracking_processing: bool = False) -> dict:
    summary = run_phase2_impl(base_dir, run_label=run_label, skip_tracking_processing=bool(skip_tracking_processing))
    return {
        "phase": "phase2",
        "run_label": str(summary.get("run_label", run_label)),
        "phase2_summary": summary,
    }


def run_phase3(base_dir: Path, run_label: str = "run_current") -> dict:
    summary = run_phase3_impl(base_dir, run_label=run_label)
    return {
        "phase": "phase3",
        "run_label": str(summary.get("run_label", run_label)),
        "phase3_summary": summary,
    }


def run_full(base_dir: Path, run_label: str = "run_current", *, skip_tracking_processing: bool = False) -> dict:
    phase1 = run_phase1(base_dir, run_label=run_label)
    phase2 = run_phase2(base_dir, run_label=run_label, skip_tracking_processing=bool(skip_tracking_processing))
    phase3 = run_phase3(base_dir, run_label=run_label)
    return {
        "run_label": run_label,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data prep consolidation pipeline entrypoint")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--run-label",
        type=str,
        default="run_current",
        help="Run folder label. Use 'latest' to reuse the newest existing pipeline run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="phase1",
        choices=["phase1", "phase2", "phase3", "full"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--skip-tracking-processing",
        action="store_true",
        help="When running phase2/full, skip Phase 2 tracking event-relative and absolute processing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_label = resolve_run_label(args.base_dir, args.run_label)

    if run_label != args.run_label:
        print(f"Resolved run label alias '{args.run_label}' -> '{run_label}'")

    if args.mode == "phase1":
        result = run_phase1(args.base_dir, run_label=run_label)
    elif args.mode == "phase2":
        result = run_phase2(
            args.base_dir,
            run_label=run_label,
            skip_tracking_processing=bool(args.skip_tracking_processing),
        )
    elif args.mode == "phase3":
        result = run_phase3(args.base_dir, run_label=run_label)
    else:
        result = run_full(
            args.base_dir,
            run_label=run_label,
            skip_tracking_processing=bool(args.skip_tracking_processing),
        )

    print(result)


if __name__ == "__main__":
    main()
