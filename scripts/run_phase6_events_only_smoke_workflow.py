from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sprint_week_utils import TransformerXTPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an events-only workflow smoke check: startup/preflight setup "
            "and optional pure postprocess validation."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Run label to validate. Use empty/latest/auto to resolve the most recent run with model artifacts.",
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=None)
    parser.add_argument("--pipeline-run-label", type=str, default=None)
    parser.add_argument("--penalty-static-for", type=float, default=None)
    parser.add_argument("--penalty-static-against", type=float, default=None)
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Run smoke preflight only and skip postprocess export validation.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip training preflight stage and validate postprocess outputs only.",
    )
    return parser.parse_args()


def _run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _assert_paths_exist(paths: list[Path], label: str) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required {label} paths:\n{lines}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_run_label(base_dir: Path, requested_run_label: str | None) -> str:
    requested = str(requested_run_label or "").strip()
    if requested and requested.lower() not in {"latest", "auto"}:
        return requested

    results_root = base_dir / "Results" / "Transformer_xT"
    models_root = base_dir / "Models" / "Transformer_xT"
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found for run-label auto resolution: {results_root}")

    run_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {results_root}")

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if (models_root / run_dir.name).exists():
            return run_dir.name

    # Fallback to newest results directory when models are not present yet.
    return run_dirs[0].name


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    python_executable = str(args.python_executable)
    resolved_run_label = _resolve_run_label(base_dir=base_dir, requested_run_label=args.run_label)

    script_train = base_dir / "scripts" / "train_phase6_transformer_entry.py"
    script_post = base_dir / "scripts" / "postprocess_phase6_outputs.py"
    _assert_paths_exist([script_train, script_post], label="script")

    run_paths = TransformerXTPaths(base_dir=base_dir, run_label=resolved_run_label)
    run_paths.ensure_all()

    print("Smoke run context:")
    print(f"- requested run_label: {args.run_label or '(auto)'}")
    print(f"- resolved run_label: {resolved_run_label}")
    print(f"- models dir: {run_paths.run_models_dir}")
    print(f"- results dir: {run_paths.run_results_dir}")
    print(f"- logs dir: {run_paths.logs_dir}")
    print(f"- skip_preflight: {bool(args.skip_preflight)}")
    print(f"- skip_postprocess: {bool(args.skip_postprocess)}")
    print(
        "- trainer loader opts: "
        f"num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}, "
        f"pin_memory={args.pin_memory}, "
        f"persistent_workers={args.persistent_workers}"
    )

    preflight_cmd = [
        python_executable,
        str(script_train),
        "--base-dir",
        str(base_dir),
        "--run-label",
        resolved_run_label,
        "--model-variant",
        "events_only",
        "--smoke-preflight-only",
    ]
    if args.sample_rows and int(args.sample_rows) > 0:
        preflight_cmd.extend(["--sample-rows", str(int(args.sample_rows))])
    if args.num_workers is not None:
        preflight_cmd.extend(["--num-workers", str(int(args.num_workers))])
    if args.prefetch_factor is not None:
        preflight_cmd.extend(["--prefetch-factor", str(int(args.prefetch_factor))])
    if args.pin_memory is True:
        preflight_cmd.append("--pin-memory")
    elif args.pin_memory is False:
        preflight_cmd.append("--no-pin-memory")
    if args.persistent_workers is True:
        preflight_cmd.append("--persistent-workers")
    elif args.persistent_workers is False:
        preflight_cmd.append("--no-persistent-workers")

    if not args.skip_preflight:
        _run_command(preflight_cmd, cwd=base_dir)

    if not args.skip_postprocess:
        postprocess_cmd = [
            python_executable,
            str(script_post),
            "--base-dir",
            str(base_dir),
            "--run-label",
            resolved_run_label,
        ]
        if args.sample_rows and int(args.sample_rows) > 0:
            postprocess_cmd.extend(["--sample-rows", str(int(args.sample_rows))])
        if args.pipeline_run_label:
            postprocess_cmd.extend(["--pipeline-run-label", str(args.pipeline_run_label)])
        if args.penalty_static_for is not None:
            postprocess_cmd.extend(["--penalty-static-for", str(float(args.penalty_static_for))])
        if args.penalty_static_against is not None:
            postprocess_cmd.extend(["--penalty-static-against", str(float(args.penalty_static_against))])

        _run_command(postprocess_cmd, cwd=base_dir)

    expected_outputs = []
    if not args.skip_preflight:
        expected_outputs.append(run_paths.logs_dir / "phase6_smoke_preflight_summary.json")
    if not args.skip_postprocess:
        expected_outputs.extend(
            [
                run_paths.run_results_dir / "oof_phase6_adjusted_predictions.parquet",
                run_paths.run_results_dir / "optimus_reim_goalie_credit_ledger_events_only.parquet",
                run_paths.run_results_dir / "optimus_reim_global_player_credit_ledger_events_only.parquet",
                run_paths.logs_dir / "phase6_postprocess_summary.json",
            ]
        )
    _assert_paths_exist(expected_outputs, label="smoke output")

    preflight_summary = {"status": "skipped"}
    if not args.skip_preflight:
        preflight_summary = _load_json(run_paths.logs_dir / "phase6_smoke_preflight_summary.json")
    summary_info = {"status": "skipped"}
    if not args.skip_postprocess:
        summary_info = _load_json(run_paths.logs_dir / "phase6_postprocess_summary.json")

    if (not args.skip_preflight) and (not isinstance(preflight_summary, dict) or not preflight_summary):
        raise RuntimeError("Smoke preflight summary is missing or malformed.")

    print("\nEvents-only workflow smoke check completed successfully.")
    print(f"Run label: {resolved_run_label}")
    print(f"Models dir: {run_paths.run_models_dir}")
    print(f"Results dir: {run_paths.run_results_dir}")
    if args.skip_preflight:
        print("Preflight stage: skipped by --skip-preflight")
    if args.skip_postprocess:
        print("Postprocess stage: skipped by --skip-postprocess")
    else:
        print(f"Postprocess adjusted rows: {int((summary_info.get('rows') or {}).get('adjusted', 0))}")
        print(f"Postprocess goalie ledger rows: {int((summary_info.get('rows') or {}).get('goalie_ledger', 0))}")
    print("Verified outputs:")
    for path in expected_outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
