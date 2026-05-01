from __future__ import annotations

from pathlib import Path


def _runs_root(base_dir: Path) -> Path:
    return Path(base_dir) / "Data" / "Pipeline Runs"


def list_run_labels(base_dir: Path) -> list[str]:
    """Return run folder names sorted by latest modified first."""
    runs_root = _runs_root(base_dir)
    if not runs_root.exists():
        return []

    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def resolve_run_label(base_dir: Path, run_label: str, *, aliases: tuple[str, ...] = ("latest", "auto")) -> str:
    """Resolve a run label alias to an existing run folder name.

    Explicit run labels are passed through unchanged unless they match an alias.
    Aliases resolve to the newest existing run folder.
    """
    if run_label not in aliases:
        return run_label

    available = list_run_labels(base_dir)
    if not available:
        runs_root = _runs_root(base_dir)
        raise FileNotFoundError(
            "No existing pipeline runs were found for alias "
            f"'{run_label}' under {runs_root}."
        )

    return available[0]


def require_artifacts_exist(artifacts: list[Path], *, stage_name: str, run_root: Path) -> None:
    """Fail fast with a clear message when required artifacts are missing."""
    missing = [str(path) for path in artifacts if not path.exists()]
    if not missing:
        return

    raise FileNotFoundError(
        f"{stage_name} prerequisites are incomplete for run root {run_root}. "
        f"Missing artifacts: {missing}"
    )
