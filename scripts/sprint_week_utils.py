from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root_from_script(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def maybe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required parquet file not found: {path}")
    return pd.read_parquet(path)


def maybe_read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def required_columns(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def infer_opp_team_id(df: pd.DataFrame, team_col: str = "team_id") -> pd.Series:
    # Works for two-team games and returns NaN if game context is incomplete.
    per_game_teams = (
        df[["game_id", team_col]]
        .dropna()
        .drop_duplicates()
        .groupby("game_id")[team_col]
        .agg(list)
    )
    game_to_opp_map: dict[tuple[int, int], float] = {}
    for game_id, teams in per_game_teams.items():
        if len(teams) != 2:
            continue
        a, b = teams
        game_to_opp_map[(game_id, a)] = b
        game_to_opp_map[(game_id, b)] = a

    return df.apply(
        lambda r: game_to_opp_map.get((r["game_id"], r[team_col]), np.nan),
        axis=1,
    )


@dataclass
class SprintPaths:
    base_dir: Path

    @property
    def results_sprint_week(self) -> Path:
        return self.base_dir / "Results" / "sprint_week"

    @property
    def logs_dir(self) -> Path:
        return self.results_sprint_week / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.results_sprint_week / "checkpoints"

    @property
    def win_models_dir(self) -> Path:
        return self.results_sprint_week / "win_models"

    @property
    def hierarchy_dir(self) -> Path:
        return self.results_sprint_week / "hierarchy"

    def ensure_all(self) -> None:
        for path in [
            self.results_sprint_week,
            self.logs_dir,
            self.checkpoints_dir,
            self.win_models_dir,
            self.hierarchy_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class TransformerXTPaths:
    base_dir: Path
    run_label: str

    @property
    def results_root(self) -> Path:
        return self.base_dir / "Results" / "Transformer_xT"

    @property
    def models_root(self) -> Path:
        return self.base_dir / "Models" / "Transformer_xT"

    @property
    def run_results_dir(self) -> Path:
        return self.results_root / self.run_label

    @property
    def run_models_dir(self) -> Path:
        return self.models_root / self.run_label

    @property
    def logs_dir(self) -> Path:
        return self.run_results_dir / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_models_dir / "checkpoints"

    @property
    def inspections_dir(self) -> Path:
        return self.run_results_dir / "inspections"

    @property
    def metrics_dir(self) -> Path:
        return self.run_results_dir / "metrics"

    def ensure_all(self) -> None:
        for path in [
            self.results_root,
            self.models_root,
            self.run_results_dir,
            self.run_models_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.inspections_dir,
            self.metrics_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
