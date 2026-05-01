from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path, columns=columns)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing csv file: {path}")
    return pd.read_csv(path)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_manifest_record(
    path: Path,
    *,
    name: str,
    output_path: Path,
    rows: int,
    columns: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    ensure_parent_dir(path)
    record = {
        "generated_at_utc": utc_now_iso(),
        "name": name,
        "output": str(output_path),
        "rows": int(rows),
        "columns": columns,
    }
    if extra:
        record.update(extra)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            payload = [payload]
    else:
        payload = []

    payload.append(record)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
