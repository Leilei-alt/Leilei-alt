# analysis/stats_utils.py
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_dict_rows_to_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(values: Iterable[float]) -> dict:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot summarize empty list.")
    return {
        "count": len(vals),
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def group_and_summarize(rows: list[dict], group_key: str, value_keys: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    summary_rows = []
    for gval, grows in grouped.items():
        out = {group_key: gval}
        for key in value_keys:
            numeric_vals = [float(r[key]) for r in grows]
            stats = summarize_numeric(numeric_vals)
            out[f"{key}_mean"] = stats["mean"]
            out[f"{key}_std"] = stats["std"]
            out[f"{key}_min"] = stats["min"]
            out[f"{key}_max"] = stats["max"]
        summary_rows.append(out)

    def sort_key(x: dict):
        try:
            return float(x[group_key])
        except ValueError:
            return x[group_key]

    summary_rows.sort(key=sort_key)
    return summary_rows
