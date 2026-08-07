from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_thermo_tables(path: Path) -> list[tuple[list[str], list[list[float]]]]:
    """Parse all conventional LAMMPS thermo tables in a log file."""
    tables: list[tuple[list[str], list[list[float]]]] = []
    header: list[str] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal header, rows
        if header and rows:
            tables.append((header, rows))
        header, rows = None, []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts and parts[0] == "Step" and "Temp" in parts:
            flush()
            header = parts
            continue
        if header is None:
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            if rows:
                flush()
            continue
        if len(values) == len(header):
            rows.append(values)
    flush()
    return tables


def summarize(values: Iterable[float], tail_fraction: float) -> dict[str, float | int | None]:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return {"n": 0, "final": None, "mean": None, "std": None, "tail_mean": None, "tail_std": None, "min": None, "max": None}
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")
    import statistics

    n_tail = max(1, int(math.ceil(len(vals) * tail_fraction)))
    tail = vals[-n_tail:]
    return {
        "n": len(vals),
        "final": vals[-1],
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "tail_mean": statistics.fmean(tail),
        "tail_std": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def summarize_log(path: Path, tail_fraction: float) -> dict:
    tables = parse_thermo_tables(path)
    if not tables:
        return {"log": path.name, "error": "no thermo tables found"}
    header, rows = tables[-1]
    columns: dict[str, dict] = {}
    for index, name in enumerate(header):
        columns[name] = summarize((row[index] for row in rows), tail_fraction)
    return {
        "log": path.name,
        "table_count": len(tables),
        "selected_table_rows": len(rows),
        "columns": columns,
    }


def summarize_run(run_dir: Path, tail_fraction: float) -> dict:
    logs = sorted(run_dir.glob("[0-9][0-9]_*.log"))
    return {
        "run_dir": str(run_dir),
        "stages": {path.stem: summarize_log(path, tail_fraction) for path in logs},
    }


def write_flat_csv(path: Path, summary: dict) -> None:
    rows: list[dict[str, object]] = []
    for stage, payload in summary.get("stages", {}).items():
        columns = payload.get("columns", {})
        for column, stats in columns.items():
            rows.append({"stage": stage, "column": column, **stats})
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "column", "n", "final", "mean", "std", "tail_mean", "tail_std", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(run_dir: Path, tail_fraction: float) -> dict:
    summary = summarize_run(run_dir, tail_fraction)
    (run_dir / "thermo_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_flat_csv(run_dir / "thermo_summary.csv", summary)
    return summary
