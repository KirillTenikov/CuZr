#!/usr/bin/env python3
"""Extract ACE and MACE validation-error convergence histories.

Important terminology:
- MACE log lines are validation-set metrics printed during training.
- Pacemaker/ACE uses a file commonly named `test_metrics.txt`; in the
  Paper 1 workflow this file corresponds to the *validation split used for
  training monitoring*, not the untouched 4242-structure hold-out test set.

These histories diagnose optimization/convergence. They are not a substitute
for independent held-out DFT validation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MACE_PATTERN = re.compile(
    r"Epoch\s+(\d+):.*?"
    r"RMSE_E_per_atom=\s*([0-9.eE+-]+)\s*meV,\s*"
    r"RMSE_F=\s*([0-9.eE+-]+)\s*meV"
)


def parse_name_path(values):
    result = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Expected NAME=PATH, got: {value}"
            )
        name, path = value.split("=", 1)
        result.append((name.strip(), Path(path.strip())))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mace-log",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for MACE_A/B/C/D",
    )
    parser.add_argument(
        "--ace-metrics",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for ACE514/ACE1352 pacemaker test_metrics.txt",
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def parse_mace(name, path):
    rows = []
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = MACE_PATTERN.search(line)
        if match:
            rows.append(
                {
                    "model": name,
                    "step": int(match.group(1)),
                    "E_RMSE_meV_atom": float(match.group(2)),
                    "F_RMSE_meV_A": float(match.group(3)),
                }
            )
    if not rows:
        raise RuntimeError(f"No MACE convergence records found in {path}")
    return rows


def parse_ace(name, path):
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    required = {"iter_num", "rmse_epa", "rmse_f_comp"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{path}: missing ACE columns {sorted(missing)}"
        )

    # Repeated iter_num entries can occur at final checkpoints.
    df = df.drop_duplicates(subset=["iter_num"], keep="last")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": name,
                "step": int(row["iter_num"]),
                "E_RMSE_meV_atom": float(row["rmse_epa"]) * 1000.0,
                "F_RMSE_meV_A": float(row["rmse_f_comp"]) * 1000.0,
            }
        )
    return rows


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mace_rows = []
    for name, path in parse_name_path(args.mace_log):
        mace_rows.extend(parse_mace(name, path))

    ace_rows = []
    for name, path in parse_name_path(args.ace_metrics):
        ace_rows.extend(parse_ace(name, path))

    if mace_rows:
        mace = pd.DataFrame(mace_rows).sort_values(["model", "step"])
        mace.to_csv(
            outdir / "mace_validation_convergence.csv",
            index=False,
        )
        print("\nMACE endpoints")
        print(mace.groupby("model").tail(1).to_string(index=False))

    if ace_rows:
        ace = pd.DataFrame(ace_rows).sort_values(["model", "step"])
        ace.to_csv(
            outdir / "ace_validation_convergence.csv",
            index=False,
        )
        print("\nACE endpoints")
        print(ace.groupby("model").tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
