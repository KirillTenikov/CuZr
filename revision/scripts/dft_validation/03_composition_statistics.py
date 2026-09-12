#!/usr/bin/env python3
"""Summarize per-structure MACE DFT errors by application-relevant composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = ["MACE_A", "MACE_B", "MACE_C", "MACE_D"]
COMPOSITION_ORDER = ["Cu36Zr64", "Cu50Zr50", "Cu64Zr36"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Per-structure CSVs produced by 02_mace_target_inference.py",
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def metrics(group: pd.DataFrame):
    return {
        "n_structures": len(group),
        "E_RMSE_meV_atom": float(
            np.sqrt(np.mean(group["dE_pa_meV"] ** 2))
        ),
        "E_abs_median_meV_atom": float(
            group["abs_dE_pa_meV"].median()
        ),
        "E_abs_p90_meV_atom": float(
            group["abs_dE_pa_meV"].quantile(0.90)
        ),
        "E_abs_p95_meV_atom": float(
            group["abs_dE_pa_meV"].quantile(0.95)
        ),
        "F_RMSE_meV_A": float(
            np.sqrt(
                np.average(
                    group["force_rmse_meV_A"] ** 2,
                    weights=group["n_atoms"],
                )
            )
        ),
        "F_struct_median_meV_A": float(
            group["force_rmse_meV_A"].median()
        ),
        "F_struct_p90_meV_A": float(
            group["force_rmse_meV_A"].quantile(0.90)
        ),
        "F_struct_p95_meV_A": float(
            group["force_rmse_meV_A"].quantile(0.95)
        ),
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        if len(df) != 127:
            raise RuntimeError(
                f"{path}: expected 127 rows, found {len(df)}"
            )
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(outdir / "mace_target_per_structure.csv", index=False)

    comp_rows = []
    for (model, composition), group in all_df.groupby(
        ["model", "composition"], sort=False
    ):
        row = {
            "model": model,
            "composition": composition,
            "x_Cu": float(group["x_Cu"].iloc[0]),
        }
        row.update(metrics(group))
        comp_rows.append(row)

    comp = pd.DataFrame(comp_rows)
    comp["model"] = pd.Categorical(
        comp["model"], MODEL_ORDER, ordered=True
    )
    comp["composition"] = pd.Categorical(
        comp["composition"], COMPOSITION_ORDER, ordered=True
    )
    comp = comp.sort_values(["model", "composition"])
    comp.to_csv(
        outdir / "mace_target_composition_summary.csv",
        index=False,
    )

    overall_rows = []
    for model, group in all_df.groupby("model", sort=False):
        row = {"model": model}
        row.update(metrics(group))
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall["model"] = pd.Categorical(
        overall["model"], MODEL_ORDER, ordered=True
    )
    overall = overall.sort_values("model")
    overall.to_csv(
        outdir / "mace_target_overall_summary.csv",
        index=False,
    )

    # A compact manuscript-facing table: RMSE only.
    manuscript = comp[
        [
            "model",
            "composition",
            "n_structures",
            "E_RMSE_meV_atom",
            "F_RMSE_meV_A",
        ]
    ].copy()
    manuscript.to_csv(
        outdir / "mace_target_manuscript_table.csv",
        index=False,
    )

    print("\nComposition summary")
    print(comp.to_string(index=False))
    print("\nOverall 127-structure summary")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
