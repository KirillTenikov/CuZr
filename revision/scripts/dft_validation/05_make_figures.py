#!/usr/bin/env python3
"""Create Paper 1 DFT-validation and training-convergence figures.

Each plot is generated as a separate figure. Matplotlib's default color cycle
is used intentionally; no custom style/color mapping is required for the
analysis itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--mace-convergence")
    parser.add_argument("--ace-convergence")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save_line_plot(
    df,
    x,
    y,
    group,
    xlabel,
    ylabel,
    title,
    path,
    dpi,
    xticks=None,
    xticklabels=None,
    logy=False,
):
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    for name, subset in df.groupby(group, sort=False):
        ax.plot(subset[x], subset[y], marker="o" if x == "x_Cu" else None, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks, xticklabels)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    xticks = [0.359375, 0.5, 0.640625]
    xticklabels = ["Cu36Zr64", "Cu50Zr50", "Cu64Zr36"]

    save_line_plot(
        summary,
        "x_Cu",
        "E_RMSE_meV_atom",
        "model",
        r"Доля Cu, $x_{\mathrm{Cu}}$",
        "RMSE энергии, мэВ/атом",
        "DFT test: MACE по составу",
        outdir / "mace_target_energy_rmse.png",
        args.dpi,
        xticks,
        xticklabels,
    )
    save_line_plot(
        summary,
        "x_Cu",
        "F_RMSE_meV_A",
        "model",
        r"Доля Cu, $x_{\mathrm{Cu}}$",
        "RMSE сил, мэВ/Å",
        "DFT test: MACE по составу",
        outdir / "mace_target_force_rmse.png",
        args.dpi,
        xticks,
        xticklabels,
    )

    if args.mace_convergence:
        mace = pd.read_csv(args.mace_convergence)
        save_line_plot(
            mace,
            "step",
            "E_RMSE_meV_atom",
            "model",
            "Эпоха",
            "RMSE энергии, мэВ/атом",
            "Сходимость MACE: энергия",
            outdir / "mace_energy_convergence.png",
            args.dpi,
        )
        save_line_plot(
            mace,
            "step",
            "F_RMSE_meV_A",
            "model",
            "Эпоха",
            "RMSE сил, мэВ/Å",
            "Сходимость MACE: силы",
            outdir / "mace_force_convergence.png",
            args.dpi,
        )

    if args.ace_convergence:
        ace = pd.read_csv(args.ace_convergence)
        save_line_plot(
            ace,
            "step",
            "E_RMSE_meV_atom",
            "model",
            "Итерация",
            "RMSE энергии, мэВ/атом",
            "Сходимость ACE: энергия",
            outdir / "ace_energy_convergence.png",
            args.dpi,
            logy=True,
        )
        save_line_plot(
            ace,
            "step",
            "F_RMSE_meV_A",
            "model",
            "Итерация",
            "RMSE сил, мэВ/Å",
            "Сходимость ACE: силы",
            outdir / "ace_force_convergence.png",
            args.dpi,
            logy=True,
        )


if __name__ == "__main__":
    main()
