#!/usr/bin/env python3
"""Run detailed MACE inference on the 127 application-relevant DFT structures.

The selected structures are the N=128 frames with n_Cu = 46, 64, 82,
corresponding to x_Cu = 0.359375, 0.500000, 0.640625.

Per-structure outputs:
  dE_pa_meV          signed energy error per atom
  abs_dE_pa_meV      absolute energy error per atom
  force_rmse_meV_A   RMSE over all 3N force components
  force_mae_meV_A    MAE over all 3N force components
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    TARGET_LABELS,
    iter_extxyz,
    make_mace_batch,
    select_target_frames,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", required=True, help="Compiled TorchScript MACE model")
    parser.add_argument("--model-name", required=True, choices=["MACE_A", "MACE_B", "MACE_C", "MACE_D"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write the growing CSV every N structures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, int(args.threads)))

    frames = select_target_frames(iter_extxyz(args.test))
    counts = {
        int(n_cu): sum(fr["n_Cu"] == n_cu for fr in frames)
        for n_cu in sorted({fr["n_Cu"] for fr in frames})
    }
    print(
        f"{args.model_name}: selected {len(frames)} structures; "
        f"n_Cu counts={counts}",
        flush=True,
    )

    if len(frames) != 127:
        raise RuntimeError(
            f"Expected 127 target structures for the frozen Paper 1 dataset; "
            f"found {len(frames)}"
        )

    model = torch.jit.load(args.model, map_location="cpu").eval()
    rmax = float(model.r_max)

    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for i, fr in enumerate(frames, start=1):
        data = make_mace_batch([fr], rmax)
        ref_forces = torch.from_numpy(fr["forces"].astype(np.float32))

        result = model(data, training=False, compute_force=True)
        pred_energy = float(result["energy"].detach().cpu()[0])
        pred_forces = result["forces"].detach().cpu()

        force_diff = (
            pred_forces - ref_forces
        ).numpy().astype(np.float64)

        dE_pa_meV = (
            (pred_energy - fr["energy"]) / fr["n"] * 1000.0
        )

        rows.append(
            {
                "model": args.model_name,
                "frame": fr["frame"],
                "composition": TARGET_LABELS[fr["n_Cu"]],
                "n_atoms": fr["n"],
                "n_Cu": fr["n_Cu"],
                "n_Zr": fr["n_Zr"],
                "x_Cu": fr["x_Cu"],
                "E_dft_eV": fr["energy"],
                "E_pred_eV": pred_energy,
                "dE_pa_meV": dE_pa_meV,
                "abs_dE_pa_meV": abs(dE_pa_meV),
                "force_rmse_meV_A": float(
                    np.sqrt(np.mean(force_diff ** 2)) * 1000.0
                ),
                "force_mae_meV_A": float(
                    np.mean(np.abs(force_diff)) * 1000.0
                ),
            }
        )

        if i % args.checkpoint_every == 0 or i == len(frames):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(
                f"{args.model_name}: {i}/{len(frames)} "
                f"elapsed={time.time() - t0:.1f} s",
                flush=True,
            )

        del data, ref_forces, result, pred_forces, force_diff
        gc.collect()

    df = pd.DataFrame(rows)

    print(f"\n{args.model_name} summary")
    for composition, group in df.groupby("composition", sort=False):
        e_rmse = float(np.sqrt(np.mean(group["dE_pa_meV"] ** 2)))
        f_rmse = float(
            np.sqrt(
                np.average(
                    group["force_rmse_meV_A"] ** 2,
                    weights=group["n_atoms"],
                )
            )
        )
        print(
            composition,
            "n=", len(group),
            "E_RMSE=", f"{e_rmse:.6f}",
            "F_RMSE=", f"{f_rmse:.6f}",
            "F_p95=", f"{group['force_rmse_meV_A'].quantile(.95):.6f}",
        )

    print("wrote", out_path)


if __name__ == "__main__":
    main()
