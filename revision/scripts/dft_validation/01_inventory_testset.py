#!/usr/bin/env python3
"""Inventory the untouched 4242-structure DFT test set.

Outputs:
  testset_inventory.json
  exact_composition_counts.csv
  target_composition_frames.csv

No phase labels are inferred. The test file does not contain reliable
liquid/amorphous/crystal metadata, so the revision uses composition only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from common import TARGET_LABELS, TARGET_NATOMS, TARGET_NCU, iter_extxyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test.extxyz")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    natoms_counts = Counter()
    composition_counts = Counter()
    header_key_counts = Counter()
    target_rows = []

    # Header-key inventory without a second full parser dependency.
    with Path(args.test).open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            natoms = int(line)
            header = handle.readline().rstrip("\n")
            header_key_counts.update(re.findall(r"(\w+)=", header))
            for _ in range(natoms):
                handle.readline()

    n_frames = 0
    n_atoms_total = 0
    for fr in iter_extxyz(args.test):
        n_frames += 1
        n_atoms_total += fr["n"]
        natoms_counts[fr["n"]] += 1
        composition_counts[(fr["n_Cu"], fr["n_Zr"], fr["n"])] += 1

        if fr["n"] == TARGET_NATOMS and fr["n_Cu"] in TARGET_NCU:
            target_rows.append(
                {
                    "frame": fr["frame"],
                    "n_atoms": fr["n"],
                    "n_Cu": fr["n_Cu"],
                    "n_Zr": fr["n_Zr"],
                    "x_Cu": fr["x_Cu"],
                    "composition": TARGET_LABELS[fr["n_Cu"]],
                }
            )

    exact = pd.DataFrame(
        [
            {
                "n_atoms": natoms,
                "n_Cu": n_cu,
                "n_Zr": n_zr,
                "x_Cu": n_cu / natoms,
                "n_structures": count,
            }
            for (n_cu, n_zr, natoms), count in composition_counts.items()
        ]
    ).sort_values(["n_atoms", "x_Cu"])
    exact.to_csv(outdir / "exact_composition_counts.csv", index=False)

    target = pd.DataFrame(target_rows)
    target.to_csv(outdir / "target_composition_frames.csv", index=False)

    inventory = {
        "test_file": str(Path(args.test).resolve()),
        "n_structures": n_frames,
        "n_atoms_total": n_atoms_total,
        "distinct_exact_compositions": len(composition_counts),
        "natoms_counts": {str(k): int(v) for k, v in sorted(natoms_counts.items())},
        "header_key_counts": dict(header_key_counts),
        "target_selection": {
            "n_atoms": TARGET_NATOMS,
            "n_Cu_values": list(TARGET_NCU),
            "labels": {str(k): v for k, v in TARGET_LABELS.items()},
            "n_selected": len(target_rows),
            "counts": (
                target.groupby("composition").size().astype(int).to_dict()
                if len(target) else {}
            ),
        },
        "phase_classification": "not_performed",
        "phase_note": (
            "No reliable phase label is present in test.extxyz; "
            "the Paper 1 revision does not infer liquid/amorphous/crystal labels."
        ),
    }

    (outdir / "testset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
