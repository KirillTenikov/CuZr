#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REVISION_ROOT = HERE.parents[1]
sys.path.insert(0, str(REVISION_ROOT / "src"))

from paper1_revision.config import load_potentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paired Paper 1 amorphous-validation matrix without executing it.")
    parser.add_argument("--matrix", type=Path, default=REVISION_ROOT / "config" / "matrix.json")
    parser.add_argument("--repo-root", type=Path, default=REVISION_ROOT.parent)
    parser.add_argument("--results-root", type=Path, default=REVISION_ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.matrix.read_text(encoding="utf-8"))
    potentials = config["potentials"]
    registry = load_potentials(REVISION_ROOT / "config" / "potentials.json")
    unavailable = [pid for pid in potentials if pid not in registry or not registry[pid].enabled or not registry[pid].path.strip()]
    if unavailable:
        raise SystemExit(
            "The matrix was not generated because these potentials are not configured and enabled: "
            + ", ".join(unavailable)
        )
    seeds = config["seeds"]
    natoms_values = config["natoms"]
    compositions = config["compositions"]
    runner = REVISION_ROOT / "scripts" / "run_glass_preparation.py"

    commands: list[list[str]] = []
    for comp in compositions:
        density = comp.get("initial_density_g_cm3")
        if density is None:
            raise SystemExit(f"Set initial_density_g_cm3 for {comp['id']} in {args.matrix} before generating the matrix.")
        for natoms in natoms_values:
            for seed in seeds:
                for potential in potentials:
                    command = [
                        sys.executable,
                        str(runner),
                        "--repo-root", str(args.repo_root),
                        "--results-root", str(args.results_root),
                        "--potential", potential,
                        "--composition", comp["id"],
                        "--natoms", str(natoms),
                        "--seed", str(seed),
                        "--initial-density", str(density),
                    ]
                    commands.append(command)

    for command in commands:
        print("[generate]", " ".join(command))
        subprocess.run(command, check=True)
    print(f"Generated {len(commands)} run directories. No simulations were executed.")


if __name__ == "__main__":
    main()
