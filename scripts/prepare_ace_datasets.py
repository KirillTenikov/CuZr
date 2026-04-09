#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

AUTO_ENERGY_KEYS = ["REF_energy", "energy", "Energy", "free_energy"]
AUTO_FORCE_KEYS = ["REF_forces", "forces", "force", "Forces"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CuZr extxyz splits into pacemaker-compatible pandas DataFrames."
    )
    p.add_argument("--train-input", required=True, help="Path to train_split.extxyz")
    p.add_argument("--valid-input", required=True, help="Path to valid_split.extxyz")
    p.add_argument("--test-input", required=False, default=None, help="Optional path to test.extxyz")
    p.add_argument("--output-dir", required=True, help="Output directory for *.pckl.gzip files")
    p.add_argument(
        "--energy-key",
        default="auto",
        help="Energy key in Atoms.info. Use 'auto' to try common names.",
    )
    p.add_argument(
        "--forces-key",
        default="auto",
        help="Forces key in Atoms.arrays. Use 'auto' to try common names.",
    )
    p.add_argument(
        "--e0-json",
        default=None,
        help='Optional JSON string or JSON file with isolated atom energies, e.g. {"Cu": -3.7, "Zr": -6.2}.',
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def load_frames(path: Path) -> List[Atoms]:
    frames = read(path.as_posix(), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError(f"No structures found in {path}")
    return frames


def resolve_energy_key(atoms: Atoms, requested: str) -> str:
    if requested != "auto":
        if requested not in atoms.info:
            raise KeyError(
                f"Requested energy key '{requested}' not found. Available info keys: {list(atoms.info.keys())}"
            )
        return requested

    for key in AUTO_ENERGY_KEYS:
        if key in atoms.info:
            return key
    raise KeyError(
        f"Could not auto-detect energy key. Available info keys: {list(atoms.info.keys())}"
    )


def resolve_forces_key(atoms: Atoms, requested: str) -> str:
    if requested != "auto":
        if requested not in atoms.arrays:
            raise KeyError(
                f"Requested forces key '{requested}' not found. Available array keys: {list(atoms.arrays.keys())}"
            )
        return requested

    for key in AUTO_FORCE_KEYS:
        if key in atoms.arrays:
            return key
    raise KeyError(
        f"Could not auto-detect forces key. Available array keys: {list(atoms.arrays.keys())}"
    )


def parse_e0_json(raw: str | None) -> Dict[str, float] | None:
    if raw is None:
        return None
    candidate = Path(raw)
    if candidate.exists():
        return {str(k): float(v) for k, v in json.loads(candidate.read_text(encoding="utf-8")).items()}
    return {str(k): float(v) for k, v in json.loads(raw).items()}


def corrected_energy(atoms: Atoms, energy: float, e0: Dict[str, float] | None) -> float:
    if not e0:
        return float(energy)

    ref = 0.0
    for sym in atoms.get_chemical_symbols():
        if sym not in e0:
            raise KeyError(f"Missing isolated atom energy for element '{sym}' in e0 mapping.")
        ref += float(e0[sym])
    return float(energy - ref)


def convert_split(
    input_path: Path,
    output_path: Path,
    energy_key_arg: str,
    forces_key_arg: str,
    e0: Dict[str, float] | None,
    overwrite: bool,
) -> Dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = load_frames(input_path)
    first = frames[0]
    energy_key = resolve_energy_key(first, energy_key_arg)
    forces_key = resolve_forces_key(first, forces_key_arg)

    rows = []
    natoms_total = 0
    for at in frames:
        energy = float(at.info[energy_key])
        forces = np.asarray(at.arrays[forces_key], dtype=float)
        if forces.shape != (len(at), 3):
            raise ValueError(
                f"Forces for one structure have shape {forces.shape}, expected ({len(at)}, 3)."
            )
        at_copy = at.copy()
        rows.append(
            {
                "ase_atoms": at_copy,
                "energy": energy,
                "forces": forces,
                "energy_corrected": corrected_energy(at_copy, energy, e0),
            }
        )
        natoms_total += len(at_copy)

    df = pd.DataFrame(rows)
    df.to_pickle(output_path, compression="gzip", protocol=4)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "n_structures": int(len(df)),
        "n_atoms_total": int(natoms_total),
        "energy_key": energy_key,
        "forces_key": forces_key,
    }


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    e0 = parse_e0_json(args.e0_json)

    summary: Dict[str, object] = {}
    summary["train"] = convert_split(
        input_path=Path(args.train_input).resolve(),
        output_path=out_dir / "train_split.pckl.gzip",
        energy_key_arg=args.energy_key,
        forces_key_arg=args.forces_key,
        e0=e0,
        overwrite=args.overwrite,
    )
    summary["valid"] = convert_split(
        input_path=Path(args.valid_input).resolve(),
        output_path=out_dir / "valid_split.pckl.gzip",
        energy_key_arg=args.energy_key,
        forces_key_arg=args.forces_key,
        e0=e0,
        overwrite=args.overwrite,
    )
    if args.test_input:
        summary["test"] = convert_split(
            input_path=Path(args.test_input).resolve(),
            output_path=out_dir / "test.pckl.gzip",
            energy_key_arg=args.energy_key,
            forces_key_arg=args.forces_key,
            e0=e0,
            overwrite=args.overwrite,
        )

    meta_path = out_dir / "ace_dataset_conversion_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
