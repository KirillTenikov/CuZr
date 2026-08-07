from __future__ import annotations

import math
import random
from pathlib import Path

from .constants import AMU_TO_G, CM3_TO_A3, CU_MASS, ZR_MASS


def parse_composition(composition: str) -> tuple[int, int]:
    import re

    match = re.fullmatch(r"Cu(\d+)Zr(\d+)", composition.strip())
    if not match:
        raise ValueError(f"Invalid composition id: {composition!r}")
    cu_pct, zr_pct = int(match.group(1)), int(match.group(2))
    if cu_pct + zr_pct != 100:
        raise ValueError(f"Composition must sum to 100: {composition}")
    return cu_pct, zr_pct


def composition_counts(natoms: int, composition: str) -> tuple[int, int]:
    if natoms < 2:
        raise ValueError("natoms must be at least 2")
    cu_pct, _ = parse_composition(composition)
    n_cu = int(round(natoms * cu_pct / 100.0))
    n_cu = min(max(n_cu, 1), natoms - 1)
    return n_cu, natoms - n_cu


def estimate_box_length_A(n_cu: int, n_zr: int, density_g_cm3: float) -> float:
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be positive")
    mass_g = (n_cu * CU_MASS + n_zr * ZR_MASS) * AMU_TO_G
    volume_cm3 = mass_g / density_g_cm3
    volume_A3 = volume_cm3 * CM3_TO_A3
    return volume_A3 ** (1.0 / 3.0)


def write_initial_data(
    path: Path,
    natoms: int,
    composition: str,
    density_g_cm3: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Write a deterministic, non-overlapping lattice-like random alloy.

    This structure is only a safe starting condition. It is minimized, melted,
    quenched, pressure-relaxed and equilibrated before any measurement.
    """
    rng = random.Random(seed)
    n_cu, n_zr = composition_counts(natoms, composition)
    box_length = estimate_box_length_A(n_cu, n_zr, density_g_cm3)
    ngrid = math.ceil(natoms ** (1.0 / 3.0))
    spacing = box_length / ngrid
    jitter = 0.12 * spacing

    positions: list[tuple[float, float, float]] = []
    for ix in range(ngrid):
        for iy in range(ngrid):
            for iz in range(ngrid):
                if len(positions) >= natoms:
                    break
                x = (ix + 0.5) * spacing + rng.uniform(-jitter, jitter)
                y = (iy + 0.5) * spacing + rng.uniform(-jitter, jitter)
                z = (iz + 0.5) * spacing + rng.uniform(-jitter, jitter)
                positions.append((x % box_length, y % box_length, z % box_length))
            if len(positions) >= natoms:
                break
        if len(positions) >= natoms:
            break

    atom_types = [1] * n_cu + [2] * n_zr
    rng.shuffle(atom_types)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Cu-Zr initial structure for Paper 1 revision\n\n")
        handle.write(f"{natoms} atoms\n")
        handle.write("2 atom types\n\n")
        handle.write(f"0.0 {box_length:.10f} xlo xhi\n")
        handle.write(f"0.0 {box_length:.10f} ylo yhi\n")
        handle.write(f"0.0 {box_length:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        handle.write(f"1 {CU_MASS:.8f} # Cu\n")
        handle.write(f"2 {ZR_MASS:.8f} # Zr\n\n")
        handle.write("Atoms # atomic\n\n")
        for atom_id, (atype, xyz) in enumerate(zip(atom_types, positions), start=1):
            x, y, z = xyz
            handle.write(f"{atom_id} {atype} {x:.10f} {y:.10f} {z:.10f}\n")

    return {
        "composition": composition,
        "natoms": natoms,
        "n_cu": n_cu,
        "n_zr": n_zr,
        "cu_fraction_actual": n_cu / natoms,
        "zr_fraction_actual": n_zr / natoms,
        "initial_density_g_cm3": density_g_cm3,
        "box_length_A": box_length,
        "grid": ngrid,
        "spacing_A": spacing,
        "seed": seed,
    }
