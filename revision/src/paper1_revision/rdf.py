from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _smooth(values, window: int = 7):
    import numpy as np

    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def first_peak_and_minimum(r, g, min_peak_r: float = 1.5) -> tuple[int, int]:
    import numpy as np

    r = np.asarray(r, dtype=float)
    gs = _smooth(g)
    candidates = np.where(r >= min_peak_r)[0]
    if len(candidates) == 0:
        raise ValueError("No RDF bins above min_peak_r")
    start = int(candidates[0])
    peak = start + int(np.argmax(gs[start:]))
    minimum = None
    for idx in range(peak + 1, len(gs) - 1):
        if gs[idx - 1] >= gs[idx] <= gs[idx + 1]:
            minimum = idx
            break
    if minimum is None:
        minimum = min(len(gs) - 1, peak + max(3, len(gs) // 10))
    return peak, minimum


def analyze_structure(
    structure_file: Path,
    output_dir: Path,
    rmax_A: float = 8.0,
    bins: int = 400,
    type_map: tuple[str, str] = ("Cu", "Zr"),
) -> dict:
    """Compute total/partial RDFs and heuristic first-shell coordination.

    Requires ASE and NumPy. Pair counting is performed with ASE's periodic
    neighbor list and unique pairs (i < j), avoiding a dense O(N^2) matrix.
    """
    import numpy as np
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms = read(str(structure_file), format="lammps-data", style="atomic", sort_by_id=True)
    symbols = atoms.get_chemical_symbols()
    if set(symbols) != set(type_map):
        # ASE may assign generic symbols from type numbers; enforce the documented Cu/Zr map.
        arrays = atoms.arrays
        numbers = arrays.get("numbers")
        if numbers is None:
            raise ValueError("Could not determine atom types from LAMMPS data")
        unique = sorted(set(int(x) for x in numbers))
        if len(unique) != 2:
            raise ValueError(f"Expected two atom types, found {unique}")
        mapping = {unique[0]: type_map[0], unique[1]: type_map[1]}
        atoms.set_chemical_symbols([mapping[int(x)] for x in numbers])
        symbols = atoms.get_chemical_symbols()

    i, j, d = neighbor_list("ijd", atoms, rmax_A, self_interaction=False)
    unique_mask = i < j
    i, j, d = i[unique_mask], j[unique_mask], d[unique_mask]

    edges = np.linspace(0.0, rmax_A, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(atoms.get_volume())
    n_total = len(atoms)
    counts_by_element = {element: symbols.count(element) for element in type_map}

    pair_defs = [
        ("total", None, None),
        ("CuCu", "Cu", "Cu"),
        ("CuZr", "Cu", "Zr"),
        ("ZrZr", "Zr", "Zr"),
    ]
    result: dict[str, object] = {
        "structure_file": str(structure_file),
        "natoms": n_total,
        "volume_A3": volume,
        "number_density_A3": n_total / volume,
        "counts": counts_by_element,
        "rmax_A": rmax_A,
        "bins": bins,
        "pairs": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | str]] = []
    symbols_arr = np.asarray(symbols)

    for label, a, b in pair_defs:
        if label == "total":
            mask = np.ones(len(d), dtype=bool)
            ideal_pairs = n_total * (n_total - 1) / 2.0
        elif a == b:
            mask = (symbols_arr[i] == a) & (symbols_arr[j] == b)
            n_a = counts_by_element[a]
            ideal_pairs = n_a * (n_a - 1) / 2.0
        else:
            mask = ((symbols_arr[i] == a) & (symbols_arr[j] == b)) | ((symbols_arr[i] == b) & (symbols_arr[j] == a))
            ideal_pairs = counts_by_element[a] * counts_by_element[b]

        hist, _ = np.histogram(d[mask], bins=edges)
        expected = ideal_pairs * shell_volume / volume
        g = np.divide(hist, expected, out=np.zeros_like(expected), where=expected > 0)
        peak_idx, min_idx = first_peak_and_minimum(centers, g)
        cutoff = float(centers[min_idx])

        if label == "total":
            coordination = 2.0 * float(np.count_nonzero(mask & (d <= cutoff))) / n_total
        elif a == b:
            coordination = 2.0 * float(np.count_nonzero(mask & (d <= cutoff))) / counts_by_element[a]
        else:
            n_pairs_shell = float(np.count_nonzero(mask & (d <= cutoff)))
            coordination = {
                f"{a}_around_{b}": n_pairs_shell / counts_by_element[b],
                f"{b}_around_{a}": n_pairs_shell / counts_by_element[a],
            }

        pair_payload = {
            "first_peak_r_A": float(centers[peak_idx]),
            "first_peak_g": float(g[peak_idx]),
            "first_minimum_r_A": cutoff,
            "coordination": coordination,
            "pair_count_within_rmax": int(np.count_nonzero(mask)),
        }
        result["pairs"][label] = pair_payload
        for radius, value in zip(centers, g):
            all_rows.append({"pair": label, "r_A": float(radius), "g_r": float(value)})

    with (output_dir / "rdf_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "r_A", "g_r"])
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "structure_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
