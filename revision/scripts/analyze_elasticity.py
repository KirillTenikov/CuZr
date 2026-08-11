#!/usr/bin/env python3
"""
Analyze result.json files produced by run_elasticity.py.

Primary estimates:
  K  from -d<P>/d(epsilon_v)
  G  from -dP_xy/d(gamma_xy), etc.

LAMMPS reports a pressure tensor whose sign is opposite the conventional
Cauchy stress used in Hooke's law, hence the minus signs above.

Cross-checks:
  K and G from quadratic energy-vs-strain curvature.

Outputs:
  elasticity_points.csv
  elasticity_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


BAR_TO_GPA = 1.0e-4
EV_A3_TO_GPA = 160.21766208


def load_results(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/*/result.json")):
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        rows.append(data)

    if not rows:
        raise RuntimeError(f"No result.json files found below {root}")

    return rows


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    coeff = np.polyfit(x, y, 1)
    slope, intercept = float(coeff[0]), float(coeff[1])
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": int(len(x)),
    }


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> dict:
    # y = a x^2 + b x + c
    a, b, c = [float(v) for v in np.polyfit(x, y, 2)]
    pred = a * x * x + b * x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    return {
        "a": a,
        "b": b,
        "c": c,
        "r2": r2,
        "n": int(len(x)),
    }


def find_reference(rows: list[dict]) -> dict:
    refs = [r for r in rows if r["mode"] == "reference"]
    if len(refs) != 1:
        raise RuntimeError(f"Expected exactly one reference result, found {len(refs)}")
    return refs[0]


def with_reference(rows: list[dict], mode: str, reference: dict) -> list[dict]:
    vals = [r for r in rows if r["mode"] == mode]
    return [reference] + vals


def pressure_mean_bar(row: dict) -> float:
    p = row["pressure_bar"]
    return (p["xx"] + p["yy"] + p["zz"]) / 3.0


def write_points_csv(root: Path, rows: list[dict]) -> None:
    out = root / "elasticity_points.csv"
    fields = [
        "mode", "strain", "pe_eV", "volume_A3",
        "pxx_bar", "pyy_bar", "pzz_bar",
        "pxy_bar", "pxz_bar", "pyz_bar",
        "mean_pressure_bar", "path",
    ]

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda z: (z["mode"], z["strain"])):
            p = r["pressure_bar"]
            w.writerow({
                "mode": r["mode"],
                "strain": r["strain"],
                "pe_eV": r["pe_eV"],
                "volume_A3": r["volume_A3"],
                "pxx_bar": p["xx"],
                "pyy_bar": p["yy"],
                "pzz_bar": p["zz"],
                "pxy_bar": p["xy"],
                "pxz_bar": p["xz"],
                "pyz_bar": p["yz"],
                "mean_pressure_bar": pressure_mean_bar(r),
                "path": r["_path"],
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze relaxed-ion elasticity results.")
    ap.add_argument("root", type=Path, help="One potential/seed elasticity result directory.")
    ap.add_argument(
        "--warn-r2",
        type=float,
        default=0.995,
        help="Warn when a stress-strain linear fit has R^2 below this value.",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    rows = load_results(root)
    reference = find_reference(rows)
    write_points_csv(root, rows)

    V0 = float(reference["volume_A3"])
    E0 = float(reference["pe_eV"])
    p0 = reference["pressure_bar"]
    mean_p0 = pressure_mean_bar(reference)

    summary = {
        "reference": {
            "volume_A3": V0,
            "pe_eV": E0,
            "pressure_bar": p0,
            "mean_pressure_bar": mean_p0,
        },
        "stress_strain": {},
        "energy_curvature": {},
        "derived_isotropic": {},
        "warnings": [],
    }

    # ---------- Bulk modulus from pressure-vs-volumetric-strain ----------
    bulk = with_reference(rows, "bulk", reference)
    if len(bulk) >= 3:
        eps = np.array(
            [0.0 if r["mode"] == "reference" else float(r["strain"]) for r in bulk],
            dtype=float,
        )
        pmean = np.array([pressure_mean_bar(r) for r in bulk], dtype=float)

        fit = linear_fit(eps, pmean)
        K_stress_GPa = -fit["slope"] * BAR_TO_GPA
        summary["stress_strain"]["bulk"] = {
            "fit_pressure_bar_vs_volumetric_strain": fit,
            "K_GPa": K_stress_GPa,
        }

        energies = np.array([float(r["pe_eV"]) for r in bulk], dtype=float)
        efit = quadratic_fit(eps, energies)
        K_energy_GPa = (2.0 * efit["a"] / V0) * EV_A3_TO_GPA
        summary["energy_curvature"]["bulk"] = {
            "fit_energy_eV_vs_volumetric_strain": efit,
            "K_GPa": K_energy_GPa,
        }

        if fit["r2"] < args.warn_r2:
            summary["warnings"].append(
                f"Bulk stress-strain R^2={fit['r2']:.6f} < {args.warn_r2:.6f}"
            )
        if K_stress_GPa <= 0:
            summary["warnings"].append(f"Non-positive stress-derived bulk modulus: {K_stress_GPa:.6g} GPa")
    else:
        summary["warnings"].append("Bulk results incomplete; K not calculated.")

    # ---------- Shear moduli ----------
    shear_component = {"xy": "xy", "xz": "xz", "yz": "yz"}
    shear_values = []
    shear_energy_values = []

    for mode, component in shear_component.items():
        pts = with_reference(rows, mode, reference)
        if len(pts) < 3:
            summary["warnings"].append(f"{mode} shear results incomplete; G_{mode} not calculated.")
            continue

        gamma = np.array(
            [0.0 if r["mode"] == "reference" else float(r["strain"]) for r in pts],
            dtype=float,
        )
        pshear = np.array([float(r["pressure_bar"][component]) for r in pts], dtype=float)

        fit = linear_fit(gamma, pshear)
        G_stress_GPa = -fit["slope"] * BAR_TO_GPA
        shear_values.append(G_stress_GPa)

        summary["stress_strain"][mode] = {
            f"fit_p{component}_bar_vs_engineering_shear": fit,
            "G_GPa": G_stress_GPa,
        }

        energies = np.array([float(r["pe_eV"]) for r in pts], dtype=float)
        efit = quadratic_fit(gamma, energies)
        G_energy_GPa = (2.0 * efit["a"] / V0) * EV_A3_TO_GPA
        shear_energy_values.append(G_energy_GPa)

        summary["energy_curvature"][mode] = {
            "fit_energy_eV_vs_engineering_shear": efit,
            "G_GPa": G_energy_GPa,
        }

        if fit["r2"] < args.warn_r2:
            summary["warnings"].append(
                f"{mode} stress-strain R^2={fit['r2']:.6f} < {args.warn_r2:.6f}"
            )
        if G_stress_GPa <= 0:
            summary["warnings"].append(f"Non-positive stress-derived G_{mode}: {G_stress_GPa:.6g} GPa")

    # ---------- Isotropic averages and E, nu ----------
    K = summary["stress_strain"].get("bulk", {}).get("K_GPa")
    if shear_values:
        G = statistics.fmean(shear_values)
        G_sd = statistics.stdev(shear_values) if len(shear_values) > 1 else 0.0
        summary["derived_isotropic"]["G_mean_GPa"] = G
        summary["derived_isotropic"]["G_directional_sd_GPa"] = G_sd
        summary["derived_isotropic"]["G_components_GPa"] = shear_values

        if K is not None and (3.0 * K + G) != 0.0:
            E = 9.0 * K * G / (3.0 * K + G)
            nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
            summary["derived_isotropic"].update({
                "K_GPa": K,
                "G_GPa": G,
                "E_GPa": E,
                "nu": nu,
            })

    if shear_energy_values:
        summary["derived_isotropic"]["G_energy_mean_GPa"] = statistics.fmean(shear_energy_values)

    K_energy = summary["energy_curvature"].get("bulk", {}).get("K_GPa")
    if K is not None and K_energy is not None and K != 0:
        summary["derived_isotropic"]["K_energy_vs_stress_relative_difference"] = abs(K_energy - K) / abs(K)

    G_stress = summary["derived_isotropic"].get("G_mean_GPa")
    G_energy = summary["derived_isotropic"].get("G_energy_mean_GPa")
    if G_stress is not None and G_energy is not None and G_stress != 0:
        summary["derived_isotropic"]["G_energy_vs_stress_relative_difference"] = abs(G_energy - G_stress) / abs(G_stress)

    out = root / "elasticity_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote: {root / 'elasticity_points.csv'}")
    print(f"Wrote: {out}")
    print()

    if K is not None:
        print(f"K = {K:.6f} GPa")
    if G_stress is not None:
        print(f"G = {G_stress:.6f} GPa")
    if "E_GPa" in summary["derived_isotropic"]:
        print(f"E = {summary['derived_isotropic']['E_GPa']:.6f} GPa")
        print(f"nu = {summary['derived_isotropic']['nu']:.8f}")

    if summary["warnings"]:
        print("\nWARNINGS:")
        for w in summary["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
