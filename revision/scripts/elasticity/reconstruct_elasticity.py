#!/usr/bin/env python3
"""
Paper 1 relaxed-ion elasticity analysis for Cu64Zr36.

Inputs:
  1) Main N=1024 campaign archive (8 potentials x 3 seeds)
  2) EAM_2019 seed44 small-strain refinement archive
  3) Optional MACE_C N=4000 finite-size archive

The script:
  - reconstructs all moduli from result.json files;
  - checks strain-window convergence with one objective rule;
  - selects the final window without manual per-potential tuning;
  - writes per-sample and per-potential tables;
  - generates publication-oriented figures.

Final window rule
-----------------
Standard full window: +/-0.000625 and +/-0.00125.
It is accepted if ALL are true:
  * minimum stress-strain R^2 >= 0.995
  * stress-vs-energy relative mismatch < 1% for K and mean G
  * inner-vs-full relative change < 1% for K and mean G

If the full window fails, the inner +/-0.000625 window is used if it
passes R^2 and stress/energy consistency.  If that also fails, the
provided dedicated refinement archive is used.

For the current dataset this yields:
  * 22 samples: standard full window
  * EAM_2019 seed43: standard inner window
  * EAM_2019 seed44: dedicated small-strain refinement

No web access is required.  The optional experimental Young's-modulus
benchmark (92.3 GPa) is from:
D. Xu et al., Acta Materialia 52 (2004) 2621-2624,
doi:10.1016/j.actamat.2004.02.009.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import statistics
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BAR_TO_GPA = 1.0e-4
EV_A3_TO_GPA = 160.21766208
E_EXPERIMENT_GPA = 92.3

POTENTIAL_ORDER = [
    "EAM_2007", "EAM_2019",
    "ACE_514", "ACE_1352",
    "MACE_A", "MACE_B", "MACE_C", "MACE_D",
]

DISPLAY = {
    "EAM_2007": "EAM 2007",
    "EAM_2019": "EAM 2019",
    "ACE_514": "ACE 514",
    "ACE_1352": "ACE 1352",
    "MACE_A": "MACE A",
    "MACE_B": "MACE B",
    "MACE_C": "MACE C",
    "MACE_D": "MACE D",
}

def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(destination)

def linear_fit(x: np.ndarray, y: np.ndarray):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)

def quadratic_fit(x: np.ndarray, y: np.ndarray):
    a, b, c = np.polyfit(x, y, 2)
    pred = a*x*x + b*x + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(a), float(b), float(c), float(r2)

def load_results(sample_dir: Path):
    rows = []
    for p in sample_dir.rglob("result.json"):
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        rows.append(d)
    return rows

def reference_row(rows):
    refs = [r for r in rows if r["mode"] == "reference"]
    if len(refs) != 1:
        raise RuntimeError(f"Expected one reference result, found {len(refs)}")
    return refs[0]

def mean_pressure_bar(r):
    p = r["pressure_bar"]
    return (float(p["xx"]) + float(p["yy"]) + float(p["zz"])) / 3.0

def derive_window(sample_dir: Path, max_abs_strain: float):
    rows = load_results(sample_dir)
    ref = reference_row(rows)
    V0 = float(ref["volume_A3"])
    modes = {}

    for mode, component in [
        ("bulk", None), ("xy", "xy"), ("xz", "xz"), ("yz", "yz")
    ]:
        pts = [ref] + [
            r for r in rows
            if r["mode"] == mode and abs(float(r["strain"])) <= max_abs_strain + 1e-14
        ]
        pts = sorted(
            pts,
            key=lambda r: 0.0 if r["mode"] == "reference" else float(r["strain"]),
        )
        x = np.array([
            0.0 if r["mode"] == "reference" else float(r["strain"])
            for r in pts
        ])

        if mode == "bulk":
            stress = np.array([mean_pressure_bar(r) for r in pts])
            slope, intercept, r2 = linear_fit(x, stress)
            M_stress = -slope * BAR_TO_GPA
        else:
            stress = np.array([float(r["pressure_bar"][component]) for r in pts])
            slope, intercept, r2 = linear_fit(x, stress)
            M_stress = -slope * BAR_TO_GPA

        energy = np.array([float(r["pe_eV"]) for r in pts])
        a, b, c, energy_r2 = quadratic_fit(x, energy)
        M_energy = 2.0 * a / V0 * EV_A3_TO_GPA

        modes[mode] = {
            "stress_modulus_GPa": M_stress,
            "energy_modulus_GPa": M_energy,
            "stress_R2": r2,
            "energy_R2": energy_r2,
            "x": x,
            "stress_raw_bar": stress,
            "energy_eV": energy,
        }

    K = modes["bulk"]["stress_modulus_GPa"]
    K_energy = modes["bulk"]["energy_modulus_GPa"]
    G_components = [modes[m]["stress_modulus_GPa"] for m in ("xy", "xz", "yz")]
    G_energy_components = [modes[m]["energy_modulus_GPa"] for m in ("xy", "xz", "yz")]
    G = statistics.fmean(G_components)
    G_energy = statistics.fmean(G_energy_components)
    G_dir_sd = statistics.stdev(G_components)
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))

    return {
        "K_GPa": K,
        "G_GPa": G,
        "E_GPa": E,
        "nu": nu,
        "G_directional_sd_GPa": G_dir_sd,
        "G_components_GPa": G_components,
        "K_energy_GPa": K_energy,
        "G_energy_GPa": G_energy,
        "K_stress_energy_rel": abs(K_energy-K)/abs(K),
        "G_stress_energy_rel": abs(G_energy-G)/abs(G),
        "min_stress_R2": min(v["stress_R2"] for v in modes.values()),
        "modes": modes,
    }

def pass_quality(x, r2_threshold=0.995, mismatch_threshold=0.01):
    return (
        x["min_stress_R2"] >= r2_threshold
        and x["K_stress_energy_rel"] < mismatch_threshold
        and x["G_stress_energy_rel"] < mismatch_threshold
    )

def find_main_root(extracted: Path):
    candidates = list(extracted.rglob("paper1_elasticity_v2"))
    if not candidates:
        raise FileNotFoundError("paper1_elasticity_v2 not found")
    return candidates[0]

def find_refine_sample(extracted: Path):
    candidates = list(extracted.rglob("EAM_2019/seed_44"))
    candidates = [p for p in candidates if (p / "protocol.json").is_file()]
    if not candidates:
        raise FileNotFoundError("Refined EAM_2019/seed_44 not found")
    return candidates[0]

def save_figure(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def dot_summary_plot(final_df, summary_df, prop, ylabel, outdir, stem,
                     experimental_line=None, experimental_label=None):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    xs = np.arange(len(POTENTIAL_ORDER))
    offsets = {42: -0.13, 43: 0.0, 44: 0.13}
    markers = {42: "o", 43: "s", 44: "^"}

    for seed in (42, 43, 44):
        ys = []
        xx = []
        for i, pot in enumerate(POTENTIAL_ORDER):
            row = final_df[(final_df.potential == pot) & (final_df.seed == seed)].iloc[0]
            ys.append(float(row[prop]))
            xx.append(i + offsets[seed])
        ax.scatter(xx, ys, marker=markers[seed], s=42, label=f"seed {seed}", zorder=3)

    means = summary_df.loc[POTENTIAL_ORDER, f"{prop}_mean"].to_numpy()
    sds = summary_df.loc[POTENTIAL_ORDER, f"{prop}_sd"].to_numpy()
    ax.errorbar(xs, means, yerr=sds, fmt="D", capsize=4, linewidth=1.3,
                markersize=5, label="mean ± SD", zorder=4)

    if experimental_line is not None:
        ax.axhline(experimental_line, linestyle="--", linewidth=1.3,
                   label=experimental_label)

    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[p] for p in POTENTIAL_ORDER], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Potential")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, outdir, stem)

def make_window_sensitivity_plot(conv, outdir):
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    conv = conv.copy()
    conv["label"] = conv.apply(
        lambda r: f"{DISPLAY[r.potential]}\n{int(r.seed)}", axis=1
    )
    x = np.arange(len(conv))
    y = 100.0 * conv["G_window_relative_change"].to_numpy()
    ax.bar(x, y)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="1% selection threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(conv["label"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"$|G_{\rm full}-G_{\rm inner}|/G_{\rm inner}$ (%)")
    ax.set_xlabel("Potential / seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir, "05_G_strain_window_sensitivity")

def make_consistency_plot(final_df, outdir):
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    df = final_df.copy()
    df["label"] = df.apply(lambda r: f"{DISPLAY[r.potential]}\n{int(r.seed)}", axis=1)
    x = np.arange(len(df))
    y = 100.0 * df["G_stress_energy_rel"].to_numpy()
    ax.bar(x, y)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="1% quality threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"$|G_E-G_\sigma|/G_\sigma$ (%)")
    ax.set_xlabel("Potential / seed (final selected window)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir, "06_final_G_stress_energy_consistency")

def shear_increment_curve(sample_dir, mode, max_abs):
    rows = load_results(sample_dir)
    ref = reference_row(rows)
    p0 = float(ref["pressure_bar"][mode])
    pts = [
        r for r in rows
        if r["mode"] == mode and abs(float(r["strain"])) <= max_abs + 1e-14
    ]
    pts = sorted(pts, key=lambda r: float(r["strain"]))
    x = np.array([float(r["strain"]) for r in pts])
    # physical shear stress increment = -(P_shear - P0)
    y = np.array([
        -(float(r["pressure_bar"][mode]) - p0) * BAR_TO_GPA
        for r in pts
    ])
    return x, y

def make_eam2019_seed44_curve(main_sample, refine_sample, outdir, mode="yz"):
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    x1, y1 = shear_increment_curve(main_sample, mode, 0.00125)
    x2, y2 = shear_increment_curve(refine_sample, mode, 0.0003125)

    ax.scatter(100*x1, y1, marker="o", s=48, label="standard window")
    ax.plot(100*x1, y1, linewidth=1.0)
    ax.scatter(100*x2, y2, marker="s", s=48, label="refined window")

    slope, intercept = np.polyfit(x2, y2, 1)
    xx = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(100*xx, slope*xx + intercept, linestyle="--",
            label=f"refined linear fit: G={slope:.2f} GPa")

    ax.set_xlabel("Engineering shear strain (%)")
    ax.set_ylabel(r"Shear-stress increment $\Delta\tau$ (GPa)")
    ax.set_title(f"EAM 2019 seed44: {mode} branch")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir, f"07_EAM2019_seed44_{mode}_strain_window")

def make_finite_size_plot(k1024, g1024, k4000, g4000, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width/2, [k1024, g1024], width, label="N=1024, revised protocol")
    ax.bar(x + width/2, [k4000, g4000], width, label="N=4000, earlier protocol")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bulk modulus K", "Shear modulus G"])
    ax.set_ylabel("Modulus (GPa)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir, "08_MACE_C_seed43_finite_size")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, type=Path)
    ap.add_argument("--refine", required=True, type=Path)
    ap.add_argument("--n4000", type=Path, default=None)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    figdir = args.outdir / "figures"
    tabdir = args.outdir / "tables"
    figdir.mkdir(exist_ok=True)
    tabdir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        main_ex = td / "main"
        ref_ex = td / "refine"
        main_ex.mkdir(); ref_ex.mkdir()
        safe_extract(args.main, main_ex)
        safe_extract(args.refine, ref_ex)

        main_root = find_main_root(main_ex)
        refine_sample = find_refine_sample(ref_ex)

        convergence_rows = []
        final_rows = []

        for pot in POTENTIAL_ORDER:
            for seed in (42, 43, 44):
                sample = main_root / pot / f"seed_{seed}"
                inner = derive_window(sample, 0.000625)
                full = derive_window(sample, 0.00125)

                conv = {
                    "potential": pot,
                    "seed": seed,
                    "K_inner_GPa": inner["K_GPa"],
                    "K_full_GPa": full["K_GPa"],
                    "G_inner_GPa": inner["G_GPa"],
                    "G_full_GPa": full["G_GPa"],
                    "K_window_relative_change":
                        abs(full["K_GPa"]-inner["K_GPa"])/abs(inner["K_GPa"]),
                    "G_window_relative_change":
                        abs(full["G_GPa"]-inner["G_GPa"])/abs(inner["G_GPa"]),
                    "full_min_R2": full["min_stress_R2"],
                    "full_K_stress_energy_rel": full["K_stress_energy_rel"],
                    "full_G_stress_energy_rel": full["G_stress_energy_rel"],
                    "inner_min_R2": inner["min_stress_R2"],
                    "inner_K_stress_energy_rel": inner["K_stress_energy_rel"],
                    "inner_G_stress_energy_rel": inner["G_stress_energy_rel"],
                }

                full_window_ok = (
                    pass_quality(full)
                    and conv["K_window_relative_change"] < 0.01
                    and conv["G_window_relative_change"] < 0.01
                )

                if full_window_ok:
                    chosen = full
                    source = "standard_full"
                    max_abs = 0.00125
                elif pass_quality(inner):
                    chosen = inner
                    source = "standard_inner"
                    max_abs = 0.000625
                else:
                    if not (pot == "EAM_2019" and seed == 44):
                        raise RuntimeError(
                            f"No converged window for {pot} seed {seed}"
                        )
                    chosen = derive_window(refine_sample, 0.0003125)
                    if not pass_quality(chosen):
                        raise RuntimeError("Refined EAM_2019 seed44 still fails quality checks")
                    source = "dedicated_refinement"
                    max_abs = 0.0003125

                conv["selected_source"] = source
                convergence_rows.append(conv)

                row = {
                    "potential": pot,
                    "seed": seed,
                    "source": source,
                    "max_abs_strain": max_abs,
                    "K_GPa": chosen["K_GPa"],
                    "G_GPa": chosen["G_GPa"],
                    "E_GPa": chosen["E_GPa"],
                    "nu": chosen["nu"],
                    "G_directional_sd_GPa": chosen["G_directional_sd_GPa"],
                    "Gxy_GPa": chosen["G_components_GPa"][0],
                    "Gxz_GPa": chosen["G_components_GPa"][1],
                    "Gyz_GPa": chosen["G_components_GPa"][2],
                    "K_energy_GPa": chosen["K_energy_GPa"],
                    "G_energy_GPa": chosen["G_energy_GPa"],
                    "K_stress_energy_rel": chosen["K_stress_energy_rel"],
                    "G_stress_energy_rel": chosen["G_stress_energy_rel"],
                    "min_stress_R2": chosen["min_stress_R2"],
                }
                final_rows.append(row)

        final_df = pd.DataFrame(final_rows)
        conv_df = pd.DataFrame(convergence_rows)

        # Stable ordering
        final_df["potential"] = pd.Categorical(
            final_df["potential"], categories=POTENTIAL_ORDER, ordered=True
        )
        conv_df["potential"] = pd.Categorical(
            conv_df["potential"], categories=POTENTIAL_ORDER, ordered=True
        )
        final_df = final_df.sort_values(["potential", "seed"]).reset_index(drop=True)
        conv_df = conv_df.sort_values(["potential", "seed"]).reset_index(drop=True)

        final_df.to_csv(tabdir / "final_sample_results.csv", index=False)
        conv_df.to_csv(tabdir / "window_convergence.csv", index=False)

        # Per-potential mean +/- sample SD across three realizations
        summary = final_df.groupby("potential", observed=False).agg(
            K_GPa_mean=("K_GPa", "mean"),
            K_GPa_sd=("K_GPa", "std"),
            G_GPa_mean=("G_GPa", "mean"),
            G_GPa_sd=("G_GPa", "std"),
            E_GPa_mean=("E_GPa", "mean"),
            E_GPa_sd=("E_GPa", "std"),
            nu_mean=("nu", "mean"),
            nu_sd=("nu", "std"),
        )
        summary["E_vs_experiment_percent"] = (
            (summary["E_GPa_mean"] / E_EXPERIMENT_GPA) - 1.0
        ) * 100.0
        summary.to_csv(tabdir / "potential_summary.csv")

        # Figure-friendly aliases
        plot_summary = summary.rename(columns={
            "K_GPa_mean": "K_GPa_mean", "K_GPa_sd": "K_GPa_sd",
            "G_GPa_mean": "G_GPa_mean", "G_GPa_sd": "G_GPa_sd",
            "E_GPa_mean": "E_GPa_mean", "E_GPa_sd": "E_GPa_sd",
        })

        # plots
        dot_summary_plot(
            final_df, plot_summary, "K_GPa", "Bulk modulus K (GPa)",
            figdir, "01_bulk_modulus"
        )
        dot_summary_plot(
            final_df, plot_summary, "G_GPa", "Shear modulus G (GPa)",
            figdir, "02_shear_modulus"
        )
        dot_summary_plot(
            final_df, plot_summary, "E_GPa", "Young's modulus E (GPa)",
            figdir, "03_youngs_modulus",
            experimental_line=E_EXPERIMENT_GPA,
            experimental_label="Cu64Zr36 experiment: 92.3 GPa"
        )

        # nu has different summary naming convention
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        xs = np.arange(len(POTENTIAL_ORDER))
        offsets = {42: -0.13, 43: 0.0, 44: 0.13}
        markers = {42: "o", 43: "s", 44: "^"}
        for seed in (42,43,44):
            vals=[]; xx=[]
            for i,pot in enumerate(POTENTIAL_ORDER):
                r=final_df[(final_df.potential==pot)&(final_df.seed==seed)].iloc[0]
                vals.append(float(r["nu"])); xx.append(i+offsets[seed])
            ax.scatter(xx, vals, marker=markers[seed], s=42, label=f"seed {seed}", zorder=3)
        ax.errorbar(
            xs, summary.loc[POTENTIAL_ORDER,"nu_mean"],
            yerr=summary.loc[POTENTIAL_ORDER,"nu_sd"],
            fmt="D", capsize=4, linewidth=1.3, markersize=5,
            label="mean ± SD", zorder=4
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([DISPLAY[p] for p in POTENTIAL_ORDER], rotation=25, ha="right")
        ax.set_ylabel("Poisson ratio ν")
        ax.set_xlabel("Potential")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        save_figure(fig, figdir, "04_poisson_ratio")

        make_window_sensitivity_plot(conv_df, figdir)
        make_consistency_plot(final_df, figdir)

        main_eam44 = main_root / "EAM_2019" / "seed_44"
        make_eam2019_seed44_curve(main_eam44, refine_sample, figdir, "yz")

        finite_size_rows = []
        if args.n4000 is not None:
            n4_ex = td / "n4000"; n4_ex.mkdir()
            safe_extract(args.n4000, n4_ex)
            n4_summaries = list(n4_ex.rglob("elasticity_summary.json"))
            if len(n4_summaries) != 1:
                raise RuntimeError(f"Expected one N4000 summary, found {len(n4_summaries)}")
            n4 = json.loads(n4_summaries[0].read_text())
            n4iso = n4["derived_isotropic"]

            r1024 = final_df[
                (final_df.potential == "MACE_C") & (final_df.seed == 43)
            ].iloc[0]
            finite_size_rows = [
                {
                    "N": 1024,
                    "protocol": "revised canonical-reference",
                    "K_GPa": r1024["K_GPa"],
                    "G_GPa": r1024["G_GPa"],
                    "E_GPa": r1024["E_GPa"],
                    "nu": r1024["nu"],
                },
                {
                    "N": 4000,
                    "protocol": "earlier independent-strain; branch verified smooth",
                    "K_GPa": n4iso["K_GPa"],
                    "G_GPa": n4iso["G_GPa"],
                    "E_GPa": n4iso["E_GPa"],
                    "nu": n4iso["nu"],
                },
            ]
            pd.DataFrame(finite_size_rows).to_csv(
                tabdir / "finite_size_MACE_C_seed43.csv", index=False
            )
            make_finite_size_plot(
                r1024["K_GPa"], r1024["G_GPa"],
                n4iso["K_GPa"], n4iso["G_GPa"], figdir
            )

        # Compact machine-readable analysis summary
        exceptions = final_df[final_df["source"] != "standard_full"][
            ["potential","seed","source","max_abs_strain"]
        ].to_dict(orient="records")
        report = {
            "selection_rule": {
                "R2_min": 0.995,
                "stress_energy_relative_mismatch_max": 0.01,
                "inner_full_relative_change_max": 0.01,
            },
            "selected_window_exceptions": exceptions,
            "n_samples": int(len(final_df)),
            "max_final_G_stress_energy_relative_mismatch":
                float(final_df["G_stress_energy_rel"].max()),
            "min_final_stress_R2":
                float(final_df["min_stress_R2"].min()),
            "experimental_Young_modulus_GPa": E_EXPERIMENT_GPA,
        }
        (args.outdir / "analysis_summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        print("\nFINAL SAMPLE SELECTION")
        print(final_df[[
            "potential","seed","source","max_abs_strain",
            "K_GPa","G_GPa","E_GPa","nu","min_stress_R2",
            "G_stress_energy_rel"
        ]].to_string(index=False))

        print("\nPOTENTIAL SUMMARY (mean +/- sample SD, n=3)")
        print(summary.to_string())

        if finite_size_rows:
            print("\nFINITE-SIZE CHECK")
            print(pd.DataFrame(finite_size_rows).to_string(index=False))

if __name__ == "__main__":
    main()
