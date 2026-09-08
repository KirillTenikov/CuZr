#!/usr/bin/env python3
"""
Second-stage analysis for Paper 1 Cu64Zr36 elasticity.

Run AFTER 01_reconstruct_elasticity.py.

Inputs:
  --analysis-dir : output directory produced by 01_reconstruct_elasticity.py
  --old-archive  : optional earlier-protocol elasticity archive, used only
                   for a quality-control comparison

Outputs:
  publication/tables/
  publication/figures/
  publication/RESULTS_SUMMARY.md

No internet access is required.
"""

from __future__ import annotations
import argparse
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORDER = [
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

def savefig(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def one_way_variance_fraction(df, prop):
    grand = float(df[prop].mean())
    ss_between = 0.0
    ss_within = 0.0
    for _, g in df.groupby("potential"):
        ss_between += len(g) * (float(g[prop].mean()) - grand) ** 2
        ss_within += float(((g[prop] - g[prop].mean()) ** 2).sum())
    total = ss_between + ss_within
    return {
        "property": prop,
        "grand_mean": grand,
        "between_potential_fraction": ss_between / total,
        "within_potential_fraction": ss_within / total,
    }

def parse_old_archive(path: Path):
    rows = []
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith("elasticity_summary.json"):
                continue
            d = json.load(tf.extractfile(member))
            parts = member.name.split("/")
            seed_token = next(x for x in parts if x.startswith("seed_"))
            seed = int(seed_token.split("_", 1)[1])
            potential = parts[-2]

            r2 = []
            for mode in ("bulk", "xy", "xz", "yz"):
                block = d["stress_strain"][mode]
                fit_key = next(k for k in block if k.startswith("fit_"))
                r2.append(float(block[fit_key]["r2"]))

            iso = d["derived_isotropic"]
            rows.append({
                "potential": potential,
                "seed": seed,
                "old_min_stress_R2": min(r2),
                "old_K_GPa": float(iso["K_GPa"]),
                "old_G_GPa": float(iso["G_GPa"]),
                "old_G_stress_energy_rel":
                    float(iso["G_energy_vs_stress_relative_difference"]),
                "old_warning_count": len(d.get("warnings", [])),
            })
    out = pd.DataFrame(rows)
    if len(out) != 24:
        raise RuntimeError(f"Expected 24 old summaries, found {len(out)}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--old-archive", type=Path, default=None)
    args = ap.parse_args()

    source_tables = args.analysis_dir / "tables"
    final = pd.read_csv(source_tables / "final_sample_results.csv")
    summary = pd.read_csv(source_tables / "potential_summary.csv")
    conv = pd.read_csv(source_tables / "window_convergence.csv")
    finite = pd.read_csv(source_tables / "finite_size_MACE_C_seed43.csv")

    out = args.analysis_dir / "publication"
    tables = out / "tables"
    figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # Stable ordering.
    final["potential"] = pd.Categorical(final["potential"], ORDER, ordered=True)
    conv["potential"] = pd.Categorical(conv["potential"], ORDER, ordered=True)
    final = final.sort_values(["potential", "seed"]).reset_index(drop=True)
    conv = conv.sort_values(["potential", "seed"]).reset_index(drop=True)

    # Add coefficients of variation and directional-shear statistics.
    grouped = final.groupby("potential", observed=False)
    enhanced = grouped.agg(
        K_mean=("K_GPa", "mean"),
        K_sd=("K_GPa", "std"),
        G_mean=("G_GPa", "mean"),
        G_sd=("G_GPa", "std"),
        E_mean=("E_GPa", "mean"),
        E_sd=("E_GPa", "std"),
        nu_mean=("nu", "mean"),
        nu_sd=("nu", "std"),
        G_directional_sd_mean=("G_directional_sd_GPa", "mean"),
        min_R2=("min_stress_R2", "min"),
        max_G_stress_energy_rel=("G_stress_energy_rel", "max"),
    )
    for p in ("K", "G", "E"):
        enhanced[f"{p}_CV_percent"] = 100.0 * enhanced[f"{p}_sd"] / enhanced[f"{p}_mean"]
    enhanced["G_directional_CV_percent"] = (
        100.0 * enhanced["G_directional_sd_mean"] / enhanced["G_mean"]
    )
    enhanced.to_csv(tables / "potential_statistics_enhanced.csv")

    # Variance partition: descriptive one-way decomposition, not an inferential ANOVA.
    variance = pd.DataFrame([
        one_way_variance_fraction(final, "K_GPa"),
        one_way_variance_fraction(final, "G_GPa"),
        one_way_variance_fraction(final, "E_GPa"),
        one_way_variance_fraction(final, "nu"),
    ])
    variance["between_potential_percent"] = 100.0 * variance["between_potential_fraction"]
    variance["within_potential_percent"] = 100.0 * variance["within_potential_fraction"]
    variance.to_csv(tables / "variance_partition.csv", index=False)

    # Window-sensitivity table.
    ws = conv[[
        "potential", "seed",
        "K_window_relative_change", "G_window_relative_change",
        "full_min_R2", "full_K_stress_energy_rel",
        "full_G_stress_energy_rel", "selected_source"
    ]].copy()
    ws["K_window_change_percent"] = 100.0 * ws["K_window_relative_change"]
    ws["G_window_change_percent"] = 100.0 * ws["G_window_relative_change"]
    ws.to_csv(tables / "strain_window_diagnostics.csv", index=False)

    # Ranking tables by each mechanical property (descriptive only).
    rank = enhanced.reset_index()[[
        "potential", "K_mean", "G_mean", "E_mean", "nu_mean",
        "K_CV_percent", "G_CV_percent", "E_CV_percent"
    ]].copy()
    rank["K_rank_stiffest"] = rank["K_mean"].rank(ascending=False, method="min").astype(int)
    rank["G_rank_stiffest"] = rank["G_mean"].rank(ascending=False, method="min").astype(int)
    rank["E_rank_stiffest"] = rank["E_mean"].rank(ascending=False, method="min").astype(int)
    rank.to_csv(tables / "mechanical_property_ranking.csv", index=False)

    # Figure A: K-G map; seeds faintly represented by marker positions,
    # potential means labeled.
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.scatter(final["K_GPa"], final["G_GPa"], s=28, alpha=0.45)
    for pot in ORDER:
        r = enhanced.loc[pot]
        ax.scatter([r.K_mean], [r.G_mean], s=80, marker="D")
        ax.annotate(DISPLAY[pot], (r.K_mean, r.G_mean),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Bulk modulus K (GPa)")
    ax.set_ylabel("Shear modulus G (GPa)")
    ax.set_title("Mechanical-response map of Cu64Zr36 glass")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, figs, "09_K_vs_G_map")

    # Figure B: within-potential CV.
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(ORDER))
    width = 0.25
    ax.bar(x-width, enhanced.loc[ORDER, "K_CV_percent"], width, label="K")
    ax.bar(x,       enhanced.loc[ORDER, "G_CV_percent"], width, label="G")
    ax.bar(x+width, enhanced.loc[ORDER, "E_CV_percent"], width, label="E")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[p] for p in ORDER], rotation=25, ha="right")
    ax.set_ylabel("Coefficient of variation across seeds (%)")
    ax.set_title("Realization-to-realization reproducibility")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, figs, "10_seed_reproducibility_CV")

    old_quality = None
    if args.old_archive is not None:
        old = parse_old_archive(args.old_archive)
        merged = old.merge(
            final[[
                "potential", "seed", "min_stress_R2",
                "G_stress_energy_rel", "K_GPa", "G_GPa", "source"
            ]],
            on=["potential", "seed"], how="inner"
        )
        merged = merged.rename(columns={
            "min_stress_R2": "final_min_stress_R2",
            "G_stress_energy_rel": "final_G_stress_energy_rel",
            "K_GPa": "final_K_GPa",
            "G_GPa": "final_G_GPa",
        })
        merged.to_csv(tables / "old_vs_final_quality.csv", index=False)
        old_quality = merged

        # Figure C: stress-fit R2 old vs final.
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        ax.scatter(merged["old_min_stress_R2"], merged["final_min_stress_R2"], s=48)
        lo = min(float(merged.old_min_stress_R2.min()), 0.67)
        ax.plot([lo, 1.0], [lo, 1.0], linestyle="--", linewidth=1.1)
        ax.axvline(0.995, linestyle=":", linewidth=1.1)
        ax.axhline(0.995, linestyle=":", linewidth=1.1)
        ax.set_xlim(lo, 1.002)
        ax.set_ylim(lo, 1.002)
        ax.set_xlabel("Earlier protocol/window: minimum stress-fit R²")
        ax.set_ylabel("Final analysis: minimum stress-fit R²")
        ax.set_title("Elasticity fit quality before and after revision")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        savefig(fig, figs, "11_old_vs_final_R2")

        # Figure D: stress-energy mismatch old vs final, logarithmic axes.
        tiny = 1e-7
        xold = np.maximum(merged["old_G_stress_energy_rel"].to_numpy(), tiny)
        ynew = np.maximum(merged["final_G_stress_energy_rel"].to_numpy(), tiny)
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        ax.scatter(100*xold, 100*ynew, s=48)
        mn = min(float((100*xold).min()), float((100*ynew).min()))
        mx = max(float((100*xold).max()), float((100*ynew).max()))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.1)
        ax.axhline(1.0, linestyle=":", linewidth=1.1)
        ax.axvline(1.0, linestyle=":", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Earlier protocol/window: |G_E-Gσ|/Gσ (%)")
        ax.set_ylabel("Final analysis: |G_E-Gσ|/Gσ (%)")
        ax.set_title("Stress-energy consistency")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        savefig(fig, figs, "12_old_vs_final_stress_energy")

    # Finite size calculations.
    n1024 = finite.loc[finite["N"] == 1024].iloc[0]
    n4000 = finite.loc[finite["N"] == 4000].iloc[0]
    fs_changes = {
        prop: 100.0 * (float(n4000[prop]) - float(n1024[prop])) / float(n1024[prop])
        for prop in ("K_GPa", "G_GPa", "E_GPa", "nu")
    }
    pd.DataFrame([{
        "property": prop,
        "relative_change_percent_N4000_vs_N1024": val
    } for prop, val in fs_changes.items()]).to_csv(
        tables / "finite_size_relative_changes.csv", index=False
    )

    # Plain-language reproducible summary.
    vp = variance.set_index("property")
    max_non_eam19_g = float(
        ws[ws["potential"].astype(str) != "EAM_2019"]["G_window_change_percent"].max()
    )
    e19_43 = ws[(ws["potential"].astype(str) == "EAM_2019") & (ws.seed == 43)].iloc[0]
    e19_44 = ws[(ws["potential"].astype(str) == "EAM_2019") & (ws.seed == 44)].iloc[0]

    lines = []
    lines.append("# Paper 1 elasticity: final statistical analysis\n")
    lines.append("## Core result\n")
    lines.append(
        "The dominant source of variation is the interatomic potential, not the independent "
        "glass realization. A descriptive one-way variance decomposition attributes "
        f"{vp.loc['K_GPa','between_potential_percent']:.2f}% of K variance, "
        f"{vp.loc['G_GPa','between_potential_percent']:.2f}% of G variance, and "
        f"{vp.loc['E_GPa','between_potential_percent']:.2f}% of E variance to differences "
        "between potentials. This is descriptive, not an inferential ANOVA."
    )
    lines.append("\n## Final mean ± sample SD (n=3)\n")
    lines.append("| Potential | K (GPa) | G (GPa) | E (GPa) | ν |")
    lines.append("|---|---:|---:|---:|---:|")
    for pot in ORDER:
        r = enhanced.loc[pot]
        lines.append(
            f"| {DISPLAY[pot]} | {r.K_mean:.2f} ± {r.K_sd:.2f} | "
            f"{r.G_mean:.2f} ± {r.G_sd:.2f} | "
            f"{r.E_mean:.2f} ± {r.E_sd:.2f} | "
            f"{r.nu_mean:.4f} ± {r.nu_sd:.4f} |"
        )

    lines.append("\n## Strain-window convergence\n")
    lines.append(
        "The common full strain window is retained for 22/24 samples. "
        f"For every model except EAM 2019, halving the window changes G by at most "
        f"{max_non_eam19_g:.3f}%. EAM 2019 seed43 changes by "
        f"{e19_43.G_window_change_percent:.3f}% and therefore uses the inner standard window; "
        f"EAM 2019 seed44 changes by {e19_44.G_window_change_percent:.3f}% and requires the "
        "dedicated smaller-strain refinement."
    )

    lines.append("\n## Numerical quality\n")
    lines.append(
        f"Across the final 24-sample set, the worst minimum stress-fit R² is "
        f"{final.min_stress_R2.min():.6f}, and the largest mean-G stress/energy mismatch is "
        f"{100*final.G_stress_energy_rel.max():.3f}%."
    )
    if old_quality is not None:
        lines.append(
            f"In the earlier archive, {int((old_quality.old_warning_count>0).sum())}/24 samples "
            f"carried at least one warning, the worst stress-fit R² was "
            f"{old_quality.old_min_stress_R2.min():.6f}, and the largest mean-G stress/energy "
            f"mismatch was {100*old_quality.old_G_stress_energy_rel.max():.1f}%. "
            "The comparison includes both the protocol change and the revised smaller default "
            "strain window, so it should be described as an improvement of the revised workflow, "
            "not attributed to a single change in isolation."
        )

    lines.append("\n## Finite-size check\n")
    lines.append(
        f"For MACE C seed43, changing N=1024 to N=4000 changes K by "
        f"{fs_changes['K_GPa']:.3f}%, G by {fs_changes['G_GPa']:.3f}%, and E by "
        f"{fs_changes['E_GPa']:.3f}%. The N=4000 calculation used the earlier strain "
        "initialization but was independently verified to remain on a smooth elastic branch; "
        "it is therefore a finite-size consistency check, not part of the 24-sample statistics."
    )

    lines.append("\n## Interpretation for Paper 1\n")
    lines.append(
        "1. Potential choice is the principal determinant of relaxed-ion glass stiffness.\n"
        "2. Bulk modulus is especially reproducible across independent realizations; shear and "
        "Young's moduli show somewhat larger but still modest seed scatter.\n"
        "3. EAM 2019 is uniquely strain-window-sensitive in the corrected dataset; this is not "
        "a generic feature of EAM, ACE, or MACE.\n"
        "4. The MACE C finite-size check indicates that the N=1024 modulus is not dominated by "
        "a gross finite-size artifact.\n"
        "5. These mechanical results should later be integrated with DFT errors, crystalline/B2 "
        "tests, density, RDF/S(q), and local-order statistics before assigning an overall "
        "potential ranking."
    )

    (out / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))

if __name__ == "__main__":
    main()
