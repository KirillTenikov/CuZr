#!/usr/bin/env python3
"""
Run Paper 1 validation workflow for multiple Cu-Zr potentials.

Converted from Notebook_05_NCL_validation_multi_potentials.ipynb.

What it does:
- registers MACE / EAM / ACE potential specs via `cuzr_setup_multi`
- runs smoke tests
- runs EOS scans for FCC Cu / HCP Zr / B2 CuZr
- runs melt -> quench -> minimize -> NVE validation pipeline
- estimates local bulk modulus from EOS
- computes vacancy formation energies
- computes RDF and an approximate S(q) proxy
- writes compact result tables and plots to disk

This version is made Docker-friendlier by:
- resolving paths explicitly
- importing helper modules from src/ only
- avoiding dependence on the notebook working directory
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(extra) not in sys.path and extra.exists():
        sys.path.insert(0, str(extra))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.geometry.analysis import Analysis
from pyiron_atomistics import Project
from pyiron_atomistics.atomistics.structure.atoms import Atoms

from src.path_utils import resolve_path


def import_helper(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        fallback_paths = [PROJECT_ROOT / "src"]
        missing_msg = ", ".join(str(p) for p in fallback_paths)
        raise ModuleNotFoundError(
            f"Could not import helper module '{module_name}'. Looked via sys.path and expected it under: {missing_msg}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Notebook 05 as a Python script")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev")
    parser.add_argument("--project-path", default="../cu_zr_mlip_project", help="pyiron project path")
    parser.add_argument("--results-dir", default="outputs/notebook05_validation", help="Directory for results")
    parser.add_argument("--pots", default="all")
    parser.add_argument("--include-ace", action="store_true")
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--project-tag", default=None)
    parser.add_argument(
        "--helper-module",
        default=os.environ.get("CUZR_HELPER_MODULE", "cuzr_setup_multi"),
        help="Helper module name, default: cuzr_setup_multi",
    )
    return parser.parse_args()


def configure_threads(mode_dev: bool) -> None:
    if mode_dev:
        os.environ.setdefault("OMP_NUM_THREADS", "6")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def register_potentials(cz, include_ace: bool) -> List[Dict[str, Any]]:
    potential_specs = [
        cz.make_mace_spec("MACE_A", "CuZr_MACE_A_compiled.model-lammps.pt"),
        cz.make_mace_spec("MACE_B", "CuZr_MACE_B_compiled.model-lammps.pt"),
        cz.make_mace_spec("MACE_C", "CuZr_MACE_C_compiled.model-lammps.pt"),
        cz.make_mace_spec("MACE_D", "CuZr_MACE_D_compiled.model-lammps.pt"),
        cz.make_pyiron_spec("EAM_Mendelev_2019_CuZr", "EAM_Mendelev_2019_CuZr__MO_945018740343_000"),
        cz.make_pyiron_spec("2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1", "2007--Mendelev-M-I--Cu-Zr--LAMMPS--ipr1"),
    ]
    if include_ace:
        potential_specs.extend(
            [
                cz.make_ace_spec("ACE_A", "CuZr_ACE_A.yace"),
                cz.make_ace_spec("ACE_B", "CuZr_ACE_B.yace"),
                cz.make_ace_spec("ACE_C", "CuZr_ACE_C.yace"),
                cz.make_ace_spec("ACE_D", "CuZr_ACE_D.yace"),
            ]
        )
    cz.register_potentials(potential_specs)
    return list(cz.POTENTIALS)


def select_potentials(all_pots: Sequence[Dict[str, Any]], pots_arg: str) -> List[Dict[str, Any]]:
    if pots_arg.strip().lower() == "all":
        return list(all_pots)
    wanted = {x.strip() for x in pots_arg.split(",") if x.strip()}
    selected = [p for p in all_pots if p["id"] in wanted]
    missing = sorted(wanted - {p["id"] for p in selected})
    if missing:
        raise ValueError(f"Unknown potentials requested: {missing}")
    return selected


def make_fcc_cu(a: float = 3.615, rep=(4, 4, 4)) -> Atoms:
    basis = Atoms(symbols=["Cu"] * 4, positions=[[0.0, 0.0, 0.0], [0.0, 0.5 * a, 0.5 * a], [0.5 * a, 0.0, 0.5 * a], [0.5 * a, 0.5 * a, 0.0]], cell=[a, a, a], pbc=True)
    return basis.repeat(rep)


def make_hcp_zr(a: float = 3.23, c_over_a: float = 1.593, rep=(4, 4, 3)) -> Atoms:
    c = c_over_a * a
    cell = np.array([[a, 0.0, 0.0], [0.5 * a, np.sqrt(3) / 2 * a, 0.0], [0.0, 0.0, c]])
    basis = Atoms(symbols=["Zr", "Zr"], scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, 1 / 3, 1 / 2]], cell=cell, pbc=True)
    return basis.repeat(rep)


def make_b2_cuzr(a: float = 3.2, rep=(6, 6, 6)) -> Atoms:
    structure = Atoms(symbols=["Cu", "Zr"], positions=[[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a]], cell=[a, a, a], pbc=True)
    return structure.repeat(rep)


def make_glass_seed(a: float = 3.2, rep=(8, 4, 4)) -> Atoms:
    return make_b2_cuzr(a=a, rep=rep)


def jname(prefix: str, pot_spec: Dict[str, Any], mode_dev: bool) -> str:
    run_tag = "dev" if mode_dev else "prod"
    return f"{run_tag}_{prefix}_{pot_spec['id']}"


def isotropic_scale(structure: Atoms, scale: float) -> Atoms:
    s = structure.copy()
    s.set_cell(np.array(s.cell) * scale, scale_atoms=True)
    return s


def get_last_energy(job) -> float:
    try:
        return float(job.output.energy_pot[-1])
    except Exception:
        try:
            return float(job.output.energy_tot[-1])
        except Exception:
            return np.nan


def get_last_temp(job) -> float:
    try:
        return float(job.output.temperature[-1])
    except Exception:
        return np.nan


def get_last_press(job) -> float:
    for attr in ["pressures", "pressure", "press"]:
        try:
            arr = getattr(job.output, attr)
            return float(arr[-1])
        except Exception:
            pass
    return np.nan


def static_summary(job, label: str, pot_id: str, structure_name: str, scale: float | None = None) -> Dict[str, Any]:
    e_last = get_last_energy(job)
    return {"label": label, "pot_id": pot_id, "structure": structure_name, "scale": scale, "n_atoms": len(job.structure), "energy_last_eV": e_last, "energy_per_atom_eV": e_last / len(job.structure)}


def md_summary(job, label: str, pot_id: str, structure_name: str) -> Dict[str, Any]:
    e_last = get_last_energy(job)
    return {"label": label, "pot_id": pot_id, "structure": structure_name, "n_atoms": len(job.structure), "energy_last_eV": e_last, "energy_per_atom_eV": e_last / len(job.structure), "temp_last_K": get_last_temp(job), "press_last": get_last_press(job)}


def md_from_last(pr: Project, job_name: str, prev_job, pot_spec: Dict[str, Any], T: float, steps: int, cores: int, thermo: int, dump_every: int, neigh_every: int, timestep_fs: float = 1.0):
    struct = prev_job.get_structure(iteration_step=-1)
    return cz.run_md(pr=pr, job_name=job_name, structure=struct, pot_spec=pot_spec, T=T, steps=steps, timestep_fs=timestep_fs, cores=cores, delete_existing=True, thermo=thermo, dump_every=dump_every, neigh_every=neigh_every)


def run_nve(pr: Project, job_name: str, structure: Atoms, pot_spec: Dict[str, Any], steps: int, cores: int, thermo: int, neigh_every: int, timestep_fs: float = 1.0):
    job = cz.make_lammps_job(pr, job_name, structure, pot_spec, delete_existing=False, cores=cores)
    job.calc_md(temperature=None, pressure=None, n_ionic_steps=steps, time_step=timestep_fs)
    job.input.control["fix___ensemble"] = "all nve"
    job.input.control["variable___thermotime"] = f"equal {int(thermo)}"
    job.input.control["neighbor"] = "2.0 bin"
    job.input.control["neigh_modify"] = f"every {int(neigh_every)} delay 0 check yes"
    job = cz.load_or_run(pr, job)
    return job


def run_minimize(pr: Project, job_name: str, structure: Atoms, pot_spec: Dict[str, Any], cores: int):
    job = cz.make_lammps_job(pr, job_name, structure, pot_spec, delete_existing=False, cores=cores)
    job.calc_minimize(ionic_energy_tolerance=0.0, ionic_force_tolerance=1e-4, max_iter=200, n_print=100)
    job = cz.load_or_run(pr, job)
    return job


def run_eos_scan(pr: Project, structure: Atoms, structure_name: str, pot_spec: Dict[str, Any], scales: Iterable[float], mode_dev: bool, cores: int) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(scales):
        struct_s = isotropic_scale(structure, float(s))
        job = cz.run_static(pr=pr, job_name=jname(f"eos_{structure_name}_{i:02d}", pot_spec, mode_dev), structure=struct_s, pot_spec=pot_spec, cores=cores, delete_existing=False)
        rows.append(static_summary(job, "eos", pot_spec["id"], structure_name, scale=float(s)))
    return pd.DataFrame(rows)


def ncl_validation_pipeline(pr: Project, structure0: Atoms, pot_spec: Dict[str, Any], mode_dev: bool, cores: int, T_melt: int, steps_melt: int, quench_ts: Sequence[int], steps_each: int, nve_steps: int, thermo: int, dump_every: int, neigh_every: int) -> Dict[str, Any]:
    out = {"pot_id": pot_spec["id"], "n_atoms": len(structure0)}
    j0 = cz.run_static(pr, jname("static_glass_seed", pot_spec, mode_dev), structure0, pot_spec, cores=cores, delete_existing=False)
    out["E_static_per_atom_eV"] = get_last_energy(j0) / len(structure0)
    jm = cz.run_md(pr=pr, job_name=jname("melt", pot_spec, mode_dev), structure=structure0, pot_spec=pot_spec, T=T_melt, steps=steps_melt, timestep_fs=1.0, cores=cores, delete_existing=False, thermo=thermo, dump_every=dump_every, neigh_every=neigh_every)
    out["E_melt_per_atom_eV"] = get_last_energy(jm) / len(structure0)
    out["T_melt_last_K"] = get_last_temp(jm)
    prev = jm
    for T in quench_ts:
        jq = md_from_last(pr=pr, job_name=jname(f"quench_{T}K", pot_spec, mode_dev), prev_job=prev, pot_spec=pot_spec, T=T, steps=steps_each, cores=cores, thermo=thermo, dump_every=dump_every, neigh_every=neigh_every)
        out[f"E_{T}K_per_atom_eV"] = get_last_energy(jq) / len(structure0)
        out[f"T_{T}K_last_K"] = get_last_temp(jq)
        prev = jq
    jmin = run_minimize(pr, jname("glass_min", pot_spec, mode_dev), prev.get_structure(iteration_step=-1), pot_spec, cores=cores)
    out["E_glass_min_per_atom_eV"] = get_last_energy(jmin) / len(structure0)
    jnve = run_nve(pr, jname("glass_nve", pot_spec, mode_dev), jmin.get_structure(iteration_step=-1), pot_spec, steps=nve_steps, cores=cores, thermo=thermo, neigh_every=neigh_every)
    out["E_nve_per_atom_eV"] = get_last_energy(jnve) / len(structure0)
    out["T_nve_last_K"] = get_last_temp(jnve)
    compressed = isotropic_scale(jmin.get_structure(iteration_step=-1), 0.97)
    jcomp = cz.run_static(pr=pr, job_name=jname("glass_compressed", pot_spec, mode_dev), structure=compressed, pot_spec=pot_spec, cores=cores, delete_existing=True)
    out["E_compressed_per_atom_eV"] = get_last_energy(jcomp) / len(compressed)
    return out


EV_PER_A3_TO_GPA = 160.21766208


def get_structure_volume(structure: Atoms) -> float:
    return float(abs(np.linalg.det(np.array(structure.cell))))


def local_bulk_modulus_from_eos(structure: Atoms, eos_slice: pd.DataFrame) -> float:
    df = eos_slice.sort_values("scale").copy()
    if len(df) < 3:
        return np.nan
    V0 = get_structure_volume(structure)
    df["volume_A3"] = V0 * (df["scale"].astype(float) ** 3)
    x = df["volume_A3"].to_numpy(dtype=float)
    y = df["energy_last_eV"].to_numpy(dtype=float)
    i0 = int(np.argmin(y))
    lo = max(0, i0 - 1)
    hi = min(len(df), i0 + 2)
    if hi - lo < 3:
        lo = max(0, hi - 3)
        hi = min(len(df), lo + 3)
    xfit = x[lo:hi]
    yfit = y[lo:hi]
    if len(xfit) < 3:
        return np.nan
    a, b, _ = np.polyfit(xfit, yfit, 2)
    if a <= 0:
        return np.nan
    vmin = -b / (2 * a)
    b0_eva3 = vmin * (2 * a)
    return b0_eva3 * EV_PER_A3_TO_GPA


def make_vacancy_structure(structure: Atoms, atom_index: int = 0) -> Atoms:
    s = structure.copy()
    del s[atom_index]
    return s


def vacancy_formation_energy(pr: Project, base_structure: Atoms, structure_name: str, pot_spec: Dict[str, Any], mode_dev: bool, cores: int, repeat=(3, 3, 3), atom_index: int = 0) -> Dict[str, Any]:
    perfect = base_structure.repeat(repeat)
    jperf = run_minimize(pr, jname(f"vac_perfect_{structure_name}", pot_spec, mode_dev), perfect, pot_spec, cores=cores)
    e_perfect = get_last_energy(jperf)
    n_perfect = len(jperf.structure)
    defect = make_vacancy_structure(jperf.get_structure(iteration_step=-1), atom_index=atom_index)
    jdef = run_minimize(pr, jname(f"vac_defect_{structure_name}", pot_spec, mode_dev), defect, pot_spec, cores=cores)
    e_defect = get_last_energy(jdef)
    e_form = e_defect - ((n_perfect - 1) / n_perfect) * e_perfect
    return {"pot_id": pot_spec["id"], "structure": structure_name, "repeat": "x".join(map(str, repeat)), "n_perfect": n_perfect, "e_vac_form_eV": e_form}


def safe_last_structure(job):
    try:
        return job.get_structure(iteration_step=-1)
    except Exception:
        return job.get_structure()


def structure_factor_from_rdf(r, g_r, rho_number, q_values):
    r = np.asarray(r, dtype=float)
    g_r = np.asarray(g_r, dtype=float)
    q_values = np.asarray(q_values, dtype=float)
    h = g_r - 1.0
    s_q = []
    for q in q_values:
        if q == 0:
            s_q.append(np.nan)
            continue
        integrand = r * h * np.sin(q * r)
        integral = np.trapz(integrand, r)
        s_q.append(1.0 + 4.0 * np.pi * rho_number * integral / q)
    return np.asarray(s_q)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}", flush=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}", flush=True)


def plot_eos(eos_df: pd.DataFrame, out_dir: Path) -> None:
    for structure_name in ["FCC_Cu", "HCP_Zr", "B2_CuZr"]:
        sub = eos_df[eos_df["structure"] == structure_name]
        plt.figure(figsize=(6, 4))
        for pid, g in sub.groupby("pot_id"):
            plt.plot(g["scale"], g["energy_per_atom_eV"], marker="o", label=pid)
        plt.xlabel("isotropic scale")
        plt.ylabel("energy / atom (eV)")
        plt.title(structure_name)
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"eos_{structure_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[saved] {path}", flush=True)


def plot_rdf(rdf_store: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    if not rdf_store:
        return
    plt.figure(figsize=(6, 4))
    for pid, vals in rdf_store.items():
        plt.plot(vals["r"], vals["g_r"], label=pid)
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("RDF of minimized Cu–Zr glass")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "rdf_glass.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}", flush=True)


def plot_sq(sq_store: Dict[str, Dict[str, np.ndarray]], out_dir: Path) -> None:
    if not sq_store:
        return
    plt.figure(figsize=(6, 4))
    for pid, vals in sq_store.items():
        plt.plot(vals["q"], vals["s_q"], label=pid)
    plt.xlabel("q (1/Å)")
    plt.ylabel("S(q) proxy")
    plt.title("Approximate structure-factor comparison")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "sq_proxy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}", flush=True)


def get_nve_drift_for_potential(pot_id: str, ncl_df: pd.DataFrame) -> float:
    sub = ncl_df[ncl_df["pot_id"] == pot_id]
    if len(sub) == 0:
        return np.nan
    try:
        return float(sub["E_nve_per_atom_eV"].iloc[0] - sub["E_glass_min_per_atom_eV"].iloc[0])
    except Exception:
        return np.nan


def main() -> int:
    global cz
    args = parse_args()
    cz = import_helper(args.helper_module)
    mode_dev = args.mode == "dev"
    configure_threads(mode_dev)

    project_path = resolve_path(args.project_path, base_dir=PROJECT_ROOT)
    results_dir = resolve_path(args.results_dir, base_dir=PROJECT_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)
    pr = Project(str(project_path))

    all_pots = register_potentials(cz, include_ace=args.include_ace)
    pots_to_run = select_potentials(all_pots, args.pots)

    pot_df = pd.DataFrame([{"id": p["id"], "mode": p["mode"], "family": p.get("family", ""), "name": p.get("name", ""), "model_file": p.get("model_file", "")} for p in pots_to_run])
    save_dataframe(pot_df, results_dir / "potentials_selected.csv")

    cores = args.cores if args.cores is not None else (1 if mode_dev else 4)
    smoke_md_steps = 50 if mode_dev else 200
    smoke_thermo = 10 if mode_dev else 20
    eos_points = 7
    eos_scales = np.linspace(0.96, 1.04, eos_points)
    T_melt = 2000
    if mode_dev:
        steps_melt = 500
        quench_ts = (1200, 600, 300)
        steps_each = 500
        nve_steps = 1000
        thermo = 100
        dump_every = 10
        neigh_every = 10
    else:
        steps_melt = 30000
        quench_ts = (1500, 1000, 700, 500, 300)
        steps_each = 20000
        nve_steps = 20000
        thermo = 1000
        dump_every = 5000
        neigh_every = 10

    settings = {
        "mode": args.mode,
        "project_path": str(project_path),
        "results_dir": str(results_dir),
        "pots": [p["id"] for p in pots_to_run],
        "cores": cores,
        "smoke_md_steps": smoke_md_steps,
        "smoke_thermo": smoke_thermo,
        "eos_scales": [float(x) for x in eos_scales],
        "T_melt": T_melt,
        "quench_ts": list(quench_ts),
        "steps_melt": steps_melt,
        "steps_each": steps_each,
        "nve_steps": nve_steps,
        "thermo": thermo,
        "dump_every": dump_every,
        "neigh_every": neigh_every,
        "project_tag": args.project_tag,
        "helper_module": args.helper_module,
    }
    save_json(settings, results_dir / "run_settings.json")

    fcc_cu = make_fcc_cu()
    hcp_zr = make_hcp_zr()
    b2_cuzr = make_b2_cuzr()
    glass_seed = make_glass_seed()

    smoke_rows = []
    for p in pots_to_run:
        print("SMOKE TEST:", p["id"], flush=True)
        j_static = cz.run_static(pr=pr, job_name=jname("smoke_static_b2", p, mode_dev), structure=b2_cuzr, pot_spec=p, cores=cores, delete_existing=True)
        smoke_rows.append(static_summary(j_static, "smoke_static", p["id"], "B2_CuZr"))
        j_md = cz.run_md(pr=pr, job_name=jname("smoke_md_b2", p, mode_dev), structure=b2_cuzr, pot_spec=p, T=300, steps=smoke_md_steps, timestep_fs=1.0, cores=cores, delete_existing=True, thermo=smoke_thermo, dump_every=10, neigh_every=10)
        smoke_rows.append(md_summary(j_md, "smoke_md", p["id"], "B2_CuZr"))
    smoke_df = pd.DataFrame(smoke_rows)
    save_dataframe(smoke_df, results_dir / "smoke_validation.csv")

    eos_frames = []
    for p in pots_to_run:
        print("EOS:", p["id"], flush=True)
        eos_frames.append(run_eos_scan(pr, fcc_cu, "FCC_Cu", p, eos_scales, mode_dev, cores))
        eos_frames.append(run_eos_scan(pr, hcp_zr, "HCP_Zr", p, eos_scales, mode_dev, cores))
        eos_frames.append(run_eos_scan(pr, b2_cuzr, "B2_CuZr", p, eos_scales, mode_dev, cores))
    eos_df = pd.concat(eos_frames, ignore_index=True)
    save_dataframe(eos_df, results_dir / "eos_validation.csv")
    plot_eos(eos_df, results_dir)

    ncl_rows = []
    for p in pots_to_run:
        print("RUNNING NCL PIPELINE:", p["id"], flush=True)
        ncl_rows.append(ncl_validation_pipeline(pr=pr, structure0=glass_seed, pot_spec=p, mode_dev=mode_dev, cores=cores, T_melt=T_melt, steps_melt=steps_melt, quench_ts=quench_ts, steps_each=steps_each, nve_steps=nve_steps, thermo=thermo, dump_every=dump_every, neigh_every=neigh_every))
    ncl_df = pd.DataFrame(ncl_rows)
    save_dataframe(ncl_df, results_dir / "ncl_validation.csv")

    structure_map = {"FCC_Cu": fcc_cu, "HCP_Zr": hcp_zr, "B2_CuZr": b2_cuzr}
    bulk_rows = []
    for pot_id in sorted(eos_df["pot_id"].unique()):
        for sname, structure in structure_map.items():
            sub = eos_df[(eos_df["pot_id"] == pot_id) & (eos_df["structure"] == sname)].copy()
            bulk_rows.append({"pot_id": pot_id, "structure": sname, "bulk_modulus_GPa_est": local_bulk_modulus_from_eos(structure, sub)})
    bulk_df = pd.DataFrame(bulk_rows)
    save_dataframe(bulk_df, results_dir / "bulk_modulus_estimates.csv")

    vac_rows = []
    vacuum_targets = [("FCC_Cu", fcc_cu, (4, 4, 4)), ("HCP_Zr", hcp_zr, (4, 4, 3))]
    for p in pots_to_run:
        print("VACANCY:", p["id"], flush=True)
        for sname, structure, rep in vacuum_targets:
            try:
                vac_rows.append(vacancy_formation_energy(pr, structure, sname, p, mode_dev=mode_dev, cores=cores, repeat=rep))
            except Exception as exc:
                vac_rows.append({"pot_id": p["id"], "structure": sname, "repeat": "x".join(map(str, rep)), "n_perfect": np.nan, "e_vac_form_eV": np.nan, "error": str(exc)})
    vac_df = pd.DataFrame(vac_rows)
    save_dataframe(vac_df, results_dir / "vacancy_formation.csv")

    rdf_store: Dict[str, Dict[str, np.ndarray]] = {}
    sq_store: Dict[str, Dict[str, np.ndarray]] = {}
    q_grid = np.linspace(0.5, 16.0, 300)
    for p in pots_to_run:
        pid = p["id"]
        try:
            jmin = pr.load(jname("glass_min", p, mode_dev))
            s = safe_last_structure(jmin)
            ana = Analysis(s)
            r, g = ana.get_rdf(rmax=8.0, nbins=200)
            rdf_store[pid] = {"r": np.asarray(r), "g_r": np.asarray(g)}
            volume = get_structure_volume(s)
            rho = len(s) / volume
            sq = structure_factor_from_rdf(rdf_store[pid]["r"], rdf_store[pid]["g_r"], rho, q_grid)
            sq_store[pid] = {"q": q_grid, "s_q": sq}
        except Exception as exc:
            print(f"RDF / S(q) skipped for {pid}: {exc}", flush=True)

    plot_rdf(rdf_store, results_dir)
    plot_sq(sq_store, results_dir)
    for pid, vals in rdf_store.items():
        np.savez(results_dir / f"rdf_{pid}.npz", r=vals["r"], g_r=vals["g_r"])
    for pid, vals in sq_store.items():
        np.savez(results_dir / f"sq_{pid}.npz", q=vals["q"], s_q=vals["s_q"])

    summary_rows = []
    for p in pots_to_run:
        pid = p["id"]
        bulk_vals = bulk_df[bulk_df["pot_id"] == pid].set_index("structure")["bulk_modulus_GPa_est"].to_dict()
        vac_vals = vac_df[vac_df["pot_id"] == pid].set_index("structure")["e_vac_form_eV"].to_dict()
        summary_rows.append({"pot_id": pid, "bulk_FCC_Cu_GPa_est": bulk_vals.get("FCC_Cu", np.nan), "bulk_HCP_Zr_GPa_est": bulk_vals.get("HCP_Zr", np.nan), "bulk_B2_CuZr_GPa_est": bulk_vals.get("B2_CuZr", np.nan), "vac_FCC_Cu_eV": vac_vals.get("FCC_Cu", np.nan), "vac_HCP_Zr_eV": vac_vals.get("HCP_Zr", np.nan), "glass_pipeline_done": bool(pid in set(ncl_df["pot_id"])), "rdf_available": pid in rdf_store, "nve_energy_shift_per_atom_eV": get_nve_drift_for_potential(pid, ncl_df)})
    paper1_validation_df = pd.DataFrame(summary_rows)
    save_dataframe(paper1_validation_df, results_dir / "paper1_validation_summary.csv")

    print("\nDone.", flush=True)
    print(f"pyiron project: {pr.path}", flush=True)
    print(f"results dir:    {results_dir.resolve()}", flush=True)
    print("\nSelected potentials:", flush=True)
    print(", ".join([p["id"] for p in pots_to_run]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
