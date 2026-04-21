#!/usr/bin/env python3
"""
Run Paper 1 validation workflow for multiple Cu-Zr potentials.

Single-file Paper 1 validation driver for Cu-Zr potentials.
This version merges the earlier base workflow and the production wrapper into one maintainable script. It compares crystalline,
amorphous, and short-time dynamical readiness metrics across Cu-Zr MLIPs and EAM
baselines. It also includes a small MD-DMS-oriented precheck so that Paper 1
outputs can serve as a clean bridge toward the oscillatory-shear workflow used
in Paper 2.

Main blocks:
- smoke tests
- EOS for FCC Cu / HCP Zr / B2 CuZr
- local bulk modulus estimate
- vacancy formation energies for FCC Cu / HCP Zr
- melt -> quench -> minimize -> NVE for each selected glass composition
- composition-aware RDF and approximate S(q)
- compact Paper 1 summary table
- short MD-DMS precheck:
    * sinusoidal xy shear on a minimized glass
    * global shear-stress / strain signal export
    * per-atom stress export for downstream Fourier / phase analysis

This single-file version avoids the old wrapper/base self-import problem while keeping the same helper-module API
as the existing Cu-Zr pyiron + LAMMPS workflow.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
from ase.io import write, read
from pyiron_atomistics import Project, ase_to_pyiron
from ase import Atoms

# Compatibility shim: some pyiron-side helpers expect a structure method named
# get_species_symbols(), while ASE Atoms provides get_chemical_symbols().
# Keep ASE Atoms for stable copy()/repeat() behavior, but add the missing alias.
if not hasattr(Atoms, "get_species_symbols"):
    def _get_species_symbols(self):
        return self.get_chemical_symbols()
    Atoms.get_species_symbols = _get_species_symbols


from src.path_utils import resolve_path

GLASS_COMPOSITIONS: Dict[str, Tuple[float, float]] = {
    "Cu64Zr36": (0.64, 0.36),
    "Cu50Zr50": (0.50, 0.50),
    "Cu36Zr64": (0.36, 0.64),
}

DEFAULT_MACE_FILES = {
    "MACE_A": "CuZr_MACE_A_compiled.model-lammps.pt",
    "MACE_B": "CuZr_MACE_B_compiled.model-lammps.pt",
    "MACE_C": "CuZr_MACE_C_compiled.model-lammps.pt",
    "MACE_D": "CuZr_MACE_D_compiled.model-lammps.pt",
}

DEFAULT_EAM_NAMES = {
    "EAM_Mendelev_2019_CuZr": "EAM_Mendelev_2019_CuZr__MO_945018740343_000",
    "2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1": "2007--Mendelev-M-I--Cu-Zr--LAMMPS--ipr1",
}


# ----------------------------------------------------------------------
# ASE / pyiron structure compatibility layer
# ----------------------------------------------------------------------
# We keep ASE Atoms for easy repeat()/copy() behavior during seed/crystal
# generation, but pyiron-backed jobs expect structure methods like
# get_species_symbols() and to_hdf(). Provide a minimal adapter so the
# workflow can persist ASE structures into pyiron HDF without switching the
# whole script back to pyiron Atoms.

def _ase_get_species_symbols(self):
    return list(dict.fromkeys(self.get_chemical_symbols()))

def _ase_to_pyiron(self):
    return ase_to_pyiron(self.copy())

def _ase_to_hdf(self, hdf, group_name="structure"):
    return _ase_to_pyiron(self).to_hdf(hdf, group_name=group_name)

if not hasattr(Atoms, "get_species_symbols"):
    Atoms.get_species_symbols = _ase_get_species_symbols

if not hasattr(Atoms, "to_hdf"):
    Atoms.to_hdf = _ase_to_hdf

# Some pyiron/LAMMPS code paths access `structure.velocities` directly,
# while ASE exposes get_velocities()/set_velocities() methods instead.
# Provide a compatibility property so restarted MD jobs can reuse ASE atoms.
if not hasattr(Atoms, "velocities"):
    Atoms.velocities = property(Atoms.get_velocities, Atoms.set_velocities)


# Some pyiron/LAMMPS code paths check `structure.is_skewed()` to decide
# whether the cell is already triclinic. ASE Atoms does not provide this
# helper, so emulate pyiron's expectation by checking for non-zero
# off-diagonal cell components.
def _ase_is_skewed(self):
    cell = np.asarray(self.cell)
    return not np.allclose(cell, np.diag(np.diag(cell)))

if not hasattr(Atoms, "is_skewed"):
    Atoms.is_skewed = _ase_is_skewed


# Some pyiron/LAMMPS code paths expect a pyiron-style get_species_objects()
# method on the structure. ASE Atoms does not provide this, so delegate via
# a temporary pyiron conversion.
def _ase_get_species_objects(self):
    return ase_to_pyiron(self.copy()).get_species_objects()

if not hasattr(Atoms, "get_species_objects"):
    Atoms.get_species_objects = _ase_get_species_objects


# Helper module handle used by the run_* helpers below.
cz = None

def import_helper(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        fallback_paths = [PROJECT_ROOT / "src"]
        missing_msg = ", ".join(str(p) for p in fallback_paths)
        raise ModuleNotFoundError(
            f"Could not import helper module '{module_name}'. Looked via sys.path and expected it under: {missing_msg}"
        )

def maybe_resolve_model_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    p = Path(os.path.expanduser(raw))
    if p.is_absolute():
        return str(p)
    repo_rel = (PROJECT_ROOT / p).resolve()
    if repo_rel.exists():
        return str(repo_rel)
    return raw

class CommaListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed = [x.strip() for x in str(values).split(",") if x.strip()]
        setattr(namespace, self.dest, parsed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production Paper 1 Cu-Zr potential validation workflow"
    )
    parser.add_argument("--mode", choices=["dev", "prod"], default="prod")
    parser.add_argument("--project-path", default="../cu_zr_mlip_project")
    parser.add_argument("--results-dir", default="outputs/paper1_validation_prod")
    parser.add_argument("--pots", default="all", help="Comma-separated ids or 'all'")
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument(
        "--helper-module",
        default=os.environ.get("CUZR_HELPER_MODULE", "cuzr_setup_multi"),
    )
    parser.add_argument(
        "--glass-compositions",
        default=",".join(GLASS_COMPOSITIONS.keys()),
        help="Comma-separated glass composition IDs",
    )
    parser.add_argument(
        "--glass-rep",
        default="10,10,25",
        help="Replication for amorphous seed, e.g. 10,10,25",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Model-file overrides
    parser.add_argument("--mace-a-file", default=DEFAULT_MACE_FILES["MACE_A"])
    parser.add_argument("--mace-b-file", default=DEFAULT_MACE_FILES["MACE_B"])
    parser.add_argument("--mace-c-file", default=DEFAULT_MACE_FILES["MACE_C"])
    parser.add_argument("--mace-d-file", default=DEFAULT_MACE_FILES["MACE_D"])
    parser.add_argument(
        "--ace-514-file",
        default=os.environ.get("CUZR_ACE_514_FILE", ""),
        help="Path to trained ACE 514 potential file for LAMMPS/pace validation",
    )
    parser.add_argument(
        "--ace-1352-file",
        default=os.environ.get("CUZR_ACE_1352_FILE", ""),
        help="Path to trained ACE 1352 potential file for LAMMPS/pace validation",
    )
    parser.add_argument(
        "--skip-eam",
        action="store_true",
        help="Do not include EAM baselines in this run",
    )

    # Block control
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-eos", action="store_true")
    parser.add_argument("--skip-glass", action="store_true")
    parser.add_argument("--skip-vacancy", action="store_true")
    parser.add_argument("--skip-rdf-sq", action="store_true")
    parser.add_argument("--skip-mddms-precheck", action="store_true")

    # Runtime overrides
    parser.add_argument("--smoke-md-steps", type=int, default=None)
    parser.add_argument("--smoke-thermo", type=int, default=None)
    parser.add_argument("--eos-points", type=int, default=7)
    parser.add_argument("--eos-min-scale", type=float, default=0.96)
    parser.add_argument("--eos-max-scale", type=float, default=1.04)
    parser.add_argument("--t-melt", type=int, default=2000)
    parser.add_argument("--quench-ts", default=None, help="Comma-separated temperatures, e.g. 1500,1000,700,500,300")
    parser.add_argument("--steps-melt", type=int, default=None)
    parser.add_argument("--steps-each", type=int, default=None)
    parser.add_argument("--nve-steps", type=int, default=None)
    parser.add_argument("--thermo", type=int, default=None)
    parser.add_argument("--dump-every", type=int, default=None)
    parser.add_argument("--neigh-every", type=int, default=10)

    # MD-DMS precheck overrides
    parser.add_argument("--mddms-precheck-composition", default="Cu64Zr36")
    parser.add_argument("--mddms-temperature-K", type=float, default=300.0)
    parser.add_argument("--mddms-strain-amplitude", type=float, default=0.01)
    parser.add_argument("--mddms-period-ps", type=float, default=50.0)
    parser.add_argument("--mddms-cycles", type=int, default=2)
    parser.add_argument("--mddms-stress-every", type=int, default=10)
    parser.add_argument("--mddms-atom-dump-every", type=int, default=250)

    return parser.parse_args()

def configure_threads(mode_dev: bool) -> None:
    if mode_dev:
        os.environ.setdefault("OMP_NUM_THREADS", "6")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

def register_potentials(cz, args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        cz.make_mace_spec("MACE_A", maybe_resolve_model_path(args.mace_a_file) or DEFAULT_MACE_FILES["MACE_A"]),
        cz.make_mace_spec("MACE_B", maybe_resolve_model_path(args.mace_b_file) or DEFAULT_MACE_FILES["MACE_B"]),
        cz.make_mace_spec("MACE_C", maybe_resolve_model_path(args.mace_c_file) or DEFAULT_MACE_FILES["MACE_C"]),
        cz.make_mace_spec("MACE_D", maybe_resolve_model_path(args.mace_d_file) or DEFAULT_MACE_FILES["MACE_D"]),
    ]

    if not args.skip_eam:
        for pid, name in DEFAULT_EAM_NAMES.items():
            specs.append(cz.make_pyiron_spec(pid, name))

    ace_514 = maybe_resolve_model_path(args.ace_514_file)
    ace_1352 = maybe_resolve_model_path(args.ace_1352_file)
    if ace_514:
        specs.append(cz.make_ace_spec("ACE_514", ace_514))
    if ace_1352:
        specs.append(cz.make_ace_spec("ACE_1352", ace_1352))

    cz.register_potentials(specs)
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

def parse_glass_composition_ids(raw: str) -> List[str]:
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not ids:
        raise ValueError("No glass compositions requested")
    unknown = sorted(set(ids) - set(GLASS_COMPOSITIONS))
    if unknown:
        raise ValueError(f"Unknown glass composition IDs: {unknown}")
    return ids

def parse_rep(raw: str) -> Tuple[int, int, int]:
    vals = tuple(int(x.strip()) for x in raw.split(","))
    if len(vals) != 3:
        raise ValueError("--glass-rep must have exactly three integers, e.g. 5,5,5")
    return vals

def parse_quench_ts(raw: str | None, mode_dev: bool) -> Tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return (1200, 600, 300) if mode_dev else (1500, 1000, 700, 500, 300)
    vals = tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    if not vals:
        raise ValueError("--quench-ts produced an empty temperature list")
    return vals
def defaulted(value: Any, default: Any) -> Any:
    return default if value is None else value
def configure_runtime(mode_dev: bool, args: argparse.Namespace) -> Dict[str, Any]:
    cores = args.cores if args.cores is not None else (1 if mode_dev else 4)
    if mode_dev:
        smoke_md_steps = defaulted(args.smoke_md_steps, 50)
        smoke_thermo = defaulted(args.smoke_thermo, 10)
        steps_melt = defaulted(args.steps_melt, 500)
        steps_each = defaulted(args.steps_each, 500)
        nve_steps = defaulted(args.nve_steps, 1000)
        thermo = defaulted(args.thermo, 100)
        dump_every = defaulted(args.dump_every, 10)
    else:
        smoke_md_steps = defaulted(args.smoke_md_steps, 200)
        smoke_thermo = defaulted(args.smoke_thermo, 20)
        steps_melt = defaulted(args.steps_melt, 30000)
        steps_each = defaulted(args.steps_each, 20000)
        nve_steps = defaulted(args.nve_steps, 20000)
        thermo = defaulted(args.thermo, 1000)
        dump_every = defaulted(args.dump_every, 5000)

    return {
        "cores": int(cores),
        "smoke_md_steps": int(smoke_md_steps),
        "smoke_thermo": int(smoke_thermo),
        "eos_scales": np.linspace(float(args.eos_min_scale), float(args.eos_max_scale), int(args.eos_points)),
        "T_melt": int(args.t_melt),
        "quench_ts": parse_quench_ts(args.quench_ts, mode_dev),
        "steps_melt": int(steps_melt),
        "steps_each": int(steps_each),
        "nve_steps": int(nve_steps),
        "thermo": int(thermo),
        "dump_every": int(dump_every),
        "neigh_every": int(args.neigh_every),
    }

# NOTE:
# Use ASE Atoms for seed/crystal builders here. pyiron_atomistics.Atoms can trip over
# empty constraint lists during copy()/repeat() in some environments, which caused the
# glass-seed preparation failure seen on the live instance.

def make_fcc_cu(a: float = 3.615, rep=(4, 4, 4)) -> Atoms:
    basis = Atoms(
        symbols=["Cu"] * 4,
        positions=[[0.0, 0.0, 0.0], [0.0, 0.5 * a, 0.5 * a], [0.5 * a, 0.0, 0.5 * a], [0.5 * a, 0.5 * a, 0.0]],
        cell=[a, a, a],
        pbc=True,
    )
    return basis.repeat(rep)

def make_hcp_zr(a: float = 3.23, c_over_a: float = 1.593, rep=(4, 4, 3)) -> Atoms:
    c = c_over_a * a
    cell = np.array([[a, 0.0, 0.0], [0.5 * a, np.sqrt(3) / 2 * a, 0.0], [0.0, 0.0, c]])
    basis = Atoms(symbols=["Zr", "Zr"], scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, 1 / 3, 1 / 2]], cell=cell, pbc=True)
    return basis.repeat(rep)

def make_b2_cuzr(a: float = 3.2, rep=(6, 6, 6)) -> Atoms:
    structure = Atoms(symbols=["Cu", "Zr"], positions=[[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a]], cell=[a, a, a], pbc=True)
    return structure.repeat(rep)

def make_glass_seed(composition_id: str, a: float = 3.2, rep=(5, 5, 5), rng_seed: int = 42) -> Atoms:
    if composition_id not in GLASS_COMPOSITIONS:
        raise ValueError(f"Unknown composition_id: {composition_id}")

    cu_frac, zr_frac = GLASS_COMPOSITIONS[composition_id]
    total_frac = cu_frac + zr_frac
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(f"Fractions for {composition_id} do not sum to 1.0")

    structure = make_b2_cuzr(a=a, rep=rep)
    n_atoms = len(structure)
    n_zr = int(round(zr_frac * n_atoms))
    n_zr = max(0, min(n_atoms, n_zr))

    rng = np.random.default_rng(rng_seed)
    zr_indices = set(rng.choice(n_atoms, size=n_zr, replace=False).tolist())
    new_symbols = ["Zr" if i in zr_indices else "Cu" for i in range(n_atoms)]
    structure.set_chemical_symbols(new_symbols)
    return structure

def composition_from_structure(structure: Atoms) -> Dict[str, Any]:
    symbols = list(structure.get_chemical_symbols())
    n_total = len(symbols)
    n_cu = sum(1 for s in symbols if s == "Cu")
    n_zr = sum(1 for s in symbols if s == "Zr")
    return {
        "n_atoms": n_total,
        "n_cu": n_cu,
        "n_zr": n_zr,
        "x_cu": n_cu / n_total if n_total else np.nan,
        "x_zr": n_zr / n_total if n_total else np.nan,
    }

def _short_pot_tag(pot_id: str) -> str:
    aliases = {
        "EAM_Mendelev_2019_CuZr": "EAM19",
        "2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1": "EAM07",
        "ACE_514": "ACE514",
        "ACE_1352": "ACE1352",
    }
    if pot_id in aliases:
        return aliases[pot_id]
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", pot_id)
    return cleaned[:16] if len(cleaned) > 16 else cleaned


def jname(prefix: str, pot_spec: Dict[str, Any], mode_dev: bool, composition_id: str | None = None) -> str:
    run_tag = "dev" if mode_dev else "prod"
    comp = f"_{composition_id}" if composition_id else ""
    name = f"{run_tag}_{prefix}{comp}_{_short_pot_tag(pot_spec['id'])}"
    return name[:50]

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
    return {
        "label": label,
        "pot_id": pot_id,
        "structure": structure_name,
        "scale": scale,
        "n_atoms": len(job.structure),
        "energy_last_eV": e_last,
        "energy_per_atom_eV": e_last / len(job.structure),
    }

def md_summary(job, label: str, pot_id: str, structure_name: str) -> Dict[str, Any]:
    e_last = get_last_energy(job)
    return {
        "label": label,
        "pot_id": pot_id,
        "structure": structure_name,
        "n_atoms": len(job.structure),
        "energy_last_eV": e_last,
        "energy_per_atom_eV": e_last / len(job.structure),
        "temp_last_K": get_last_temp(job),
        "press_last": get_last_press(job),
    }

def run_static_resume(pr, job_name: str, structure, pot_spec: Dict[str, Any], cores: int):
    job = cz.make_lammps_job(pr, job_name, structure, pot_spec, delete_existing=False, cores=cores)
    job.calc_static()
    job = cz.load_or_run(pr, job)
    return job

def md_from_last(
    pr,
    job_name: str,
    prev_job,
    pot_spec: Dict[str, Any],
    T: float,
    steps: int,
    cores: int,
    thermo: int,
    dump_every: int,
    neigh_every: int,
    timestep_fs: float = 1.0,
):
    struct = prev_job.get_structure(iteration_step=-1)
    return cz.run_md(
        pr=pr,
        job_name=job_name,
        structure=struct,
        pot_spec=pot_spec,
        T=T,
        steps=steps,
        timestep_fs=timestep_fs,
        cores=cores,
        delete_existing=False,
        thermo=thermo,
        dump_every=dump_every,
        neigh_every=neigh_every,
    )

def run_nve(
    pr: Project,
    job_name: str,
    structure: Atoms,
    pot_spec: Dict[str, Any],
    steps: int,
    cores: int,
    thermo: int,
    neigh_every: int,
    timestep_fs: float = 1.0,
):
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

def cache_minimized_glass_structure(cache_dir: Path, pot_id: str, composition_id: str, structure: Atoms) -> Path | None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / f"glass_min_{pot_id}_{composition_id}.extxyz"
        write(out, structure, format="extxyz")
        return out
    except Exception:
        return None

def load_cached_glass_structure(cache_dir: Path, pot_id: str, composition_id: str) -> Atoms | None:
    p = cache_dir / f"glass_min_{pot_id}_{composition_id}.extxyz"
    if not p.exists():
        return None
    try:
        return read(p.as_posix(), format="extxyz")
    except Exception:
        return None

def run_eos_scan(
    pr: Project,
    structure: Atoms,
    structure_name: str,
    pot_spec: Dict[str, Any],
    scales: Iterable[float],
    mode_dev: bool,
    cores: int,
) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(scales):
        struct_s = isotropic_scale(structure, float(s))
        job = cz.run_static(
            pr=pr,
            job_name=jname(f"eos_{structure_name}_{i:02d}", pot_spec, mode_dev),
            structure=struct_s,
            pot_spec=pot_spec,
            cores=cores,
            delete_existing=False,
        )
        rows.append(static_summary(job, "eos", pot_spec["id"], structure_name, scale=float(s)))
    return pd.DataFrame(rows)

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

def vacancy_formation_energy(
    pr: Project,
    base_structure: Atoms,
    structure_name: str,
    pot_spec: Dict[str, Any],
    mode_dev: bool,
    cores: int,
    repeat=(3, 3, 3),
    atom_index: int = 0,
) -> Dict[str, Any]:
    perfect = base_structure.repeat(repeat)
    jperf = run_minimize(pr, jname(f"vac_perfect_{structure_name}", pot_spec, mode_dev), perfect, pot_spec, cores=cores)
    e_perfect = get_last_energy(jperf)
    n_perfect = len(jperf.structure)
    defect = make_vacancy_structure(jperf.get_structure(iteration_step=-1), atom_index=atom_index)
    jdef = run_minimize(pr, jname(f"vac_defect_{structure_name}", pot_spec, mode_dev), defect, pot_spec, cores=cores)
    e_defect = get_last_energy(jdef)
    e_form = e_defect - ((n_perfect - 1) / n_perfect) * e_perfect
    return {
        "pot_id": pot_spec["id"],
        "structure": structure_name,
        "repeat": "x".join(map(str, repeat)),
        "n_perfect": n_perfect,
        "e_vac_form_eV": e_form,
    }

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

def plot_rdf(rdf_store: Dict[Tuple[str, str], Dict[str, np.ndarray]], out_dir: Path) -> None:
    if not rdf_store:
        return
    for composition_id in sorted({key[1] for key in rdf_store}):
        plt.figure(figsize=(6, 4))
        for (pid, cid), vals in rdf_store.items():
            if cid != composition_id:
                continue
            plt.plot(vals["r"], vals["g_r"], label=pid)
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title(f"RDF of minimized glass: {composition_id}")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"rdf_glass_{composition_id}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[saved] {path}", flush=True)

def plot_sq(sq_store: Dict[Tuple[str, str], Dict[str, np.ndarray]], out_dir: Path) -> None:
    if not sq_store:
        return
    for composition_id in sorted({key[1] for key in sq_store}):
        plt.figure(figsize=(6, 4))
        for (pid, cid), vals in sq_store.items():
            if cid != composition_id:
                continue
            plt.plot(vals["q"], vals["s_q"], label=pid)
        plt.xlabel("q (1/Å)")
        plt.ylabel("S(q) proxy")
        plt.title(f"Approximate structure factor: {composition_id}")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"sq_proxy_{composition_id}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[saved] {path}", flush=True)

def get_nve_drift_for_potential_and_composition(pot_id: str, composition_id: str, ncl_df: pd.DataFrame) -> float:
    sub = ncl_df[(ncl_df["pot_id"] == pot_id) & (ncl_df["composition_id"] == composition_id)]
    if len(sub) == 0:
        return np.nan
    try:
        return float(sub["E_nve_per_atom_eV"].iloc[0] - sub["E_glass_min_per_atom_eV"].iloc[0])
    except Exception:
        return np.nan

def glass_formation_energy_from_refs(
    glass_energy_per_atom_eV: float,
    x_cu: float,
    x_zr: float,
    ref_cu_energy_per_atom_eV: float,
    ref_zr_energy_per_atom_eV: float,
) -> float:
    if np.isnan(glass_energy_per_atom_eV) or np.isnan(ref_cu_energy_per_atom_eV) or np.isnan(ref_zr_energy_per_atom_eV):
        return np.nan
    return glass_energy_per_atom_eV - (x_cu * ref_cu_energy_per_atom_eV + x_zr * ref_zr_energy_per_atom_eV)

def ncl_validation_pipeline(
    pr,
    structure0,
    composition_id: str,
    pot_spec: Dict[str, Any],
    mode_dev: bool,
    cores: int,
    T_melt: int,
    steps_melt: int,
    quench_ts: Sequence[int],
    steps_each: int,
    nve_steps: int,
    thermo: int,
    dump_every: int,
    neigh_every: int,
    cache_dir: Path | None = None,
) -> tuple[Dict[str, Any], Atoms]:
    comp = composition_from_structure(structure0)
    out: Dict[str, Any] = {
        "pot_id": pot_spec["id"],
        "composition_id": composition_id,
        **comp,
        "n_atoms": len(structure0),
    }

    j0 = run_static_resume(
        pr,
        jname("static_glass_seed", pot_spec, mode_dev, composition_id=composition_id),
        structure0,
        pot_spec,
        cores=cores,
    )
    out["E_static_per_atom_eV"] = get_last_energy(j0) / len(structure0)

    jm = cz.run_md(
        pr=pr,
        job_name=jname("melt", pot_spec, mode_dev, composition_id=composition_id),
        structure=structure0,
        pot_spec=pot_spec,
        T=T_melt,
        steps=steps_melt,
        timestep_fs=1.0,
        cores=cores,
        delete_existing=False,
        thermo=thermo,
        dump_every=dump_every,
        neigh_every=neigh_every,
    )
    out["E_melt_per_atom_eV"] = get_last_energy(jm) / len(structure0)
    out["T_melt_last_K"] = get_last_temp(jm)

    prev = jm
    for T in quench_ts:
        jq = md_from_last(
            pr=pr,
            job_name=jname(f"quench_{T}K", pot_spec, mode_dev, composition_id=composition_id),
            prev_job=prev,
            pot_spec=pot_spec,
            T=T,
            steps=steps_each,
            cores=cores,
            thermo=thermo,
            dump_every=dump_every,
            neigh_every=neigh_every,
        )
        out[f"E_{T}K_per_atom_eV"] = get_last_energy(jq) / len(structure0)
        out[f"T_{T}K_last_K"] = get_last_temp(jq)
        prev = jq

    jmin = run_minimize(
        pr,
        jname("glass_min", pot_spec, mode_dev, composition_id=composition_id),
        prev.get_structure(iteration_step=-1),
        pot_spec,
        cores=cores,
    )
    out["E_glass_min_per_atom_eV"] = get_last_energy(jmin) / len(structure0)

    jnve = run_nve(
        pr,
        jname("glass_nve", pot_spec, mode_dev, composition_id=composition_id),
        jmin.get_structure(iteration_step=-1),
        pot_spec,
        steps=nve_steps,
        cores=cores,
        thermo=thermo,
        neigh_every=neigh_every,
    )
    out["E_nve_per_atom_eV"] = get_last_energy(jnve) / len(structure0)
    out["T_nve_last_K"] = get_last_temp(jnve)

    s_min = jmin.get_structure(iteration_step=-1)
    compressed = isotropic_scale(s_min, 0.97)
    jcomp = run_static_resume(
        pr,
        jname("glass_compressed", pot_spec, mode_dev, composition_id=composition_id),
        compressed,
        pot_spec,
        cores=cores,
    )
    out["E_compressed_per_atom_eV"] = get_last_energy(jcomp) / len(compressed)
    out["job_glass_min"] = jname("glass_min", pot_spec, mode_dev, composition_id=composition_id)
    if cache_dir is not None:
        cache_minimized_glass_structure(cache_dir, pot_spec["id"], composition_id, s_min)
    return out, s_min

def run_mddms_precheck(
    pr,
    structure,
    composition_id: str,
    pot_spec: Dict[str, Any],
    mode_dev: bool,
    cores: int,
    temperature_K: float,
    strain_amplitude: float,
    period_ps: float,
    cycles: int,
    stress_every: int,
    atom_dump_every: int,
) -> Dict[str, Any]:
    timestep_fs = 1.0
    period_steps = int(round(period_ps * 1000.0 / timestep_fs))
    total_steps = int(cycles * period_steps)
    if period_steps <= 0 or total_steps <= 0:
        raise ValueError("MD-DMS precheck needs positive period_steps and total_steps")

    job_name = jname("mddms_precheck", pot_spec, mode_dev, composition_id=composition_id)
    job = cz.make_lammps_job(pr, job_name, structure, pot_spec, delete_existing=False, cores=cores)
    job.calc_md(temperature=temperature_K, n_ionic_steps=total_steps, time_step=timestep_fs)
    job.input.control["variable___thermotime"] = f"equal {max(10, int(stress_every))}"
    job.input.control["neighbor"] = "1.0 bin"
    job.input.control["neigh_modify"] = "every 10 delay 0 check yes"
    job.input.control["variable___mddms_A"] = f"equal {float(strain_amplitude):.8f}"
    job.input.control["variable___mddms_period"] = f"equal {int(period_steps)}"
    job.input.control["variable___mddms_gamma"] = "equal v_mddms_A*sin(2.0*PI*step/v_mddms_period)"
    job.input.control["fix___mddms_deform"] = "all deform 1 xy variable v_mddms_gamma remap x units box"
    job.input.control["compute___mddms_stress"] = "all stress/atom NULL"
    job.input.control["compute___mddms_voronoi"] = "all voronoi/atom"
    job.input.control["fix___mddms_print"] = (
        f'all print {int(stress_every)} "${{step}} ${{time}} ${{pxy}} ${{v_mddms_gamma}}" '
        'file system_stress_strain.dat screen no'
    )
    job.input.control["variable___dumptime"] = f"equal {int(atom_dump_every)}"
    job.input.control["dump___mddms_atoms"] = (
        "all custom ${dumptime} dump.mddms_precheck.lammpstrj "
        "id type xsu ysu zsu vx vy vz "
        "c_mddms_voronoi "
        "c_mddms_stress[1] c_mddms_stress[2] c_mddms_stress[3] "
        "c_mddms_stress[4] c_mddms_stress[5] c_mddms_stress[6]"
    )
    job.input.control["dump_modify___mddms_atoms"] = "sort id"
    job = cz.load_or_run(pr, job)

    wd = Path(job.working_directory)
    stress_file = wd / "system_stress_strain.dat"
    atom_dump = wd / "dump.mddms_precheck.lammpstrj"
    out = {
        "pot_id": pot_spec["id"],
        "composition_id": composition_id,
        "job_name": job_name,
        "temperature_K": float(temperature_K),
        "strain_amplitude": float(strain_amplitude),
        "period_ps": float(period_ps),
        "cycles": int(cycles),
        "total_steps": int(total_steps),
        "stress_every": int(stress_every),
        "atom_dump_every": int(atom_dump_every),
        "working_directory": str(wd),
        "stress_file": str(stress_file),
        "atom_dump_file": str(atom_dump),
        "stress_file_exists": stress_file.exists(),
        "atom_dump_exists": atom_dump.exists(),
        "stress_file_size_B": stress_file.stat().st_size if stress_file.exists() else np.nan,
        "atom_dump_size_B": atom_dump.stat().st_size if atom_dump.exists() else np.nan,
        "E_last_eV": get_last_energy(job),
        "T_last_K": get_last_temp(job),
        "P_last_bar_like": get_last_press(job),
    }
    if stress_file.exists():
        try:
            sig = pd.read_csv(
                stress_file,
                delim_whitespace=True,
                header=None,
                names=["step", "time_ps", "pxy_bar", "gamma_xy"],
            )
            out["signal_rows"] = int(len(sig))
            out["gamma_abs_max"] = float(np.max(np.abs(sig["gamma_xy"]))) if len(sig) else np.nan
            out["pxy_abs_max"] = float(np.max(np.abs(sig["pxy_bar"]))) if len(sig) else np.nan
        except Exception as exc:  # pragma: no cover
            out["signal_rows"] = np.nan
            out["signal_read_error"] = str(exc)
    return out

def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df
def error_row(block: str, pot_id: str, structure: str | None = None, composition_id: str | None = None, exc: Exception | str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {"pot_id": pot_id, "block": block, "error": str(exc)}
    if structure is not None:
        row["structure"] = structure
    if composition_id is not None:
        row["composition_id"] = composition_id
    return row
def save_progress(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    save_dataframe(pd.DataFrame(rows), path)

def main() -> int:
    global cz
    args = parse_args()
    mode_dev = args.mode == "dev"
    configure_threads(mode_dev)
    cz = import_helper(args.helper_module)

    project_path = resolve_path(args.project_path, base_dir=PROJECT_ROOT)
    results_dir = resolve_path(args.results_dir, base_dir=PROJECT_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)
    pr = Project(str(project_path))

    all_pots = register_potentials(cz, args)
    pots_to_run = select_potentials(all_pots, args.pots)
    glass_composition_ids = parse_glass_composition_ids(args.glass_compositions)
    glass_rep = parse_rep(args.glass_rep)
    runtime = configure_runtime(mode_dev, args)

    settings = {
        "mode": args.mode,
        "project_path": str(project_path),
        "results_dir": str(results_dir),
        "pots": [p["id"] for p in pots_to_run],
        "glass_compositions": glass_composition_ids,
        "glass_rep": list(glass_rep),
        "seed": args.seed,
        "helper_module": args.helper_module,
        "skip_eam": bool(args.skip_eam),
        "skip_smoke": bool(args.skip_smoke),
        "skip_eos": bool(args.skip_eos),
        "skip_glass": bool(args.skip_glass),
        "skip_vacancy": bool(args.skip_vacancy),
        "skip_rdf_sq": bool(args.skip_rdf_sq),
        "skip_mddms_precheck": bool(args.skip_mddms_precheck),
        "mace_a_file": maybe_resolve_model_path(args.mace_a_file) or args.mace_a_file,
        "mace_b_file": maybe_resolve_model_path(args.mace_b_file) or args.mace_b_file,
        "mace_c_file": maybe_resolve_model_path(args.mace_c_file) or args.mace_c_file,
        "mace_d_file": maybe_resolve_model_path(args.mace_d_file) or args.mace_d_file,
        "ace_514_file": maybe_resolve_model_path(args.ace_514_file) or "",
        "ace_1352_file": maybe_resolve_model_path(args.ace_1352_file) or "",
        **{k: (list(v) if isinstance(v, tuple) else v.tolist() if isinstance(v, np.ndarray) else v) for k, v in runtime.items()},
        "mddms_precheck_composition": args.mddms_precheck_composition,
        "mddms_temperature_K": float(args.mddms_temperature_K),
        "mddms_strain_amplitude": float(args.mddms_strain_amplitude),
        "mddms_period_ps": float(args.mddms_period_ps),
        "mddms_cycles": int(args.mddms_cycles),
        "mddms_stress_every": int(args.mddms_stress_every),
        "mddms_atom_dump_every": int(args.mddms_atom_dump_every),
    }
    save_json(settings, results_dir / "run_settings.json")

    pot_df = pd.DataFrame(
        [
            {
                "id": p["id"],
                "mode": p["mode"],
                "family": p.get("family", ""),
                "name": p.get("name", ""),
                "model_file": p.get("model_file", ""),
            }
            for p in pots_to_run
        ]
    )
    save_dataframe(pot_df, results_dir / "potentials_selected.csv")

    glass_meta_rows = []
    for i, cid in enumerate(glass_composition_ids):
        seed_structure = make_glass_seed(cid, rep=glass_rep, rng_seed=args.seed + i)
        comp = composition_from_structure(seed_structure)
        glass_meta_rows.append({"composition_id": cid, **comp, "rep": "x".join(map(str, glass_rep))})
    save_dataframe(pd.DataFrame(glass_meta_rows), results_dir / "glass_compositions_selected.csv")

    fcc_cu = make_fcc_cu()
    hcp_zr = make_hcp_zr()
    b2_cuzr = make_b2_cuzr()
    glass_seeds = {
        cid: make_glass_seed(cid, rep=glass_rep, rng_seed=args.seed + i)
        for i, cid in enumerate(glass_composition_ids)
    }
    glass_cache_dir = results_dir / "glass_min_cache"
    glass_min_cache: Dict[Tuple[str, str], Atoms] = {}

    smoke_rows: List[Dict[str, Any]] = []
    if not args.skip_smoke:
        for p in pots_to_run:
            print("SMOKE TEST:", p["id"], flush=True)
            try:
                j_static = run_static_resume(
                    pr=pr,
                    job_name=jname("smoke_static_b2", p, mode_dev),
                    structure=b2_cuzr,
                    pot_spec=p,
                    cores=runtime["cores"],
                )
                smoke_rows.append(static_summary(j_static, "smoke_static", p["id"], "B2_CuZr"))
            except Exception as exc:
                smoke_rows.append(error_row("smoke_static", p["id"], structure="B2_CuZr", exc=exc))
            save_progress(smoke_rows, results_dir / "smoke_validation.csv")
            try:
                j_md = cz.run_md(
                    pr=pr,
                    job_name=jname("smoke_md_b2", p, mode_dev),
                    structure=b2_cuzr,
                    pot_spec=p,
                    T=300,
                    steps=runtime["smoke_md_steps"],
                    timestep_fs=1.0,
                    cores=runtime["cores"],
                    delete_existing=False,
                    thermo=runtime["smoke_thermo"],
                    dump_every=10,
                    neigh_every=10,
                )
                smoke_rows.append(md_summary(j_md, "smoke_md", p["id"], "B2_CuZr"))
            except Exception as exc:
                smoke_rows.append(error_row("smoke_md", p["id"], structure="B2_CuZr", exc=exc))
            save_progress(smoke_rows, results_dir / "smoke_validation.csv")

    eos_rows: List[Dict[str, Any]] = []
    if not args.skip_eos:
        structure_map = [(fcc_cu, "FCC_Cu"), (hcp_zr, "HCP_Zr"), (b2_cuzr, "B2_CuZr")]
        for p in pots_to_run:
            print("EOS:", p["id"], flush=True)
            for structure, sname in structure_map:
                for i, s in enumerate(runtime["eos_scales"]):
                    try:
                        struct_s = isotropic_scale(structure, float(s))
                        job = run_static_resume(
                            pr=pr,
                            job_name=jname(f"eos_{sname}_{i:02d}", p, mode_dev),
                            structure=struct_s,
                            pot_spec=p,
                            cores=runtime["cores"],
                        )
                        eos_rows.append(static_summary(job, "eos", p["id"], sname, scale=float(s)))
                    except Exception as exc:
                        eos_rows.append({
                            "pot_id": p["id"],
                            "stage": "eos",
                            "structure": sname,
                            "scale": float(s),
                            "energy_eV": np.nan,
                            "energy_per_atom_eV": np.nan,
                            "volume_A3": np.nan,
                            "error": str(exc),
                        })
                    save_progress(eos_rows, results_dir / "eos_validation.csv")
        eos_df = pd.DataFrame(eos_rows)
        if len(eos_df):
            plot_df = eos_df[pd.notna(eos_df.get("energy_per_atom_eV", pd.Series(dtype=float)))]
            if len(plot_df):
                plot_eos(plot_df, results_dir)
    else:
        eos_df = pd.DataFrame()

    ncl_rows: List[Dict[str, Any]] = []
    if not args.skip_glass:
        for p in pots_to_run:
            for composition_id in glass_composition_ids:
                print(f"RUNNING GLASS PIPELINE: {p['id']} / {composition_id}", flush=True)
                try:
                    row, s_min = ncl_validation_pipeline(
                        pr=pr,
                        structure0=glass_seeds[composition_id],
                        composition_id=composition_id,
                        pot_spec=p,
                        mode_dev=mode_dev,
                        cores=runtime["cores"],
                        T_melt=runtime["T_melt"],
                        steps_melt=runtime["steps_melt"],
                        quench_ts=runtime["quench_ts"],
                        steps_each=runtime["steps_each"],
                        nve_steps=runtime["nve_steps"],
                        thermo=runtime["thermo"],
                        dump_every=runtime["dump_every"],
                        neigh_every=runtime["neigh_every"],
                        cache_dir=glass_cache_dir,
                    )
                    ncl_rows.append(row)
                    glass_min_cache[(p["id"], composition_id)] = s_min
                except Exception as exc:
                    ncl_rows.append(error_row("glass_pipeline", p["id"], composition_id=composition_id, exc=exc))
                save_progress(ncl_rows, results_dir / "ncl_validation.csv")
    ncl_df = pd.DataFrame(ncl_rows)

    mddms_rows: List[Dict[str, Any]] = []
    if not args.skip_mddms_precheck:
        precheck_cid = args.mddms_precheck_composition.strip()
        if precheck_cid not in glass_composition_ids:
            print(
                f"MD-DMS precheck composition '{precheck_cid}' was not requested in --glass-compositions; using '{glass_composition_ids[0]}' instead.",
                flush=True,
            )
            precheck_cid = glass_composition_ids[0]
        for p in pots_to_run:
            print(f"MD-DMS PRECHECK: {p['id']} / {precheck_cid}", flush=True)
            try:
                s_pre = glass_min_cache.get((p["id"], precheck_cid))
                if s_pre is None:
                    s_pre = load_cached_glass_structure(glass_cache_dir, p["id"], precheck_cid)
                if s_pre is None:
                    raise RuntimeError(f"No minimized glass structure cached for {p['id']} / {precheck_cid}")
                mddms_rows.append(
                    run_mddms_precheck(
                        pr=pr,
                        structure=s_pre,
                        composition_id=precheck_cid,
                        pot_spec=p,
                        mode_dev=mode_dev,
                        cores=runtime["cores"],
                        temperature_K=args.mddms_temperature_K,
                        strain_amplitude=args.mddms_strain_amplitude,
                        period_ps=args.mddms_period_ps,
                        cycles=args.mddms_cycles,
                        stress_every=args.mddms_stress_every,
                        atom_dump_every=args.mddms_atom_dump_every,
                    )
                )
            except Exception as exc:
                mddms_rows.append(error_row("mddms_precheck", p["id"], composition_id=precheck_cid, exc=exc))
            save_progress(mddms_rows, results_dir / "mddms_precheck_summary.csv")

    bulk_rows: List[Dict[str, Any]] = []
    if not args.skip_eos and len(eos_df):
        structure_map = {"FCC_Cu": fcc_cu, "HCP_Zr": hcp_zr, "B2_CuZr": b2_cuzr}
        for pot_id in sorted(eos_df["pot_id"].dropna().unique()):
            for sname, structure in structure_map.items():
                sub = eos_df[(eos_df["pot_id"] == pot_id) & (eos_df["structure"] == sname)].copy()
                try:
                    bulk_rows.append(
                        {
                            "pot_id": pot_id,
                            "structure": sname,
                            "bulk_modulus_GPa_est": local_bulk_modulus_from_eos(structure, sub),
                        }
                    )
                except Exception as exc:
                    bulk_rows.append({"pot_id": pot_id, "structure": sname, "bulk_modulus_GPa_est": np.nan, "error": str(exc)})
        save_dataframe(pd.DataFrame(bulk_rows), results_dir / "bulk_modulus_estimates.csv")
    bulk_df = pd.DataFrame(bulk_rows)

    vac_rows: List[Dict[str, Any]] = []
    if not args.skip_vacancy:
        vacuum_targets = [("FCC_Cu", fcc_cu, (4, 4, 4)), ("HCP_Zr", hcp_zr, (4, 4, 3))]
        for p in pots_to_run:
            print("VACANCY:", p["id"], flush=True)
            for sname, structure, rep in vacuum_targets:
                try:
                    vac_rows.append(vacancy_formation_energy(pr, structure, sname, p, mode_dev=mode_dev, cores=runtime["cores"], repeat=rep))
                except Exception as exc:
                    vac_rows.append(
                        {
                            "pot_id": p["id"],
                            "structure": sname,
                            "repeat": "x".join(map(str, rep)),
                            "n_perfect": np.nan,
                            "e_vac_form_eV": np.nan,
                            "error": str(exc),
                        }
                    )
                save_progress(vac_rows, results_dir / "vacancy_formation.csv")
    vac_df = pd.DataFrame(vac_rows)

    # Derived glass formation energies
    ensure_columns(ncl_df, ["pot_id", "x_cu", "x_zr", "E_glass_min_per_atom_eV"])
    ref_energy_map: Dict[str, Dict[str, float]] = {}
    if len(eos_df):
        for pid in sorted(eos_df["pot_id"].dropna().unique()):
            ref_energy_map[pid] = {}
            for sname in ["FCC_Cu", "HCP_Zr"]:
                sub = eos_df[(eos_df["pot_id"] == pid) & (eos_df["structure"] == sname)].copy()
                if len(sub) == 0:
                    ref_energy_map[pid][sname] = np.nan
                else:
                    ref_energy_map[pid][sname] = float(sub.sort_values("energy_per_atom_eV").iloc[0]["energy_per_atom_eV"])
    if len(ncl_df):
        ncl_df["glass_form_energy_per_atom_eV"] = ncl_df.apply(
            lambda row: glass_formation_energy_from_refs(
                glass_energy_per_atom_eV=float(row["E_glass_min_per_atom_eV"]) if pd.notna(row["E_glass_min_per_atom_eV"]) else np.nan,
                x_cu=float(row["x_cu"]) if pd.notna(row["x_cu"]) else np.nan,
                x_zr=float(row["x_zr"]) if pd.notna(row["x_zr"]) else np.nan,
                ref_cu_energy_per_atom_eV=ref_energy_map.get(row["pot_id"], {}).get("FCC_Cu", np.nan),
                ref_zr_energy_per_atom_eV=ref_energy_map.get(row["pot_id"], {}).get("HCP_Zr", np.nan),
            ),
            axis=1,
        )
        save_dataframe(ncl_df, results_dir / "ncl_validation.csv")

    rdf_store: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    sq_store: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    rdf_rows: List[Dict[str, Any]] = []
    if not args.skip_rdf_sq and len(ncl_df):
        q_grid = np.linspace(0.5, 16.0, 300)
        for p in pots_to_run:
            pid = p["id"]
            for composition_id in glass_composition_ids:
                try:
                    s = glass_min_cache.get((pid, composition_id))
                    if s is None:
                        s = load_cached_glass_structure(glass_cache_dir, pid, composition_id)
                    if s is None:
                        raise RuntimeError(f"No minimized glass structure cached for {pid} / {composition_id}")
                    ana = Analysis(s)
                    r, g = ana.get_rdf(rmax=8.0, nbins=200)
                    rdf_store[(pid, composition_id)] = {"r": np.asarray(r), "g_r": np.asarray(g)}
                    volume = get_structure_volume(s)
                    rho = len(s) / volume
                    sq = structure_factor_from_rdf(rdf_store[(pid, composition_id)]["r"], rdf_store[(pid, composition_id)]["g_r"], rho, q_grid)
                    sq_store[(pid, composition_id)] = {"q": q_grid, "s_q": sq}
                    rdf_rows.append(
                        {
                            "pot_id": pid,
                            "composition_id": composition_id,
                            "r_peak_A": float(r[np.argmax(g)]),
                            "g_peak": float(np.max(g)),
                        }
                    )
                except Exception as exc:
                    rdf_rows.append(error_row("rdf_sq", pid, composition_id=composition_id, exc=exc))
                save_progress(rdf_rows, results_dir / "rdf_summary.csv")
        if rdf_store:
            plot_rdf(rdf_store, results_dir)
            for (pid, cid), vals in rdf_store.items():
                np.savez(results_dir / f"rdf_{pid}_{cid}.npz", r=vals["r"], g_r=vals["g_r"])
        if sq_store:
            plot_sq(sq_store, results_dir)
            for (pid, cid), vals in sq_store.items():
                np.savez(results_dir / f"sq_{pid}_{cid}.npz", q=vals["q"], s_q=vals["s_q"])

    summary_rows: List[Dict[str, Any]] = []
    if len(ncl_df):
        bulk_df = ensure_columns(bulk_df, ["pot_id", "structure", "bulk_modulus_GPa_est"])
        vac_df = ensure_columns(vac_df, ["pot_id", "structure", "e_vac_form_eV"])
        mddms_df = pd.DataFrame(mddms_rows)
        for p in pots_to_run:
            pid = p["id"]
            bulk_vals = bulk_df[bulk_df["pot_id"] == pid].set_index("structure")["bulk_modulus_GPa_est"].to_dict() if len(bulk_df) else {}
            vac_vals = vac_df[vac_df["pot_id"] == pid].set_index("structure")["e_vac_form_eV"].to_dict() if len(vac_df) else {}
            for composition_id in glass_composition_ids:
                sub = ncl_df[(ncl_df["pot_id"] == pid) & (ncl_df["composition_id"] == composition_id)]
                if len(sub) == 0:
                    continue
                row = sub.iloc[0]
                summary_rows.append(
                    {
                        "pot_id": pid,
                        "composition_id": composition_id,
                        "bulk_FCC_Cu_GPa_est": bulk_vals.get("FCC_Cu", np.nan),
                        "bulk_HCP_Zr_GPa_est": bulk_vals.get("HCP_Zr", np.nan),
                        "bulk_B2_CuZr_GPa_est": bulk_vals.get("B2_CuZr", np.nan),
                        "vac_FCC_Cu_eV": vac_vals.get("FCC_Cu", np.nan),
                        "vac_HCP_Zr_eV": vac_vals.get("HCP_Zr", np.nan),
                        "glass_n_atoms": row.get("n_atoms", np.nan),
                        "glass_x_cu": row.get("x_cu", np.nan),
                        "glass_x_zr": row.get("x_zr", np.nan),
                        "glass_min_energy_per_atom_eV": row.get("E_glass_min_per_atom_eV", np.nan),
                        "glass_form_energy_per_atom_eV": row.get("glass_form_energy_per_atom_eV", np.nan),
                        "nve_energy_shift_per_atom_eV": get_nve_drift_for_potential_and_composition(pid, composition_id, ncl_df),
                        "rdf_available": (pid, composition_id) in rdf_store,
                        "sq_available": (pid, composition_id) in sq_store,
                        "mddms_precheck_available": bool(
                            len(mddms_df[(mddms_df.get("pot_id") == pid) & (mddms_df.get("composition_id") == composition_id)])
                        ) if len(mddms_df) else False,
                    }
                )
        save_dataframe(pd.DataFrame(summary_rows), results_dir / "paper1_validation_summary.csv")

    print("\nDone.", flush=True)
    print(f"pyiron project: {pr.path}", flush=True)
    print(f"results dir:    {results_dir.resolve()}", flush=True)
    print("\nSelected potentials:", flush=True)
    print(", ".join([p["id"] for p in pots_to_run]), flush=True)
    print("Selected glass compositions:", flush=True)
    print(", ".join(glass_composition_ids), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
