#!/usr/bin/env python3
"""
Direct LAMMPS validation driver for Cu-Zr potentials.

Direct LAMMPS validation driver for Cu-Zr potentials.

Direct replacement for the pyiron-based validation workflow:
- smoke static
- smoke short MD
- EOS for FCC Cu / HCP Zr / B2 CuZr
- vacancy formation for FCC Cu / HCP Zr
- basic amorphous melt/quench/minimize sanity checks
- RDF peak summary for minimized glasses

No pyiron dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.data import atomic_masses, atomic_numbers

DEFAULT_MACE_FILES = {
    "MACE_A": "models/raw/mace_A.model-mliap_lammps.pt",
    "MACE_B": "models/raw/mace_B.model-mliap_lammps.pt",
    "MACE_C": "models/raw/mace_C.model-mliap_lammps.pt",
    "MACE_D": "models/raw/mace_D.model-mliap_lammps.pt",
}
DEFAULT_ACE_FILES = {
    "ACE_514": "",
    "ACE_1352": "",
}
DEFAULT_EAM_FILES = {
    "EAM_Mendelev_2019_CuZr": "models/raw/eam/Cu-Zr_4.eam.fs",
    "2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1": "models/raw/eam/CuZr_mm.eam.fs",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PotentialSpec:
    id: str
    family: str
    model_file: str
    pair_style: str
    pair_coeff: str


def maybe_resolve_model_path(raw: Optional[str]) -> Optional[str]:
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
    cwd_rel = (Path.cwd() / p).resolve()
    if cwd_rel.exists():
        return str(cwd_rel)
    return raw


def infer_eam_pair_style(model_file: str) -> str:
    low = model_file.lower()
    if low.endswith(".eam.fs"):
        return "eam/fs"
    if low.endswith(".eam.alloy"):
        return "eam/alloy"
    if low.endswith(".eam"):
        return "eam"
    raise ValueError(f"Cannot infer EAM pair_style from file: {model_file}")


def make_mace_spec(id_: str, model_file: str, elements: Tuple[str, str] = ("Cu", "Zr")) -> PotentialSpec:
    return PotentialSpec(
        id=id_,
        family="MACE",
        model_file=model_file,
        pair_style=f"mliap unified {model_file} 0",
        pair_coeff=f"* * {' '.join(elements)}",
    )


def make_ace_spec(id_: str, model_file: str, elements: Tuple[str, str] = ("Cu", "Zr")) -> PotentialSpec:
    return PotentialSpec(
        id=id_,
        family="ACE",
        model_file=model_file,
        pair_style="pace",
        pair_coeff=f"* * {model_file} {' '.join(elements)}",
    )


def make_eam_spec(id_: str, model_file: str, elements: Tuple[str, str] = ("Cu", "Zr")) -> PotentialSpec:
    return PotentialSpec(
        id=id_,
        family="EAM",
        model_file=model_file,
        pair_style=infer_eam_pair_style(model_file),
        pair_coeff=f"* * {model_file} {' '.join(elements)}",
    )


def ensure_path_exists(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Model file does not exist: {path_str}")
    return str(p.resolve())


def build_potentials(args: argparse.Namespace) -> List[PotentialSpec]:
    pots: List[PotentialSpec] = []
    if not args.skip_mace:
        for pid, default in DEFAULT_MACE_FILES.items():
            raw = getattr(args, f"{pid.lower()}_file")
            resolved = maybe_resolve_model_path(raw) or default
            if Path(str(resolved)).exists():
                pots.append(make_mace_spec(pid, ensure_path_exists(str(resolved))))
    if not args.skip_eam:
        eam_2019 = maybe_resolve_model_path(args.eam_2019_file) or DEFAULT_EAM_FILES["EAM_Mendelev_2019_CuZr"]
        eam_2007 = maybe_resolve_model_path(args.eam_2007_file) or DEFAULT_EAM_FILES["2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1"]
        pots.append(make_eam_spec("EAM_Mendelev_2019_CuZr", ensure_path_exists(str(eam_2019))))
        pots.append(make_eam_spec("2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1", ensure_path_exists(str(eam_2007))))
    if not args.skip_ace:
        for pid, default in DEFAULT_ACE_FILES.items():
            raw = getattr(args, f"{pid.lower()}_file")
            resolved = maybe_resolve_model_path(raw) or default
            if resolved and Path(str(resolved)).exists():
                pots.append(make_ace_spec(pid, ensure_path_exists(str(resolved))))
    return pots


def select_potentials(all_pots: Sequence[PotentialSpec], raw: str) -> List[PotentialSpec]:
    if raw.strip().lower() == "all":
        return list(all_pots)
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    selected = [p for p in all_pots if p.id in wanted]
    missing = sorted(wanted - {p.id for p in selected})
    if missing:
        raise ValueError(f"Unknown potentials requested: {missing}")
    return selected


def make_b2_cuzr(a: float = 3.2) -> Atoms:
    cell = np.diag([a, a, a])
    scaled_positions = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    atoms = Atoms(symbols=["Cu", "Zr"], cell=cell, pbc=True)
    atoms.set_scaled_positions(scaled_positions)
    return atoms


def crystal_structures() -> Dict[str, Atoms]:
    fcc_cu = bulk("Cu", "fcc", a=3.615, cubic=True)
    hcp_zr = bulk("Zr", "hcp", a=3.232, c=5.147)
    b2_cuzr = make_b2_cuzr()
    return {
        "FCC_Cu": fcc_cu,
        "HCP_Zr": hcp_zr,
        "B2_CuZr": b2_cuzr,
    }


def replicated(atoms: Atoms, reps: Tuple[int, int, int]) -> Atoms:
    return atoms.repeat(reps)


def sanitize_atoms(atoms: Atoms) -> Atoms:
    clean = Atoms(
        symbols=list(atoms.get_chemical_symbols()),
        positions=np.array(atoms.get_positions(), dtype=float),
        cell=np.array(atoms.cell),
        pbc=np.array(atoms.pbc, dtype=bool),
    )
    return clean


def parse_thermo_table(log_path: Path) -> Dict[str, float]:
    header: Optional[List[str]] = None
    last_row: Optional[List[str]] = None
    with log_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Step "):
                header = line.split()
                last_row = None
                continue
            if header is None:
                continue
            parts = line.split()
            if len(parts) != len(header):
                continue
            try:
                [float(x) for x in parts]
            except ValueError:
                continue
            last_row = parts
    if header is None or last_row is None:
        raise RuntimeError(f"Could not parse thermo table from {log_path}")
    result: Dict[str, float] = {}
    for k, v in zip(header, last_row):
        try:
            result[k] = float(v)
        except ValueError:
            pass
    return result



def unique_species_in_order(atoms: Atoms) -> List[str]:
    species: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in species:
            species.append(sym)
    return species


def species_mapping_for_structure(atoms: Atoms) -> List[str]:
    present = set(atoms.get_chemical_symbols())
    canonical = [s for s in ("Cu", "Zr") if s in present]
    return canonical if canonical else unique_species_in_order(atoms)


def pair_coeff_for_structure(potential: PotentialSpec, atoms: Atoms) -> str:
    species_order = species_mapping_for_structure(atoms)
    if potential.family == "EAM":
        return f"* * {potential.model_file} {' '.join(species_order)}"
    if potential.family == "ACE":
        return f"* * {potential.model_file} {' '.join(species_order)}"
    if potential.family == "MACE":
        return f"* * {' '.join(species_order)}"
    return potential.pair_coeff


def mass_commands_for_structure(atoms: Atoms) -> List[str]:
    cmds: List[str] = []
    for i, elem in enumerate(species_mapping_for_structure(atoms), start=1):
        mass = float(atomic_masses[atomic_numbers[elem]])
        cmds.append(f"mass {i} {mass:.8f}")
    return cmds

def locate_lammps_exe(explicit: Optional[str]) -> str:
    candidates = [
        explicit,
        os.environ.get("LAMMPS_EXE"),
        shutil.which("lmp"),
        shutil.which("lmp_mpi"),
        shutil.which("lmp_serial"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(Path(c).resolve())
        if c and shutil.which(c):
            return str(Path(shutil.which(c)).resolve())
    raise FileNotFoundError("Could not find a LAMMPS executable. Set --lammps-exe or LAMMPS_EXE.")


def write_lammps_data(path: Path, atoms: Atoms) -> None:
    write(str(path), sanitize_atoms(atoms), format="lammps-data", atom_style="atomic")


def run_lammps_case(
    lammps_exe: str,
    workdir: Path,
    structure: Atoms,
    potential: PotentialSpec,
    run_name: str,
    mode: str,
    timestep_fs: float,
    md_steps: int,
    thermo_every: int,
    seed: int,
    cpu_mpi_ranks: int = 20,
    cpu_omp_threads: int = 2,
    mace_omp_threads: int = 4,
    mace_kokkos_gpus: int = 1,
    min_etol: float = 1.0e-12,
    min_ftol: float = 1.0e-12,
    min_maxiter: int = 2000,
    min_maxeval: int = 10000,
) -> Dict[str, float]:
    workdir.mkdir(parents=True, exist_ok=True)
    data_file = workdir / f"{run_name}.data"
    input_file = workdir / f"{run_name}.in"
    log_file = workdir / f"{run_name}.log"
    dump_file = workdir / f"{run_name}.lammpstrj"

    atoms = sanitize_atoms(structure)
    write_lammps_data(data_file, atoms)

    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"read_data {data_file.name}",
        f"pair_style {potential.pair_style}",
        f"pair_coeff {pair_coeff_for_structure(potential, atoms)}",
        *mass_commands_for_structure(atoms),
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        f"timestep {timestep_fs/1000.0:.8f}",
        f"thermo {thermo_every}",
        "thermo_style custom step temp pe etotal press vol atoms",
    ]
    if mode == "static":
        lines += [
            "run 0",
        ]
    elif mode == "md":
        lines += [
            f"velocity all create 300.0 {seed} mom yes rot yes dist gaussian",
            "fix int all nvt temp 300.0 300.0 0.1",
            f"dump d1 all custom {max(1, md_steps)} {dump_file.name} id type x y z",
            f"run {md_steps}",
            "unfix int",
            "undump d1",
        ]
    elif mode == "minimize":
        lines += [
            "min_style cg",
            f"minimize {min_etol:.6e} {min_ftol:.6e} {int(min_maxiter)} {int(min_maxeval)}",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    input_file.write_text("\n".join(lines) + "\n")
    env = os.environ.copy()
    cmd: List[str]
    if potential.family == "MACE":
        env["OMP_NUM_THREADS"] = str(max(1, int(mace_omp_threads)))
        cmd = [
            lammps_exe,
            "-k", "on", "g", str(max(1, int(mace_kokkos_gpus))),
            "-sf", "kk",
            "-pk", "kokkos", "newton", "on", "neigh", "half",
        ]
    else:
        env["OMP_NUM_THREADS"] = str(max(1, int(cpu_omp_threads)))
        if int(cpu_mpi_ranks) > 1:
            cmd = ["mpiexec", "-np", str(int(cpu_mpi_ranks)), lammps_exe]
        else:
            cmd = [lammps_exe]
    cmd += ["-in", input_file.name, "-log", log_file.name, "-echo", "screen"]
    cp = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed for {run_name}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    thermo = parse_thermo_table(log_file)

    def pop_first(keys, default=math.nan):
        for key in keys:
            if key in thermo:
                return thermo.pop(key)
        return default

    thermo["volume_A3"] = pop_first(["Volume", "Vol", "volume", "vol"])
    thermo["energy_eV"] = pop_first(["PotEng", "Pe", "pe", "poteng"])
    thermo["etotal_eV"] = pop_first(["TotEng", "Etotal", "etotal", "toteng"])
    thermo["temp_K"] = pop_first(["Temp", "temp"])
    thermo["pressure_bar"] = pop_first(["Press", "press"])
    thermo["atoms"] = int(pop_first(["Atoms", "atoms"], len(atoms)))
    thermo["energy_per_atom_eV"] = (
        thermo["energy_eV"] / thermo["atoms"] if thermo["atoms"] else math.nan
    )
    return thermo


def append_rows(csv_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_smoke(
    args: argparse.Namespace,
    lammps_exe: str,
    potentials: Sequence[PotentialSpec],
    results_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    b2 = replicated(make_b2_cuzr(), args.smoke_rep)
    for pot in potentials:
        print(f"SMOKE TEST: {pot.id}")
        for block, mode in [("smoke_static", "static"), ("smoke_md", "md")]:
            case_dir = results_dir / "tmp" / pot.id / block
            try:
                thermo = run_lammps_case(
                    lammps_exe=lammps_exe,
                    workdir=case_dir,
                    structure=b2,
                    potential=pot,
                    run_name=f"{pot.id}_{block}",
                    mode=mode,
                    timestep_fs=args.timestep_fs,
                    md_steps=args.smoke_md_steps,
                    thermo_every=max(1, min(args.smoke_md_steps, args.smoke_thermo)),
                    seed=args.seed,
                )
                rows.append(
                    {
                        "pot_id": pot.id,
                        "block": block,
                        "error": "",
                        "structure": "B2_CuZr",
                        "energy_eV": thermo["energy_eV"],
                        "energy_per_atom_eV": thermo["energy_per_atom_eV"],
                        "volume_A3": thermo["volume_A3"],
                        "temp_K": thermo["temp_K"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "pot_id": pot.id,
                        "block": block,
                        "error": str(e),
                        "structure": "B2_CuZr",
                        "energy_eV": "",
                        "energy_per_atom_eV": "",
                        "volume_A3": "",
                        "temp_K": "",
                    }
                )
    append_rows(results_dir / "smoke_validation.csv", rows)


def scale_structure(atoms: Atoms, scale: float) -> Atoms:
    scaled = sanitize_atoms(atoms)
    scaled.set_cell(np.array(scaled.cell) * scale, scale_atoms=True)
    return scaled




def make_vacancy_structure(atoms: Atoms, atom_index: int = 0) -> Atoms:
    s = sanitize_atoms(atoms)
    if atom_index < 0 or atom_index >= len(s):
        raise IndexError(f"Vacancy atom_index {atom_index} out of range for {len(s)} atoms")
    del s[atom_index]
    return s

def parse_composition_id(comp: str) -> Tuple[int, int]:
    m = re.fullmatch(r"Cu(\d+)Zr(\d+)", comp.strip())
    if not m:
        raise ValueError(f"Invalid composition id: {comp}")
    cu = int(m.group(1))
    zr = int(m.group(2))
    if cu + zr != 100:
        raise ValueError(f"Composition must sum to 100: {comp}")
    return cu, zr


def build_glass_seed(composition_id: str, reps: Tuple[int, int, int], seed: int) -> Atoms:
    cu_pct, zr_pct = parse_composition_id(composition_id)
    base = replicated(make_b2_cuzr(), reps)
    n = len(base)
    n_cu = int(round(n * cu_pct / 100.0))
    n_cu = max(1, min(n - 1, n_cu))
    symbols = np.array(["Cu"] * n_cu + ["Zr"] * (n - n_cu), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)
    base = sanitize_atoms(base)
    base.set_chemical_symbols(symbols.tolist())
    return base


def parse_last_lammpstrj(path: Path, type_to_symbol: Dict[int, str]) -> Atoms:
    text = path.read_text().strip().splitlines()
    i = 0
    last = None
    while i < len(text):
        if text[i].startswith('ITEM: TIMESTEP'):
            timestep = int(text[i+1].strip())
            assert text[i+2].startswith('ITEM: NUMBER OF ATOMS')
            natoms = int(text[i+3].strip())
            assert text[i+4].startswith('ITEM: BOX BOUNDS')
            xlo, xhi = map(float, text[i+5].split()[:2])
            ylo, yhi = map(float, text[i+6].split()[:2])
            zlo, zhi = map(float, text[i+7].split()[:2])
            header = text[i+8].split()[2:]
            rows = [text[i+9+j].split() for j in range(natoms)]
            last = (timestep, natoms, (xlo, xhi, ylo, yhi, zlo, zhi), header, rows)
            i = i + 9 + natoms
        else:
            i += 1
    if last is None:
        raise RuntimeError(f'Could not parse dump file: {path}')
    _, natoms, bounds, header, rows = last
    col = {name: idx for idx, name in enumerate(header)}
    req = ['id', 'type', 'x', 'y', 'z']
    for r in req:
        if r not in col:
            raise RuntimeError(f'Dump file missing column {r}: {path}')
    rows = sorted(rows, key=lambda r: int(r[col['id']]))
    symbols = [type_to_symbol[int(r[col['type']])] for r in rows]
    positions = np.array([[float(r[col['x']]), float(r[col['y']]), float(r[col['z']])] for r in rows], dtype=float)
    xlo, xhi, ylo, yhi, zlo, zhi = bounds
    cell = np.diag([xhi - xlo, yhi - ylo, zhi - zlo])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms


def total_rdf_summary(atoms: Atoms, rmax: float, bins: int) -> Dict[str, float]:
    d = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(len(atoms), k=1)
    dist = d[iu]
    dist = dist[dist < rmax]
    edges = np.linspace(0.0, rmax, bins + 1)
    counts, _ = np.histogram(dist, bins=edges)
    dr = edges[1] - edges[0]
    r = 0.5 * (edges[:-1] + edges[1:])
    rho = len(atoms) / atoms.get_volume()
    shell_vol = 4.0 * np.pi * r * r * dr
    expected = 0.5 * len(atoms) * rho * shell_vol
    g = np.divide(counts, expected, out=np.zeros_like(r), where=expected > 0)
    valid = r > 1.5 * dr
    if not np.any(valid):
        return {"rdf_peak_r_A": math.nan, "rdf_peak_g": math.nan}
    rv = r[valid]
    gv = g[valid]
    imax = int(np.argmax(gv))
    return {"rdf_peak_r_A": float(rv[imax]), "rdf_peak_g": float(gv[imax])}


def density_g_cm3(atoms: Atoms) -> float:
    mass_amu = sum(float(atomic_masses[atomic_numbers[s]]) for s in atoms.get_chemical_symbols())
    return mass_amu * 1.66053906660 / atoms.get_volume()


def run_glass_case(
    args: argparse.Namespace,
    lammps_exe: str,
    workdir: Path,
    structure: Atoms,
    potential: PotentialSpec,
    run_name: str,
) -> Tuple[Dict[str, float], Path, Dict[int, str]]:
    workdir.mkdir(parents=True, exist_ok=True)
    data_file = workdir / f"{run_name}.data"
    input_file = workdir / f"{run_name}.in"
    log_file = workdir / f"{run_name}.log"
    dump_file = workdir / f"{run_name}_final.lammpstrj"

    atoms = sanitize_atoms(structure)
    write_lammps_data(data_file, atoms)
    species = species_mapping_for_structure(atoms)
    type_to_symbol = {i + 1: s for i, s in enumerate(species)}

    lines = [
        'units metal',
        'atom_style atomic',
        'boundary p p p',
        f'read_data {data_file.name}',
        f'pair_style {potential.pair_style}',
        f'pair_coeff {pair_coeff_for_structure(potential, atoms)}',
        *mass_commands_for_structure(atoms),
        'neighbor 2.0 bin',
        'neigh_modify every 1 delay 0 check yes',
        f'timestep {args.timestep_fs/1000.0:.8f}',
        f'thermo {max(1, int(args.glass_thermo))}',
        'thermo_style custom step temp pe etotal press vol atoms',
        f'velocity all create {float(args.glass_temp_high):.6f} {int(args.seed)} mom yes rot yes dist gaussian',
        f'fix fmelt all nvt temp {float(args.glass_temp_high):.6f} {float(args.glass_temp_high):.6f} {float(args.glass_tdamp_ps):.6f}',
        f'run {int(args.glass_melt_steps)}',
        'unfix fmelt',
        f'fix fquench all nvt temp {float(args.glass_temp_high):.6f} {float(args.glass_temp_low):.6f} {float(args.glass_tdamp_ps):.6f}',
        f'run {int(args.glass_quench_steps)}',
        'unfix fquench',
        'min_style cg',
        f'minimize 1.0e-12 1.0e-12 {int(args.glass_min_maxiter)} {int(args.glass_min_maxeval)}',
        f'dump d1 all custom 1 {dump_file.name} id type x y z',
        'dump_modify d1 sort id',
        'run 0',
        'undump d1',
    ]
    input_file.write_text("\n".join(lines) + "\n")

    env = os.environ.copy()
    if potential.family == 'MACE':
        env['OMP_NUM_THREADS'] = str(max(1, int(args.mace_omp_threads)))
        cmd = [
            lammps_exe,
            '-k', 'on', 'g', str(max(1, int(args.mace_kokkos_gpus))),
            '-sf', 'kk',
            '-pk', 'kokkos', 'newton', 'on', 'neigh', 'half',
        ]
    else:
        env['OMP_NUM_THREADS'] = str(max(1, int(args.cpu_omp_threads)))
        if int(args.cpu_mpi_ranks) > 1:
            cmd = ['mpiexec', '-np', str(int(args.cpu_mpi_ranks)), lammps_exe]
        else:
            cmd = [lammps_exe]
    cmd += ['-in', input_file.name, '-log', log_file.name, '-echo', 'screen']
    cp = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed for {run_name}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    thermo = parse_thermo_table(log_file)
    def pop_first(keys, default=math.nan):
        for key in keys:
            if key in thermo:
                return thermo.pop(key)
        return default
    thermo['volume_A3'] = pop_first(['Volume','Vol','volume','vol'])
    thermo['energy_eV'] = pop_first(['PotEng','Pe','pe','poteng'])
    thermo['etotal_eV'] = pop_first(['TotEng','Etotal','etotal','toteng'])
    thermo['temp_K'] = pop_first(['Temp','temp'])
    thermo['pressure_bar'] = pop_first(['Press','press'])
    thermo['atoms'] = int(pop_first(['Atoms','atoms'], len(atoms)))
    thermo['energy_per_atom_eV'] = thermo['energy_eV'] / thermo['atoms'] if thermo['atoms'] else math.nan
    return thermo, dump_file, type_to_symbol


def run_glass(
    args: argparse.Namespace,
    lammps_exe: str,
    potentials: Sequence[PotentialSpec],
    results_dir: Path,
) -> None:
    glass_rows: List[Dict[str, object]] = []
    rdf_rows: List[Dict[str, object]] = []
    comps = [c.strip() for c in str(args.glass_compositions).split(',') if c.strip()]
    for pot in potentials:
        for i, comp in enumerate(comps):
            print(f"GLASS: {pot.id} / {comp}")
            case_dir = results_dir / 'tmp' / pot.id / 'glass' / comp
            try:
                atoms0 = build_glass_seed(comp, args.glass_rep, args.seed + i)
                thermo, dump_file, type_to_symbol = run_glass_case(args, lammps_exe, case_dir, atoms0, pot, f"{pot.id}_{comp}_glass")
                final_atoms = parse_last_lammpstrj(dump_file, type_to_symbol)
                cu_pct, zr_pct = parse_composition_id(comp)
                glass_rows.append({
                    'pot_id': pot.id,
                    'composition_id': comp,
                    'x_cu': cu_pct/100.0,
                    'x_zr': zr_pct/100.0,
                    'n_atoms': len(final_atoms),
                    'energy_eV': thermo['energy_eV'],
                    'energy_per_atom_eV': thermo['energy_per_atom_eV'],
                    'volume_A3': thermo['volume_A3'],
                    'density_g_cm3': density_g_cm3(final_atoms),
                    'temp_K': thermo['temp_K'],
                    'error': '',
                })
                if not args.skip_rdf_sq:
                    rdf = total_rdf_summary(final_atoms, float(args.rdf_rmax_A), int(args.rdf_bins))
                    rdf_rows.append({
                        'pot_id': pot.id,
                        'composition_id': comp,
                        'rdf_peak_r_A': rdf['rdf_peak_r_A'],
                        'rdf_peak_g': rdf['rdf_peak_g'],
                        'error': '',
                    })
            except Exception as e:
                glass_rows.append({
                    'pot_id': pot.id,
                    'composition_id': comp,
                    'x_cu': '',
                    'x_zr': '',
                    'n_atoms': '',
                    'energy_eV': '',
                    'energy_per_atom_eV': '',
                    'volume_A3': '',
                    'density_g_cm3': '',
                    'temp_K': '',
                    'error': str(e),
                })
                if not args.skip_rdf_sq:
                    rdf_rows.append({
                        'pot_id': pot.id,
                        'composition_id': comp,
                        'rdf_peak_r_A': '',
                        'rdf_peak_g': '',
                        'error': str(e),
                    })
    append_rows(results_dir / 'glass_validation.csv', glass_rows)
    if rdf_rows:
        append_rows(results_dir / 'rdf_summary.csv', rdf_rows)


def run_eos(
    args: argparse.Namespace,
    lammps_exe: str,
    potentials: Sequence[PotentialSpec],
    results_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    structures = crystal_structures()
    scales = np.linspace(args.eos_min_scale, args.eos_max_scale, args.eos_points)
    for pot in potentials:
        for sname, atoms in structures.items():
            print(f"EOS: {pot.id} / {sname}")
            for scale in scales:
                case_dir = results_dir / "tmp" / pot.id / "eos" / sname / f"{scale:.6f}"
                try:
                    thermo = run_lammps_case(
                        lammps_exe=lammps_exe,
                        workdir=case_dir,
                        structure=scale_structure(atoms, float(scale)),
                        potential=pot,
                        run_name=f"{pot.id}_{sname}_{scale:.6f}",
                        mode="static",
                        timestep_fs=args.timestep_fs,
                        md_steps=0,
                        thermo_every=1,
                        seed=args.seed,
                    )
                    rows.append(
                        {
                            "pot_id": pot.id,
                            "stage": "eos",
                            "structure": sname,
                            "scale": scale,
                            "energy_eV": thermo["energy_eV"],
                            "energy_per_atom_eV": thermo["energy_per_atom_eV"],
                            "volume_A3": thermo["volume_A3"],
                            "error": "",
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "pot_id": pot.id,
                            "stage": "eos",
                            "structure": sname,
                            "scale": scale,
                            "energy_eV": "",
                            "energy_per_atom_eV": "",
                            "volume_A3": "",
                            "error": str(e),
                        }
                    )
    append_rows(results_dir / "eos_validation.csv", rows)




def run_vacancy(
    args: argparse.Namespace,
    lammps_exe: str,
    potentials: Sequence[PotentialSpec],
    results_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    vacancy_targets = [
        ("FCC_Cu", crystal_structures()["FCC_Cu"], (4, 4, 4)),
        ("HCP_Zr", crystal_structures()["HCP_Zr"], (4, 4, 3)),
    ]
    for pot in potentials:
        print(f"VACANCY: {pot.id}")
        for sname, base_atoms, rep in vacancy_targets:
            try:
                perfect = replicated(base_atoms, rep)
                perfect_thermo = run_lammps_case(
                    lammps_exe=lammps_exe,
                    workdir=results_dir / "tmp" / pot.id / "vacancy" / sname / "perfect",
                    structure=perfect,
                    potential=pot,
                    run_name=f"{pot.id}_{sname}_vac_perfect",
                    mode="minimize",
                    timestep_fs=args.timestep_fs,
                    md_steps=0,
                    thermo_every=1,
                    seed=args.seed,
                )
                defect = make_vacancy_structure(perfect, atom_index=args.vacancy_atom_index)
                defect_thermo = run_lammps_case(
                    lammps_exe=lammps_exe,
                    workdir=results_dir / "tmp" / pot.id / "vacancy" / sname / "defect",
                    structure=defect,
                    potential=pot,
                    run_name=f"{pot.id}_{sname}_vac_defect",
                    mode="minimize",
                    timestep_fs=args.timestep_fs,
                    md_steps=0,
                    thermo_every=1,
                    seed=args.seed,
                )
                n_perfect = len(perfect)
                e_perfect = float(perfect_thermo["energy_eV"])
                e_defect = float(defect_thermo["energy_eV"])
                e_form = e_defect - ((n_perfect - 1) / n_perfect) * e_perfect
                rows.append({
                    "pot_id": pot.id,
                    "structure": sname,
                    "repeat": "x".join(map(str, rep)),
                    "n_perfect": n_perfect,
                    "e_perfect_eV": e_perfect,
                    "e_defect_eV": e_defect,
                    "e_vac_form_eV": e_form,
                    "error": "",
                })
            except Exception as e:
                rows.append({
                    "pot_id": pot.id,
                    "structure": sname,
                    "repeat": "x".join(map(str, rep)),
                    "n_perfect": "",
                    "e_perfect_eV": "",
                    "e_defect_eV": "",
                    "e_vac_form_eV": "",
                    "error": str(e),
                })
    append_rows(results_dir / "vacancy_formation.csv", rows)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with csv_path.open("w", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(x: object) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def dedupe_rows(rows: Sequence[Dict[str, str]], key_fields: Sequence[str]) -> List[Dict[str, str]]:
    latest: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        latest[key] = row
    return list(latest.values())


def estimate_bulk_modulus_from_eos_rows(
    eos_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    eos_rows = dedupe_rows(eos_rows, ["pot_id", "structure", "scale"])
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in eos_rows:
        if str(row.get("error", "")).strip():
            continue
        pot_id = str(row.get("pot_id", "")).strip()
        structure = str(row.get("structure", "")).strip()
        if not pot_id or not structure:
            continue
        grouped.setdefault((pot_id, structure), []).append(row)

    out_rows: List[Dict[str, object]] = []
    for (pot_id, structure), rows in sorted(grouped.items()):
        pts = []
        for row in rows:
            v = maybe_float(row.get("volume_A3", ""))
            e = maybe_float(row.get("energy_eV", ""))
            if np.isfinite(v) and np.isfinite(e):
                pts.append((v, e))
        if len(pts) < 3:
            out_rows.append({
                "pot_id": pot_id,
                "structure": structure,
                "fit_points": len(pts),
                "v0_A3": "",
                "e0_eV": "",
                "bulk_modulus_GPa": "",
                "error": "Not enough EOS points for quadratic fit",
            })
            continue

        pts.sort(key=lambda x: x[0])
        V = np.array([p[0] for p in pts], dtype=float)
        E = np.array([p[1] for p in pts], dtype=float)

        imin = int(np.argmin(E))
        lo = max(0, imin - 2)
        hi = min(len(V), imin + 3)
        if hi - lo < 3:
            lo = max(0, len(V) - 3)
            hi = len(V)
        Vfit = V[lo:hi]
        Efit = E[lo:hi]

        try:
            a, b, c = np.polyfit(Vfit, Efit, 2)
            if abs(a) < 1e-14:
                raise ValueError("Quadratic coefficient too small")
            v0 = -b / (2.0 * a)
            e0 = a * v0 * v0 + b * v0 + c
            bulk_gpa = v0 * (2.0 * a) * 160.21766208
            out_rows.append({
                "pot_id": pot_id,
                "structure": structure,
                "fit_points": len(Vfit),
                "v0_A3": float(v0),
                "e0_eV": float(e0),
                "bulk_modulus_GPa": float(bulk_gpa),
                "error": "",
            })
        except Exception as e:
            out_rows.append({
                "pot_id": pot_id,
                "structure": structure,
                "fit_points": len(Vfit),
                "v0_A3": "",
                "e0_eV": "",
                "bulk_modulus_GPa": "",
                "error": str(e),
            })

    return out_rows


def generate_summary_outputs(results_dir: Path) -> None:
    smoke_rows = dedupe_rows(read_csv_rows(results_dir / "smoke_validation.csv"), ["pot_id", "block", "structure"])
    eos_rows = dedupe_rows(read_csv_rows(results_dir / "eos_validation.csv"), ["pot_id", "structure", "scale"])
    vac_rows = dedupe_rows(read_csv_rows(results_dir / "vacancy_formation.csv"), ["pot_id", "structure", "repeat"])
    glass_rows = dedupe_rows(read_csv_rows(results_dir / "glass_validation.csv"), ["pot_id", "composition_id"])
    rdf_rows = dedupe_rows(read_csv_rows(results_dir / "rdf_summary.csv"), ["pot_id", "composition_id"])

    bulk_rows = estimate_bulk_modulus_from_eos_rows(eos_rows) if eos_rows else []
    if bulk_rows:
        write_rows(results_dir / "bulk_modulus_estimates.csv", bulk_rows)

    smoke_map: Dict[Tuple[str, str], Dict[str, str]] = {
        (str(r.get("pot_id", "")), str(r.get("block", ""))): r
        for r in smoke_rows
        if str(r.get("structure", "")) == "B2_CuZr"
    }
    bulk_map: Dict[Tuple[str, str], Dict[str, object]] = {
        (str(r.get("pot_id", "")), str(r.get("structure", ""))): r for r in bulk_rows
    }
    vac_map: Dict[Tuple[str, str], Dict[str, str]] = {
        (str(r.get("pot_id", "")), str(r.get("structure", ""))): r for r in vac_rows
    }
    glass_map: Dict[Tuple[str, str], Dict[str, str]] = {
        (str(r.get("pot_id", "")), str(r.get("composition_id", ""))): r for r in glass_rows
    }
    rdf_map: Dict[Tuple[str, str], Dict[str, str]] = {
        (str(r.get("pot_id", "")), str(r.get("composition_id", ""))): r for r in rdf_rows
    }

    pot_ids = sorted({
        *(str(r.get("pot_id", "")) for r in smoke_rows),
        *(str(r.get("pot_id", "")) for r in eos_rows),
        *(str(r.get("pot_id", "")) for r in vac_rows),
        *(str(r.get("pot_id", "")) for r in glass_rows),
        *(str(r.get("pot_id", "")) for r in rdf_rows),
    })

    summary_rows: List[Dict[str, object]] = []
    for pot_id in pot_ids:
        row: Dict[str, object] = {"pot_id": pot_id}

        s_static = smoke_map.get((pot_id, "smoke_static"), {})
        s_md = smoke_map.get((pot_id, "smoke_md"), {})
        row["smoke_static_B2_e_per_atom_eV"] = s_static.get("energy_per_atom_eV", "")
        row["smoke_static_B2_error"] = s_static.get("error", "")
        row["smoke_md_B2_e_per_atom_eV"] = s_md.get("energy_per_atom_eV", "")
        row["smoke_md_B2_temp_K"] = s_md.get("temp_K", "")
        row["smoke_md_B2_error"] = s_md.get("error", "")

        for structure, short in [("FCC_Cu", "fcc_cu"), ("HCP_Zr", "hcp_zr"), ("B2_CuZr", "b2_cuzr")]:
            b = bulk_map.get((pot_id, structure), {})
            row[f"{short}_v0_A3"] = b.get("v0_A3", "")
            row[f"{short}_e0_eV"] = b.get("e0_eV", "")
            row[f"{short}_bulk_modulus_GPa"] = b.get("bulk_modulus_GPa", "")
            row[f"{short}_bulk_error"] = b.get("error", "")

        for structure, short in [("FCC_Cu", "fcc_cu"), ("HCP_Zr", "hcp_zr")]:
            v = vac_map.get((pot_id, structure), {})
            row[f"{short}_vac_eV"] = v.get("e_vac_form_eV", "")
            row[f"{short}_vac_error"] = v.get("error", "")

        for comp in ["Cu64Zr36", "Cu50Zr50", "Cu36Zr64"]:
            g = glass_map.get((pot_id, comp), {})
            r = rdf_map.get((pot_id, comp), {})
            tag = comp.lower()
            row[f"glass_{tag}_e_per_atom_eV"] = g.get("energy_per_atom_eV", "")
            row[f"glass_{tag}_density_g_cm3"] = g.get("density_g_cm3", "")
            row[f"glass_{tag}_error"] = g.get("error", "")
            row[f"rdf_{tag}_peak_r_A"] = r.get("rdf_peak_r_A", "")
            row[f"rdf_{tag}_peak_g"] = r.get("rdf_peak_g", "")
            row[f"rdf_{tag}_error"] = r.get("error", "")

        summary_rows.append(row)

    if summary_rows:
        write_rows(results_dir / "paper1_validation_summary.csv", summary_rows)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct LAMMPS validation without pyiron")
    p.add_argument("--mode", choices=["dev", "prod"], default="prod")
    p.add_argument("--results-dir", default="outputs/paper1_validation_direct")
    p.add_argument("--lammps-exe", default=os.environ.get("LAMMPS_EXE", ""))
    p.add_argument("--cpu-mpi-ranks", type=int, default=20, help="MPI ranks for CPU-driven EAM/ACE jobs")
    p.add_argument("--cpu-omp-threads", type=int, default=2, help="OpenMP threads per CPU MPI rank for EAM/ACE jobs")
    p.add_argument("--mace-omp-threads", type=int, default=4, help="OpenMP threads for MACE ML-IAP runs")
    p.add_argument("--mace-kokkos-gpus", type=int, default=1, help="Number of GPUs for MACE Kokkos launch")
    p.add_argument("--pots", default="all")
    p.add_argument("--skip-mace", action="store_true")
    p.add_argument("--skip-ace", action="store_true")
    p.add_argument("--skip-eam", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--skip-eos", action="store_true")
    p.add_argument("--skip-vacancy", action="store_true")
    p.add_argument("--skip-glass", action="store_true")
    p.add_argument("--skip-rdf-sq", action="store_true")
    p.add_argument("--skip-mddms-precheck", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--smoke-md-steps", type=int, default=200)
    p.add_argument("--smoke-thermo", type=int, default=20)
    p.add_argument("--smoke-rep", default="4,4,4")
    p.add_argument("--eos-points", type=int, default=7)
    p.add_argument("--vacancy-atom-index", type=int, default=0)
    p.add_argument("--eos-min-scale", type=float, default=0.96)
    p.add_argument("--eos-max-scale", type=float, default=1.04)
    p.add_argument("--glass-rep", default="5,5,5")
    p.add_argument("--glass-compositions", default="Cu64Zr36,Cu50Zr50,Cu36Zr64")
    p.add_argument("--glass-temp-high", type=float, default=2000.0)
    p.add_argument("--glass-temp-low", type=float, default=300.0)
    p.add_argument("--glass-melt-steps", type=int, default=5000)
    p.add_argument("--glass-quench-steps", type=int, default=5000)
    p.add_argument("--glass-thermo", type=int, default=200)
    p.add_argument("--glass-tdamp-ps", type=float, default=0.1)
    p.add_argument("--glass-min-maxiter", type=int, default=4000)
    p.add_argument("--glass-min-maxeval", type=int, default=20000)
    p.add_argument("--rdf-rmax-A", type=float, default=8.0)
    p.add_argument("--rdf-bins", type=int, default=200)
    p.add_argument("--mace_a_file", default=DEFAULT_MACE_FILES["MACE_A"])
    p.add_argument("--mace_b_file", default=DEFAULT_MACE_FILES["MACE_B"])
    p.add_argument("--mace_c_file", default=DEFAULT_MACE_FILES["MACE_C"])
    p.add_argument("--mace_d_file", default=DEFAULT_MACE_FILES["MACE_D"])
    p.add_argument("--ace_514_file", default=DEFAULT_ACE_FILES["ACE_514"])
    p.add_argument("--ace_1352_file", default=DEFAULT_ACE_FILES["ACE_1352"])
    p.add_argument("--eam-2019-file", default=DEFAULT_EAM_FILES["EAM_Mendelev_2019_CuZr"])
    p.add_argument("--eam-2007-file", default=DEFAULT_EAM_FILES["2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1"])
    args = p.parse_args()
    args.smoke_rep = tuple(int(x) for x in str(args.smoke_rep).split(","))
    if len(args.smoke_rep) != 3:
        raise ValueError("--smoke-rep must have exactly three integers")
    args.glass_rep = tuple(int(x) for x in str(args.glass_rep).split(","))
    if len(args.glass_rep) != 3:
        raise ValueError("--glass-rep must have exactly three integers")
    return args


def main() -> int:
    args = parse_args()
    lammps_exe = locate_lammps_exe(args.lammps_exe)
    all_pots = build_potentials(args)
    pots = select_potentials(all_pots, args.pots)

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "run_settings.json", vars(args))
    append_rows(results_dir / "potentials_selected.csv", [
        {"pot_id": p.id, "family": p.family, "pair_style": p.pair_style, "model_file": p.model_file}
        for p in pots
    ])

    if not args.skip_smoke:
        run_smoke(args, lammps_exe, pots, results_dir)
    if not args.skip_eos:
        run_eos(args, lammps_exe, pots, results_dir)

    if not args.skip_vacancy:
        run_vacancy(args, lammps_exe, pots, results_dir)
    if not args.skip_glass:
        run_glass(args, lammps_exe, pots, results_dir)

    generate_summary_outputs(results_dir)

    print("\nDone.")
    print(f"results dir: {results_dir}")
    print("Selected potentials:")
    print(", ".join(p.id for p in pots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
