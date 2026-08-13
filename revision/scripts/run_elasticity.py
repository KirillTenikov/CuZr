#!/usr/bin/env python3
"""
Generate and optionally execute zero-temperature relaxed-ion elasticity
calculations from an already relaxed amorphous structure.

Each strained calculation starts independently from the same source structure.
The strained cell is held fixed while atomic coordinates are minimized.

Primary strain conventions
--------------------------
bulk : volumetric strain, epsilon_v = Delta V / V0
xy   : engineering shear gamma_xy = Delta(xy tilt) / Ly
xz   : engineering shear gamma_xz = Delta(xz tilt) / Lz
yz   : engineering shear gamma_yz = Delta(yz tilt) / Lz

LAMMPS pressure is recorded with `compute pressure NULL virial`, so the kinetic
term is excluded.  The analysis script accounts for the LAMMPS pressure/stress
sign convention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import subprocess
from pathlib import Path


DEFAULT_STRAINS = (-0.005, -0.0025, 0.0025, 0.005)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        repo = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def strain_tag(value: float) -> str:
    if abs(value) < 5e-15:
        return "zero"
    sign = "p" if value > 0 else "m"
    body = f"{abs(value):.6f}".replace(".", "p")
    return f"{sign}{body}"


def deformation_block(mode: str, strain: float) -> str:
    if mode == "reference":
        return "# Unstrained reference."

    if mode == "bulk":
        if 1.0 + strain <= 0.0:
            raise ValueError(f"Invalid volumetric strain {strain}")
        scale = (1.0 + strain) ** (1.0 / 3.0)
        return (
            "# strain = DeltaV/V0; identical linear scale in x,y,z\n"
            f"change_box all "
            f"x scale {scale:.16g} "
            f"y scale {scale:.16g} "
            f"z scale {scale:.16g} "
            f"remap units box"
        )

    if mode == "xy":
        return (
            "change_box all triclinic\n"
            f"variable dtilt equal ({strain:.16g})*ly\n"
            'change_box all xy delta ${dtilt} remap units box'
        )

    if mode == "xz":
        return (
            "change_box all triclinic\n"
            f"variable dtilt equal ({strain:.16g})*lz\n"
            'change_box all xz delta ${dtilt} remap units box'
        )

    if mode == "yz":
        return (
            "change_box all triclinic\n"
            f"variable dtilt equal ({strain:.16g})*lz\n"
            'change_box all yz delta ${dtilt} remap units box'
        )

    raise ValueError(f"Unknown mode: {mode}")


def make_input(
    data_file: Path,
    pair_style: str,
    pair_coeff: str,
    mode: str,
    strain: float,
    etol: float,
    ftol: float,
    maxiter: int,
    maxeval: int,
) -> str:
    deform = deformation_block(mode, strain)
    return f"""\
clear
units metal
dimension 3
boundary p p p
atom_style atomic

read_data "{data_file}"

pair_style {pair_style}
pair_coeff {pair_coeff}

# The source is an inherent structure.  Remove any stored velocities anyway so
# ordinary thermo output cannot accidentally contain a kinetic pressure term.
velocity all set 0.0 0.0 0.0

# Virial-only pressure tensor; order is xx yy zz xy xz yz.
compute pvir all pressure NULL virial

thermo 25
thermo_style custom step pe vol lx ly lz xy xz yz c_pvir c_pvir[1] c_pvir[2] c_pvir[3] c_pvir[4] c_pvir[5] c_pvir[6]
thermo_modify flush yes

# Apply the deformation once.  No fix box/relax is used afterwards: the
# strained cell must remain fixed while internal coordinates relax.
{deform}

min_style cg
minimize {etol:.16g} {ftol:.16g} {maxiter} {maxeval}

# Refresh thermo computes at the final minimized coordinates.
run 0

variable E   equal pe
variable V   equal vol
variable LX  equal lx
variable LY  equal ly
variable LZ  equal lz
variable XY  equal xy
variable XZ  equal xz
variable YZ  equal yz
variable PXX equal c_pvir[1]
variable PYY equal c_pvir[2]
variable PZZ equal c_pvir[3]
variable PXY equal c_pvir[4]
variable PXZ equal c_pvir[5]
variable PYZ equal c_pvir[6]

print "ELASTIC_RESULT mode={mode} strain={strain:.16g} pe=${{E}} vol=${{V}} lx=${{LX}} ly=${{LY}} lz=${{LZ}} xy=${{XY}} xz=${{XZ}} yz=${{YZ}} pxx=${{PXX}} pyy=${{PYY}} pzz=${{PZZ}} pxy=${{PXY}} pxz=${{PXZ}} pyz=${{PYZ}}"

write_data relaxed.data
"""


def parse_result_marker(text: str) -> dict:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("ELASTIC_RESULT ")
    ]
    if not lines:
        raise RuntimeError("No ELASTIC_RESULT marker found in LAMMPS output.")

    tokens = lines[-1].split()[1:]
    raw = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        raw[key] = value

    required = {
        "mode",
        "strain",
        "pe",
        "vol",
        "lx",
        "ly",
        "lz",
        "xy",
        "xz",
        "yz",
        "pxx",
        "pyy",
        "pzz",
        "pxy",
        "pxz",
        "pyz",
    }
    missing = required - raw.keys()
    if missing:
        raise RuntimeError(
            f"Incomplete ELASTIC_RESULT marker; missing: {sorted(missing)}"
        )

    return {
        "mode": raw["mode"],
        "strain": float(raw["strain"]),
        "pe_eV": float(raw["pe"]),
        "volume_A3": float(raw["vol"]),
        "box_A": {
            "lx": float(raw["lx"]),
            "ly": float(raw["ly"]),
            "lz": float(raw["lz"]),
            "xy": float(raw["xy"]),
            "xz": float(raw["xz"]),
            "yz": float(raw["yz"]),
        },
        "pressure_bar": {
            "xx": float(raw["pxx"]),
            "yy": float(raw["pyy"]),
            "zz": float(raw["pzz"]),
            "xy": float(raw["pxy"]),
            "xz": float(raw["pxz"]),
            "yz": float(raw["pyz"]),
        },
    }


def run_job(
    job_dir: Path,
    input_text: str,
    lmp_cmd: str,
    execute: bool,
    force: bool,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)

    inp = job_dir / "in.elastic"
    stdout_file = job_dir / "stdout.txt"
    result_file = job_dir / "result.json"

    if result_file.exists() and not force:
        print(f"[SKIP] {job_dir} already complete", flush=True)
        return

    existing = [p for p in job_dir.iterdir()]
    if existing and not force and not result_file.exists():
        raise RuntimeError(
            f"Incomplete/non-empty directory exists: {job_dir}\n"
            "Inspect it first; use --force only if you intentionally want to overwrite."
        )

    inp.write_text(input_text)

    if not execute:
        print(f"[PREPARED] {job_dir}", flush=True)
        return

    cmd = shlex.split(lmp_cmd) + ["-in", inp.name]
    print(f"[RUN] {job_dir}", flush=True)

    # Stream LAMMPS output live to the terminal/tmux while preserving an exact
    # per-job copy in stdout.txt.  LAMMPS input uses `thermo_modify flush yes`,
    # so thermo progress is visible promptly during long minimizations.
    with stdout_file.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=job_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed in {job_dir}; inspect {stdout_file}"
        )

    result = parse_result_marker(stdout_file.read_text())
    result["lammps_command"] = cmd
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[OK]  {job_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate/execute zero-temperature relaxed-ion elasticity jobs."
    )
    p.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Source relaxed LAMMPS data file.",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output root for one potential/seed.",
    )
    p.add_argument("--label", required=True, help="Human-readable potential label.")
    p.add_argument(
        "--pair-style",
        required=True,
        help='LAMMPS pair_style arguments, e.g. "mliap unified /path/model.pt 0".',
    )
    p.add_argument(
        "--pair-coeff",
        default="* * Cu Zr",
        help='LAMMPS pair_coeff arguments (default: "* * Cu Zr").',
    )
    p.add_argument(
        "--lmp-cmd",
        default="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half",
        help="LAMMPS executable plus accelerator arguments.",
    )
    p.add_argument(
        "--strains",
        nargs="+",
        type=float,
        default=list(DEFAULT_STRAINS),
        help="Non-zero strain amplitudes; use symmetric +/- values.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=("bulk", "xy", "xz", "yz"),
        default=("bulk", "xy", "xz", "yz"),
    )
    p.add_argument("--etol", type=float, default=1.0e-12)
    p.add_argument("--ftol", type=float, default=1.0e-8)
    p.add_argument("--maxiter", type=int, default=10000)
    p.add_argument("--maxeval", type=int, default=100000)
    p.add_argument("--execute", action="store_true", help="Actually run LAMMPS.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite incomplete job directories.",
    )
    args = p.parse_args()

    data_file = args.data.expanduser().resolve()
    if not data_file.is_file():
        raise SystemExit(f"Input data file not found: {data_file}")

    if any(abs(x) < 1e-15 for x in args.strains):
        raise SystemExit(
            "--strains should contain only non-zero values; reference strain is added automatically."
        )

    positives = sorted(x for x in args.strains if x > 0)
    negatives = sorted(-x for x in args.strains if x < 0)
    if positives != negatives:
        raise SystemExit(
            "Use symmetric +/- strain amplitudes, e.g. -0.005 -0.0025 0.0025 0.005."
        )

    root = args.out.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "label": args.label,
        "source_data": str(data_file),
        "source_data_sha256": sha256_file(data_file),
        "pair_style": args.pair_style,
        "pair_coeff": args.pair_coeff,
        "lammps_command": args.lmp_cmd,
        "strains": args.strains,
        "modes": list(args.modes),
        "strain_definition": {
            "bulk": "volumetric strain epsilon_v = DeltaV/V0",
            "xy": "engineering shear gamma_xy = Delta(xy tilt)/Ly",
            "xz": "engineering shear gamma_xz = Delta(xz tilt)/Lz",
            "yz": "engineering shear gamma_yz = Delta(yz tilt)/Lz",
        },
        "relaxation": {
            "temperature": "0 K inherent-structure calculation",
            "cell": "fixed after imposed deformation",
            "internal_coordinates": "conjugate-gradient minimization",
            "min_style": "cg",
            "etol": args.etol,
            "ftol": args.ftol,
            "maxiter": args.maxiter,
            "maxeval": args.maxeval,
        },
        "stress": "LAMMPS compute pressure NULL virial; kinetic term excluded",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    jobs = [("reference", 0.0)]
    for mode in args.modes:
        for strain in args.strains:
            jobs.append((mode, strain))

    for mode, strain in jobs:
        job_dir = root / mode / strain_tag(strain)
        input_text = make_input(
            data_file=data_file,
            pair_style=args.pair_style,
            pair_coeff=args.pair_coeff,
            mode=mode,
            strain=strain,
            etol=args.etol,
            ftol=args.ftol,
            maxiter=args.maxiter,
            maxeval=args.maxeval,
        )
        run_job(
            job_dir=job_dir,
            input_text=input_text,
            lmp_cmd=args.lmp_cmd,
            execute=args.execute,
            force=args.force,
        )


if __name__ == "__main__":
    main()
