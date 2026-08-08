from __future__ import annotations

import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Potential, Protocol
from .potentials import potential_block


@dataclass(frozen=True)
class RunSpec:
    composition: str
    natoms: int
    seed: int
    initial_density_g_cm3: float
    potential_id: str
    run_dir: Path


def _header(data_file: str, potential: Potential, repo_root: Path) -> str:
    return f"""units metal
atom_style atomic
boundary p p p
newton on
read_data {data_file}

{potential_block(potential, repo_root)}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
"""


def _thermo(protocol: Protocol) -> str:
    return f"""timestep {protocol.timestep_ps:.8f}
thermo {protocol.thermo_every_steps}
thermo_style custom step time temp pe ke etotal press pxx pyy pzz pxy pxz pyz vol density lx ly lz
thermo_modify flush yes
"""


def _checkpoint(stage: str, checkpoint_every_steps: int) -> str:
    if checkpoint_every_steps <= 0:
        return ""
    return f"restart {checkpoint_every_steps} checkpoints/{stage}.restart.*\n"


def generate_inputs(
    run_spec: RunSpec,
    protocol: Protocol,
    potential: Potential,
    repo_root: Path,
    include_box_relax: bool = True,
    include_nve: bool = True,
    checkpoint_every_steps: int = 0,
) -> list[str]:
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be >= 0")

    run_dir = run_spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_every_steps > 0:
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    generated: list[str] = []
    melt_steps = protocol.steps(protocol.melt_ps)
    quench_steps = protocol.steps(protocol.quench_ps)
    npt_steps = protocol.steps(protocol.relax_npt_ps)
    nvt_steps = protocol.steps(protocol.equilibrate_nvt_ps)

    stage00 = f"""{_header('initial.data', potential, repo_root)}{_thermo(protocol)}
# Gentle cleanup of the nonphysical starting lattice.
min_style cg
minimize {protocol.minimize_etol:.8e} {protocol.minimize_ftol:.8e} {protocol.minimize_maxiter} {protocol.minimize_maxeval}
velocity all create {protocol.temperature_high_K:.6f} {run_spec.seed + 101} mom yes rot yes dist gaussian
fix melt all nvt temp {protocol.temperature_high_K:.6f} {protocol.temperature_high_K:.6f} {protocol.tdamp_ps:.6f}
{_checkpoint('00', checkpoint_every_steps)}run {melt_steps}
unfix melt
fix quench all nvt temp {protocol.temperature_high_K:.6f} {protocol.temperature_low_K:.6f} {protocol.tdamp_ps:.6f}
run {quench_steps}
unfix quench
write_data 00_after_quench.data
write_restart 00_after_quench.restart
"""
    (run_dir / "00_prepare_melt_quench.in").write_text(stage00, encoding="utf-8")
    generated.append("00_prepare_melt_quench.in")

    stage01 = f"""{_header('00_after_quench.data', potential, repo_root)}{_thermo(protocol)}
velocity all create {protocol.temperature_low_K:.6f} {run_spec.seed + 202} mom yes rot yes dist gaussian
fix relax all npt temp {protocol.temperature_low_K:.6f} {protocol.temperature_low_K:.6f} {protocol.tdamp_ps:.6f} iso {protocol.pressure_bar:.6f} {protocol.pressure_bar:.6f} {protocol.pdamp_ps:.6f}
{_checkpoint('01', checkpoint_every_steps)}run {npt_steps}
unfix relax
write_data 01_after_relax_npt.data
write_restart 01_after_relax_npt.restart
"""
    (run_dir / "01_relax_npt.in").write_text(stage01, encoding="utf-8")
    generated.append("01_relax_npt.in")

    stage02 = f"""{_header('01_after_relax_npt.data', potential, repo_root)}{_thermo(protocol)}
velocity all create {protocol.temperature_low_K:.6f} {run_spec.seed + 303} mom yes rot yes dist gaussian
fix eq all nvt temp {protocol.temperature_low_K:.6f} {protocol.temperature_low_K:.6f} {protocol.tdamp_ps:.6f}
{_checkpoint('02', checkpoint_every_steps)}run {nvt_steps}
unfix eq
write_data 02_after_equilibrate_nvt.data
write_restart 02_after_equilibrate_nvt.restart
"""
    (run_dir / "02_equilibrate_nvt.in").write_text(stage02, encoding="utf-8")
    generated.append("02_equilibrate_nvt.in")

    stage03 = f"""{_header('02_after_equilibrate_nvt.data', potential, repo_root)}{_thermo(protocol)}
# Fixed-cell inherent structure: atoms are minimized while the NVT cell is retained.
min_style cg
minimize {protocol.minimize_etol:.8e} {protocol.minimize_ftol:.8e} {protocol.minimize_maxiter} {protocol.minimize_maxeval}
run 0
write_data 03_inherent_fixed_cell.data
"""
    (run_dir / "03_inherent_fixed_cell.in").write_text(stage03, encoding="utf-8")
    generated.append("03_inherent_fixed_cell.in")
    if include_box_relax:
        stage04 = f"""{_header('03_inherent_fixed_cell.data', potential, repo_root)}{_thermo(protocol)}
# Separately labelled zero-pressure cell-relaxed inherent structure.
fix relaxbox all box/relax iso {protocol.pressure_bar:.6f} vmax 0.001
min_style cg
minimize {protocol.minimize_etol:.8e} {protocol.minimize_ftol:.8e} {protocol.minimize_maxiter} {protocol.minimize_maxeval}
unfix relaxbox
run 0
write_data 04_inherent_box_relaxed.data
"""
        (run_dir / "04_inherent_box_relaxed.in").write_text(stage04, encoding="utf-8")
        generated.append("04_inherent_box_relaxed.in")
    if include_nve:
        pre_steps = protocol.steps(protocol.nve_preequilibrate_ps)
        nve_steps = protocol.steps(protocol.nve_stability_ps)
        stage05 = f"""{_header('02_after_equilibrate_nvt.data', potential, repo_root)}{_thermo(protocol)}
# Short production-like stability check. This is not a transport calculation.
velocity all create {protocol.temperature_low_K:.6f} {run_spec.seed + 505} mom yes rot yes dist gaussian
fix pre all nvt temp {protocol.temperature_low_K:.6f} {protocol.temperature_low_K:.6f} {protocol.tdamp_ps:.6f}
run {pre_steps}
unfix pre
reset_timestep 0
fix integ all nve
run {nve_steps}
unfix integ
write_data 05_after_nve_stability.data
"""
        (run_dir / "05_nve_stability.in").write_text(stage05, encoding="utf-8")
        generated.append("05_nve_stability.in")
    return generated


def write_shell_script(run_dir: Path, input_files: list[str], lmp_command: str) -> Path:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"LMP_CMD={shlex.quote(lmp_command)}",
        "",
        "run_stage() {",
        '  local input="$1"',
        '  local log="${input%.in}.log"',
        '  echo "[paper1-revision] $input -> $log"',
        '  ${LMP_CMD} -log "$log" -in "$input"',
        "}",
        "",
    ]
    lines.extend(f"run_stage {shlex.quote(name)}" for name in input_files)
    lines.append("")
    path = run_dir / "run_lammps.sh"
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def execute(run_dir: Path, input_files: list[str], lmp_command: str) -> None:
    for input_name in input_files:
        log_name = input_name.replace(".in", ".log")
        command = shlex.split(lmp_command) + ["-log", log_name, "-in", input_name]
        print("[execute]", " ".join(shlex.quote(x) for x in command), flush=True)
        subprocess.run(command, cwd=run_dir, check=True)


def run_spec_dict(spec: RunSpec) -> dict:
    payload = asdict(spec)
    payload["run_dir"] = str(spec.run_dir)
    return payload
