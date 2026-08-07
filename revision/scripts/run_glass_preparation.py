#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve()
REVISION_ROOT = HERE.parents[1]
sys.path.insert(0, str(REVISION_ROOT / "src"))

from paper1_revision.config import load_potentials, load_protocol
from paper1_revision.lammps import RunSpec, execute, generate_inputs, run_spec_dict, write_shell_script
from paper1_revision.manifest import git_info, sha256, write_json
from paper1_revision.potentials import resolve_model_path
from paper1_revision.structure import write_initial_data
from paper1_revision.thermo import write_summary


def default_run_dir(results_root: Path, composition: str, natoms: int, seed: int, potential_id: str) -> Path:
    return results_root / composition / f"N{natoms}" / f"seed_{seed}" / potential_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally execute the additive Paper 1 glass-preparation workflow.")
    parser.add_argument("--repo-root", type=Path, default=REVISION_ROOT.parent)
    parser.add_argument("--protocol", type=Path, default=REVISION_ROOT / "config" / "protocol.json")
    parser.add_argument("--potentials", type=Path, default=REVISION_ROOT / "config" / "potentials.json")
    parser.add_argument("--potential", default="MACE_C")
    parser.add_argument("--composition", default="Cu64Zr36")
    parser.add_argument("--natoms", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--initial-density", type=float, default=7.20)
    parser.add_argument("--results-root", type=Path, default=REVISION_ROOT / "results")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--initial-data", type=Path, help="Use a pre-generated shared initial.data instead of generating one.")
    parser.add_argument("--lmp-command", help="Override the command stored for the selected potential.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--no-box-relax", action="store_true")
    parser.add_argument("--with-nve", action="store_true", help="Also generate and run the optional NVE stability stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    protocol = load_protocol(args.protocol.resolve())
    potentials = load_potentials(args.potentials.resolve())
    if args.potential not in potentials:
        raise SystemExit(f"Unknown potential {args.potential!r}. Available: {', '.join(potentials)}")
    potential = potentials[args.potential]
    if not potential.enabled:
        raise SystemExit(f"Potential {potential.id} is disabled in {args.potentials}; set its path and enabled=true first.")

    run_dir = (args.run_dir or default_run_dir(args.results_root, args.composition, args.natoms, args.seed, potential.id)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Run directory is not empty: {run_dir}\n"
            "For safety, this workflow never overwrites an existing run. Move it, archive it, or choose --run-dir."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_path = run_dir / "initial.data"
    if args.initial_data:
        source = args.initial_data.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, initial_path)
        structure_meta = {"source_initial_data": str(source), "source_sha256": sha256(source)}
    else:
        structure_meta = write_initial_data(initial_path, args.natoms, args.composition, args.initial_density, args.seed)

    spec = RunSpec(
        composition=args.composition,
        natoms=args.natoms,
        seed=args.seed,
        initial_density_g_cm3=args.initial_density,
        potential_id=potential.id,
        run_dir=run_dir,
    )
    input_files = generate_inputs(
        spec,
        protocol,
        potential,
        repo_root,
        include_box_relax=not args.no_box_relax,
        include_nve=args.with_nve,
    )
    lmp_command = args.lmp_command or potential.lmp_command
    shell_script = write_shell_script(run_dir, input_files, lmp_command)
    model_path = resolve_model_path(repo_root, potential.path)

    manifest = {
        "workflow_version": "0.2.0",
        "run": run_spec_dict(spec),
        "protocol": asdict(protocol),
        "derived": {
            "quench_ps": protocol.quench_ps,
            "melt_steps": protocol.steps(protocol.melt_ps),
            "quench_steps": protocol.steps(protocol.quench_ps),
            "relax_npt_steps": protocol.steps(protocol.relax_npt_ps),
            "equilibrate_nvt_steps": protocol.steps(protocol.equilibrate_nvt_ps),
        },
        "potential": asdict(potential),
        "potential_resolved_path": str(model_path),
        "potential_sha256": sha256(model_path),
        "initial_structure": structure_meta,
        "initial_data_sha256": sha256(initial_path),
        "input_files": input_files,
        "lmp_command": lmp_command,
        "git": git_info(repo_root),
    }
    write_json(run_dir / "manifest.json", manifest)

    print(json.dumps({"run_dir": str(run_dir), "input_files": input_files, "shell_script": str(shell_script)}, indent=2))
    if args.execute:
        execute(run_dir, input_files, lmp_command)
    if args.summarize or args.execute:
        write_summary(run_dir, protocol.tail_fraction)
        print(f"Wrote {run_dir / 'thermo_summary.json'}")


if __name__ == "__main__":
    main()
