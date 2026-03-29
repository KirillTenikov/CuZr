\
    from __future__ import annotations

    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from typing import Any, Dict, List

    from .ace_pacemaker_input import build_pacemaker_input
    from .common import ensure_dir, save_json, save_yaml


    def _copy_if_exists(src: Path, dst: Path) -> None:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


    def _copy_tree_if_exists(src: Path, dst: Path) -> None:
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


    def build_ace_commands(cfg: Dict[str, Any], run_dir: Path) -> List[List[str]]:
        backend_cfg = cfg["backend"]
        backend = backend_cfg["name"]
        if backend != "pacemaker":
            raise ValueError(
                f"Unsupported ACE backend: {backend}. Expected 'pacemaker' for the production path."
            )

        pacemaker_exe = backend_cfg.get("pacemaker_executable", "pacemaker")
        input_yaml = run_dir / "pacemaker_input.yaml"
        commands: List[List[str]] = []

        if backend_cfg.get("prepare_data", True):
            commands.append([pacemaker_exe, "--prepare-data", str(input_yaml)])

        if backend_cfg.get("no_fit", False):
            commands.append([pacemaker_exe, str(input_yaml), "--no-fit"])
        else:
            commands.append([pacemaker_exe, str(input_yaml)])

        return commands


    def collect_pacemaker_outputs(run_dir: Path) -> None:
        artifacts = ensure_dir(run_dir / "artifacts")

        _copy_if_exists(run_dir / "log.txt", artifacts / "log.txt")
        _copy_if_exists(run_dir / "output_potential.yaml", artifacts / "output_potential.yaml")
        _copy_if_exists(
            run_dir / "interim_potential_best_cycle.yaml",
            artifacts / "interim_potential_best_cycle.yaml",
        )
        _copy_tree_if_exists(run_dir / "report", artifacts / "report")

        # Convenience aliases for downstream usage.
        if (run_dir / "interim_potential_best_cycle.yaml").exists():
            _copy_if_exists(
                run_dir / "interim_potential_best_cycle.yaml",
                artifacts / "ace_best.yaml",
            )
        elif (run_dir / "output_potential.yaml").exists():
            _copy_if_exists(run_dir / "output_potential.yaml", artifacts / "ace_best.yaml")

        # Optional prepared-data caches if pacemaker generates them.
        for name in ["fitting_data_info.pckl.gzip", "test_data_info.pckl.gzip"]:
            _copy_if_exists(run_dir / name, artifacts / name)


    def run_ace_training(cfg: Dict[str, Any], run_dir: Path) -> int:
        ensure_dir(run_dir / "logs")
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "artifacts")

        input_cfg = build_pacemaker_input(cfg, run_dir)
        input_yaml = run_dir / "pacemaker_input.yaml"
        save_yaml(input_cfg, input_yaml)

        commands = build_ace_commands(cfg, run_dir)
        save_json(
            {
                "backend": "pacemaker",
                "commands": commands,
                "input_yaml": str(input_yaml),
            },
            run_dir / "artifacts" / "ace_backend_commands.json",
        )

        env = os.environ.copy()
        train_parent = str(Path(cfg["data"]["train_path"]).resolve().parent)
        env.setdefault("PACEMAKERDATAPATH", train_parent)

        for cmd in commands:
            print("Launching ACE command:", flush=True)
            print(" ".join(cmd), flush=True)
            completed = subprocess.run(
                cmd,
                check=False,
                cwd=str(run_dir),
                env=env,
            )
            if completed.returncode != 0:
                return completed.returncode

        collect_pacemaker_outputs(run_dir)
        return 0
