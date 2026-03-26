from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .common import ensure_dir


def build_ace_command(cfg: Dict[str, Any], run_dir: Path) -> List[str]:
    """
    Replace this wrapper with the real ACE backend command once finalized.
    Right now it assumes a Python entrypoint supplied in the YAML config.
    """
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    backend_cfg = cfg["backend"]

    backend = backend_cfg["name"]
    python_exe = train_cfg.get("python_executable") or sys.executable

    if backend == "placeholder_python":
        cmd = [
            python_exe,
            backend_cfg["entrypoint"],
            "--train-path",
            data_cfg["train_path"],
            "--valid-path",
            data_cfg["valid_path"],
            "--output-dir",
            str(run_dir),
            "--cutoff",
            str(model_cfg["cutoff"]),
            "--basis-size",
            str(model_cfg["basis_size"]),
            "--max-epochs",
            str(train_cfg["max_epochs"]),
            "--batch-size",
            str(train_cfg["batch_size"]),
            "--lr",
            str(train_cfg["lr"]),
            "--seed",
            str(train_cfg.get("seed", 42)),
        ]
        extra_args = backend_cfg.get("extra_args", [])
        if extra_args:
            cmd.extend([str(x) for x in extra_args])
        return cmd

    raise ValueError(f"Unsupported ACE backend: {backend}")


def run_ace_training(cfg: Dict[str, Any], run_dir: Path) -> int:
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "artifacts")

    cmd = build_ace_command(cfg, run_dir)
    print("Launching ACE command:", flush=True)
    print(" ".join(cmd), flush=True)

    env = os.environ.copy()
    completed = subprocess.run(cmd, check=False, cwd=str(Path.cwd()), env=env)
    return completed.returncode
