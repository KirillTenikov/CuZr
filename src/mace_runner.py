from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .common import ensure_dir


def build_mace_command(cfg: Dict[str, Any], run_dir: Path) -> List[str]:
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    cmd = [
        "python",
        "-m",
        "mace.cli.run_train",
        "--name", cfg["run"]["name"],
        "--train_file", data_cfg["train_path"],
        "--valid_file", data_cfg["valid_path"],
        "--energy_key", data_cfg.get("energy_key", "energy"),
        "--forces_key", data_cfg.get("forces_key", "forces"),
        "--model", model_cfg.get("model_type", "MACE"),
        "--hidden_irreps", model_cfg["hidden_irreps"],
        "--r_max", str(model_cfg["r_max"]),
        "--num_interactions", str(model_cfg["num_interactions"]),
        "--max_num_epochs", str(train_cfg["max_epochs"]),
        "--batch_size", str(train_cfg["batch_size"]),
        "--valid_batch_size", str(train_cfg.get("valid_batch_size", train_cfg["batch_size"])),
        "--lr", str(train_cfg["lr"]),
        "--weight_decay", str(train_cfg.get("weight_decay", 0.0)),
        "--seed", str(train_cfg.get("seed", 42)),
        "--default_dtype", train_cfg.get("default_dtype", "float32"),
        "--device", train_cfg.get("device", "cuda"),
        "--checkpoints_dir", str(run_dir / "checkpoints"),
        "--results_dir", str(run_dir / "artifacts"),
        "--log_dir", str(run_dir / "logs"),
    ]

    if data_cfg.get("test_path"):
        cmd.extend(["--test_file", data_cfg["test_path"]])
    if data_cfg.get("stress_key"):
        cmd.extend(["--stress_key", data_cfg["stress_key"]])

    optional_args = {
        "--ema": train_cfg.get("ema"),
        "--ema_decay": train_cfg.get("ema_decay"),
        "--scheduler": train_cfg.get("scheduler"),
        "--energy_weight": train_cfg.get("energy_weight"),
        "--forces_weight": train_cfg.get("forces_weight"),
        "--stress_weight": train_cfg.get("stress_weight"),
        "--patience": train_cfg.get("patience"),
    }

    for key, value in optional_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(key)
        else:
            cmd.extend([key, str(value)])

    return cmd


def run_mace_training(cfg: Dict[str, Any], run_dir: Path) -> int:
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "artifacts")

    cmd = build_mace_command(cfg, run_dir)
    print("Launching MACE command:")
    print(" ".join(cmd))

    completed = subprocess.run(cmd, check=False)
    return completed.returncode
