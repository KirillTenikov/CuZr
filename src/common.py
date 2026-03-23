from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class RunPaths:
    run_dir: Path
    log_dir: Path
    ckpt_dir: Path
    artifacts_dir: Path


def make_run_dirs(base_output: str | Path, run_name: str) -> RunPaths:
    run_dir = ensure_dir(Path(base_output) / run_name)
    log_dir = ensure_dir(run_dir / "logs")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    return RunPaths(
        run_dir=run_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        artifacts_dir=artifacts_dir,
    )


def copy_config_to_run(config_path: str | Path, run_dir: str | Path) -> None:
    shutil.copy2(config_path, Path(run_dir) / "config_used.yaml")
