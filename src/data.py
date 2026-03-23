from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def resolve_dataset_paths(cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    data_cfg = cfg["data"]
    return {
        "train": Path(data_cfg["train_path"]),
        "valid": Path(data_cfg["valid_path"]),
        "test": Path(data_cfg["test_path"]) if data_cfg.get("test_path") else None,
    }


def sanity_check_paths(paths: Dict[str, Optional[Path]]) -> None:
    for name, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found: {path}")
