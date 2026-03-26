from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .path_utils import resolve_path


def _resolve_dataset_path(raw: str | Path | None, config_dir: Path) -> Optional[Path]:
    if not raw:
        return None

    raw_str = str(raw)
    direct = resolve_path(raw_str, base_dir=config_dir)
    if direct.exists():
        return direct

    data_root = os.environ.get("CUZR_DATA_ROOT")
    if data_root:
        alt = resolve_path(raw_str, base_dir=data_root)
        if alt.exists():
            return alt

    return direct


def resolve_dataset_paths(cfg: Dict[str, Any], config_dir: str | Path | None = None) -> Dict[str, Optional[Path]]:
    data_cfg = cfg["data"]
    base = Path(config_dir or Path.cwd()).resolve()
    return {
        "train": _resolve_dataset_path(data_cfg["train_path"], base),
        "valid": _resolve_dataset_path(data_cfg["valid_path"], base),
        "test": _resolve_dataset_path(data_cfg.get("test_path"), base),
    }


def normalize_config_dataset_paths(cfg: Dict[str, Any], config_dir: str | Path | None = None) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    resolved = resolve_dataset_paths(cfg, config_dir=config_dir)
    cfg["data"]["train_path"] = str(resolved["train"])
    cfg["data"]["valid_path"] = str(resolved["valid"])
    if resolved["test"] is not None:
        cfg["data"]["test_path"] = str(resolved["test"])
    return cfg


def sanity_check_paths(paths: Dict[str, Optional[Path]]) -> None:
    for name, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found: {path}")
