#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    copy_config_to_run,
    load_yaml,
    make_run_dirs,
    runtime_metadata,
    save_json,
    set_seed,
    timestamp,
)
from src.data import normalize_config_dataset_paths, resolve_dataset_paths, sanity_check_paths
from src.path_utils import resolve_path
from src.mace_runner import run_mace_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MACE model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional override for run name")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for cfg.run.output_root (useful in Docker mount layouts)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config, must_exist=True)
    cfg = load_yaml(config_path)
    config_dir = config_path.parent
    cfg = normalize_config_dataset_paths(cfg, config_dir=config_dir)

    if args.output_root:
        cfg["run"]["output_root"] = str(resolve_path(args.output_root))
    else:
        cfg["run"]["output_root"] = str(resolve_path(cfg["run"]["output_root"], base_dir=config_dir))

    base_name = args.run_name or cfg["run"]["name"]
    run_name = f"{base_name}_{timestamp()}"
    paths = make_run_dirs(cfg["run"]["output_root"], run_name)

    copy_config_to_run(config_path, paths.run_dir)
    set_seed(cfg["training"].get("seed", 42))

    dataset_paths = resolve_dataset_paths(cfg, config_dir=config_dir)
    sanity_check_paths(dataset_paths)

    meta = {
        "trainer": "mace",
        "run_name": run_name,
        "config_path": str(config_path),
        "resolved_datasets": {k: str(v) if v is not None else None for k, v in dataset_paths.items()},
        "runtime": runtime_metadata(),
    }
    save_json(meta, paths.run_dir / "run_meta.json")

    return_code = run_mace_training(cfg, paths.run_dir)
    if return_code != 0:
        print(f"MACE training failed with exit code {return_code}", file=sys.stderr, flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
