#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train_ace.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import copy_config_to_run, load_yaml, make_run_dirs, save_json, set_seed, timestamp
from src.data import resolve_dataset_paths, sanity_check_paths
from src.ace_runner import run_ace_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ACE model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional override for run name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    base_name = args.run_name or cfg["run"]["name"]
    run_name = f"{base_name}_{timestamp()}"
    paths = make_run_dirs(cfg["run"]["output_root"], run_name)

    copy_config_to_run(args.config, paths.run_dir)
    set_seed(cfg["training"].get("seed", 42))

    dataset_paths = resolve_dataset_paths(cfg)
    sanity_check_paths(dataset_paths)

    meta = {
        "trainer": "ace",
        "run_name": run_name,
        "config_path": args.config,
    }
    save_json(meta, paths.run_dir / "run_meta.json")

    return_code = run_ace_training(cfg, paths.run_dir)
    if return_code != 0:
        print(f"ACE training failed with exit code {return_code}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
