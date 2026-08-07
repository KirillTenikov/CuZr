#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REVISION_ROOT = HERE.parents[1]
sys.path.insert(0, str(REVISION_ROOT / "src"))

from paper1_revision.config import load_protocol
from paper1_revision.thermo import write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LAMMPS thermo logs for one revision run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=REVISION_ROOT / "config" / "protocol.json")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    summary = write_summary(args.run_dir.resolve(), protocol.tail_fraction)
    print(f"Summarized {len(summary.get('stages', {}))} stages in {args.run_dir}")


if __name__ == "__main__":
    main()
