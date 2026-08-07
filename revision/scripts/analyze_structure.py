#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REVISION_ROOT = HERE.parents[1]
sys.path.insert(0, str(REVISION_ROOT / "src"))

from paper1_revision.rdf import analyze_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate total/partial RDFs and first-shell coordination for a prepared glass.")
    parser.add_argument("structure", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rmax", type=float, default=8.0)
    parser.add_argument("--bins", type=int, default=400)
    args = parser.parse_args()
    structure = args.structure.resolve()
    output_dir = (args.output_dir or structure.parent / "structure_analysis").resolve()
    result = analyze_structure(structure, output_dir, args.rmax, args.bins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
