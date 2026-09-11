#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:-.}
OUTPUT=${2:-analysis_72}
WORKERS=${WORKERS:-2}
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
python "$SCRIPT_DIR/01_inventory_qc.py" --input "$INPUT" --output "$OUTPUT"
python "$SCRIPT_DIR/02_state_statistics.py" --output "$OUTPUT"
python "$SCRIPT_DIR/03_pair_structure.py" --input "$INPUT" --output "$OUTPUT"
python "$SCRIPT_DIR/04_voronoi.py" --input "$INPUT" --output "$OUTPUT" --workers "$WORKERS"
python "$SCRIPT_DIR/05_descriptor_statistics.py" --output "$OUTPUT"
python "$SCRIPT_DIR/06_make_figures.py" --output "$OUTPUT"
