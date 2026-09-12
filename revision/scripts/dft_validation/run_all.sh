#!/usr/bin/env bash
set -euo pipefail

# Paper 1 detailed DFT validation.
#
# Required environment variables:
#   TEST_EXTXYZ
#   MACE_A_MODEL
#   MACE_B_MODEL
#   MACE_C_MODEL
#   MACE_D_MODEL
#
# Optional:
#   OUTDIR              default: revision/results/dft_validation
#   MACE_A_LOG ...      training logs for convergence extraction
#   ACE514_METRICS ...  pacemaker test_metrics.txt for validation history
#
# This script does not download data or models.

OUTDIR="${OUTDIR:-revision/results/dft_validation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTDIR}/inventory" "${OUTDIR}/per_structure" "${OUTDIR}/summary" "${OUTDIR}/figures" "${OUTDIR}/convergence"

python "${SCRIPT_DIR}/01_inventory_testset.py" \
  --test "${TEST_EXTXYZ}" \
  --outdir "${OUTDIR}/inventory"

for model in MACE_A MACE_B MACE_C MACE_D; do
  model_var="${model}_MODEL"
  model_path="${!model_var}"
  python "${SCRIPT_DIR}/02_mace_target_inference.py" \
    --test "${TEST_EXTXYZ}" \
    --model "${model_path}" \
    --model-name "${model}" \
    --out "${OUTDIR}/per_structure/${model}_target_dft.csv"
done

python "${SCRIPT_DIR}/03_composition_statistics.py" \
  --inputs \
    "${OUTDIR}/per_structure/MACE_A_target_dft.csv" \
    "${OUTDIR}/per_structure/MACE_B_target_dft.csv" \
    "${OUTDIR}/per_structure/MACE_C_target_dft.csv" \
    "${OUTDIR}/per_structure/MACE_D_target_dft.csv" \
  --outdir "${OUTDIR}/summary"

convergence_args=()
for model in MACE_A MACE_B MACE_C MACE_D; do
  var="${model}_LOG"
  if [[ -n "${!var:-}" ]]; then
    convergence_args+=(--mace-log "${model}=${!var}")
  fi
done
if [[ -n "${ACE514_METRICS:-}" ]]; then
  convergence_args+=(--ace-metrics "ACE514=${ACE514_METRICS}")
fi
if [[ -n "${ACE1352_METRICS:-}" ]]; then
  convergence_args+=(--ace-metrics "ACE1352=${ACE1352_METRICS}")
fi

if (( ${#convergence_args[@]} > 0 )); then
  python "${SCRIPT_DIR}/04_training_convergence.py" \
    "${convergence_args[@]}" \
    --outdir "${OUTDIR}/convergence"
fi

figure_args=(
  --summary "${OUTDIR}/summary/mace_target_composition_summary.csv"
  --outdir "${OUTDIR}/figures"
)

if [[ -f "${OUTDIR}/convergence/mace_validation_convergence.csv" ]]; then
  figure_args+=(--mace-convergence "${OUTDIR}/convergence/mace_validation_convergence.csv")
fi
if [[ -f "${OUTDIR}/convergence/ace_validation_convergence.csv" ]]; then
  figure_args+=(--ace-convergence "${OUTDIR}/convergence/ace_validation_convergence.csv")
fi

python "${SCRIPT_DIR}/05_make_figures.py" "${figure_args[@]}"

echo "DFT validation workflow complete: ${OUTDIR}"
