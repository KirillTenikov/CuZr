#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 1 ACE production queue for GPU/Kokkos execution.
#
# Why this wrapper exists:
#   * the generic glass-preparation generator emits `pair_style pace`;
#   * LAMMPS/Kokkos selects pace/kk with `-sf kk`;
#   * pace/kk on NVIDIA GPUs requires the ACE `product` evaluator.
#
# Therefore this queue generates the standard Paper 1 inputs, changes only
# `pair_style pace` -> `pair_style pace product`, and executes the unchanged
# preparation protocol with Kokkos on one GPU.

REPO_ROOT="${REPO_ROOT:-/workspace/CuZr}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/revision/results}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"

NATOMS=1024
CHECKPOINT_STEPS=5000
POTENTIAL="${POTENTIAL:-ACE_514}"

DENSITY_CU64ZR36="7.20"
DENSITY_CU50ZR50="${DENSITY_CU50ZR50:-}"
DENSITY_CU36ZR64="${DENSITY_CU36ZR64:-}"

DRY_RUN=0

# This is the command that was physically smoke-tested on the A100.
LMP_GPU=(lmp -k on g 1 -sf kk -pk kokkos newton on neigh half)
LMP_COMMAND_STRING="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"

usage() {
  cat <<'USAGE'
Usage:
  ./run_ace_queue.sh \
    --potential ACE_514|ACE_1352 \
    --density-cu50 VALUE \
    --density-cu36 VALUE \
    [--dry-run]

Examples:
  ./run_ace_queue.sh --potential ACE_514  --density-cu50 7.20 --density-cu36 6.90 --dry-run
  ./run_ace_queue.sh --potential ACE_514  --density-cu50 7.20 --density-cu36 6.90
  ./run_ace_queue.sh --potential ACE_1352 --density-cu50 7.20 --density-cu36 6.90

The Cu64Zr36 starting density is fixed at 7.20 g/cm^3.
The default checkpoint interval is 5000 steps, matching the MACE campaign.

The queue is conservative:
  * completed runs are skipped;
  * non-empty incomplete run directories block the queue;
  * no existing run directory is overwritten.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --potential)
      POTENTIAL="${2:?missing value}"
      shift 2
      ;;
    --density-cu50)
      DENSITY_CU50ZR50="${2:?missing value}"
      shift 2
      ;;
    --density-cu36)
      DENSITY_CU36ZR64="${2:?missing value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${POTENTIAL}" in
  ACE_514|ACE_1352) ;;
  *)
    echo "ERROR: --potential must be ACE_514 or ACE_1352, got: ${POTENTIAL}" >&2
    exit 2
    ;;
esac

[[ -n "${DENSITY_CU50ZR50}" ]] || {
  echo "ERROR: set Cu50Zr50 density with --density-cu50." >&2
  exit 2
}
[[ -n "${DENSITY_CU36ZR64}" ]] || {
  echo "ERROR: set Cu36Zr64 density with --density-cu36." >&2
  exit 2
}

mkdir -p "${LOG_ROOT}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
POTENTIAL_TAG="$(printf '%s' "${POTENTIAL}" | tr '[:upper:]' '[:lower:]')"
QUEUE_LOG="${LOG_ROOT}/paper1_${POTENTIAL_TAG}_queue_${STAMP}.log"
STATUS_FILE="${LOG_ROOT}/paper1_${POTENTIAL_TAG}_queue_${STAMP}.tsv"

exec > >(tee -a "${QUEUE_LOG}") 2>&1
trap 'rc=$?; echo "=== QUEUE FAILED rc=${rc}: $(date -Is) ==="; exit "${rc}"' ERR INT TERM

cd "${REPO_ROOT}"
source /workspace/cuzr_runtime.env

command -v python >/dev/null
command -v lmp >/dev/null
command -v nvidia-smi >/dev/null

case "${POTENTIAL}" in
  ACE_514)
    MODEL_FILE="${ACE_514_FILE:?ACE_514_FILE is not set; source /workspace/cuzr_runtime.env}"
    ;;
  ACE_1352)
    MODEL_FILE="${ACE_1352_FILE:?ACE_1352_FILE is not set; source /workspace/cuzr_runtime.env}"
    ;;
esac

[[ -s "${MODEL_FILE}" ]] || {
  echo "ERROR: ACE model file does not exist or is empty: ${MODEL_FILE}" >&2
  exit 3
}

printf "timestamp\tcomposition\tseed\tstatus\trun_dir\n" > "${STATUS_FILE}"

echo "============================================================"
echo "Paper 1 ACE GPU production queue"
echo "Started:             $(date -Is)"
echo "Git commit:          $(git rev-parse --short HEAD)"
echo "Potential:           ${POTENTIAL}"
echo "Model file:          ${MODEL_FILE}"
echo "Atoms:               ${NATOMS}"
echo "Seeds:               42 43 44"
echo "Checkpoint interval: ${CHECKPOINT_STEPS} steps"
echo "Cu64Zr36 density:    ${DENSITY_CU64ZR36}"
echo "Cu50Zr50 density:    ${DENSITY_CU50ZR50}"
echo "Cu36Zr64 density:    ${DENSITY_CU36ZR64}"
echo "ACE evaluator:       product"
echo "LAMMPS command:      ${LMP_COMMAND_STRING}"
echo "Queue log:           ${QUEUE_LOG}"
echo "============================================================"

nvidia-smi \
  --query-gpu=name,uuid,memory.total,temperature.gpu,power.limit,compute_mode \
  --format=csv,noheader || true

run_dir_for() {
  printf '%s/%s/N%s/seed_%s/%s' \
    "${RESULTS_ROOT}" "$1" "${NATOMS}" "$2" "${POTENTIAL}"
}

run_is_complete() {
  local d="$1"
  [[ -s "${d}/thermo_summary.json" ]] &&
  [[ -s "${d}/02_after_equilibrate_nvt.data" ]] &&
  [[ -s "${d}/04_inherent_box_relaxed.data" ]]
}

run_dir_has_content() {
  local d="$1"
  [[ -d "${d}" ]] && find "${d}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

record_status() {
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$1" "$2" "$3" "$4" >> "${STATUS_FILE}"
}

patch_ace_inputs_for_gpu() {
  local run_dir="$1"
  local input
  local count=0

  shopt -s nullglob
  for input in "${run_dir}"/[0-9][0-9]_*.in; do
    if grep -qx 'pair_style pace' "${input}"; then
      sed -i 's/^pair_style pace$/pair_style pace product/' "${input}"
      count=$((count + 1))
    fi
  done
  shopt -u nullglob

  if (( count == 0 )); then
    echo "ERROR: no generated ACE inputs containing 'pair_style pace' were found." >&2
    return 30
  fi

  if grep -Hn '^pair_style pace$' "${run_dir}"/[0-9][0-9]_*.in 2>/dev/null; then
    echo "ERROR: an ACE input still contains unpatched 'pair_style pace'." >&2
    return 31
  fi

  echo "[PATCH] ${count} ACE input file(s): pair_style pace -> pair_style pace product"

  # Separate provenance record; do not alter the generator's manifest schema.
  python - "${run_dir}" "${POTENTIAL}" "${LMP_COMMAND_STRING}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
potential = sys.argv[2]
lmp_command = sys.argv[3]
payload = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "potential": potential,
    "reason": "PACE Kokkos GPU execution requires the product evaluator.",
    "generated_pair_style": "pace",
    "executed_pair_style": "pace product (selected as pace/kk by -sf kk)",
    "lmp_command": lmp_command,
}
(run_dir / "ace_gpu_execution.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

execute_stages() {
  local run_dir="$1"
  local input
  local log
  local stage

  local stages=(
    00_prepare_melt_quench.in
    01_relax_npt.in
    02_equilibrate_nvt.in
    03_inherent_fixed_cell.in
    04_inherent_box_relaxed.in
  )

  (
    cd "${run_dir}"
    for input in "${stages[@]}"; do
      [[ -s "${input}" ]] || {
        echo "ERROR: missing generated input: ${run_dir}/${input}" >&2
        exit 32
      }
      stage="${input%.in}"
      log="${stage}.log"
      echo "[LAMMPS] ${stage}"
      printf '[CMD] '
      printf '%q ' "${LMP_GPU[@]}" -log "${log}" -in "${input}"
      printf '\n'
      "${LMP_GPU[@]}" -log "${log}" -in "${input}"
    done
  )
}

write_thermo_summary() {
  local run_dir="$1"

  PYTHONPATH="${REPO_ROOT}/revision/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python - "${run_dir}" "${REPO_ROOT}/revision/config/protocol.json" <<'PY'
import sys
from pathlib import Path

from paper1_revision.config import load_protocol
from paper1_revision.thermo import write_summary

run_dir = Path(sys.argv[1])
protocol_path = Path(sys.argv[2])
protocol = load_protocol(protocol_path)
write_summary(run_dir, protocol.tail_fraction)
print(f"[SUMMARY] Wrote {run_dir / 'thermo_summary.json'}")
print(f"[SUMMARY] Wrote {run_dir / 'thermo_summary.csv'}")
PY
}

run_one() {
  local composition="$1"
  local seed="$2"
  local density="$3"
  local run_dir
  run_dir="$(run_dir_for "${composition}" "${seed}")"

  echo
  echo "============================================================"
  echo "${POTENTIAL} / ${composition} / N${NATOMS} / seed ${seed}"
  echo "Initial density: ${density} g/cm^3"
  echo "Run directory:   ${run_dir}"
  echo "============================================================"

  if run_is_complete "${run_dir}"; then
    echo "[SKIP] already complete"
    record_status "${composition}" "${seed}" "SKIPPED_COMPLETE" "${run_dir}"
    return 0
  fi

  if run_dir_has_content "${run_dir}"; then
    echo "[BLOCKED] non-empty incomplete run directory: ${run_dir}" >&2
    echo "The queue will not overwrite it." >&2
    record_status "${composition}" "${seed}" "BLOCKED_INCOMPLETE" "${run_dir}"
    return 20
  fi

  local generate_cmd=(
    python revision/scripts/run_glass_preparation.py
    --repo-root .
    --potential "${POTENTIAL}"
    --composition "${composition}"
    --natoms "${NATOMS}"
    --seed "${seed}"
    --initial-density "${density}"
    --checkpoint-every-steps "${CHECKPOINT_STEPS}"
    --lmp-command "${LMP_COMMAND_STRING}"
  )

  printf '[GENERATE] '
  printf '%q ' "${generate_cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] would patch generated ACE inputs to 'pair_style pace product'"
    echo "[DRY-RUN] would execute stages 00-04 with: ${LMP_COMMAND_STRING}"
    record_status "${composition}" "${seed}" "DRY_RUN" "${run_dir}"
    return 0
  fi

  record_status "${composition}" "${seed}" "STARTED" "${run_dir}"

  "${generate_cmd[@]}"
  patch_ace_inputs_for_gpu "${run_dir}"
  execute_stages "${run_dir}"
  write_thermo_summary "${run_dir}"

  if ! run_is_complete "${run_dir}"; then
    echo "[ERROR] calculation finished but completion markers are missing." >&2
    record_status "${composition}" "${seed}" "FAILED_VERIFY" "${run_dir}"
    return 21
  fi

  echo "[OK] completed and verified"
  record_status "${composition}" "${seed}" "COMPLETED" "${run_dir}"
}

density_for() {
  case "$1" in
    Cu64Zr36) printf '%s' "${DENSITY_CU64ZR36}" ;;
    Cu50Zr50) printf '%s' "${DENSITY_CU50ZR50}" ;;
    Cu36Zr64) printf '%s' "${DENSITY_CU36ZR64}" ;;
    *) return 2 ;;
  esac
}

for composition in Cu64Zr36 Cu50Zr50 Cu36Zr64; do
  density="$(density_for "${composition}")"
  for seed in 42 43 44; do
    run_one "${composition}" "${seed}" "${density}"
  done
done

echo
echo "=== ${POTENTIAL} QUEUE COMPLETE: $(date -Is) ==="
echo "Status: ${STATUS_FILE}"
