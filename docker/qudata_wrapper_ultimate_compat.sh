#!/usr/bin/env bash
set -euo pipefail

# qudata_wrapper_ultimate_compat_conda.sh
#
# Same overall flow as the compatibility wrapper, but after startup it sources
# the Python/LAMMPS environment helper written by startup_md_conda_pyiron.sh.
#
# Important:
# - creates exact legacy-compatible MACE aliases expected by the current
#   Paper 1 validation script
# - uses the conda/micromamba Python created by startup_md_conda_pyiron.sh

REPO_URL="${REPO_URL:-https://github.com/KirillTenikov/CuZr.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/CuZr}"

LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/logs}"
MARKER_DIR="${MARKER_DIR:-${WORKSPACE_ROOT}/.markers}"

MODELS_ROOT="${MODELS_ROOT:-${WORKSPACE_ROOT}/models}"
RAW_MODELS_DIR="${RAW_MODELS_DIR:-${MODELS_ROOT}/raw}"
CONVERTED_MODELS_DIR="${CONVERTED_MODELS_DIR:-${MODELS_ROOT}/converted}"

POTENTIALS_ROOT="${POTENTIALS_ROOT:-${WORKSPACE_ROOT}/potentials}"
POTENTIALS_MACE_DIR="${POTENTIALS_MACE_DIR:-${POTENTIALS_ROOT}/mace}"
POTENTIALS_ACE_DIR="${POTENTIALS_ACE_DIR:-${POTENTIALS_ROOT}/ace}"
POTENTIALS_EAM_DIR="${POTENTIALS_EAM_DIR:-${POTENTIALS_ROOT}/eam}"

DATA_DIR="${DATA_DIR:-${WORKSPACE_ROOT}/data}"

SYNC_POTENTIALS_INTO_REPO="${SYNC_POTENTIALS_INTO_REPO:-1}"
REPO_RUNTIME_DIR="${REPO_RUNTIME_DIR:-${REPO_DIR}/runtime}"
REPO_POTENTIALS_DIR="${REPO_POTENTIALS_DIR:-${REPO_RUNTIME_DIR}/potentials}"

REPO_MACE_A_ALIAS="${REPO_MACE_A_ALIAS:-${REPO_DIR}/CuZr_MACE_A_compiled.model-lammps.pt}"
REPO_MACE_B_ALIAS="${REPO_MACE_B_ALIAS:-${REPO_DIR}/CuZr_MACE_B_compiled.model-lammps.pt}"
REPO_MACE_C_ALIAS="${REPO_MACE_C_ALIAS:-${REPO_DIR}/CuZr_MACE_C_compiled.model-lammps.pt}"
REPO_MACE_D_ALIAS="${REPO_MACE_D_ALIAS:-${REPO_DIR}/CuZr_MACE_D_compiled.model-lammps.pt}"

PYIRON_ROOT="${PYIRON_ROOT:-${WORKSPACE_ROOT}/pyiron}"
PYIRON_PROJECTS_DIR="${PYIRON_PROJECTS_DIR:-${PYIRON_ROOT}/projects}"
PYIRON_RESOURCES_DIR="${PYIRON_RESOURCES_DIR:-${PYIRON_ROOT}/resources}"
PYIRON_LAMMPS_DIR="${PYIRON_LAMMPS_DIR:-${PYIRON_RESOURCES_DIR}/lammps}"
PYIRON_LAMMPS_BIN_DIR="${PYIRON_LAMMPS_BIN_DIR:-${PYIRON_LAMMPS_DIR}/bin}"
PYIRON_LAMMPS_POT_DIR="${PYIRON_LAMMPS_POT_DIR:-${PYIRON_LAMMPS_DIR}/potentials}"
PYIRON_CONFIG_FILE="${PYIRON_CONFIG_FILE:-${HOME}/.pyiron}"
PYIRON_DB_FILE="${PYIRON_DB_FILE:-${PYIRON_ROOT}/pyiron.db}"
PYIRON_PROJECT_CHECK_ENABLED="${PYIRON_PROJECT_CHECK_ENABLED:-True}"

MODEL_BASE_URL="${MODEL_BASE_URL:-}"
DATA_BASE_URL="${DATA_BASE_URL:-}"

RUN_STARTUP_MD="${RUN_STARTUP_MD:-1}"
CONVERT_MACE_MODELS="${CONVERT_MACE_MODELS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"

STARTUP_SCRIPT_REL="${STARTUP_SCRIPT_REL:-docker/startup_md.sh}"
STARTUP_SCRIPT="${REPO_DIR}/${STARTUP_SCRIPT_REL}"
PYTHON_ENV_FILE="${PYTHON_ENV_FILE:-/workspace/cuzr_python.env}"

MACE_A_NAME="${MACE_A_NAME:-mace_A.model}"
MACE_B_NAME="${MACE_B_NAME:-mace_B.model}"
MACE_C_NAME="${MACE_C_NAME:-mace_C.model}"
MACE_D_NAME="${MACE_D_NAME:-mace_D.model}"
ACE_514_NAME="${ACE_514_NAME:-ace_514.yaml}"
ACE_1352_NAME="${ACE_1352_NAME:-ace_1352.yaml}"

CANONICAL_MACE_A_NAME="${CANONICAL_MACE_A_NAME:-mace_A-mliap_lammps.pt}"
CANONICAL_MACE_B_NAME="${CANONICAL_MACE_B_NAME:-mace_B-mliap_lammps.pt}"
CANONICAL_MACE_C_NAME="${CANONICAL_MACE_C_NAME:-mace_C-mliap_lammps.pt}"
CANONICAL_MACE_D_NAME="${CANONICAL_MACE_D_NAME:-mace_D-mliap_lammps.pt}"
CANONICAL_ACE_514_NAME="${CANONICAL_ACE_514_NAME:-ace_514.yaml}"
CANONICAL_ACE_1352_NAME="${CANONICAL_ACE_1352_NAME:-ace_1352.yaml}"

VALIDATION_MACE_A_NAME="${VALIDATION_MACE_A_NAME:-CuZr_MACE_A_compiled.model-lammps.pt}"
VALIDATION_MACE_B_NAME="${VALIDATION_MACE_B_NAME:-CuZr_MACE_B_compiled.model-lammps.pt}"
VALIDATION_MACE_C_NAME="${VALIDATION_MACE_C_NAME:-CuZr_MACE_C_compiled.model-lammps.pt}"
VALIDATION_MACE_D_NAME="${VALIDATION_MACE_D_NAME:-CuZr_MACE_D_compiled.model-lammps.pt}"

VALIDATION_ACE_514_NAME="${VALIDATION_ACE_514_NAME:-CuZr_ACE_514.yace}"
VALIDATION_ACE_1352_NAME="${VALIDATION_ACE_1352_NAME:-CuZr_ACE_1352.yace}"

TRAIN_SPLIT_NAME="${TRAIN_SPLIT_NAME:-train_split.extxyz}"
VALID_SPLIT_NAME="${VALID_SPLIT_NAME:-valid_split.extxyz}"
TEST_NAME="${TEST_NAME:-test.extxyz}"

MACE_A_URL="${MACE_A_URL:-}"
MACE_B_URL="${MACE_B_URL:-}"
MACE_C_URL="${MACE_C_URL:-}"
MACE_D_URL="${MACE_D_URL:-}"
ACE_514_URL="${ACE_514_URL:-}"
ACE_1352_URL="${ACE_1352_URL:-}"

TRAIN_SPLIT_URL="${TRAIN_SPLIT_URL:-}"
VALID_SPLIT_URL="${VALID_SPLIT_URL:-}"
TEST_URL="${TEST_URL:-}"

STARTUP_DONE_MARKER="${MARKER_DIR}/startup_md.done"
CONVERSION_DONE_MARKER="${MARKER_DIR}/mace_conversion.done"
POTENTIALS_DONE_MARKER="${MARKER_DIR}/potentials_ready.done"
PYIRON_DONE_MARKER="${MARKER_DIR}/pyiron_ready.done"
WRAPPER_DONE_MARKER="${MARKER_DIR}/wrapper.done"

POTENTIALS_ENV_FILE="${POTENTIALS_ROOT}/potential_paths.env"
POTENTIALS_MANIFEST_JSON="${POTENTIALS_ROOT}/potential_manifest.json"
PYIRON_ENV_FILE="${PYIRON_ROOT}/pyiron_paths.env"
READY_SUMMARY_TXT="${LOG_DIR}/validation_ready.txt"
PYIRON_NOTE_TXT="${PYIRON_LAMMPS_POT_DIR}/README_custom_potentials.txt"

mkdir -p \
  "${LOG_DIR}" \
  "${MARKER_DIR}" \
  "${RAW_MODELS_DIR}" \
  "${CONVERTED_MODELS_DIR}" \
  "${POTENTIALS_MACE_DIR}" \
  "${POTENTIALS_ACE_DIR}" \
  "${POTENTIALS_EAM_DIR}" \
  "${DATA_DIR}" \
  "${PYIRON_PROJECTS_DIR}" \
  "${PYIRON_LAMMPS_BIN_DIR}" \
  "${PYIRON_LAMMPS_POT_DIR}"

exec > >(tee -a "${LOG_DIR}/wrapper.log") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
fail() { log "ERROR: $*"; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -z "${url}" ]]; then
    fail "No download URL provided for ${out}"
  fi
  if [[ -f "${out}" && "${FORCE_REDOWNLOAD}" != "1" ]]; then
    log "Already exists, skipping: ${out}"
    return 0
  fi
  mkdir -p "$(dirname "${out}")"
  log "Downloading ${url} -> ${out}"
  curl -L --fail --retry 5 --retry-delay 5 --retry-connrefused -o "${out}" "${url}"
}

build_release_url() {
  local base_url="$1"
  local filename="$2"
  if [[ -z "${base_url}" ]]; then
    echo ""
  else
    echo "${base_url%/}/${filename}"
  fi
}

make_link() {
  local src="$1"
  local dst="$2"
  [[ -e "${src}" ]] || fail "Cannot link missing source: ${src}"
  mkdir -p "$(dirname "${dst}")"
  ln -sfn "${src}" "${dst}"
  log "Linked ${dst} -> ${src}"
}

clone_or_update_repo() {
  require_command git
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning repo: ${REPO_URL} -> ${REPO_DIR}"
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
  else
    log "Repo already exists, updating: ${REPO_DIR}"
    git -C "${REPO_DIR}" fetch --all
    git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
    git -C "${REPO_DIR}" pull --ff-only origin "${REPO_BRANCH}"
  fi
  log "Repo HEAD: $(git -C "${REPO_DIR}" rev-parse --short HEAD)"
}

download_model_assets() {
  require_command curl
  local mace_a_url="${MACE_A_URL:-$(build_release_url "${MODEL_BASE_URL}" "${MACE_A_NAME}")}"
  local mace_b_url="${MACE_B_URL:-$(build_release_url "${MODEL_BASE_URL}" "${MACE_B_NAME}")}"
  local mace_c_url="${MACE_C_URL:-$(build_release_url "${MODEL_BASE_URL}" "${MACE_C_NAME}")}"
  local mace_d_url="${MACE_D_URL:-$(build_release_url "${MODEL_BASE_URL}" "${MACE_D_NAME}")}"
  local ace_514_url="${ACE_514_URL:-$(build_release_url "${MODEL_BASE_URL}" "${ACE_514_NAME}")}"
  local ace_1352_url="${ACE_1352_URL:-$(build_release_url "${MODEL_BASE_URL}" "${ACE_1352_NAME}")}"
  download_if_missing "${mace_a_url}" "${RAW_MODELS_DIR}/${MACE_A_NAME}"
  download_if_missing "${mace_b_url}" "${RAW_MODELS_DIR}/${MACE_B_NAME}"
  download_if_missing "${mace_c_url}" "${RAW_MODELS_DIR}/${MACE_C_NAME}"
  download_if_missing "${mace_d_url}" "${RAW_MODELS_DIR}/${MACE_D_NAME}"
  download_if_missing "${ace_514_url}" "${RAW_MODELS_DIR}/${ACE_514_NAME}"
  download_if_missing "${ace_1352_url}" "${RAW_MODELS_DIR}/${ACE_1352_NAME}"
  log "Raw model directory contents:"
  ls -lh "${RAW_MODELS_DIR}"
}

download_dataset_assets() {
  if [[ "${DOWNLOAD_DATASETS}" != "1" ]]; then
    log "Dataset download disabled"
    return 0
  fi
  require_command curl
  local train_url="${TRAIN_SPLIT_URL:-$(build_release_url "${DATA_BASE_URL}" "${TRAIN_SPLIT_NAME}")}"
  local valid_url="${VALID_SPLIT_URL:-$(build_release_url "${DATA_BASE_URL}" "${VALID_SPLIT_NAME}")}"
  local test_url="${TEST_URL:-$(build_release_url "${DATA_BASE_URL}" "${TEST_NAME}")}"
  download_if_missing "${train_url}" "${DATA_DIR}/${TRAIN_SPLIT_NAME}"
  download_if_missing "${valid_url}" "${DATA_DIR}/${VALID_SPLIT_NAME}"
  download_if_missing "${test_url}" "${DATA_DIR}/${TEST_NAME}"
  log "Data directory contents:"
  ls -lh "${DATA_DIR}"
}

run_startup_md() {
  [[ -f "${STARTUP_SCRIPT}" ]] || fail "Startup script not found: ${STARTUP_SCRIPT}"
  if [[ -f "${STARTUP_DONE_MARKER}" ]]; then
    log "startup_md already completed earlier"
  else
    log "Running startup script: ${STARTUP_SCRIPT}"
    chmod +x "${STARTUP_SCRIPT}"
    bash "${STARTUP_SCRIPT}"
    touch "${STARTUP_DONE_MARKER}"
    log "startup_md finished successfully"
  fi
  if [[ -f "${PYTHON_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    . "${PYTHON_ENV_FILE}"
    log "Sourced Python/LAMMPS env helper: ${PYTHON_ENV_FILE}"
    log "Using PYTHON_BIN=${PYTHON_BIN:-python}"
  else
    fail "Expected Python env helper missing: ${PYTHON_ENV_FILE}"
  fi
}

convert_mace_if_needed() {
  local raw_model="$1"
  local converted_target="$2"
  local pyexe="${PYTHON_BIN:-python}"
  [[ -f "${raw_model}" ]] || fail "Raw MACE model not found: ${raw_model}"
  if [[ -f "${converted_target}" && "${FORCE_REDOWNLOAD}" != "1" ]]; then
    log "Already converted: ${converted_target}"
    return 0
  fi
  log "Converting MACE model with ${pyexe}: ${raw_model}"
  "${pyexe}" -m mace.cli.create_lammps_model "${raw_model}" --format=mliap
  local produced="${raw_model}-mliap_lammps.pt"
  [[ -f "${produced}" ]] || fail "Expected converted model missing: ${produced}"
  if [[ "${produced}" != "${converted_target}" ]]; then
    mv -f "${produced}" "${converted_target}"
    log "Moved converted model to canonical converted path: ${converted_target}"
  fi
}

convert_all_mace_models() {
  if [[ "${CONVERT_MACE_MODELS}" != "1" ]]; then
    log "MACE conversion disabled"
    return 0
  fi
  if [[ -f "${CONVERSION_DONE_MARKER}" ]]; then
    log "MACE conversion already completed earlier"
    return 0
  fi
  local pyexe="${PYTHON_BIN:-python}"
  log "Checking Python / torch before MACE conversion"
  "${pyexe}" - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
PY
  convert_mace_if_needed "${RAW_MODELS_DIR}/${MACE_A_NAME}" "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_A_NAME}"
  convert_mace_if_needed "${RAW_MODELS_DIR}/${MACE_B_NAME}" "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_B_NAME}"
  convert_mace_if_needed "${RAW_MODELS_DIR}/${MACE_C_NAME}" "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_C_NAME}"
  convert_mace_if_needed "${RAW_MODELS_DIR}/${MACE_D_NAME}" "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_D_NAME}"
  touch "${CONVERSION_DONE_MARKER}"
  log "Converted MACE models:"
  ls -lh "${CONVERTED_MODELS_DIR}"
}

create_canonical_potential_paths() {
  if [[ -f "${POTENTIALS_DONE_MARKER}" ]]; then
    log "Canonical potential paths already prepared earlier"
    return 0
  fi
  make_link "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_A_NAME}" "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_A_NAME}"
  make_link "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_B_NAME}" "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_B_NAME}"
  make_link "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_C_NAME}" "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_C_NAME}"
  make_link "${CONVERTED_MODELS_DIR}/${CANONICAL_MACE_D_NAME}" "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_D_NAME}"

  make_link "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_A_NAME}" "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_A_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_B_NAME}" "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_B_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_C_NAME}" "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_C_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_D_NAME}" "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_D_NAME}"

  make_link "${RAW_MODELS_DIR}/${ACE_514_NAME}" "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}"
  make_link "${RAW_MODELS_DIR}/${ACE_1352_NAME}" "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}" "${POTENTIALS_ACE_DIR}/${VALIDATION_ACE_514_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}" "${POTENTIALS_ACE_DIR}/${VALIDATION_ACE_1352_NAME}"

  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_A_NAME}" "${REPO_MACE_A_ALIAS}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_B_NAME}" "${REPO_MACE_B_ALIAS}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_C_NAME}" "${REPO_MACE_C_ALIAS}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_D_NAME}" "${REPO_MACE_D_ALIAS}"

  if [[ "${SYNC_POTENTIALS_INTO_REPO}" == "1" ]]; then
    mkdir -p "${REPO_POTENTIALS_DIR}/mace" "${REPO_POTENTIALS_DIR}/ace" "${REPO_POTENTIALS_DIR}/eam"
    make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_A_NAME}" "${REPO_POTENTIALS_DIR}/mace/${VALIDATION_MACE_A_NAME}"
    make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_B_NAME}" "${REPO_POTENTIALS_DIR}/mace/${VALIDATION_MACE_B_NAME}"
    make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_C_NAME}" "${REPO_POTENTIALS_DIR}/mace/${VALIDATION_MACE_C_NAME}"
    make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_D_NAME}" "${REPO_POTENTIALS_DIR}/mace/${VALIDATION_MACE_D_NAME}"
    make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}" "${REPO_POTENTIALS_DIR}/ace/${CANONICAL_ACE_514_NAME}"
    make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}" "${REPO_POTENTIALS_DIR}/ace/${CANONICAL_ACE_1352_NAME}"
  fi

  write_potential_env_file
  write_potential_manifest_json
  write_ready_summary

  touch "${POTENTIALS_DONE_MARKER}"
  log "Canonical and validation-compatible potential paths prepared"
}

setup_pyiron_layout() {
  if [[ -f "${PYIRON_DONE_MARKER}" ]]; then
    log "pyiron layout already prepared earlier"
    return 0
  fi
  mkdir -p "${PYIRON_PROJECTS_DIR}" "${PYIRON_LAMMPS_BIN_DIR}" "${PYIRON_LAMMPS_POT_DIR}"
  cat > "${PYIRON_CONFIG_FILE}" <<EOF
[DEFAULT]
FILE = ${PYIRON_DB_FILE}
PROJECT_PATHS = ${PYIRON_PROJECTS_DIR}
RESOURCE_PATHS = ${PYIRON_RESOURCES_DIR}
PROJECT_CHECK_ENABLED = ${PYIRON_PROJECT_CHECK_ENABLED}
EOF

  cat > "${PYIRON_ENV_FILE}" <<EOF
export PYIRONCONFIG="${PYIRON_CONFIG_FILE}"
export PYIRONPROJECTPATHS="${PYIRON_PROJECTS_DIR}"
export PYIRONRESOURCEPATHS="${PYIRON_RESOURCES_DIR}"
export PYIRON_PROJECTS_DIR="${PYIRON_PROJECTS_DIR}"
export PYIRON_RESOURCES_DIR="${PYIRON_RESOURCES_DIR}"
export PYIRON_LAMMPS_BIN_DIR="${PYIRON_LAMMPS_BIN_DIR}"
export PYIRON_LAMMPS_POT_DIR="${PYIRON_LAMMPS_POT_DIR}"
EOF

  cat > "${PYIRON_LAMMPS_BIN_DIR}/run_lammps_custom.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
lmp -in control.inp
EOF
  chmod +x "${PYIRON_LAMMPS_BIN_DIR}/run_lammps_custom.sh"

  cat > "${PYIRON_LAMMPS_BIN_DIR}/run_lammps_custom_mpi.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
NCORES="${1:-1}"
mpiexec -n "${NCORES}" lmp -in control.inp
EOF
  chmod +x "${PYIRON_LAMMPS_BIN_DIR}/run_lammps_custom_mpi.sh"

  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_A_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_MACE_A_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_B_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_MACE_B_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_C_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_MACE_C_NAME}"
  make_link "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_D_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_MACE_D_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${CANONICAL_ACE_514_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${CANONICAL_ACE_1352_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_ACE_514_NAME}"
  make_link "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}" "${PYIRON_LAMMPS_POT_DIR}/${VALIDATION_ACE_1352_NAME}"

  cat > "${PYIRON_NOTE_TXT}" <<EOF
Custom potentials have been mirrored here for pyiron-facing workflows.

Created by wrapper:
  Projects:   ${PYIRON_PROJECTS_DIR}
  Resources:  ${PYIRON_RESOURCES_DIR}
  Config:     ${PYIRON_CONFIG_FILE}

Important compatibility note:
- The current Paper 1 validation workflow expects legacy-compatible MACE filenames:
    ${VALIDATION_MACE_A_NAME}
    ${VALIDATION_MACE_B_NAME}
    ${VALIDATION_MACE_C_NAME}
    ${VALIDATION_MACE_D_NAME}
- Those exact aliases are mirrored here.
- ACE file paths are also exported through:
    CUZR_ACE_514_FILE
    CUZR_ACE_1352_FILE
EOF

  touch "${PYIRON_DONE_MARKER}"
  log "pyiron layout prepared"
}

write_potential_env_file() {
  cat > "${POTENTIALS_ENV_FILE}" <<EOF
export WORKSPACE_ROOT="${WORKSPACE_ROOT}"
export REPO_DIR="${REPO_DIR}"
export RAW_MODELS_DIR="${RAW_MODELS_DIR}"
export CONVERTED_MODELS_DIR="${CONVERTED_MODELS_DIR}"
export POTENTIALS_ROOT="${POTENTIALS_ROOT}"
export POTENTIALS_MACE_DIR="${POTENTIALS_MACE_DIR}"
export POTENTIALS_ACE_DIR="${POTENTIALS_ACE_DIR}"
export POTENTIALS_EAM_DIR="${POTENTIALS_EAM_DIR}"

export MACE_A_LAMMPS_PT="${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_A_NAME}"
export MACE_B_LAMMPS_PT="${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_B_NAME}"
export MACE_C_LAMMPS_PT="${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_C_NAME}"
export MACE_D_LAMMPS_PT="${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_D_NAME}"

export CUZR_MACE_A_FILE="${REPO_MACE_A_ALIAS}"
export CUZR_MACE_B_FILE="${REPO_MACE_B_ALIAS}"
export CUZR_MACE_C_FILE="${REPO_MACE_C_ALIAS}"
export CUZR_MACE_D_FILE="${REPO_MACE_D_ALIAS}"

export ACE_514_YAML="${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}"
export ACE_1352_YAML="${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}"
export CUZR_ACE_514_FILE="${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}"
export CUZR_ACE_1352_FILE="${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}"

export REPO_RUNTIME_POTENTIALS_DIR="${REPO_POTENTIALS_DIR}"
export PYTHON_ENV_FILE="${PYTHON_ENV_FILE}"
EOF
}

write_potential_manifest_json() {
  cat > "${POTENTIALS_MANIFEST_JSON}" <<EOF
{
  "workspace_root": "${WORKSPACE_ROOT}",
  "repo_dir": "${REPO_DIR}",
  "raw_models_dir": "${RAW_MODELS_DIR}",
  "converted_models_dir": "${CONVERTED_MODELS_DIR}",
  "potentials_root": "${POTENTIALS_ROOT}",
  "python_env_file": "${PYTHON_ENV_FILE}",
  "mace": {
    "canonical": {
      "mace_A": "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_A_NAME}",
      "mace_B": "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_B_NAME}",
      "mace_C": "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_C_NAME}",
      "mace_D": "${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_D_NAME}"
    },
    "validation_aliases": {
      "MACE_A": "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_A_NAME}",
      "MACE_B": "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_B_NAME}",
      "MACE_C": "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_C_NAME}",
      "MACE_D": "${POTENTIALS_MACE_DIR}/${VALIDATION_MACE_D_NAME}"
    }
  },
  "ace": {
    "ace_514": "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}",
    "ace_1352": "${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}"
  },
  "pyiron": {
    "config_file": "${PYIRON_CONFIG_FILE}",
    "db_file": "${PYIRON_DB_FILE}",
    "projects_dir": "${PYIRON_PROJECTS_DIR}",
    "resources_dir": "${PYIRON_RESOURCES_DIR}",
    "lammps_bin_dir": "${PYIRON_LAMMPS_BIN_DIR}",
    "lammps_potential_dir": "${PYIRON_LAMMPS_POT_DIR}"
  }
}
EOF
}

write_ready_summary() {
  cat > "${READY_SUMMARY_TXT}" <<EOF
Validation environment is ready.

Repo:
  ${REPO_DIR}

Validation-script-compatible MACE aliases:
  ${REPO_MACE_A_ALIAS}
  ${REPO_MACE_B_ALIAS}
  ${REPO_MACE_C_ALIAS}
  ${REPO_MACE_D_ALIAS}

Canonical potentials:
  MACE A:    ${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_A_NAME}
  MACE B:    ${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_B_NAME}
  MACE C:    ${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_C_NAME}
  MACE D:    ${POTENTIALS_MACE_DIR}/${CANONICAL_MACE_D_NAME}
  ACE 514:   ${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_514_NAME}
  ACE 1352:  ${POTENTIALS_ACE_DIR}/${CANONICAL_ACE_1352_NAME}

Python env helper:
  ${PYTHON_ENV_FILE}

pyiron:
  Config:       ${PYIRON_CONFIG_FILE}
  DB:           ${PYIRON_DB_FILE}
  Projects:     ${PYIRON_PROJECTS_DIR}
  Resources:    ${PYIRON_RESOURCES_DIR}
  LAMMPS bin:   ${PYIRON_LAMMPS_BIN_DIR}
  LAMMPS pots:  ${PYIRON_LAMMPS_POT_DIR}

Typical next step:
  cd ${REPO_DIR}
  source ${PYTHON_ENV_FILE}
  source ${POTENTIALS_ENV_FILE}
  source ${PYIRON_ENV_FILE}
  python scripts/paper1_validate_potentials.py
EOF
}

final_summary() {
  log "Wrapper final summary"
  log "Repo dir:           ${REPO_DIR}"
  log "Raw models dir:     ${RAW_MODELS_DIR}"
  log "Converted dir:      ${CONVERTED_MODELS_DIR}"
  log "Potentials root:    ${POTENTIALS_ROOT}"
  log "pyiron projects:    ${PYIRON_PROJECTS_DIR}"
  log "pyiron resources:   ${PYIRON_RESOURCES_DIR}"
  if command -v lmp >/dev/null 2>&1; then
    log "LAMMPS executable: $(command -v lmp)"
  else
    log "WARNING: lmp not found on PATH"
  fi
  local pyexe="${PYTHON_BIN:-python}"
  "${pyexe}" - <<'PY'
import os, torch
print("Torch:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("PYTHON_BIN:", os.environ.get("PYTHON_BIN", "<unset>"))
PY
  touch "${WRAPPER_DONE_MARKER}"
}

main() {
  log "Wrapper started"
  clone_or_update_repo
  download_model_assets
  download_dataset_assets
  if [[ "${RUN_STARTUP_MD}" == "1" ]]; then
    run_startup_md
  else
    log "RUN_STARTUP_MD=0, skipping startup_md"
  fi
  convert_all_mace_models
  create_canonical_potential_paths
  setup_pyiron_layout
  final_summary
  log "Wrapper finished successfully"
}

main "$@"
