#!/usr/bin/env bash
set -euo pipefail

# startup_md_lammps.sh
#
# Runtime bootstrap for CuZr validation on a fresh instance:
# 1) verify baked Python env
# 2) build/install LAMMPS if needed
# 3) download ACE / MACE / EAM potentials
# 4) convert raw MACE models to ML-IAP format
# 5) prepare canonical directories and convenient env files

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

download_if_missing() {
  local url="$1"
  local dst="$2"
  if [[ -s "$dst" ]]; then
    log "Already present: $dst"
    return 0
  fi
  log "Downloading: $url -> $dst"
  mkdir -p "$(dirname "$dst")"
  curl -L --fail --retry 5 --retry-delay 3 -o "$dst" "$url"
}

find_new_pt_after_conversion() {
  local src="$1"
  local before_file="$2"
  local after_file="$3"

  find "$(dirname "$src")" -maxdepth 1 -type f -name "*.pt" | sort > "$after_file"

  local new_file=""
  if command -v comm >/dev/null 2>&1; then
    new_file="$(comm -13 "$before_file" "$after_file" | tail -n 1 || true)"
  fi

  if [[ -z "$new_file" ]]; then
    local d b stem
    d="$(dirname "$src")"
    b="$(basename "$src")"
    stem="${b%.model}"
    for cand in \
      "${src}-mliap_lammps.pt" \
      "${src}-lammps.pt" \
      "${d}/${b}-mliap_lammps.pt" \
      "${d}/${b}-lammps.pt" \
      "${d}/${stem}-mliap_lammps.pt" \
      "${d}/${stem}-lammps.pt"
    do
      if [[ -f "$cand" ]]; then
        new_file="$cand"
        break
      fi
    done
  fi

  if [[ -z "$new_file" ]]; then
    new_file="$(find "$(dirname "$src")" -maxdepth 1 -type f -name "*.pt" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
  fi

  [[ -n "$new_file" ]] || return 1
  echo "$new_file"
}

convert_mace_to_mliap() {
  local label="$1"
  local raw_src="$2"
  local final_dst="$3"

  if [[ -s "$final_dst" ]]; then
    log "Converted MACE already present: $final_dst"
    return 0
  fi

  [[ -f "$raw_src" ]] || die "Raw MACE model not found: $raw_src"

  local before_file after_file new_pt
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  find "$(dirname "$raw_src")" -maxdepth 1 -type f -name "*.pt" | sort > "$before_file"

  log "Converting ${label} to ML-IAP: ${raw_src}"
  "${PYTHON_BIN}" -m mace.cli.create_lammps_model "$raw_src" --format=mliap

  new_pt="$(find_new_pt_after_conversion "$raw_src" "$before_file" "$after_file" || true)"
  rm -f "$before_file" "$after_file"

  [[ -n "$new_pt" ]] || die "Could not find converted ML-IAP .pt for $raw_src"

  mkdir -p "$(dirname "$final_dst")"
  if [[ "$new_pt" != "$final_dst" ]]; then
    mv -f "$new_pt" "$final_dst"
  fi
  [[ -s "$final_dst" ]] || die "Converted ML-IAP model missing after move: $final_dst"

  log "Prepared ${label} -> ${final_dst}"
}

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_DIR}/CuZr}"
REPO_URL="${REPO_URL:-https://github.com/KirillTenikov/CuZr.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"

CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"

LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/opt/lammps}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build}"
LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps-install}"
BUILD_JOBS="${BUILD_JOBS:-4}"
GPU_ARCH_FLAG="${GPU_ARCH_FLAG:--DKokkos_ARCH_AMPERE80=ON}"

MODEL_BASE_URL="${MODEL_BASE_URL:-https://github.com/KirillTenikov/CuZr/releases/download/dataset-v1}"

RAW_DIR="${RAW_DIR:-${WORKSPACE_DIR}/models/raw}"
RAW_EAM_DIR="${RAW_EAM_DIR:-${RAW_DIR}/eam}"
CONVERTED_DIR="${CONVERTED_DIR:-${WORKSPACE_DIR}/models/converted}"

POTENTIALS_DIR="${POTENTIALS_DIR:-${WORKSPACE_DIR}/potentials}"
POTENTIALS_MACE_DIR="${POTENTIALS_MACE_DIR:-${POTENTIALS_DIR}/mace}"
POTENTIALS_ACE_DIR="${POTENTIALS_ACE_DIR:-${POTENTIALS_DIR}/ace}"
POTENTIALS_EAM_DIR="${POTENTIALS_EAM_DIR:-${POTENTIALS_DIR}/eam}"

MACE_A_NAME="${MACE_A_NAME:-mace_A.model}"
MACE_B_NAME="${MACE_B_NAME:-mace_B.model}"
MACE_C_NAME="${MACE_C_NAME:-mace_C.model}"
MACE_D_NAME="${MACE_D_NAME:-mace_D.model}"
ACE_514_NAME="${ACE_514_NAME:-ace_514.yaml}"
ACE_1352_NAME="${ACE_1352_NAME:-ace_1352.yaml}"
EAM_2019_NAME="${EAM_2019_NAME:-Cu-Zr_4.eam.fs}"

MACE_A_RAW="${RAW_DIR}/${MACE_A_NAME}"
MACE_B_RAW="${RAW_DIR}/${MACE_B_NAME}"
MACE_C_RAW="${RAW_DIR}/${MACE_C_NAME}"
MACE_D_RAW="${RAW_DIR}/${MACE_D_NAME}"
ACE_514_RAW="${RAW_DIR}/${ACE_514_NAME}"
ACE_1352_RAW="${RAW_DIR}/${ACE_1352_NAME}"
EAM_2019_RAW="${RAW_EAM_DIR}/${EAM_2019_NAME}"

MACE_A_MLIAP="${RAW_DIR}/mace_A.model-mliap_lammps.pt"
MACE_B_MLIAP="${RAW_DIR}/mace_B.model-mliap_lammps.pt"
MACE_C_MLIAP="${RAW_DIR}/mace_C.model-mliap_lammps.pt"
MACE_D_MLIAP="${RAW_DIR}/mace_D.model-mliap_lammps.pt"

EAM_2007_SYSTEM="${EAM_2007_SYSTEM:-${LAMMPS_INSTALL_DIR}/share/lammps/potentials/CuZr_mm.eam.fs}"
EAM_2007_RAW="${RAW_EAM_DIR}/CuZr_mm.eam.fs"

POTENTIAL_PATHS_ENV="${POTENTIALS_DIR}/potential_paths.env"
PYTHON_ENV_FILE="${WORKSPACE_DIR}/cuzr_python.env"
RUNTIME_ENV_FILE="${WORKSPACE_DIR}/cuzr_runtime.env"

export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Repo missing, cloning ${REPO_URL} -> ${REPO_DIR}"
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

log "Verifying baked Python environment"
"${PYTHON_BIN}" - <<'PY'
import shutil
import numpy
import torch
import Cython
import ase
import mace
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Cython:", Cython.__version__)
print("ASE:", ase.__version__)
print("MACE:", mace.__file__)
print("cythonize:", shutil.which("cythonize"))
assert shutil.which("cythonize"), "cythonize not found"
PY

if [[ ! -x "${LAMMPS_INSTALL_DIR}/bin/lmp" ]]; then
  log "Checking LAMMPS source tree"
  [[ -d "${LAMMPS_SRC_DIR}/cmake" ]] || die "LAMMPS source tree missing: ${LAMMPS_SRC_DIR}"

  log "Configuring LAMMPS"
  rm -rf "${LAMMPS_BUILD_DIR}"
  mkdir -p "${LAMMPS_BUILD_DIR}"
  cd "${LAMMPS_BUILD_DIR}"

  cmake -G Ninja "${LAMMPS_SRC_DIR}/cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${LAMMPS_INSTALL_DIR}" \
    -D CMAKE_C_STANDARD=17 \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KIM=OFF \
    -D DOWNLOAD_KIM=OFF \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    ${GPU_ARCH_FLAG} \
    -D PKG_MANYBODY=ON \
    -D PKG_MEAM=ON \
    -D PKG_KSPACE=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-DUMP=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_EXTRA-MOLECULE=ON \
    -D PKG_EXTRA-PAIR=ON \
    -D PKG_MISC=ON \
    -D PKG_REPLICA=ON \
    -D PKG_RIGID=ON \
    -D FFT=FFTW3 \
    -D FFTW3_INCLUDE_DIR=/usr/include \
    -D FFTW3_LIBRARY=/usr/lib/x86_64-linux-gnu/libfftw3.so \
    -D Python_EXECUTABLE="${PYTHON_BIN}"

  log "Building and installing LAMMPS"
  cmake --build . --parallel "${BUILD_JOBS}"
  cmake --install .
  cmake --build . --target install-python --parallel 1
else
  log "LAMMPS already installed: ${LAMMPS_INSTALL_DIR}/bin/lmp"
fi

export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RAW_DIR}" "${RAW_EAM_DIR}" "${CONVERTED_DIR}" \
         "${POTENTIALS_MACE_DIR}" "${POTENTIALS_ACE_DIR}" "${POTENTIALS_EAM_DIR}"

download_if_missing "${MODEL_BASE_URL}/${MACE_A_NAME}" "${MACE_A_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_B_NAME}" "${MACE_B_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_C_NAME}" "${MACE_C_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_D_NAME}" "${MACE_D_RAW}"
download_if_missing "${MODEL_BASE_URL}/${ACE_514_NAME}" "${ACE_514_RAW}"
download_if_missing "${MODEL_BASE_URL}/${ACE_1352_NAME}" "${ACE_1352_RAW}"
download_if_missing "${MODEL_BASE_URL}/${EAM_2019_NAME}" "${EAM_2019_RAW}"

[[ -f "${EAM_2007_SYSTEM}" ]] || die "Default 2007 EAM not found in LAMMPS install: ${EAM_2007_SYSTEM}"
cp -f "${EAM_2007_SYSTEM}" "${EAM_2007_RAW}"

convert_mace_to_mliap "MACE_A" "${MACE_A_RAW}" "${MACE_A_MLIAP}"
convert_mace_to_mliap "MACE_B" "${MACE_B_RAW}" "${MACE_B_MLIAP}"
convert_mace_to_mliap "MACE_C" "${MACE_C_RAW}" "${MACE_C_MLIAP}"
convert_mace_to_mliap "MACE_D" "${MACE_D_RAW}" "${MACE_D_MLIAP}"

ln -sfn "${MACE_A_MLIAP}" "${POTENTIALS_MACE_DIR}/mace_A.model-mliap_lammps.pt"
ln -sfn "${MACE_B_MLIAP}" "${POTENTIALS_MACE_DIR}/mace_B.model-mliap_lammps.pt"
ln -sfn "${MACE_C_MLIAP}" "${POTENTIALS_MACE_DIR}/mace_C.model-mliap_lammps.pt"
ln -sfn "${MACE_D_MLIAP}" "${POTENTIALS_MACE_DIR}/mace_D.model-mliap_lammps.pt"

ln -sfn "${ACE_514_RAW}" "${POTENTIALS_ACE_DIR}/ace_514.yaml"
ln -sfn "${ACE_1352_RAW}" "${POTENTIALS_ACE_DIR}/ace_1352.yaml"

ln -sfn "${EAM_2019_RAW}" "${POTENTIALS_EAM_DIR}/Cu-Zr_4.eam.fs"
ln -sfn "${EAM_2007_RAW}" "${POTENTIALS_EAM_DIR}/CuZr_mm.eam.fs"

cat > "${POTENTIAL_PATHS_ENV}" <<EOF
export CUZR_MODELS_RAW="${RAW_DIR}"
export CUZR_MODELS_CONVERTED="${CONVERTED_DIR}"
export CUZR_POTENTIALS_DIR="${POTENTIALS_DIR}"

export EAM_2019_FILE="${EAM_2019_RAW}"
export EAM_2007_FILE="${EAM_2007_RAW}"

export MACE_A_RAW="${MACE_A_RAW}"
export MACE_B_RAW="${MACE_B_RAW}"
export MACE_C_RAW="${MACE_C_RAW}"
export MACE_D_RAW="${MACE_D_RAW}"

export MACE_A_MLIAP="${MACE_A_MLIAP}"
export MACE_B_MLIAP="${MACE_B_MLIAP}"
export MACE_C_MLIAP="${MACE_C_MLIAP}"
export MACE_D_MLIAP="${MACE_D_MLIAP}"

export ACE_514_FILE="${ACE_514_RAW}"
export ACE_1352_FILE="${ACE_1352_RAW}"
EOF

cat > "${PYTHON_ENV_FILE}" <<EOF
export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LAMMPS_INSTALL_DIR}/lib/python3.11/site-packages:\${PYTHONPATH:-}"
EOF

cat > "${RUNTIME_ENV_FILE}" <<EOF
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LAMMPS_INSTALL_DIR}/lib/python3.11/site-packages:\${PYTHONPATH:-}"
source "${POTENTIAL_PATHS_ENV}"
EOF

log "Prepared MACE ML-IAP models:"
ls -lh "${RAW_DIR}"/*mliap_lammps.pt 2>/dev/null || true

log "Prepared ACE files:"
ls -lh "${RAW_DIR}"/*.yaml 2>/dev/null || true

log "Prepared EAM files:"
ls -lh "${RAW_EAM_DIR}"/*.eam.fs 2>/dev/null || true

log "Done. Source this next time:"
echo "  source ${RUNTIME_ENV_FILE}"
