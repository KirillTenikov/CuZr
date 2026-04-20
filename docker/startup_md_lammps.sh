#!/usr/bin/env bash
set -euo pipefail

# startup_md_lammps_with_potentials.sh
#
# Plan C runtime script:
#   1) verify baked Python env
#   2) build/install LAMMPS if needed
#   3) download ACE/MACE potentials
#   4) convert MACE models to LAMMPS-ready form
#   5) write convenient runtime env files and canonical paths
#
# Assumes the Docker image already contains:
#   - /opt/cuzr-mamba Python env with pyiron + torch + MACE stack
#   - /opt/lammps source tree already unpacked

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

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
    for cand in       "${src}-mliap_lammps.pt"       "${src}-lammps.pt"       "${d}/${b}-mliap_lammps.pt"       "${d}/${b}-lammps.pt"       "${d}/${stem}-mliap_lammps.pt"       "${d}/${stem}-lammps.pt"; do
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

compile_mace_model() {
  local label="$1"
  local raw_src="$2"
  local final_dst="$3"

  if [[ -s "$final_dst" ]]; then
    log "Compiled MACE already present: $final_dst"
    return 0
  fi

  [[ -f "$raw_src" ]] || die "Raw MACE model not found: $raw_src"

  local before_file after_file new_pt
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  find "$(dirname "$raw_src")" -maxdepth 1 -type f -name "*.pt" | sort > "$before_file"

  log "Converting $label from $raw_src"
  "${PYTHON_BIN}" -m mace.cli.create_lammps_model "$raw_src" --format=mliap

  new_pt="$(find_new_pt_after_conversion "$raw_src" "$before_file" "$after_file" || true)"
  rm -f "$before_file" "$after_file"

  [[ -n "$new_pt" ]] || die "Could not find converted .pt output for $raw_src"

  mkdir -p "$(dirname "$final_dst")"
  if [[ "$new_pt" != "$final_dst" ]]; then
    mv -f "$new_pt" "$final_dst"
  fi

  [[ -s "$final_dst" ]] || die "Converted model missing after move: $final_dst"
  log "Prepared $label -> $final_dst"
}

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_DIR}/CuZr}"
REPO_URL="${REPO_URL:-https://github.com/KirillTenikov/CuZr.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"

CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"
PIP_BIN="${PIP_BIN:-${CUZR_ENV_PREFIX}/bin/pip}"

LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/opt/lammps}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build}"
LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps-install}"
BUILD_JOBS="${BUILD_JOBS:-4}"
GPU_ARCH_FLAG="${GPU_ARCH_FLAG:--DKokkos_ARCH_AMPERE80=ON}"

MODEL_BASE_URL="${MODEL_BASE_URL:-https://github.com/KirillTenikov/CuZr/releases/download/dataset-v1}"
RAW_DIR="${RAW_DIR:-${WORKSPACE_DIR}/models/raw}"
CONVERTED_DIR="${CONVERTED_DIR:-${WORKSPACE_DIR}/models/converted}"
POTENTIALS_DIR="${POTENTIALS_DIR:-${WORKSPACE_DIR}/potentials}"
POTENTIALS_MACE_DIR="${POTENTIALS_MACE_DIR:-${POTENTIALS_DIR}/mace}"
POTENTIALS_ACE_DIR="${POTENTIALS_ACE_DIR:-${POTENTIALS_DIR}/ace}"

MACE_A_NAME="${MACE_A_NAME:-mace_A.model}"
MACE_B_NAME="${MACE_B_NAME:-mace_B.model}"
MACE_C_NAME="${MACE_C_NAME:-mace_C.model}"
MACE_D_NAME="${MACE_D_NAME:-mace_D.model}"
ACE_514_NAME="${ACE_514_NAME:-ace_514.yaml}"
ACE_1352_NAME="${ACE_1352_NAME:-ace_1352.yaml}"

MACE_A_RAW="${RAW_DIR}/${MACE_A_NAME}"
MACE_B_RAW="${RAW_DIR}/${MACE_B_NAME}"
MACE_C_RAW="${RAW_DIR}/${MACE_C_NAME}"
MACE_D_RAW="${RAW_DIR}/${MACE_D_NAME}"
ACE_514_RAW="${RAW_DIR}/${ACE_514_NAME}"
ACE_1352_RAW="${RAW_DIR}/${ACE_1352_NAME}"

MACE_A_COMPILED="${CONVERTED_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
MACE_B_COMPILED="${CONVERTED_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
MACE_C_COMPILED="${CONVERTED_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
MACE_D_COMPILED="${CONVERTED_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"

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

print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Cython:", Cython.__version__)
print("ASE:", ase.__version__)
print("cythonize:", shutil.which("cythonize"))

assert numpy.__version__ == "1.26.4", numpy.__version__
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

mkdir -p "$(dirname "${PYTHON_ENV_FILE}")"
cat > "${PYTHON_ENV_FILE}" <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

cat > /etc/profile.d/cuzr-lammps.sh <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

log "Downloading raw ACE/MACE potentials"
mkdir -p "${RAW_DIR}" "${CONVERTED_DIR}" "${POTENTIALS_MACE_DIR}" "${POTENTIALS_ACE_DIR}" "${REPO_DIR}"

download_if_missing "${MODEL_BASE_URL}/${MACE_A_NAME}" "${MACE_A_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_B_NAME}" "${MACE_B_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_C_NAME}" "${MACE_C_RAW}"
download_if_missing "${MODEL_BASE_URL}/${MACE_D_NAME}" "${MACE_D_RAW}"
download_if_missing "${MODEL_BASE_URL}/${ACE_514_NAME}" "${ACE_514_RAW}"
download_if_missing "${MODEL_BASE_URL}/${ACE_1352_NAME}" "${ACE_1352_RAW}"

log "Converting MACE models for LAMMPS"
compile_mace_model "MACE_A" "${MACE_A_RAW}" "${MACE_A_COMPILED}"
compile_mace_model "MACE_B" "${MACE_B_RAW}" "${MACE_B_COMPILED}"
compile_mace_model "MACE_C" "${MACE_C_RAW}" "${MACE_C_COMPILED}"
compile_mace_model "MACE_D" "${MACE_D_RAW}" "${MACE_D_COMPILED}"

log "Creating canonical potential paths"
ln -sfn "${MACE_A_COMPILED}" "${POTENTIALS_MACE_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
ln -sfn "${MACE_B_COMPILED}" "${POTENTIALS_MACE_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
ln -sfn "${MACE_C_COMPILED}" "${POTENTIALS_MACE_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
ln -sfn "${MACE_D_COMPILED}" "${POTENTIALS_MACE_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"

ln -sfn "${ACE_514_RAW}" "${POTENTIALS_ACE_DIR}/ace_514.yaml"
ln -sfn "${ACE_1352_RAW}" "${POTENTIALS_ACE_DIR}/ace_1352.yaml"

ln -sfn "${MACE_A_COMPILED}" "${REPO_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
ln -sfn "${MACE_B_COMPILED}" "${REPO_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
ln -sfn "${MACE_C_COMPILED}" "${REPO_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
ln -sfn "${MACE_D_COMPILED}" "${REPO_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"
ln -sfn "${ACE_514_RAW}" "${REPO_DIR}/ace_514.yaml"
ln -sfn "${ACE_1352_RAW}" "${REPO_DIR}/ace_1352.yaml"

cat > "${POTENTIAL_PATHS_ENV}" <<EOF
export CUZR_ACE_514_FILE="${POTENTIALS_ACE_DIR}/ace_514.yaml"
export CUZR_ACE_1352_FILE="${POTENTIALS_ACE_DIR}/ace_1352.yaml"
export CUZR_MACE_A_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
export CUZR_MACE_B_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
export CUZR_MACE_C_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
export CUZR_MACE_D_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"
EOF

cat > "${RUNTIME_ENV_FILE}" <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
export CUZR_ACE_514_FILE="${POTENTIALS_ACE_DIR}/ace_514.yaml"
export CUZR_ACE_1352_FILE="${POTENTIALS_ACE_DIR}/ace_1352.yaml"
export CUZR_MACE_A_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
export CUZR_MACE_B_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
export CUZR_MACE_C_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
export CUZR_MACE_D_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"
EOF

cat >> "${PYTHON_ENV_FILE}" <<EOF
export CUZR_ACE_514_FILE="${POTENTIALS_ACE_DIR}/ace_514.yaml"
export CUZR_ACE_1352_FILE="${POTENTIALS_ACE_DIR}/ace_1352.yaml"
export CUZR_MACE_A_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_A_compiled.model-lammps.pt"
export CUZR_MACE_B_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_B_compiled.model-lammps.pt"
export CUZR_MACE_C_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_C_compiled.model-lammps.pt"
export CUZR_MACE_D_FILE="${POTENTIALS_MACE_DIR}/CuZr_MACE_D_compiled.model-lammps.pt"
EOF

log "Final verification"
which lmp
lmp -h | head -n 20 || true
"${PYTHON_BIN}" -c "import lammps; print(lammps.__file__)"

log "Prepared compiled models:"
ls -lh "${POTENTIALS_MACE_DIR}" || true

log "Prepared ACE files:"
ls -lh "${POTENTIALS_ACE_DIR}" || true

log "Done. Source this file next time:"
echo "  source ${RUNTIME_ENV_FILE}"
