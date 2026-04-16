#!/usr/bin/env bash
set -euo pipefail

LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/opt/lammps}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build}"
LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps-install}"
BUILD_JOBS="${BUILD_JOBS:-4}"
GPU_ARCH_FLAG="${GPU_ARCH_FLAG:--DKokkos_ARCH_AMPERE80=ON}"
CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"
PIP_BIN="${PIP_BIN:-${CUZR_ENV_PREFIX}/bin/pip}"
PYTHON_ENV_FILE="${PYTHON_ENV_FILE:-/workspace/cuzr_python.env}"

export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

echo "==> Verifying baked Python environment"
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

echo "==> Checking LAMMPS source tree"
test -d "${LAMMPS_SRC_DIR}/cmake"
ls -lah "${LAMMPS_SRC_DIR}" | head -n 20 || true

echo "==> Configuring LAMMPS"
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
  -D PKG_KIM=ON \
  -D DOWNLOAD_KIM=ON \
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

echo "==> Building and installing LAMMPS"
cmake --build . --parallel "${BUILD_JOBS}"
cmake --install .
cmake --build . --target install-python --parallel 1

echo "==> Writing runtime env helper"
mkdir -p "$(dirname "${PYTHON_ENV_FILE}")"
cat > "${PYTHON_ENV_FILE}" <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

cat > /etc/profile.d/cuzr-lammps.sh <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "==> Final verification"
which lmp
lmp -h | head -n 20 || true
echo "==> LAMMPS-only startup complete"
