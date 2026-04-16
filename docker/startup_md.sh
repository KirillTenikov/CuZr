#!/usr/bin/env bash
set -euo pipefail

# startup_md_conda_pyiron_v3.sh
#
# Strategy:
# - install system build deps with apt
# - install micromamba
# - create a dedicated conda-forge env for pyiron + LAMMPS Python pieces
# - install PyTorch CUDA 12.6 into that env
# - install MACE-related Python packages into that env
# - keep NumPy pinned to 1.26.4 to stay compatible with pyiron-side packages
# - ensure cythonize is available for LAMMPS ML-IAP configure
# - build and install LAMMPS using that env's Python
#
# Optional environment variables:
#   LAMMPS_BRANCH=develop
#   LAMMPS_SRC_DIR=/opt/lammps
#   LAMMPS_BUILD_DIR=/opt/lammps/build
#   LAMMPS_INSTALL_DIR=/opt/lammps-install
#   BUILD_JOBS=4
#   GPU_ARCH_FLAG=-DKokkos_ARCH_AMPERE80=ON
#   CUZR_ENV_PREFIX=/opt/cuzr-mamba
#   PYTHON_ENV_FILE=/workspace/cuzr_python.env

export DEBIAN_FRONTEND=noninteractive

LAMMPS_BRANCH="${LAMMPS_BRANCH:-develop}"
LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/opt/lammps}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build}"
LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps-install}"
BUILD_JOBS="${BUILD_JOBS:-4}"
GPU_ARCH_FLAG="${GPU_ARCH_FLAG:--DKokkos_ARCH_AMPERE80=ON}"
CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
PYTHON_ENV_FILE="${PYTHON_ENV_FILE:-/workspace/cuzr_python.env}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/usr/local/bin/micromamba}"

echo "==> Installing system packages"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  gfortran \
  git \
  wget \
  curl \
  ca-certificates \
  pkg-config \
  cmake \
  ninja-build \
  openmpi-bin \
  libopenmpi-dev \
  libfftw3-dev \
  libcurl4-openssl-dev \
  libjpeg-dev \
  libpng-dev \
  libhdf5-dev \
  libhdf5-openmpi-dev \
  unzip \
  bzip2 \
  && rm -rf /var/lib/apt/lists/*

echo "==> Installing micromamba"
if [ ! -x "${MICROMAMBA_BIN}" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /tmp bin/micromamba
  install -m 0755 /tmp/bin/micromamba "${MICROMAMBA_BIN}"
  rm -rf /tmp/bin
fi
"${MICROMAMBA_BIN}" --version

echo "==> Creating conda-forge environment for pyiron"
if [ ! -x "${CUZR_ENV_PREFIX}/bin/python" ]; then
  "${MICROMAMBA_BIN}" create -y -p "${CUZR_ENV_PREFIX}" -c conda-forge \
    python=3.11 \
    pip \
    numpy=1.26.4 \
    scipy \
    pandas \
    matplotlib \
    ase \
    h5py \
    mpi4py \
    h5io \
    sqlalchemy \
    pysqa \
    pyiron \
    pyiron_base \
    pyiron_atomistics \
    pylammpsmpi \
    structuretoolkit \
    cython
else
  echo "Environment already exists at ${CUZR_ENV_PREFIX}; skipping create"
fi

PYTHON_BIN="${CUZR_ENV_PREFIX}/bin/python"
PIP_BIN="${CUZR_ENV_PREFIX}/bin/pip"

export PYTHON_BIN
export PIP_BIN
export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

echo "==> Python: ${PYTHON_BIN}"
echo "==> Pip:    ${PIP_BIN}"
"${PYTHON_BIN}" --version
"${PIP_BIN}" --version

echo "==> Preflight checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
fi

echo "==> Checking Python / Torch before installing PyTorch"
"${PYTHON_BIN}" - <<'PY'
import sys
print("Python executable:", sys.executable)
try:
    import torch
    print("Existing Torch:", torch.__version__)
    print("Existing Torch CUDA available:", torch.cuda.is_available())
except Exception as exc:
    print("Torch not yet available in env:", exc)
PY

echo "==> Installing PyTorch CUDA 12.6 into the environment"
"${PIP_BIN}" install --no-cache-dir \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing MACE-related Python packages"
"${PIP_BIN}" install --no-cache-dir \
  numpy==1.26.4 \
  cython \
  configargparse \
  "e3nn==0.4.4" \
  lmdb \
  matscipy \
  prettytable \
  python-hostlist \
  torch-ema \
  torchmetrics \
  "mace-torch==0.3.15" \
  cuequivariance \
  cuequivariance-torch \
  cuequivariance-ops-torch-cu12 \
  cupy-cuda12x \
  kim-property

echo "==> Verifying Cython and NumPy pin"
which cythonize
"${PYTHON_BIN}" - <<'PY'
import shutil
import Cython
import numpy
print("cythonize:", shutil.which("cythonize"))
print("Cython:", Cython.__version__)
print("NumPy after MACE install:", numpy.__version__)
assert numpy.__version__ == "1.26.4", f"Unexpected NumPy version: {numpy.__version__}"
PY

echo "==> Post-install Python sanity check"
"${PYTHON_BIN}" - <<'PY'
import sys
import numpy
import torch
import ase
print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("ASE:", ase.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
PY

echo "==> Cloning LAMMPS"
mkdir -p "$(dirname "${LAMMPS_SRC_DIR}")"
if [ ! -d "${LAMMPS_SRC_DIR}/.git" ]; then
  git clone --depth 1 --branch "${LAMMPS_BRANCH}" https://github.com/lammps/lammps.git "${LAMMPS_SRC_DIR}"
else
  echo "LAMMPS source already exists at ${LAMMPS_SRC_DIR}; skipping clone"
fi

echo "==> Configuring LAMMPS"
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

echo "==> Writing environment helper files"
mkdir -p "$(dirname "${PYTHON_ENV_FILE}")"
cat > "${PYTHON_ENV_FILE}" <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

cat >/etc/profile.d/cuzr-lammps.sh <<EOF
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

export PATH="${CUZR_ENV_PREFIX}/bin:${LAMMPS_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "==> Final verification"
"${PYTHON_BIN}" - <<'PY'
import sys
import numpy
import torch
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
PY

if command -v lmp >/dev/null 2>&1; then
  echo "==> lmp found at: $(command -v lmp)"
  lmp -h | head -n 20 || true
else
  echo "WARNING: lmp executable not found on PATH after installation"
fi

echo "==> Setup complete"
