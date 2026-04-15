#!/usr/bin/env bash
set -euo pipefail

# Startup/setup script for a container based on:
#   pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel
#
# Goal:
# - install system dependencies needed for LAMMPS + pyiron workflows
# - install Python packages needed for MACE/pyiron analysis
# - build and install LAMMPS with the packages needed for the CuZr project
#
# Main fix in this version:
# - avoid Ubuntu's externally-managed system Python
# - create a dedicated venv with --system-site-packages
# - reuse the image's preinstalled Torch/CUDA stack inside that venv
#
# Usage inside the container:
#   bash docker/startup_md.sh
#
# Optional environment variables:
#   LAMMPS_BRANCH=develop
#   LAMMPS_SRC_DIR=/opt/lammps
#   LAMMPS_BUILD_DIR=/opt/lammps/build
#   LAMMPS_INSTALL_DIR=/opt/lammps-install
#   BUILD_JOBS=4
#   GPU_ARCH_FLAG=-DKokkos_ARCH_AMPERE80=ON

export DEBIAN_FRONTEND=noninteractive

LAMMPS_BRANCH="${LAMMPS_BRANCH:-develop}"
LAMMPS_SRC_DIR="${LAMMPS_SRC_DIR:-/opt/lammps}"
LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build}"
LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps-install}"
BUILD_JOBS="${BUILD_JOBS:-4}"
GPU_ARCH_FLAG="${GPU_ARCH_FLAG:--DKokkos_ARCH_AMPERE80=ON}"

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
  python3-venv \
  && rm -rf /var/lib/apt/lists/*

echo "==> Creating Python venv with system site packages"
python3 -m venv --system-site-packages /opt/cuzr-venv
PYTHON_BIN=/opt/cuzr-venv/bin/python
PIP_BIN=/opt/cuzr-venv/bin/pip

export PYTHON_BIN
export PIP_BIN
export PATH="/opt/cuzr-venv/bin:${PATH}"

echo "==> Python: ${PYTHON_BIN}"
echo "==> Pip:    ${PIP_BIN}"
"${PYTHON_BIN}" --version
"${PIP_BIN}" --version

echo "==> Preflight checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. GPU runtime may not be attached yet."
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "WARNING: nvcc not found on PATH. This image may not be a CUDA devel image."
fi

"${PYTHON_BIN}" - <<'PY'
import sys
print("Python executable:", sys.executable)
try:
    import torch
    print("Torch:", torch.__version__)
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
except Exception as exc:
    print("WARNING: torch preflight failed:", exc)
PY

echo "==> Upgrading pip build tooling"
"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "==> Installing Python stack"
# Keep numpy<2 to stay friendly with pyiron-related packages and older scientific deps.
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "numpy<2" \
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

echo "==> Post-install Python sanity check"
"${PYTHON_BIN}" - <<'PY'
import numpy
import torch
import ase
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("ASE:", ase.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
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

echo "==> Writing environment helper"
cat >/etc/profile.d/cuzr-lammps.sh <<EOF
export PATH="${LAMMPS_INSTALL_DIR}/bin:/opt/cuzr-venv/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

export PATH="${LAMMPS_INSTALL_DIR}/bin:/opt/cuzr-venv/bin:${PATH}"
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
