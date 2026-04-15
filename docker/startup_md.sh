#!/usr/bin/env bash
set -euo pipefail

# Startup/setup script for a container based on:
#   pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
#
# Goal:
# - install system dependencies needed for LAMMPS + pyiron workflows
# - install Python packages needed for MACE/pyiron analysis
# - build and install LAMMPS with the packages needed for the CuZr project
#
# Usage inside the container:
#   bash startup_planb_pytorch28_cuda128.sh
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

PYTHON_BIN="$(command -v python)"
PIP_BIN="$(command -v pip)"

echo "==> Python: ${PYTHON_BIN}"
echo "==> Pip:    ${PIP_BIN}"
python --version
pip --version

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

echo "==> Upgrading pip build tooling"
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "==> Installing Python stack"
# Keep numpy<2 to stay friendly with pyiron-related packages and older scientific deps.
python -m pip install --no-cache-dir \
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
export PATH="${LAMMPS_INSTALL_DIR}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"
EOF

export PATH="${LAMMPS_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "==> Basic verification"
python - <<'PY'
import sys
import numpy
import torch
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
PY

if command -v lmp >/dev/null 2>&1; then
  echo "==> lmp found at: $(command -v lmp)"
  lmp -h | head -n 20 || true
else
  echo "WARNING: lmp executable not found on PATH after installation"
fi

echo "==> Setup complete"
