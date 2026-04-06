#!/usr/bin/env bash
set -Eeuo pipefail

# Install Pacemaker / python-ace and TensorPotential inside the existing MACE container.
# Intended usage inside the container:
#   cd /workspace/CuZr
#   bash docker/install_ace_backend.sh
#
# Optional overrides:
#   WORKDIR=/opt/ace-src
#   TENSORFLOW_PKG=tensorflow
#   TENSORPOTENTIAL_REPO=https://github.com/ICAMS/TensorPotential.git
#   PYACE_REPO=https://github.com/ICAMS/python-ace.git
#   TENSORPOTENTIAL_REF=main
#   PYACE_REF=master

WORKDIR="${WORKDIR:-/opt/ace-src}"
TENSORFLOW_PKG="${TENSORFLOW_PKG:-tensorflow}"
TENSORPOTENTIAL_REPO="${TENSORPOTENTIAL_REPO:-https://github.com/ICAMS/TensorPotential.git}"
PYACE_REPO="${PYACE_REPO:-https://github.com/ICAMS/python-ace.git}"
TENSORPOTENTIAL_REF="${TENSORPOTENTIAL_REF:-main}"
PYACE_REF="${PYACE_REF:-master}"


echo "Python: $(which python)"
python --version
echo "Pip:    $(which pip)"
pip --version

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "==> Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

# Pacemaker commonly needs protobuf 3.20.x compatibility.
echo "==> Installing protobuf compatibility pin"
python -m pip install --upgrade "protobuf==3.20.*"

# Install TensorFlow before TensorPotential.
echo "==> Installing TensorFlow package: ${TENSORFLOW_PKG}"
python -m pip install --upgrade "${TENSORFLOW_PKG}"

echo "==> Cloning / updating TensorPotential"
if [ ! -d TensorPotential/.git ]; then
  git clone "${TENSORPOTENTIAL_REPO}" TensorPotential
fi
git -C TensorPotential fetch --all --tags
git -C TensorPotential checkout "${TENSORPOTENTIAL_REF}"

echo "==> Installing TensorPotential"
python -m pip install --upgrade ./TensorPotential

echo "==> Cloning / updating python-ace"
if [ ! -d python-ace/.git ]; then
  git clone "${PYACE_REPO}" python-ace
fi
git -C python-ace fetch --all --tags
git -C python-ace checkout "${PYACE_REF}"

echo "==> Installing python-ace / pacemaker"
python -m pip install --upgrade ./python-ace

echo "==> Verifying installation"
python - <<'PY'
import importlib
mods = ["tensorflow", "tensorpotential"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m}: OK ({getattr(mod, '__version__', 'no __version__')})")
PY

echo "==> CLI checks"
command -v pacemaker
command -v pace_info
pacemaker --help >/dev/null
pace_info --help >/dev/null || true

echo
echo "ACE backend installation complete."
echo "If needed, verify manually with:"
echo "  which pacemaker"
echo "  pacemaker --help"
echo "  python -c 'import tensorflow as tf; print(tf.__version__)'"
