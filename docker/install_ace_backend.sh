\
    #!/usr/bin/env bash
    set -euo pipefail

    # This installs the documented pacemaker/python-ace + TensorPotential stack
    # from source, which is the route described in the python-ace docs.
    #
    # Usage inside the container:
    #   bash docker/install_ace_backend.sh

    WORKDIR="${1:-/opt/ace-src}"
    mkdir -p "${WORKDIR}"
    cd "${WORKDIR}"

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install "protobuf<3.21"

    if [ ! -d TensorPotential ]; then
      git clone https://github.com/ICAMS/TensorPotential.git
    fi
    python -m pip install --upgrade ./TensorPotential

    if [ ! -d python-ace ]; then
      git clone https://github.com/ICAMS/python-ace.git
    fi
    python -m pip install --upgrade ./python-ace

    echo
    echo "ACE backend installation complete."
    echo "You should now have pacemaker, pace_info, and related tools on PATH."
