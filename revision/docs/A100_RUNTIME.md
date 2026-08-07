# A100 runtime and environment

This document records the actual runtime design already encoded in the repository. It is not a generic installation recipe.

## 1. Authoritative source files

- `docker/Dockerfile.cuzr_md`
- `docker/startup_md_lammps.sh`

The lightweight `env/environment.yml` in the Paper 2 repository is useful for analysis, but it is not the authoritative GPU MACE/LAMMPS production environment.

## 2. Docker environment already defined

The Dockerfile uses:

| Component | Recorded value |
|---|---|
| Base image | `nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04` |
| Python | 3.11 in `/opt/cuzr-mamba` |
| NumPy | 1.26.4 |
| PyTorch | 2.10.0, CUDA 12.6 wheel |
| MACE | `mace-torch==0.3.15` |
| e3nn | 0.4.4 |
| cuEquivariance | 0.9.1 |
| CuPy | `cupy-cuda12x==13.6.0` |
| ASE | conda-forge package in the same environment |
| LAMMPS source | `develop`, downloaded to `/opt/lammps` |
| Working directory | `/workspace` |

The image includes compilers, CMake, Ninja, OpenMPI, FFTW and the Python scientific stack needed to build and run LAMMPS.

## 3. Important runtime policy

LAMMPS is **not prebuilt in the image**. After the A100 instance starts, `docker/startup_md_lammps.sh`:

- verifies Python, CUDA, Torch, ASE and MACE;
- builds LAMMPS into `/opt/lammps-install`;
- enables MPI, shared libraries, ML-IAP, ML-SNAP, Python, ML-PACE and Kokkos;
- builds for A100 using `Kokkos_ARCH_AMPERE80` by default;
- downloads the released MACE, ACE and EAM models;
- converts all four raw MACE models to ML-IAP format;
- writes `/workspace/cuzr_runtime.env` and potential-path environment variables.

## 4. A100 startup sequence

Run these commands only after the GPU instance/container is available:

```bash
cd /workspace/CuZr

# Build LAMMPS and prepare all potential files.
bash docker/startup_md_lammps.sh

# Load LAMMPS, Python-library and potential paths into the current shell.
source /workspace/cuzr_runtime.env
```

The startup script is idempotent for already prepared files: it skips existing converted models and an existing LAMMPS installation.

## 5. Mandatory environment checks

```bash
nvidia-smi

python - <<'PY'
import torch
import mace
import ase
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("MACE:", mace.__file__)
print("ASE:", ase.__version__)
PY

which lmp
lmp -h | head -n 40

env | grep -E '^(MACE_[ABCD]_MLIAP|ACE_(514|1352)_FILE|EAM_(2007|2019)_FILE)=' | sort
```

Do not begin the pilot unless CUDA is available, the GPU is identified as an A100-class device, `lmp` is found, and all eight potential variables are defined.

## 6. Why revision v0.2 uses environment variables

The startup script places models outside the Git repository, primarily under `/workspace/models/` and `/workspace/potentials/`. Revision v0.2 therefore reads:

- `${MACE_A_MLIAP}` through `${MACE_D_MLIAP}`;
- `${ACE_514_FILE}` and `${ACE_1352_FILE}`;
- `${EAM_2007_FILE}` and `${EAM_2019_FILE}`.

This avoids hard-coding incorrect repository-relative paths. The shell must source `/workspace/cuzr_runtime.env` before the revision runner is invoked.

## 7. First pilot command

From `/workspace/CuZr`, after sourcing the runtime environment:

```bash
python revision/scripts/run_glass_preparation.py \
  --repo-root . \
  --potential MACE_C \
  --composition Cu64Zr36 \
  --natoms 1024 \
  --seed 43 \
  --initial-density 7.20 \
  --execute \
  --summarize
```

The optional NVE stage is not part of the first preparation pilot. Add `--with-nve` only after the preparation stages have been accepted.

## 8. Pilot acceptance checks

Before launching any matrix, inspect:

- successful completion of every stage;
- absence of lost atoms, NaNs and extreme pressure excursions;
- reasonable density evolution;
- stable 300 K NVT temperature;
- tail-averaged pressure and density;
- fixed-cell versus box-relaxed inherent-structure pressure;
- A100 memory use and wall time;
- correct model hash and Git commit in `manifest.json`.

Record the results using [`RUN_RECORD_TEMPLATE.md`](RUN_RECORD_TEMPLATE.md).
