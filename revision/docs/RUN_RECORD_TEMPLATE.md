# A100 run record

Copy this file or its contents for each important pilot or cloud session.

## Session identity

- Date/time:
- Cloud provider / instance identifier:
- GPU model and memory:
- Host/container notes:
- Operator:

## Repository state

- Repository path:
- Branch:
- Commit hash:
- `git status --short`:
- Revision workflow version:

## Runtime state

- Docker image/tag or image identifier:
- `nvidia-smi` summary:
- Python version:
- Torch version:
- CUDA version reported by Torch:
- MACE version/path:
- LAMMPS version/date:
- LAMMPS build directory:
- `GPU_ARCH_FLAG`:
- Startup-script result:

## Potential preparation

- MACE ML-IAP files present:
- ACE files present:
- EAM files present:
- Potential hashes checked:
- `/workspace/cuzr_runtime.env` sourced: yes/no

## Run specification

- Potential:
- Composition:
- Atom count:
- Seed:
- Initial density:
- Run directory:
- NVE stage enabled: yes/no
- Box relaxation enabled: yes/no

## Outcome

- Stage 00 completed:
- Stage 01 completed:
- Stage 02 completed:
- Stage 03 completed:
- Stage 04 completed:
- Stage 05 completed:
- Lost atoms / NaNs / warnings:
- Peak GPU memory:
- Wall time by stage:

## Physical sanity checks

- Final/tail temperature:
- Final/tail density:
- Final/tail pressure:
- Pressure fluctuation:
- Fixed-cell inherent pressure:
- Box-relaxed inherent pressure:
- RDF inspected:
- Any suspicious structural feature:

## Decision

- Pilot accepted: yes/no
- Problems requiring code changes:
- Next run to launch:
