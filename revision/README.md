# Additive Paper 1 revision workflow

This folder adds the revised validation workflow without changing any existing training, conversion or submitted-paper validation scripts. The preparation stages are adapted from the proven Paper 2 pressure-relaxed workflow, but copied here so Paper 1 remains self-contained.

## Documentation

Start with [`docs/README.md`](docs/README.md). It explains:

- what already exists and remains frozen;
- what the revision adds;
- the actual A100 Docker/MACE/LAMMPS runtime;
- the preparation stages and output structure;
- which validation blocks are implemented, planned or excluded.

## Safety rule

The old repository remains a reproducible record of the submitted manuscript. All new code, inputs, logs, structures and analyses are contained under `revision/`. The runner refuses to overwrite a non-empty run directory.

## Runtime location

The simulations are intended for the A100 machine, not for a local laptop. The repository's Docker image supplies CUDA, Python, PyTorch, MACE and build tools. LAMMPS is built after the A100 instance starts by:

```bash
cd /workspace/CuZr
bash docker/startup_md_lammps.sh
source /workspace/cuzr_runtime.env
```

Potential paths are resolved from the environment variables written by that startup script. This includes all four MACE, both ACE and both EAM models.

## First A100 pilot

After the runtime checks in [`docs/A100_RUNTIME.md`](docs/A100_RUNTIME.md):

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

The optional NVE stability stage is deliberately opt-in (`--with-nve`). It should be added only after the preparation pilot is accepted.

## Outputs

Default pilot path:

```text
revision/results/Cu64Zr36/N1024/seed_43/MACE_C/
```

Key files:

- `manifest.json`: provenance, resolved settings and hashes;
- stage-separated LAMMPS input and log files;
- finite-temperature and inherent-structure data files;
- `thermo_summary.json` and `thermo_summary.csv` after execution.

## Structural analysis

```bash
python revision/scripts/analyze_structure.py \
  revision/results/Cu64Zr36/N1024/seed_43/MACE_C/02_after_equilibrate_nvt.data
```

This writes total and partial RDF curves plus a preliminary first-shell summary. Automatically detected minima must be visually checked before manuscript use.

## Matrix generation

Complete the two unresolved starting densities in `revision/config/matrix.json`, source the A100 runtime environment, and then run:

```bash
python revision/scripts/generate_matrix.py --repo-root .
```

This generates run directories only; it does not execute LAMMPS.

## Tests

The standard-library tests are useful after code changes:

```bash
python -m unittest discover -s revision/tests -v
```

They do not replace the required physical A100 pilot.
