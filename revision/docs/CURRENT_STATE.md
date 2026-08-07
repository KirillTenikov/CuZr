# Current project state

Last documentation update: 2026-08-02.

## 1. Original Paper 1 project

The repository contains the original model-training, conversion and validation work used for the submitted Paper 1. Its central comparison is:

- four MACE models: `MACE_A`, `MACE_B`, `MACE_C`, `MACE_D`;
- two in-house ACE models: `ACE_514`, `ACE_1352`;
- two EAM references: Mendelev 2007 and Mendelev 2019.

The existing repository also contains Docker/runtime files, original validation scripts, model configuration files and utility source code.

## 2. Frozen material

During the revision, the following are treated as read-only:

- all training scripts and training configurations;
- the trained model files;
- old conversion scripts;
- old validation scripts used for the submitted manuscript;
- old numerical results and figure-generation paths.

They may be read and compared against the new workflow, but they are not edited. This preserves the ability to reconstruct the submitted-paper state.

## 3. Validation already represented in the submitted work

The original project already contains or supports:

- global energy and force errors against a held-out DFT test set;
- FCC Cu, HCP Zr and B2 CuZr crystalline checks;
- equations of state and bulk properties;
- vacancy formation energies;
- basic amorphous preparation for three compositions;
- density, residual pressure and total RDF analysis;
- technical LAMMPS/MACE execution checks.

The principal limitations motivating the revision are that the amorphous block used only one 1024-atom realization per composition, structural validation was mostly based on the total RDF, and the submitted workflow was not yet aligned with the longer pressure-relaxed preparation used for production MD-DMS.

## 4. Existing runtime infrastructure

The authoritative runtime files already present in the repository are:

- `docker/Dockerfile.cuzr_md` — CUDA/Python/PyTorch/MACE build environment;
- `docker/startup_md_lammps.sh` — runtime LAMMPS build, potential download/conversion and environment-file generation.

The Docker image includes the Python/MACE stack and LAMMPS source, but LAMMPS itself is compiled after the cloud GPU instance starts. See [`A100_RUNTIME.md`](A100_RUNTIME.md).

## 5. New additive revision code

The new `revision/` directory currently provides:

- a Paper 2-derived pressure-relaxed glass-preparation workflow;
- deterministic paired initial structures;
- support for four MACE, two ACE and two EAM potentials through runtime environment variables;
- stage-separated LAMMPS input generation;
- fixed-cell and box-relaxed inherent-structure stages;
- optional short NVE stability testing;
- thermo-log aggregation;
- total and partial RDF analysis;
- matrix generation without automatic execution;
- run manifests and hashes for provenance;
- documentation and a validation roadmap.

## 6. What has not yet been physically tested

The following remain pending until the A100 instance is started:

- building the current LAMMPS source on the actual A100 machine;
- confirming ML-IAP/Kokkos execution for each MACE model;
- confirming `pair_style pace` execution for both ACE models;
- running the first 1024-atom `MACE_C`, Cu64Zr36, seed 43 pilot;
- checking pressure, density, temperature and timing across all stages;
- confirming that the generated files and summaries are scientifically sensible.

No full validation matrix should be launched before this pilot is inspected.

## 7. Remaining configuration decisions

Before matrix generation, we still need to freeze:

- initial densities for Cu50Zr50 and Cu36Zr64;
- whether the box-relaxed inherent structure is retained for every production run or only selected checks;
- the exact finalists and number of realizations for detailed elastic validation;
- experimental datasets and scattering weights for the final structure-factor comparison.
