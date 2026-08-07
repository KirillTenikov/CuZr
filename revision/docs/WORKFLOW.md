# Revision workflow and outputs

## 1. Additive design

All revision code is under `revision/`. No original training or submitted-paper file is modified.

Default output hierarchy:

```text
revision/results/
└── Cu64Zr36/
    └── N1024/
        └── seed_43/
            └── MACE_C/
```

A run directory must be empty before generation. The runner refuses to overwrite any existing content.

## 2. Preparation stages

| Stage | Input file | Main operation | Main output |
|---|---|---|---|
| 00 | `00_prepare_melt_quench.in` | initial minimization, 20 ps NVT melt, 20 ps NVT quench | `00_after_quench.data` |
| 01 | `01_relax_npt.in` | 50 ps NPT at 300 K and 0 bar | `01_after_relax_npt.data` |
| 02 | `02_equilibrate_nvt.in` | 50 ps NVT at 300 K | `02_after_equilibrate_nvt.data` |
| 03 | `03_inherent_fixed_cell.in` | coordinate minimization at retained finite-temperature cell | `03_inherent_fixed_cell.data` |
| 04 | `04_inherent_box_relaxed.in` | separate isotropic zero-pressure box relaxation | `04_inherent_box_relaxed.data` |
| 05 | `05_nve_stability.in` | optional 10 ps NVT plus 50 ps NVE stability check | `05_after_nve_stability.data` |

Stage 05 is opt-in and should not be generated for every matrix member automatically.

## 3. Paired-comparison rule

For a fixed composition, atom count and seed, all potentials should begin from the same `initial.data`. The physical protocol is identical for all models, but each potential evolves its own trajectory.

This distinguishes two effects cleanly:

- realization variability, controlled by the seed;
- potential dependence, evaluated from paired initial conditions.

## 4. Potential resolution

Potential paths are read from environment variables created by `docker/startup_md_lammps.sh`. The runner records both the configured expression and fully resolved path in `manifest.json`.

MACE runs use:

```text
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half
```

ACE and EAM runs currently use plain `lmp` unless an explicit command override is provided.

## 5. Provenance files

Every generated run contains:

- `manifest.json` with workflow version, protocol, Git state, potential path and hashes;
- `initial.data` and its hash;
- generated LAMMPS input files;
- `run_lammps.sh`;
- one LAMMPS log per executed stage;
- final data/restart files from completed stages;
- `thermo_summary.json` and `thermo_summary.csv` after summarization.

## 6. Thermodynamic quantities

The summarizer extracts final values and tail statistics from the last thermo table in each stage log. Relevant columns include:

- temperature;
- total, kinetic and potential energy;
- pressure and stress components;
- density and volume;
- box lengths.

For manuscript analysis, finite-temperature tail means and fluctuations must be kept separate from minimized inherent-structure values.

## 7. Structural analysis

`analyze_structure.py` currently computes:

- total RDF;
- Cu–Cu partial RDF;
- Cu–Zr partial RDF;
- Zr–Zr partial RDF;
- heuristic first-peak and first-minimum locations;
- preliminary first-shell coordination numbers.

Automatically detected minima are provisional and must be visually checked before final integration limits are accepted.

## 8. Matrix launch sequence

The intended order is:

1. one MACE_C pilot;
2. one composition/seed across all eight potentials;
3. three seeds across all potentials for the first composition;
4. full three-composition matrix;
5. selected 4000-atom, NVE and detailed elasticity runs.

Do not jump directly to the full matrix. Each expansion should occur only after the previous level is inspected.
