# Paper 1 revision validation plan

This directory is additive. Existing training, conversion and submitted-paper validation files remain untouched.

The maintained documentation index is [`docs/README.md`](docs/README.md). The concise status table is [`docs/VALIDATION_STATUS.md`](docs/VALIDATION_STATUS.md).

## Implemented in v0.2

- Paper 2-derived glass preparation protocol:
  - initial coordinate minimization;
  - 20 ps NVT melt at 3000 K;
  - 20 ps NVT cooling to 300 K;
  - 50 ps NPT relaxation at 300 K and 0 bar;
  - 50 ps NVT equilibration at 300 K.
- Deterministic paired initial structures by composition, size and seed.
- Runtime-environment path resolution for four MACE, two ACE and two EAM models.
- Separate fixed-cell and box-relaxed inherent structures.
- Optional short NVT + NVE stability check.
- Stage-resolved pressure, temperature, energy, density, volume and stress-component summaries.
- Total and partial RDFs, preliminary first-peak/first-minimum estimates and first-shell coordination.
- Matrix generator that creates run folders without launching simulations.
- Provenance manifest with Git state, model hash, initial-data hash and resolved protocol.
- A100 runtime, workflow, status and run-record documentation.

## Next implementation blocks

1. Run and inspect the first A100 MACE_C pilot.
2. Learning-curve extraction for MACE and ACE.
3. Held-out DFT error analysis by structural state, composition and stress-label availability.
4. B2 CuZr formation energy.
5. Elastic screening of all eight potentials, followed by detailed statistics for finalists.
6. Explicit 1024-versus-4000 finite-size comparison.
7. Experimentally weighted structure factor S(q).
8. Compact local-order/Voronoi analysis.
9. Aggregation across 3 compositions × 3 realizations × 8 potentials.

The excluded calculations remain viscosity, diffusion campaigns, Tg, melting curves, a phase diagram, new DFT calculations and model retraining.
