# Revision scope

## Scientific purpose

The revised Paper 1 should no longer read as only an internal choice of a potential. Its scientific purpose is a controlled comparison of EAM, linear ACE and nonlinear equivariant MACE models for crystalline and amorphous Cu–Zr, with emphasis on the target domain of small-strain mechanical molecular dynamics at 300 K.

The revision tests whether lower DFT force errors, especially for MACE, translate into more reliable structural, thermodynamic and mechanical predictions.

## What the revision adds

### Training and DFT analysis

- MACE training/validation learning curves;
- ACE convergence information where available;
- DFT errors resolved by composition and structural state where metadata permit;
- stress errors for configurations carrying DFT stress labels;
- error distributions and high-percentile errors, not only a global RMSE.

### Improved glass preparation

The new protocol is adapted from the proven pressure-relaxed Paper 2 workflow:

1. initial coordinate minimization;
2. 20 ps NVT melt at 3000 K;
3. 20 ps NVT cooling from 3000 K to 300 K;
4. 50 ps NPT relaxation at 300 K and 0 bar;
5. 50 ps NVT equilibration at 300 K;
6. separately labelled inherent-structure calculations.

### Statistical amorphous validation

Target matrix:

- three compositions;
- three independent seeds;
- eight potentials;
- initially 1024 atoms.

The paired design uses the same composition, atom count, seed and initial structure across potentials. Each potential then evolves independently under the same physical protocol.

### Structural validation

- total RDF;
- Cu–Cu, Cu–Zr and Zr–Zr partial RDFs;
- nearest-neighbour peak positions;
- first-shell coordination numbers;
- experimentally weighted structure factor, once implemented;
- compact local-order/Voronoi analysis.

### Size, stability and mechanics

- explicit 1024-versus-4000 finite-size comparison;
- short production-like NVE stability checks;
- elastic screening of all eight potentials;
- detailed amorphous elastic statistics for the leading two or three models;
- optional crystalline elastic constants if the workflow remains inexpensive.

### Compact additions inspired by Leimeroth et al.

- B2 CuZr formation energy;
- compact icosahedral/local-order analysis;
- stronger experimental structural comparison;
- possibly crystalline elastic constants.

The Leimeroth potential itself is not added, because our two ACE models were trained on the same underlying data and already provide the controlled ACE–MACE comparison.

## What the revision does not do

The following are deliberately outside scope:

- retraining the six MLIPs;
- replacing or cleaning old scripts;
- a new DFT campaign;
- viscosity calculations;
- a diffusion matrix;
- glass-transition-temperature determination;
- melting curves;
- a phase diagram;
- full reproduction of Leimeroth et al.'s general-purpose validation programme.

These would require separate property-specific projects and would obscure the central comparative question.

## Model-selection principle

`MACE_C` is not assumed to be universally best. The final wording should be that it is selected by the aggregate of the considered criteria, provided the revised results support that conclusion.

Mechanical validation is therefore staged:

1. inexpensive screening for all eight potentials;
2. statistically detailed validation for the leading models selected after DFT, pressure, structure and preliminary elasticity are considered together.
