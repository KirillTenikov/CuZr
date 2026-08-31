# Paper 1 — Amorphous elasticity validation protocol

## Purpose

This workflow adds a zero-temperature, relaxed-ion mechanical validation of the
Cu–Zr interatomic potentials used in Paper 1. It is designed for already
prepared amorphous configurations and therefore does **not** repeat the
melt–quench–NPT–NVT preparation campaign.

The primary comparison is intended to use the independently prepared
Cu64Zr36 glasses for each potential and seed.

## Starting structure and canonical reference

The supplied source is the fully relaxed inherent structure:

```text
04_inherent_box_relaxed.data
```

The elasticity workflow first performs one additional zero-strain minimization
using exactly the same minimization settings that will be used for the strained
states. The result is written to:

```text
reference/zero/relaxed.data
```

This file is the **canonical zero-strain reference**.

Every non-zero strain point starts independently from this same canonical
reference. Strains are never accumulated sequentially. This ensures that the
finite-difference points are perturbations of one common inherent structure
rather than independent minimizations from a less tightly minimized source,
which could select neighboring inherent-structure basins.

## Strain amplitudes

Default non-zero amplitudes:

```text
-0.00125
-0.000625
+0.000625
+0.00125
```

Together with the unstrained reference, this gives five points per deformation
mode. The smaller strain window is chosen to estimate the derivative at zero
strain while reducing the probability of leaving the reference
inherent-structure basin.

## Deformation modes

### Bulk

`strain` is the volumetric strain

\[
\epsilon_v = \frac{\Delta V}{V_0}.
\]

The three cell lengths are scaled equally by

\[
s=(1+\epsilon_v)^{1/3}.
\]

After deformation, the cell is held fixed and only the atomic coordinates are
minimized.

### Shear

Three independent engineering shears are used:

\[
\gamma_{xy},\qquad \gamma_{xz},\qquad \gamma_{yz}.
\]

They are imposed through the triclinic tilt factors:

```text
gamma_xy = Delta(xy) / Ly
gamma_xz = Delta(xz) / Lz
gamma_yz = Delta(yz) / Lz
```

All atoms are remapped affinely when the cell is deformed. The cell is then
held fixed while the internal coordinates relax.

## Relaxation

Default minimization:

```text
min_style cg
etol    = 1e-12
ftol    = 1e-8
maxiter = 10000
maxeval = 100000
```

The same minimization settings are used for the canonical zero-strain reference
and every strained state.

There is deliberately **no** `fix box/relax` after a strain is imposed. Using
box relaxation at that stage would remove the deformation whose stress response
is being measured.

## Stress definition

The workflow records

```text
compute pvir all pressure NULL virial
```

so the kinetic contribution is excluded. This avoids the residual-velocity
pressure issue encountered in some minimized structures.

LAMMPS reports a pressure tensor with the opposite sign to the conventional
Cauchy stress used in Hooke's law. Therefore the stress-derived moduli are
computed as

\[
K=-\frac{dP_\mathrm{mean}}{d\epsilon_v},
\]

and

\[
G_{xy}=-\frac{dP_{xy}}{d\gamma_{xy}},
\quad
G_{xz}=-\frac{dP_{xz}}{d\gamma_{xz}},
\quad
G_{yz}=-\frac{dP_{yz}}{d\gamma_{yz}}.
\]

For a finite amorphous sample the three shear directions need not be identical.
The isotropic estimate is

\[
G=\frac{G_{xy}+G_{xz}+G_{yz}}{3},
\]

while their directional spread is retained as a diagnostic.

## Derived isotropic elastic constants

Using the stress-derived bulk modulus \(K\) and averaged shear modulus \(G\),

\[
E=\frac{9KG}{3K+G},
\]

\[
\nu=\frac{3K-2G}{2(3K+G)}.
\]

## Energy-curvature cross-check

The analyzer independently fits the minimized potential energy versus strain.

For small volumetric strain,

\[
\Delta E \simeq \frac{1}{2} K V_0 \epsilon_v^2,
\]

and for engineering shear,

\[
\Delta E \simeq \frac{1}{2} G V_0 \gamma^2.
\]

Agreement between stress slopes and energy curvatures is used as an internal
consistency check, not as a substitute for inspecting linearity.

For the final analysis, the stress–strain linearity, energy-curvature agreement,
and continuity of the relaxed configurations should be checked together. No
strain point should be excluded solely because it changes the fitted modulus.

## Output structure

For one potential/seed:

```text
elasticity_root/
├── protocol.json
├── reference/
│   └── zero/
│       ├── in.elastic
│       ├── stdout.txt
│       ├── log.lammps
│       ├── relaxed.data
│       └── result.json
├── bulk/
├── xy/
├── xz/
├── yz/
├── elasticity_points.csv
└── elasticity_summary.json
```

The runner records the source-file SHA-256 hash, the canonical-reference path,
and, when available, the Git commit hash in `protocol.json`.

## Important rerun rule

Use a **new output root** for the revised elasticity campaign. Do not mix the
new canonical-reference calculations with the earlier campaign in the same
directory.

The earlier archive should be preserved as provenance and as a diagnostic
benchmark.

## Recommended first smoke test

Do **not** launch the full campaign immediately.

1. Generate one potential/seed input set without `--execute`.
2. Inspect `reference/zero/in.elastic`.
3. Inspect one bulk and one shear input and confirm that their `read_data`
   points to `reference/zero/relaxed.data`, not directly to the original
   `04_inherent_box_relaxed.data`.
4. Execute the single-seed calculation in a fresh output directory.
5. Run `analyze_elasticity.py`.
6. Check:
   - minimization convergence;
   - residual reference virial pressure;
   - signs of slopes;
   - stress-strain linearity;
   - directional shear spread;
   - agreement with energy-curvature estimates.
7. Only then launch the remaining seeds and potentials.

## Example: prepare MACE_D seed 42

From the repository root:

```bash
python revision/scripts/run_elasticity.py \
  --data revision/results/Cu64Zr36/N1024/seed_42/MACE_D/04_inherent_box_relaxed.data \
  --out revision/results_elasticity_v2/Cu64Zr36/N1024/seed_42/MACE_D \
  --label MACE_D \
  --pair-style "mliap unified /workspace/models/raw/mace_D.model-mliap_lammps.pt 0"
```

After inspecting the generated inputs, execute the same command in a fresh
output directory with `--execute`.

Analyze with:

```bash
python revision/scripts/analyze_elasticity.py \
  revision/results_elasticity_v2/Cu64Zr36/N1024/seed_42/MACE_D
```
