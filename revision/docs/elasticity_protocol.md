# Paper 1 — Amorphous elasticity validation protocol

## Purpose

This workflow adds a zero-temperature, relaxed-ion mechanical validation of the
Cu–Zr interatomic potentials used in Paper 1.  It is designed for already
prepared amorphous configurations and therefore does **not** repeat the
melt–quench–NPT–NVT preparation campaign.

The primary comparison is intended to use the independently prepared
Cu64Zr36 glasses for each potential and seed.

## Starting structure

Use the fully relaxed inherent structure:

```text
04_inherent_box_relaxed.data
```

Each strain point starts independently from this same unstrained source
structure.  Strains are never accumulated sequentially.

## Strain amplitudes

Default non-zero amplitudes:

```text
-0.005
-0.0025
+0.0025
+0.005
```

Together with the unstrained reference, this gives five points per deformation
mode.  The two magnitudes provide a direct check that the response remains in
the linear elastic regime.

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

All atoms are remapped affinely when the cell is deformed.  The cell is then
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

There is deliberately **no** `fix box/relax` after a strain is imposed.  Using
box relaxation at that stage would remove the deformation whose stress response
is being measured.

## Stress definition

The workflow records

```text
compute pvir all pressure NULL virial
```

so the kinetic contribution is excluded.  This avoids the residual-velocity
pressure issue encountered in some minimized structures.

LAMMPS reports a pressure tensor with the opposite sign to the conventional
Cauchy stress used in Hooke's law.  Therefore the stress-derived moduli are
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

The runner records the source-file SHA-256 hash and, when available, the Git
commit hash in `protocol.json`.

## Recommended first smoke test

Do **not** launch the full campaign immediately.

1. Allow the current production queue to finish.
2. Generate the MACE_D / Cu64Zr36 / seed 42 inputs without `--execute`.
3. Inspect one bulk and one shear input.
4. Execute the single-seed calculation.
5. Run `analyze_elasticity.py`.
6. Check:
   - minimization convergence;
   - residual reference virial pressure;
   - signs of slopes;
   - stress-strain linearity;
   - directional shear spread;
   - agreement with energy-curvature estimates.
7. Only then launch seeds 43 and 44 and later the other potentials.

## Example: prepare MACE_D seed 42

From the repository root:

```bash
python revision/scripts/run_elasticity.py \
  --data revision/results/Cu64Zr36/N1024/seed_42/MACE_D/04_inherent_box_relaxed.data \
  --out revision/results_elasticity/Cu64Zr36/N1024/seed_42/MACE_D \
  --label MACE_D \
  --pair-style "mliap unified /workspace/models/raw/mace_D.model-mliap_lammps.pt 0"
```

After inspecting the generated inputs, add `--execute` to actually run them.

Analyze with:

```bash
python revision/scripts/analyze_elasticity.py \
  revision/results_elasticity/Cu64Zr36/N1024/seed_42/MACE_D
```
