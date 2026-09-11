# Paper 1: 72-glass ensemble analysis

Reproducible analysis for the balanced matrix

`8 potentials × 3 Cu-Zr compositions × 3 seeds = 72 N=1024 glasses`.

The scripts read the `paper1_*.tar*.gz` archives directly; bulky restart files are never extracted.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
WORKERS=2 ./run_all.sh /path/to/archives analysis_72
```

## Pipeline

1. `01_inventory_qc.py` — matrix audit, density/volume/pressure, stage-03→04 branch-continuity check, minimization-log QC.
2. `02_state_statistics.py` — mean±sample-SD state summaries and descriptive sums-of-squares decomposition.
3. `03_pair_structure.py` — total/partial RDFs, composition-specific common first-shell cutoffs, coordination numbers, chemical-order parameters, and `S_NN(q)`.
4. `04_voronoi.py` — periodic unweighted geometric Voronoi topology, exact volume-closure QC, exact `<0,0,12,0>` fraction, pentagonal-face fraction, and dominant Voronoi indices.
5. `05_descriptor_statistics.py` — unified 72-row descriptor table, potential/composition/interaction/seed descriptive variance decomposition, seed reproducibility.
6. `06_make_figures.py` — diagnostic/publication-prototype figures.

## Frozen methodological choices

- Structural state: `04_inherent_box_relaxed.data`.
- Stage-03→04 continuity is measured in fractional coordinates so affine box contraction is removed.
- RDF: `dr = 0.02 Å`, `r_max = 10 Å`, exact finite-N ideal-gas normalization.
- Coordination cutoffs: one cutoff for each **composition × pair type**, derived objectively from the RDF averaged over all 8 potentials and 3 seeds. The same cutoff is then used for every potential at that composition.
- Static structure factor: number-number `S_NN(q)` from the total RDF, Lorch termination window, `q=0.5…15 Å^-1` in `0.02 Å^-1` increments. This is **not** an X-ray-weighted experimental structure factor.
- Voronoi: periodic **unweighted geometric** tessellation. A 6 Å image buffer is used for speed and is accepted only if the sum of central Voronoi volumes closes to the simulation-box volume. No empirical atomic radii and no small-face threshold are used.
- Exact icosahedron: raw Voronoi index `<0,0,12,0>` with exactly 12 faces.
- Variance decompositions are descriptive sums-of-squares partitions, **not inferential ANOVA**.

## Important interpretation caveats

Rare exact Voronoi motifs have appreciably larger realization noise than smooth observables such as density, coordination, or `S(q)`. Do not compare these unweighted Voronoi fractions directly with literature using radical/weighted Voronoi without matching definitions.
