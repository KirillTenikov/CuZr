# Paper 1 — detailed DFT validation

This folder reproduces the **additional DFT-validation block introduced during
the Paper 1 revision**.

The analysis deliberately separates two questions:

1. **Global held-out DFT accuracy** — the existing comparison on the complete
   untouched `test.extxyz` set (4242 structures; 501108 atoms), reported in
   the manuscript for ACE514, ACE1352 and MACE_A–D.
2. **Application-relevant detailed diagnostics** — new per-structure and
   composition-resolved errors for MACE_A–D on the three composition regions
   used later for the metallic-glass calculations.

The second analysis is **not** presented as a phase-resolved test. The extxyz
file does not contain reliable liquid/amorphous/crystal labels, and this
workflow does not infer such labels from structural heuristics.

## Frozen detailed subset

The detailed analysis uses N=128 test structures with:

| n_Cu | x_Cu | Paper label | structures |
|---:|---:|---|---:|
| 46 | 0.359375 | Cu36Zr64 | 42 |
| 64 | 0.500000 | Cu50Zr50 | 43 |
| 82 | 0.640625 | Cu64Zr36 | 42 |

Total: **127 independent held-out DFT structures**.

These are taken directly from the same untouched test set used for the global
validation; they were not used for training or hyperparameter selection.

## Error definitions

For structure `s` containing `N_s` atoms,

```text
dE_s/N_s = (E_ML - E_DFT) / N_s

epsilon_F,s =
sqrt[
  (1/(3 N_s)) *
  sum_(i,alpha) (F_ML(i,alpha) - F_DFT(i,alpha))^2
]
```

The composition-level energy RMSE is the RMSE of signed per-atom energy
errors across structures. The force RMSE is the pooled component RMSE,
reconstructed by weighting the squared per-structure force RMSE by `N_s`.
Because all frozen target structures contain 128 atoms, this is also
equivalent to an unweighted RMS over the structure-level squared force RMSEs.

The reported median/p90/p95 force diagnostics are percentiles of the
**per-structure** `epsilon_F,s` distribution.

## Scripts

- `01_inventory_testset.py`  
  Inventories the test set, exact compositions and target frames.
- `02_mace_target_inference.py`  
  Runs a compiled TorchScript MACE model on the 127 target structures and
  writes one row per DFT structure.
- `03_composition_statistics.py`  
  Produces composition and overall summaries, including median/p90/p95 tails.
- `04_training_convergence.py`  
  Extracts MACE and ACE validation histories from the original training logs.
- `05_make_figures.py`  
  Regenerates composition-resolved DFT plots and ACE/MACE convergence figures.
- `run_all.sh`  
  Convenience wrapper; does not download models or datasets.

## Important ACE/MACE scope distinction

The **global** 4242-structure held-out comparison in the paper includes:

```text
ACE514
ACE1352
MACE_A
MACE_B
MACE_C
MACE_D
```

The **new detailed 127-structure composition-resolved analysis** includes:

```text
MACE_A
MACE_B
MACE_C
MACE_D
```

Do not state or imply that ACE514/ACE1352 were rerun for the new detailed
127-structure block.

## Training convergence is not independent validation

The convergence curves use the validation data monitored during model
training. They show stable optimization / approach to a plateau.

They must not be described as held-out DFT-test performance.

For ACE/pacemaker specifically, the file commonly named `test_metrics.txt`
corresponds in this project to the validation split used during training
monitoring, **not** to the untouched 4242-structure external `test.extxyz`.

## Inputs

The workflow expects local paths to:

- the untouched `test.extxyz`;
- the four compiled TorchScript MACE models;
- optionally the four MACE training logs;
- optionally ACE514 / ACE1352 pacemaker `test_metrics.txt` histories.

Large models and datasets should stay in the project release/archive rather
than being committed to Git.

## Reproduction

Example:

```bash
export TEST_EXTXYZ=/path/to/test.extxyz

export MACE_A_MODEL=/path/to/mace_A_compiled.model
export MACE_B_MODEL=/path/to/mace_B_compiled.model
export MACE_C_MODEL=/path/to/mace_C_compiled.model
export MACE_D_MODEL=/path/to/mace_D_compiled.model

# Optional convergence inputs:
export MACE_A_LOG=/path/to/MACE_A.log
export MACE_B_LOG=/path/to/MACE_B.log
export MACE_C_LOG=/path/to/MACE_C.log
export MACE_D_LOG=/path/to/MACE_D.log
export ACE514_METRICS=/path/to/ACE514/test_metrics.txt
export ACE1352_METRICS=/path/to/ACE1352/test_metrics.txt

bash revision/scripts/dft_validation/run_all.sh
```

Default output tree:

```text
revision/results/dft_validation/
  inventory/
  per_structure/
  summary/
  convergence/
  figures/
```

## Frozen Paper 1 result pattern

For the 127-structure detailed subset, the revision analysis found:

- MACE_C gives the lowest **force RMSE** at each of the three target
  compositions.
- MACE_D gives the lowest **energy RMSE** at each of the three target
  compositions.
- Force errors decrease toward the Cu-rich target composition for all four
  MACE variants.
- The high-error force tail (p95 of per-structure force RMSE) is also smallest
  for MACE_C among the four detailed MACE candidates.

These observations complement, rather than replace, the global 4242-structure
held-out DFT comparison.

## Data policy

Commit code, configuration and documentation to Git.

Keep large raw inputs (DFT extxyz and model files) in GitHub Release / Zenodo.
Per-structure prediction tables can also be archived with a release if desired.
Small final manuscript-facing CSV summaries may be versioned deliberately, but
generated outputs are not required for the scripts themselves.
