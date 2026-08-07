# Paper 1 revision documentation

This directory is the navigation point for the additive Paper 1 revision workflow.

## Reading order

1. [`CURRENT_STATE.md`](CURRENT_STATE.md) — what already exists in the original project and what is frozen.
2. [`REVISION_SCOPE.md`](REVISION_SCOPE.md) — what the revision is intended to add, and what is deliberately excluded.
3. [`A100_RUNTIME.md`](A100_RUNTIME.md) — the actual Docker/Python/MACE/LAMMPS environment and the A100 startup sequence.
4. [`WORKFLOW.md`](WORKFLOW.md) — preparation stages, paired-design rules, run directories and outputs.
5. [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) — implemented, planned and excluded validation blocks.
6. [`RUN_RECORD_TEMPLATE.md`](RUN_RECORD_TEMPLATE.md) — a small record to fill in for each A100 session or important pilot.

## Governing safety rule

The existing training scripts, model-development files and submitted-paper validation scripts are historical scientific records. They are not edited, moved or refactored during the revision. All new code and documentation live under `revision/`, and all new numerical outputs live under `revision/results/` or an explicitly selected external results directory.

## Current software version

This documentation describes revision workflow **v0.2.0**. The first A100 pilot has not yet been run; therefore, generated LAMMPS inputs have been syntax-inspected and Python-tested, but not yet physically validated on the production runtime.
