# Cu-Zr training scripts scaffold

This bundle contains a clean scaffold for:
- `scripts/train_mace.py`
- `scripts/train_ace.py`
- reusable helpers in `src/`
- example YAML configs in `configs/`

## Run examples

```bash
python scripts/train_mace.py --config configs/mace_A.yaml
python scripts/train_ace.py --config configs/ace_1352.yaml
```

## Notes

- `train_mace.py` is close to usable now, but CLI options may need slight adjustment to match your installed MACE version.
- `train_ace.py` is a stable wrapper around a placeholder ACE backend. Replace `external/train_ace_backend.py` and/or `src/ace_runner.py` when you finalize the ACE training backend.
- Update dataset paths in the YAML files before running.
