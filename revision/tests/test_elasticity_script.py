from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_elasticity.py"

spec = importlib.util.spec_from_file_location("run_elasticity_script", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load elasticity runner: {SCRIPT}")
run_elasticity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_elasticity)


class ElasticityRunnerTests(unittest.TestCase):
    def test_default_strains_are_small_and_symmetric(self):
        self.assertEqual(
            run_elasticity.DEFAULT_STRAINS,
            (-0.00125, -0.000625, 0.000625, 0.00125),
        )

    def test_reference_uses_source_and_strains_use_canonical_reference(self):
        source = Path("/tmp/04_inherent_box_relaxed.data")
        canonical = Path("/tmp/reference/zero/relaxed.data")

        self.assertEqual(
            run_elasticity.input_data_for_mode("reference", source, canonical),
            source,
        )
        for mode in ("bulk", "xy", "xz", "yz"):
            self.assertEqual(
                run_elasticity.input_data_for_mode(mode, source, canonical),
                canonical,
            )

    def test_strained_input_reads_canonical_reference(self):
        canonical = Path("/tmp/reference/zero/relaxed.data")
        text = run_elasticity.make_input(
            data_file=canonical,
            pair_style="eam/alloy",
            pair_coeff="* * model.eam.alloy Cu Zr",
            mode="xy",
            strain=0.000625,
            etol=1.0e-12,
            ftol=1.0e-8,
            maxiter=10000,
            maxeval=100000,
        )
        self.assertIn(f'read_data "{canonical}"', text)
        self.assertIn("change_box all xy delta ${dtilt}", text)


if __name__ == "__main__":
    unittest.main()
