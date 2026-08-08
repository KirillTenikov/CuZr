from __future__ import annotations

import json
import sys
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper1_revision.config import Potential, load_protocol
from paper1_revision.lammps import RunSpec, generate_inputs
from paper1_revision.potentials import resolve_model_path
from paper1_revision.structure import composition_counts, estimate_box_length_A, parse_composition, write_initial_data
from paper1_revision.thermo import parse_thermo_tables, summarize_log


class StructureTests(unittest.TestCase):
    def test_parse_composition(self):
        self.assertEqual(parse_composition("Cu64Zr36"), (64, 36))
        with self.assertRaises(ValueError):
            parse_composition("Cu64Zr35")

    def test_composition_counts(self):
        n_cu, n_zr = composition_counts(1024, "Cu64Zr36")
        self.assertEqual(n_cu + n_zr, 1024)
        self.assertEqual(n_cu, 655)

    def test_initial_structure_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.data"
            b = Path(tmp) / "b.data"
            meta_a = write_initial_data(a, 64, "Cu50Zr50", 7.0, 43)
            meta_b = write_initial_data(b, 64, "Cu50Zr50", 7.0, 43)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(meta_a, meta_b)


class ProtocolTests(unittest.TestCase):
    def test_protocol_quench_duration(self):
        protocol = load_protocol(ROOT / "config" / "protocol.json")
        self.assertAlmostEqual(protocol.quench_ps, 20.0)
        self.assertEqual(protocol.steps(protocol.quench_ps), 20000)


class InputGenerationTests(unittest.TestCase):
    @staticmethod
    def _setup_case(tmp: str) -> tuple[Path, Path, Potential, RunSpec]:
        repo = Path(tmp) / "repo"
        run_dir = repo / "revision" / "results" / "pilot"
        model = repo / "models" / "raw" / "mace_C.model-mliap_lammps.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"dummy")
        potential = Potential(
            id="MACE_C",
            family="MACE",
            path="models/raw/mace_C.model-mliap_lammps.pt",
            enabled=True,
            lmp_command="lmp",
        )
        spec = RunSpec("Cu64Zr36", 1024, 43, 7.2, "MACE_C", run_dir)
        return repo, run_dir, potential, spec

    def test_paper2_derived_stage_lengths_and_nve_opt_in(self):
        protocol = load_protocol(ROOT / "config" / "protocol.json")
        with tempfile.TemporaryDirectory() as tmp:
            repo, run_dir, potential, spec = self._setup_case(tmp)
            files = generate_inputs(spec, protocol, potential, repo, include_box_relax=True, include_nve=False)
            self.assertNotIn("05_nve_stability.in", files)
            prepare = (run_dir / "00_prepare_melt_quench.in").read_text()
            self.assertIn("run 20000", prepare)
            relax = (run_dir / "01_relax_npt.in").read_text()
            self.assertIn("run 50000", relax)
            equilibrate = (run_dir / "02_equilibrate_nvt.in").read_text()
            self.assertIn("run 50000", equilibrate)

    def test_periodic_checkpoints_are_opt_in_and_stage_scoped(self):
        protocol = load_protocol(ROOT / "config" / "protocol.json")
        with tempfile.TemporaryDirectory() as tmp:
            repo, run_dir, potential, spec = self._setup_case(tmp)
            generate_inputs(
                spec,
                protocol,
                potential,
                repo,
                include_box_relax=True,
                include_nve=False,
                checkpoint_every_steps=5000,
            )
            self.assertTrue((run_dir / "checkpoints").is_dir())
            prepare = (run_dir / "00_prepare_melt_quench.in").read_text()
            relax = (run_dir / "01_relax_npt.in").read_text()
            equilibrate = (run_dir / "02_equilibrate_nvt.in").read_text()
            inherent = (run_dir / "03_inherent_fixed_cell.in").read_text()
            self.assertIn("restart 5000 checkpoints/00.restart.*", prepare)
            self.assertIn("restart 5000 checkpoints/01.restart.*", relax)
            self.assertIn("restart 5000 checkpoints/02.restart.*", equilibrate)
            self.assertNotIn("restart 5000", inherent)

    def test_negative_checkpoint_interval_is_rejected(self):
        protocol = load_protocol(ROOT / "config" / "protocol.json")
        with tempfile.TemporaryDirectory() as tmp:
            repo, _, potential, spec = self._setup_case(tmp)
            with self.assertRaises(ValueError):
                generate_inputs(spec, protocol, potential, repo, checkpoint_every_steps=-1)


class PotentialPathTests(unittest.TestCase):
    def test_runtime_environment_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            model = Path(tmp) / "models" / "mace_C.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"dummy")
            with patch.dict(os.environ, {"MACE_C_MLIAP": str(model)}, clear=False):
                self.assertEqual(resolve_model_path(repo, "${MACE_C_MLIAP}"), model.resolve())

    def test_missing_runtime_environment_variable_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                resolve_model_path(Path("/tmp/repo"), "${MACE_C_MLIAP}")


class ThermoTests(unittest.TestCase):
    def test_log_parser_uses_last_table(self):
        text = """LAMMPS
Step Time Temp Press Density
0 0 300 100 7.0
10 0.01 301 50 7.1
Loop time
Step Time Temp Press Density
0 0 300 10 7.2
10 0.01 300 0 7.3
Loop time
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.log"
            path.write_text(text)
            tables = parse_thermo_tables(path)
            self.assertEqual(len(tables), 2)
            summary = summarize_log(path, 1.0)
            self.assertEqual(summary["columns"]["Press"]["final"], 0.0)
            self.assertEqual(summary["columns"]["Density"]["tail_mean"], 7.25)


if __name__ == "__main__":
    unittest.main()
