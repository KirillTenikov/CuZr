from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GlassPreparationConfig:
    composition: str = "Cu64.5Zr35.5"
    n_atoms: int = 4000
    box_length_angstrom: float | None = None
    timestep_fs: float = 1.0
    melt_temperature_k: float = 3000.0
    melt_time_ps: float = 1000.0
    quench_target_temperature_k: float = 300.0
    quench_rate_k_per_ps: float = 1.0
    equilibration_temperature_k: float = 300.0
    equilibration_pressure_bar: float = 0.0
    equilibration_time_ps: float = 100.0
    random_seed: int = 42


@dataclass(slots=True)
class MDDMSConfig:
    strain_amplitude: float = 0.01
    period_ps: float = 50.0
    n_cycles: int = 30
    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    pressure_bar: float = 0.0
    dump_interval_steps: int = 10
    thermo_interval_steps: int = 10
    stress_sample_interval_steps: int = 1
    atom_stress_interval_steps: int = 1
    block_size_ps: float = 100.0
    lossy_fraction: float = 0.05
    lossy_target_phase_rad: float = math.pi / 2.0
    lossy_phase_half_width_rad: float = math.pi / 8.0

    @property
    def total_time_ps(self) -> float:
        return self.period_ps * self.n_cycles

    @property
    def omega_per_ps(self) -> float:
        return 2.0 * math.pi / self.period_ps

    @property
    def drive_frequency_thz(self) -> float:
        # 1 / ps = 1 THz
        return 1.0 / self.period_ps


@dataclass(slots=True)
class AnalysisConfig:
    stress_component: str = "xy"
    unwrap_phase: bool = False
    use_window_function: bool = False
    min_points_per_block: int = 50
    save_intermediate_arrays: bool = True


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    structures: Path = field(init=False)
    trajectories: Path = field(init=False)
    stress: Path = field(init=False)
    analysis: Path = field(init=False)
    figures: Path = field(init=False)
    metadata: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.structures = self.root / "structures"
        self.trajectories = self.root / "trajectories"
        self.stress = self.root / "stress"
        self.analysis = self.root / "analysis"
        self.figures = self.root / "figures"
        self.metadata = self.root / "metadata"

    def mkdirs(self) -> None:
        for path in [
            self.root,
            self.structures,
            self.trajectories,
            self.stress,
            self.analysis,
            self.figures,
            self.metadata,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class BlockResult:
    block_index: int
    t_start_ps: float
    t_end_ps: float
    system_phase_rad: float
    system_amplitude: float
    n_atoms: int
    lossy_atom_ids: np.ndarray
    atomic_phase_rad: np.ndarray
    atomic_amplitude: np.ndarray


class Paper2MDDMSWorkflow:
    def __init__(
        self,
        paths: ProjectPaths,
        prep: GlassPreparationConfig,
        md: MDDMSConfig,
        analysis: AnalysisConfig,
    ) -> None:
        self.paths = paths
        self.prep = prep
        self.md = md
        self.analysis = analysis
        self.paths.mkdirs()

    # ---------------------------------------------------------------------
    # metadata and bookkeeping
    # ---------------------------------------------------------------------
    def write_metadata(self) -> None:
        payload = {
            "glass_preparation": asdict(self.prep),
            "md_dms": asdict(self.md),
            "analysis": asdict(self.analysis),
        }
        out = self.paths.metadata / "paper2_mdms_config.json"
        out.write_text(json.dumps(payload, indent=2))

    # ---------------------------------------------------------------------
    # stage 1: structure preparation
    # ---------------------------------------------------------------------
    def build_initial_structure(self) -> Any:
        """
        Return an initial structure object.

        Expected implementation choices:
        - pyiron_atomistics structure builder
        - ASE Atoms object
        - pre-generated amorphous starting structure from a file
        """
        raise NotImplementedError("Connect this to your preferred structure builder.")

    def melt_quench_equilibrate(self, structure: Any) -> Path:
        """
        Run the glass preparation protocol and save the equilibrated structure.

        Suggested outputs:
        - final structure file for the equilibrated glass
        - thermo log from melt/quench/equilibration
        - optional RDF snapshot for validation
        """
        raise NotImplementedError("Connect this to pyiron/LAMMPS melt-quench workflow.")

    # ---------------------------------------------------------------------
    # stage 2: MD-DMS production run
    # ---------------------------------------------------------------------
    def generate_strain_series(self) -> pd.DataFrame:
        dt_ps = self.md.timestep_fs * 1e-3
        n_steps = int(round(self.md.total_time_ps / dt_ps))
        step = np.arange(n_steps + 1, dtype=int)
        time_ps = step * dt_ps
        strain = self.md.strain_amplitude * np.sin(self.md.omega_per_ps * time_ps)
        return pd.DataFrame({"step": step, "time_ps": time_ps, "strain_xy": strain})

    def run_md_dms(self, structure_path: Path) -> dict[str, Path]:
        """
        Run sinusoidal shear MD and write raw outputs.

        Expected raw outputs:
        - system stress time series
        - atomic stress time series
        - atomic positions or selected snapshots
        - optional velocities for later diagnostics

        The exact pyiron/LAMMPS implementation is intentionally left open here.
        """
        raise NotImplementedError("Connect this to your LAMMPS sinusoidal shear driver.")

    # ---------------------------------------------------------------------
    # stage 3: loading raw outputs
    # ---------------------------------------------------------------------
    def load_system_stress(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"time_ps", "sigma_xy"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required system-stress columns: {sorted(missing)}")
        return df.sort_values("time_ps").reset_index(drop=True)

    def load_atomic_stress(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        required = {"time_ps", "atom_id", "sigma_xy"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required atomic-stress columns: {sorted(missing)}")
        return df.sort_values(["time_ps", "atom_id"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # stage 4: block-wise Fourier analysis
    # ---------------------------------------------------------------------
    def block_edges_ps(self) -> np.ndarray:
        n_blocks = int(round(self.md.total_time_ps / self.md.block_size_ps))
        return np.linspace(0.0, self.md.total_time_ps, n_blocks + 1)

    def split_system_into_blocks(self, system_df: pd.DataFrame) -> list[pd.DataFrame]:
        edges = self.block_edges_ps()
        blocks: list[pd.DataFrame] = []
        for start, end in zip(edges[:-1], edges[1:]):
            mask = (system_df["time_ps"] >= start) & (system_df["time_ps"] < end)
            block = system_df.loc[mask].copy()
            if len(block) < self.analysis.min_points_per_block:
                raise ValueError(f"System block [{start}, {end}) has too few samples: {len(block)}")
            blocks.append(block)
        return blocks

    def split_atomic_into_blocks(self, atomic_df: pd.DataFrame) -> list[pd.DataFrame]:
        edges = self.block_edges_ps()
        blocks: list[pd.DataFrame] = []
        for start, end in zip(edges[:-1], edges[1:]):
            mask = (atomic_df["time_ps"] >= start) & (atomic_df["time_ps"] < end)
            block = atomic_df.loc[mask].copy()
            if len(block) == 0:
                raise ValueError(f"Atomic block [{start}, {end}) is empty.")
            blocks.append(block)
        return blocks

    def _fourier_component_at_drive(self, time_ps: np.ndarray, signal: np.ndarray) -> complex:
        if time_ps.ndim != 1 or signal.ndim != 1:
            raise ValueError("time_ps and signal must be 1D arrays")
        if len(time_ps) != len(signal):
            raise ValueError("time_ps and signal must have the same length")
        if len(time_ps) < 2:
            raise ValueError("Need at least two samples for Fourier extraction")

        dt = np.median(np.diff(time_ps))
        omega = self.md.omega_per_ps
        phase = np.exp(-1j * omega * time_ps)
        if self.analysis.use_window_function:
            window = np.hanning(len(signal))
            signal = signal * window
            phase = phase * window
        component = np.sum(signal * phase) * dt / (time_ps[-1] - time_ps[0] + dt)
        return component

    def extract_system_phase_amplitude(self, system_block: pd.DataFrame) -> tuple[float, float]:
        component = self._fourier_component_at_drive(
            system_block["time_ps"].to_numpy(),
            system_block["sigma_xy"].to_numpy(),
        )
        amplitude = 2.0 * np.abs(component)
        phase = float(np.angle(component))
        return phase, float(amplitude)

    def extract_atomic_phase_amplitude(self, atomic_block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        atom_ids = atomic_block["atom_id"].drop_duplicates().to_numpy()
        phases = np.empty(len(atom_ids), dtype=float)
        amplitudes = np.empty(len(atom_ids), dtype=float)

        for i, atom_id in enumerate(atom_ids):
            atom_df = atomic_block.loc[atomic_block["atom_id"] == atom_id]
            component = self._fourier_component_at_drive(
                atom_df["time_ps"].to_numpy(),
                atom_df["sigma_xy"].to_numpy(),
            )
            phases[i] = np.angle(component)
            amplitudes[i] = 2.0 * np.abs(component)

        return atom_ids, phases, amplitudes

    # ---------------------------------------------------------------------
    # stage 5: identify lossy atoms and partial-stress logic
    # ---------------------------------------------------------------------
    def identify_lossy_atoms(
        self,
        atom_ids: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        target = self.md.lossy_target_phase_rad
        half_width = self.md.lossy_phase_half_width_rad
        phase_distance = np.angle(np.exp(1j * (phases - target)))
        mask = np.abs(phase_distance) <= half_width
        candidate_ids = atom_ids[mask]

        if len(candidate_ids) == 0:
            n_keep = max(1, int(round(len(atom_ids) * self.md.lossy_fraction)))
            order = np.argsort(np.abs(phase_distance))
            return atom_ids[order[:n_keep]]

        if len(candidate_ids) > int(round(len(atom_ids) * self.md.lossy_fraction)):
            subset_amp = amplitudes[mask]
            order = np.argsort(subset_amp)[::-1]
            n_keep = max(1, int(round(len(atom_ids) * self.md.lossy_fraction)))
            return candidate_ids[order[:n_keep]]

        return candidate_ids

    def reconstruct_partial_stress(
        self,
        atomic_block: pd.DataFrame,
        excluded_atom_ids: Iterable[int],
    ) -> pd.DataFrame:
        excluded_atom_ids = set(int(x) for x in excluded_atom_ids)
        kept = atomic_block.loc[~atomic_block["atom_id"].isin(excluded_atom_ids)].copy()
        partial = kept.groupby("time_ps", as_index=False)["sigma_xy"].sum()
        partial = partial.rename(columns={"sigma_xy": "sigma_xy_partial"})
        return partial

    def analyze_single_block(
        self,
        block_index: int,
        system_block: pd.DataFrame,
        atomic_block: pd.DataFrame,
    ) -> BlockResult:
        system_phase, system_amplitude = self.extract_system_phase_amplitude(system_block)
        atom_ids, atom_phases, atom_amplitudes = self.extract_atomic_phase_amplitude(atomic_block)
        lossy_atom_ids = self.identify_lossy_atoms(atom_ids, atom_phases, atom_amplitudes)

        return BlockResult(
            block_index=block_index,
            t_start_ps=float(system_block["time_ps"].min()),
            t_end_ps=float(system_block["time_ps"].max()),
            system_phase_rad=system_phase,
            system_amplitude=system_amplitude,
            n_atoms=len(atom_ids),
            lossy_atom_ids=lossy_atom_ids,
            atomic_phase_rad=atom_phases,
            atomic_amplitude=atom_amplitudes,
        )

    def analyze_all_blocks(
        self,
        system_df: pd.DataFrame,
        atomic_df: pd.DataFrame,
    ) -> list[BlockResult]:
        system_blocks = self.split_system_into_blocks(system_df)
        atomic_blocks = self.split_atomic_into_blocks(atomic_df)
        if len(system_blocks) != len(atomic_blocks):
            raise ValueError("System and atomic block counts do not match")

        results: list[BlockResult] = []
        for i, (s_block, a_block) in enumerate(zip(system_blocks, atomic_blocks)):
            results.append(self.analyze_single_block(i, s_block, a_block))
        return results

    # ---------------------------------------------------------------------
    # stage 6: transient analysis across blocks
    # ---------------------------------------------------------------------
    @staticmethod
    def jaccard_overlap(ids_a: np.ndarray, ids_b: np.ndarray) -> float:
        set_a = set(int(x) for x in ids_a)
        set_b = set(int(x) for x in ids_b)
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)

    def track_lossy_atom_transience(self, block_results: list[BlockResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i, result_i in enumerate(block_results):
            for j, result_j in enumerate(block_results[i:], start=i):
                rows.append(
                    {
                        "block_i": i,
                        "block_j": j,
                        "delta_blocks": j - i,
                        "jaccard_overlap": self.jaccard_overlap(
                            result_i.lossy_atom_ids,
                            result_j.lossy_atom_ids,
                        ),
                    }
                )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # stage 7: export helpers
    # ---------------------------------------------------------------------
    def block_summary_table(self, block_results: list[BlockResult]) -> pd.DataFrame:
        rows = []
        for result in block_results:
            rows.append(
                {
                    "block_index": result.block_index,
                    "t_start_ps": result.t_start_ps,
                    "t_end_ps": result.t_end_ps,
                    "system_phase_rad": result.system_phase_rad,
                    "system_amplitude": result.system_amplitude,
                    "n_lossy_atoms": len(result.lossy_atom_ids),
                    "mean_atomic_phase_rad": float(np.mean(result.atomic_phase_rad)),
                    "std_atomic_phase_rad": float(np.std(result.atomic_phase_rad)),
                }
            )
        return pd.DataFrame(rows)

    def save_block_results(self, block_results: list[BlockResult]) -> None:
        summary = self.block_summary_table(block_results)
        summary.to_csv(self.paths.analysis / "block_summary.csv", index=False)

        overlap = self.track_lossy_atom_transience(block_results)
        overlap.to_csv(self.paths.analysis / "lossy_overlap.csv", index=False)

        if self.analysis.save_intermediate_arrays:
            for result in block_results:
                np.save(self.paths.analysis / f"block_{result.block_index:02d}_atom_ids.npy", result.lossy_atom_ids)
                np.save(self.paths.analysis / f"block_{result.block_index:02d}_atomic_phase.npy", result.atomic_phase_rad)
                np.save(self.paths.analysis / f"block_{result.block_index:02d}_atomic_amplitude.npy", result.atomic_amplitude)


def example_workflow() -> None:
    paths = ProjectPaths(root=Path("paper2_mdms"))
    prep = GlassPreparationConfig()
    md = MDDMSConfig()
    analysis = AnalysisConfig()
    workflow = Paper2MDDMSWorkflow(paths=paths, prep=prep, md=md, analysis=analysis)
    workflow.write_metadata()

    # Stage 1 and Stage 2 are the two parts you need to connect to pyiron/LAMMPS.
    # structure = workflow.build_initial_structure()
    # equilibrated_structure = workflow.melt_quench_equilibrate(structure)
    # raw_paths = workflow.run_md_dms(equilibrated_structure)

    # Stage 3 onward is analysis once raw files exist.
    # system_df = workflow.load_system_stress(raw_paths["system_stress"])
    # atomic_df = workflow.load_atomic_stress(raw_paths["atomic_stress"])
    # block_results = workflow.analyze_all_blocks(system_df, atomic_df)
    # workflow.save_block_results(block_results)


if __name__ == "__main__":
    example_workflow()
