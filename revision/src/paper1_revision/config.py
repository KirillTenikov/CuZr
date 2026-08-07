from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Protocol:
    timestep_ps: float
    temperature_high_K: float
    temperature_low_K: float
    pressure_bar: float
    melt_ps: float
    quench_rate_K_per_ps: float
    relax_npt_ps: float
    equilibrate_nvt_ps: float
    thermo_every_steps: int
    tdamp_ps: float
    pdamp_ps: float
    minimize_etol: float
    minimize_ftol: float
    minimize_maxiter: int
    minimize_maxeval: int
    tail_fraction: float
    nve_preequilibrate_ps: float
    nve_stability_ps: float

    @property
    def quench_ps(self) -> float:
        delta_t = self.temperature_high_K - self.temperature_low_K
        if delta_t <= 0:
            raise ValueError("temperature_high_K must exceed temperature_low_K")
        if self.quench_rate_K_per_ps <= 0:
            raise ValueError("quench_rate_K_per_ps must be positive")
        return delta_t / self.quench_rate_K_per_ps

    def steps(self, duration_ps: float) -> int:
        if self.timestep_ps <= 0:
            raise ValueError("timestep_ps must be positive")
        return max(0, int(round(duration_ps / self.timestep_ps)))


@dataclass(frozen=True)
class Potential:
    id: str
    family: str
    path: str
    enabled: bool
    lmp_command: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_protocol(path: Path) -> Protocol:
    data = _read_json(path)
    return Protocol(**data)


def load_potentials(path: Path) -> dict[str, Potential]:
    data = _read_json(path)
    raw = data.get("potentials")
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a 'potentials' list")
    result: dict[str, Potential] = {}
    for item in raw:
        pot = Potential(**item)
        if pot.id in result:
            raise ValueError(f"Duplicate potential id: {pot.id}")
        if pot.family.upper() not in {"MACE", "ACE", "EAM"}:
            raise ValueError(f"Unsupported family for {pot.id}: {pot.family}")
        result[pot.id] = pot
    return result
