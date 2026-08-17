from __future__ import annotations

import os
import re
from pathlib import Path

from .config import Potential

_ENV_TOKEN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def _missing_environment_variables(raw_path: str) -> list[str]:
    missing: list[str] = []
    for match in _ENV_TOKEN.finditer(raw_path):
        name = match.group("braced") or match.group("plain")
        if name not in os.environ:
            missing.append(name)
    return sorted(set(missing))


def resolve_model_path(repo_root: Path, raw_path: str) -> Path:
    """Resolve a potential path.

    Paths may be absolute, relative to the repository root, or use environment
    variables written by ``docker/startup_md_lammps.sh`` (for example,
    ``${MACE_C_MLIAP}``).
    """
    if not raw_path.strip():
        raise ValueError("Potential path is empty; set it in revision/config/potentials.json")

    missing = _missing_environment_variables(raw_path)
    if missing:
        names = ", ".join(missing)
        raise EnvironmentError(
            f"Potential path requires unset environment variable(s): {names}. "
            "On the A100 machine, run docker/startup_md_lammps.sh and then "
            "source /workspace/cuzr_runtime.env."
        )

    expanded = os.path.expandvars(raw_path)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def infer_eam_pair_style(model_path: Path) -> str:
    low = model_path.name.lower()
    if low.endswith(".eam.fs"):
        return "eam/fs"
    if low.endswith(".eam.alloy"):
        return "eam/alloy"
    if low.endswith(".eam"):
        return "eam"
    raise ValueError(f"Cannot infer EAM pair_style from {model_path}")


def potential_block(potential: Potential, repo_root: Path) -> str:
    family = potential.family.upper()
    model_path = resolve_model_path(repo_root, potential.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Potential file does not exist: {model_path}")

    if family == "MACE":
        return (
            "# MACE through the LAMMPS ML-IAP unified interface\n"
            f"pair_style mliap unified {model_path} 0\n"
            "pair_coeff * * Cu Zr\n"
        )
    if family == "ACE":
        return (
            "# ACE through pair_style pace using the product evaluator for KOKKOS/pace/kk\n"
            "pair_style pace product\n"
            f"pair_coeff * * {model_path} Cu Zr\n"
        )
    if family == "EAM":
        return (
            "# EAM reference potential\n"
            f"pair_style {infer_eam_pair_style(model_path)}\n"
            f"pair_coeff * * {model_path} Cu Zr\n"
        )
    raise ValueError(f"Unsupported potential family: {family}")
