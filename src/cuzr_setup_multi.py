"""
Cu–Zr MACE / EAM / ACE setup helpers for pyiron + LAMMPS.

Validation-/training-side helper module.
- No LAMMPS_EXE here by design.
- Does NOT create jobs on import.
- You pass your pyiron Project `pr` into helper functions.
- Supports a single unified POTENTIALS registry that can contain:
    * multiple MACE models
    * multiple ACE models
    * EAM baselines
    * placeholders

Typical usage in Notebook 05:
    from pyiron_atomistics import Project
    import cuzr_setup_updated as cz

    pr = Project("../cu_zr_mlip_project")

    POTENTIAL_SPECS = [
        cz.make_mace_spec("MACE_A", "CuZr_MACE_A.model-lammps.pt"),
        cz.make_mace_spec("MACE_B", "CuZr_MACE_B.model-lammps.pt"),
        cz.make_pyiron_spec("EAM_2019", "EAM_Mendelev_2019_CuZr__MO_945018740343_000"),
        cz.make_placeholder_spec("ACE_A"),
    ]
    cz.register_potentials(POTENTIAL_SPECS)

    job = cz.run_static(pr, "test", structure, cz.pot("MACE_A"))
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional, Iterable
import os
import copy
import pandas as pd

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_MACE_MODEL_FILE: str = os.environ.get(
    "CUZR_MACE_MODEL_FILE",
    "CuZr_DEV_r5.0_compiled.model-lammps.pt",
)

# Placeholder for optional pyiron-known ACE potential registration.
ACE_PYIRON_NAME: Optional[str] = os.environ.get("CUZR_ACE_PYIRON_NAME") or None

DEFAULT_CORES: int = 1
DEFAULT_TIMESTEP_FS: float = 1.0
DEFAULT_THERMO_STEPS: int = 50
DEFAULT_NEIGHBOR_BIN: float = 1.0
DEFAULT_NEIGH_EVERY: int = 10

# Baseline potential names used previously
EAM_NAMES: List[str] = [
    "2007--Mendelev-M-I--Cu-Zr--LAMMPS--ipr1",
    "EAM_Mendelev_2019_CuZr__MO_945018740343_000",
]

# ----------------------------
# Potential factories / registry
# ----------------------------

def make_mace_potential_df(
    model_file: str,
    elements: Tuple[str, str] = ("Cu", "Zr"),
    pair_style: str = "mace",
) -> pd.DataFrame:
    """
    Create a pyiron 'Custom' potential DataFrame for a LAMMPS-readable MACE model.

    `model_file` is resolved by pyiron via its resource paths.
    `pair_style` can be changed if your local LAMMPS build requires a specific form,
    e.g. 'mace no_domain_decomposition'.
    """
    el = list(elements)
    return pd.DataFrame(
        {
            "Name": ["MACE_CuZr"],
            "Filename": [[model_file]],
            "Model": ["Custom"],
            "Species": [el],
            "Config": [[
                f"pair_style {pair_style}\n",
                f"pair_coeff * * {model_file} " + " ".join(el) + "\n",
            ]],
        }
    )


def make_custom_lammps_potential_df(
    name: str,
    model_file: str,
    pair_style: str,
    pair_coeff: str,
    species: Tuple[str, ...],
) -> pd.DataFrame:
    """Generic helper for custom LAMMPS-backed potentials."""
    return pd.DataFrame(
        {
            "Name": [name],
            "Filename": [[model_file]],
            "Model": ["Custom"],
            "Species": [list(species)],
            "Config": [[
                f"pair_style {pair_style}\n",
                f"pair_coeff {pair_coeff}\n",
            ]],
        }
    )


def _short_pyiron_id(name: str) -> str:
    return name.split("__MO_")[0].replace("--", "_").replace(" ", "_")


def make_pyiron_spec(id_: str, name: str) -> Dict[str, Any]:
    """Potential spec for a pyiron-known potential."""
    return {"id": str(id_), "mode": "pyiron", "name": str(name)}


def make_placeholder_spec(id_: str) -> Dict[str, Any]:
    """Placeholder spec used before a model is trained/registered."""
    return {"id": str(id_), "mode": "placeholder", "name": None}


def make_mace_spec(
    id_: str,
    model_file: str,
    elements: Tuple[str, str] = ("Cu", "Zr"),
    pair_style: str = "mace",
) -> Dict[str, Any]:
    """Potential spec for one MACE model."""
    return {
        "id": str(id_),
        "mode": "custom",
        "df": make_mace_potential_df(model_file, elements=elements, pair_style=pair_style),
        "model_file": str(model_file),
        "pair_style": str(pair_style),
        "family": "MACE",
    }


def make_ace_spec(
    id_: str,
    model_file: str,
    elements: Tuple[str, str] = ("Cu", "Zr"),
    pair_style: str = "pace",
    pair_coeff: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Potential spec for one ACE model exported for LAMMPS.

    Default assumes a PACE-style LAMMPS interface:
        pair_style pace
        pair_coeff * * <model_file> Cu Zr

    Adjust `pair_style` / `pair_coeff` if your ACE-LAMMPS interface differs.
    """
    if pair_coeff is None:
        pair_coeff = f"* * {model_file} " + " ".join(elements)
    return {
        "id": str(id_),
        "mode": "custom",
        "df": make_custom_lammps_potential_df(
            name=str(id_),
            model_file=str(model_file),
            pair_style=str(pair_style),
            pair_coeff=str(pair_coeff),
            species=elements,
        ),
        "model_file": str(model_file),
        "pair_style": str(pair_style),
        "family": "ACE",
    }


def default_eam_specs() -> List[Dict[str, Any]]:
    """Return the two standard EAM baseline specs."""
    return [make_pyiron_spec(_short_pyiron_id(name), name) for name in EAM_NAMES]


def build_default_potentials(
    mace_model_file: str = DEFAULT_MACE_MODEL_FILE,
    include_eam: bool = True,
    include_ace_placeholder: bool = True,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible default registry:
      1. one MACE model
      2. EAM baselines
      3. ACE placeholder or pyiron-known ACE if provided by environment
    """
    pots: List[Dict[str, Any]] = [make_mace_spec("MACE", mace_model_file)]

    if include_eam:
        pots.extend(default_eam_specs())

    if ACE_PYIRON_NAME:
        pots.append(make_pyiron_spec("ACE", ACE_PYIRON_NAME))
    elif include_ace_placeholder:
        pots.append(make_placeholder_spec("ACE"))

    return pots


POTENTIALS: List[Dict[str, Any]] = build_default_potentials()


def register_potentials(potentials: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replace the global POTENTIALS registry with an explicit list of potential specs.

    This is the preferred interface for Notebook 05 when comparing multiple
    MACE and ACE models in one unified array.
    """
    global POTENTIALS
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()

    for p in potentials:
        q = copy.deepcopy(dict(p))
        pid = q.get("id")
        if not pid:
            raise ValueError(f"Every potential spec must have a non-empty 'id': {p}")
        if pid in seen_ids:
            raise ValueError(f"Duplicate potential id detected: {pid}")
        seen_ids.add(pid)
        normalized.append(q)

    POTENTIALS = normalized
    return POTENTIALS


def reset_default_potentials(
    mace_model_file: str = DEFAULT_MACE_MODEL_FILE,
    include_eam: bool = True,
    include_ace_placeholder: bool = True,
) -> List[Dict[str, Any]]:
    """Reset POTENTIALS to the backward-compatible default registry."""
    return register_potentials(
        build_default_potentials(
            mace_model_file=mace_model_file,
            include_eam=include_eam,
            include_ace_placeholder=include_ace_placeholder,
        )
    )


def set_mace_model_file(model_file: str) -> None:
    """
    Backward-compatible helper for the single-MACE workflow.
    Resets POTENTIALS to one MACE + default baselines.
    """
    reset_default_potentials(mace_model_file=str(model_file))


def pot(id_: str) -> Dict[str, Any]:
    for p in POTENTIALS:
        if p["id"] == id_:
            return copy.deepcopy(p)
    raise KeyError(f"Potential id not found: {id_}. Available ids: {[p['id'] for p in POTENTIALS]}")


# ----------------------------
# Apply potential to a job
# ----------------------------

def assign_potential(job, pot_spec: Dict[str, Any]) -> None:
    """Apply pot_spec to an existing pyiron Lammps job."""
    mode = pot_spec.get("mode", "").lower()
    if mode == "custom":
        job.potential = pot_spec["df"]
    elif mode == "pyiron":
        job.potential = pot_spec["name"]
    elif mode == "placeholder":
        raise ValueError(
            f"Potential '{pot_spec.get('id')}' is only a placeholder. "
            "Train/register the model first, then rebuild POTENTIALS."
        )
    else:
        raise ValueError(f"Unknown pot_spec mode: {pot_spec.get('mode')}")


# ----------------------------
# Job helpers (no global `pr`)
# ----------------------------

def make_lammps_job(
    pr,
    job_name: str,
    structure,
    pot_spec: Dict[str, Any],
    delete_existing: bool = True,
    cores: int = DEFAULT_CORES,
):
    """Create and configure a pyiron Lammps job."""
    job = pr.create.job.Lammps(job_name, delete_existing_job=delete_existing)
    job.structure = structure
    assign_potential(job, pot_spec)
    job.server.cores = int(cores)
    return job


def load_or_run(pr, job):
    try:
        j = pr.load(job.job_name)
        if j.status.finished or j.status.running:
            return j
    except Exception:
        pass
    job.run()
    return job


def run_static(
    pr,
    job_name: str,
    structure,
    pot_spec: Dict[str, Any],
    cores: int = DEFAULT_CORES,
    delete_existing: bool = True,
):
    """Run a static calculation with the given potential spec."""
    job = make_lammps_job(pr, job_name, structure, pot_spec, delete_existing=delete_existing, cores=cores)
    job.calc_static()
    job.run()
    return job


def run_md(
    pr,
    job_name: str,
    structure,
    pot_spec,
    T: float,
    steps: int,
    timestep_fs: float = DEFAULT_TIMESTEP_FS,
    cores: int = DEFAULT_CORES,
    delete_existing: bool = True,
    thermo: int = DEFAULT_THERMO_STEPS,
    dump_every: int | None = None,
    neigh_every: int = DEFAULT_NEIGH_EVERY,
    neighbor_bin: float = DEFAULT_NEIGHBOR_BIN,
):
    """Run a simple NVT/NPT pyiron MD job with a bit of LAMMPS tuning."""
    job = make_lammps_job(
        pr, job_name, structure, pot_spec,
        delete_existing=delete_existing, cores=cores
    )
    job.calc_md(temperature=T, n_ionic_steps=steps, time_step=timestep_fs)

    # Thermo output frequency
    job.input.control["variable___thermotime"] = f"equal {int(thermo)}"

    # Neighbor list tuning; keep skin moderate for ML potentials.
    job.input.control["neighbor"] = f"{float(neighbor_bin):.1f} bin"
    job.input.control["neigh_modify"] = f"every {int(neigh_every)} delay 0 check yes"

    if dump_every is None:
        for k in ["variable___dumptime", "dump___1", "dump_modify___1"]:
            try:
                del job.input.control[k]
            except Exception:
                pass
    else:
        job.input.control["variable___dumptime"] = f"equal {int(dump_every)}"
        job.input.control["dump___1"] = ("all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz")
        job.input.control["dump_modify___1"] = "sort id"
    return load_or_run(pr, job)
