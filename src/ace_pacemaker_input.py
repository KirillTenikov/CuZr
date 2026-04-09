from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

BASIS_PROFILES: Dict[str, Dict[str, Dict[str, List[int]]]] = {
    # These profiles are generic starting points. The main complexity control
    # remains `number_of_functions_per_element` (basis_size).
    "small": {
        "UNARY": {
            "nradmax_by_orders": [15, 6, 4, 3, 2, 2],
            "lmax_by_orders": [0, 3, 3, 2, 2, 1],
        },
        "BINARY": {
            "nradmax_by_orders": [15, 6, 3, 2, 2, 1],
            "lmax_by_orders": [0, 3, 2, 1, 1, 0],
        },
        "TERNARY": {
            "nradmax_by_orders": [15, 3, 3, 2, 1],
            "lmax_by_orders": [0, 2, 2, 1, 1],
        },
        "ALL": {
            "nradmax_by_orders": [15, 3, 2, 1, 1],
            "lmax_by_orders": [0, 2, 2, 1, 1],
        },
    },
    "medium": {
        "UNARY": {
            "nradmax_by_orders": [16, 7, 5, 3, 2, 2],
            "lmax_by_orders": [0, 3, 3, 2, 2, 1],
        },
        "BINARY": {
            "nradmax_by_orders": [16, 7, 4, 3, 2, 1],
            "lmax_by_orders": [0, 3, 2, 2, 1, 0],
        },
        "TERNARY": {
            "nradmax_by_orders": [16, 4, 3, 2, 1],
            "lmax_by_orders": [0, 2, 2, 1, 1],
        },
        "ALL": {
            "nradmax_by_orders": [16, 4, 3, 2, 1],
            "lmax_by_orders": [0, 2, 2, 1, 1],
        },
    },
    "large": {
        "UNARY": {
            "nradmax_by_orders": [18, 8, 5, 4, 3, 2],
            "lmax_by_orders": [0, 4, 3, 2, 2, 1],
        },
        "BINARY": {
            "nradmax_by_orders": [18, 8, 5, 3, 2, 2],
            "lmax_by_orders": [0, 3, 3, 2, 1, 1],
        },
        "TERNARY": {
            "nradmax_by_orders": [18, 5, 4, 2, 2],
            "lmax_by_orders": [0, 3, 2, 1, 1],
        },
        "ALL": {
            "nradmax_by_orders": [18, 5, 4, 2, 2],
            "lmax_by_orders": [0, 3, 2, 1, 1],
        },
    },
    "xlarge": {
        "UNARY": {
            "nradmax_by_orders": [20, 10, 6, 4, 3, 2],
            "lmax_by_orders": [0, 4, 4, 3, 2, 1],
        },
        "BINARY": {
            "nradmax_by_orders": [20, 10, 6, 4, 3, 2],
            "lmax_by_orders": [0, 4, 3, 2, 2, 1],
        },
        "TERNARY": {
            "nradmax_by_orders": [20, 6, 5, 3, 2],
            "lmax_by_orders": [0, 3, 3, 2, 1],
        },
        "ALL": {
            "nradmax_by_orders": [20, 6, 5, 3, 2],
            "lmax_by_orders": [0, 3, 3, 2, 1],
        },
    },
}

DEFAULT_WEIGHTING = {
    "type": "EnergyBasedWeightingPolicy",
    "DElow": 1.0,
    "DEup": 10.0,
    "DFup": 50.0,
    "DE": 1.0,
    "DF": 1.0,
    "wlow": 0.75,
    "energy": "convex_hull",
    "reftype": "all",
}


def infer_basis_profile(basis_size: int) -> str:
    if basis_size <= 700:
        return "small"
    if basis_size <= 1800:
        return "medium"
    if basis_size <= 4000:
        return "large"
    return "xlarge"


def build_pacemaker_input(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    del run_dir  # reserved for future use

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    backend_cfg = cfg["backend"]

    basis_size = int(model_cfg["basis_size"])
    elements = model_cfg.get("elements", ["Cu", "Zr"])
    profile_name = model_cfg.get("basis_profile", infer_basis_profile(basis_size))
    if profile_name not in BASIS_PROFILES:
        raise ValueError(f"Unknown ACE basis_profile: {profile_name}")
    basis_profile = BASIS_PROFILES[profile_name]

    weighting = dict(DEFAULT_WEIGHTING)
    weighting.update(train_cfg.get("weighting", {}))
    weighting["seed"] = int(train_cfg.get("seed", 42))

    loss = {
        "kappa": float(train_cfg.get("loss", {}).get("kappa", 0.3)),
        "L1_coeffs": float(train_cfg.get("loss", {}).get("L1_coeffs", 1e-8)),
        "L2_coeffs": float(train_cfg.get("loss", {}).get("L2_coeffs", 1e-8)),
    }

    fit: Dict[str, Any] = {
        "loss": loss,
        "weighting": weighting,
        "optimizer": train_cfg.get("optimizer", "BFGS"),
        "maxiter": int(train_cfg.get("maxiter", train_cfg.get("max_epochs", 2000))),
        "repulsion": train_cfg.get("repulsion", "auto"),
    }

    early_stopping = train_cfg.get("early_stopping")
    if early_stopping:
        fit.update(early_stopping)

    potential = {
        "deltaSplineBins": float(model_cfg.get("delta_spline_bins", 0.001)),
        "elements": list(elements),
        "embeddings": {
            "ALL": {
                "npot": model_cfg.get("embedding", {}).get(
                    "npot", "FinnisSinclairShiftedScaled"
                ),
                "fs_parameters": model_cfg.get("embedding", {}).get(
                    "fs_parameters", [1, 1, 1, 0.5]
                ),
                "ndensity": int(model_cfg.get("embedding", {}).get("ndensity", 2)),
            }
        },
        "bonds": {
            "ALL": {
                "radbase": model_cfg.get("radial_base", "SBessel"),
                "radparameters": model_cfg.get("radial_parameters", [5.25]),
                "rcut": float(model_cfg["cutoff"]),
                "dcut": float(model_cfg.get("dcut", 0.01)),
                "NameOfCutoffFunction": model_cfg.get("cutoff_function", "cos"),
            }
        },
        "functions": {
            "number_of_functions_per_element": basis_size,
            **basis_profile,
        },
    }

    input_cfg: Dict[str, Any] = {
        "cutoff": float(model_cfg["cutoff"]),
        "seed": int(train_cfg.get("seed", 42)),
        "metadata": {
            "project": "CuZr",
            "run_name": cfg["run"]["name"],
            "basis_profile": profile_name,
            "basis_size": str(basis_size),
            "purpose": "ACE training for CuZr project",
        },
        "potential": potential,
        "data": {
            "filename": str(Path(data_cfg["train_path"]).resolve()),
            "test_filename": str(Path(data_cfg["valid_path"]).resolve()),
        },
        "fit": fit,
        "backend": {
            "evaluator": backend_cfg.get("evaluator", "tensorpot"),
            "batch_size": int(
                backend_cfg.get("batch_size", train_cfg.get("batch_size", 100))
            ),
            "display_step": int(backend_cfg.get("display_step", 50)),
        },
    }
    return input_cfg
