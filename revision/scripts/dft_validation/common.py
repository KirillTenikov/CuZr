#!/usr/bin/env python3
"""Shared utilities for Paper 1 DFT validation.

The extxyz reader intentionally avoids ASE so that test-set inventory and
MACE TorchScript inference can be reproduced with a minimal dependency set.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import numpy as np


TARGET_NATOMS = 128
TARGET_NCU = (46, 64, 82)
TARGET_LABELS = {
    46: "Cu36Zr64",
    64: "Cu50Zr50",
    82: "Cu64Zr36",
}


def iter_extxyz(path: str | Path) -> Iterator[Dict]:
    """Yield structures from the Cu-Zr extxyz test file.

    Expected per-frame metadata:
      - Lattice="..."
      - energy=<eV>
      - optional pbc="T T T"

    Expected atom columns:
      symbol x y z fx fy fz [...]

    The routine only consumes the fields required by the Paper 1 DFT analysis.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                natoms = int(line)
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected frame start at frame {frame}: {line[:120]}"
                ) from exc

            header = handle.readline().strip()

            lattice_match = re.search(r'Lattice="([^"]+)"', header)
            if not lattice_match:
                raise ValueError(f"Frame {frame}: missing Lattice metadata")
            cell = np.asarray(
                [float(x) for x in lattice_match.group(1).split()],
                dtype=np.float64,
            ).reshape(3, 3)

            energy_match = re.search(r"(?:^|\s)energy=([^\s]+)", header)
            if not energy_match:
                raise ValueError(f"Frame {frame}: missing energy metadata")
            energy = float(energy_match.group(1))

            pbc_match = re.search(r'pbc="([^"]+)"', header)
            if pbc_match:
                pbc = np.asarray(
                    [x.upper().startswith("T") for x in pbc_match.group(1).split()],
                    dtype=bool,
                )
            else:
                pbc = np.asarray([False, False, False], dtype=bool)

            symbols: List[str] = []
            positions = np.empty((natoms, 3), dtype=np.float64)
            forces = np.empty((natoms, 3), dtype=np.float64)

            for i in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise EOFError(f"Unexpected EOF in frame {frame}")
                fields = atom_line.split()
                if len(fields) < 7:
                    raise ValueError(
                        f"Frame {frame}, atom {i}: expected symbol/x/y/z/fx/fy/fz"
                    )
                symbols.append(fields[0])
                positions[i] = [float(v) for v in fields[1:4]]
                forces[i] = [float(v) for v in fields[4:7]]

            n_cu = sum(symbol == "Cu" for symbol in symbols)
            n_zr = sum(symbol == "Zr" for symbol in symbols)
            if n_cu + n_zr != natoms:
                raise ValueError(
                    f"Frame {frame}: contains species other than Cu/Zr"
                )

            yield {
                "frame": frame,
                "n": natoms,
                "cell": cell,
                "pbc": pbc,
                "symbols": symbols,
                "pos": positions,
                "forces": forces,
                "energy": energy,
                "n_Cu": n_cu,
                "n_Zr": n_zr,
                "x_Cu": n_cu / natoms,
            }
            frame += 1


def select_target_frames(
    frames: Iterable[Dict],
    natoms: int = TARGET_NATOMS,
    n_cu_values: Iterable[int] = TARGET_NCU,
) -> List[Dict]:
    """Select the three application-relevant N=128 compositions."""
    target = set(int(x) for x in n_cu_values)
    return [
        fr for fr in frames
        if int(fr["n"]) == int(natoms) and int(fr["n_Cu"]) in target
    ]


def build_edges(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    rmax: float,
):
    """Build directed periodic neighbor edges for a TorchScript MACE model.

    Pair-vector convention:
        r_j - r_i + S @ cell

    A conservative integer-image bound is used and all candidate images are
    filtered by Cartesian distance.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)

    inv_cell = np.linalg.inv(cell)
    image_bounds = []
    for axis in range(3):
        if pbc[axis]:
            bound = int(math.ceil(rmax * np.linalg.norm(inv_cell[:, axis]))) + 1
        else:
            bound = 0
        image_bounds.append(bound)

    integer_shifts = np.asarray(
        list(
            itertools.product(
                *[range(-m, m + 1) for m in image_bounds]
            )
        ),
        dtype=np.int32,
    )
    cart_shifts = integer_shifts @ cell

    base = positions[None, :, :] - positions[:, None, :]
    rc2 = float(rmax) ** 2

    src_all = []
    dst_all = []
    unit_shift_all = []
    cart_shift_all = []

    for shift_int, shift_cart in zip(integer_shifts, cart_shifts):
        disp = base + shift_cart
        d2 = np.einsum("ijk,ijk->ij", disp, disp)
        mask = d2 < rc2 - 1e-10

        if np.all(shift_int == 0):
            np.fill_diagonal(mask, False)

        src, dst = np.nonzero(mask)
        if len(src) == 0:
            continue

        src_all.append(src.astype(np.int64))
        dst_all.append(dst.astype(np.int64))
        unit_shift_all.append(
            np.repeat(shift_int[None, :], len(src), axis=0)
        )
        cart_shift_all.append(
            np.repeat(shift_cart[None, :], len(src), axis=0)
        )

    if not src_all:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        np.concatenate(src_all),
        np.concatenate(dst_all),
        np.concatenate(unit_shift_all).astype(np.float32),
        np.concatenate(cart_shift_all).astype(np.float32),
    )


def make_mace_batch(frames: List[Dict], rmax: float):
    """Convert frames to the dictionary expected by compiled TorchScript MACE."""
    import torch

    positions_all = []
    attrs_all = []
    src_all = []
    dst_all = []
    shifts_all = []
    unit_shifts_all = []
    batch_all = []
    cells = []
    ptr = [0]
    offset = 0

    for graph_index, fr in enumerate(frames):
        positions = fr["pos"].astype(np.float32)

        # Model bundles used for this paper were Cu/Zr two-species models.
        node_attrs = np.zeros((fr["n"], 2), dtype=np.float32)
        node_attrs[:, 0] = np.asarray(
            [s == "Cu" for s in fr["symbols"]], dtype=np.float32
        )
        node_attrs[:, 1] = 1.0 - node_attrs[:, 0]

        src, dst, unit_shifts, cart_shifts = build_edges(
            fr["pos"], fr["cell"], fr["pbc"], rmax
        )

        positions_all.append(positions)
        attrs_all.append(node_attrs)
        src_all.append(src + offset)
        dst_all.append(dst + offset)
        unit_shifts_all.append(unit_shifts)
        shifts_all.append(cart_shifts)
        batch_all.append(
            np.full(fr["n"], graph_index, dtype=np.int64)
        )
        cells.append(fr["cell"].astype(np.float32))

        offset += fr["n"]
        ptr.append(offset)

    return {
        "positions": torch.from_numpy(np.concatenate(positions_all)),
        "node_attrs": torch.from_numpy(np.concatenate(attrs_all)),
        "edge_index": torch.from_numpy(
            np.stack([np.concatenate(src_all), np.concatenate(dst_all)])
        ),
        "shifts": torch.from_numpy(np.concatenate(shifts_all)),
        "unit_shifts": torch.from_numpy(np.concatenate(unit_shifts_all)),
        "cell": torch.from_numpy(np.stack(cells)),
        "batch": torch.from_numpy(np.concatenate(batch_all)),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "head": torch.zeros(len(frames), dtype=torch.long),
    }
