from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def find_repo_root(start: str | Path | None = None) -> Path:
    """Best-effort detection of the project root.

    Prefers explicit env var CUZR_PROJECT_ROOT, then walks upward looking for
    common repo markers.
    """
    env_root = os.environ.get("CUZR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(start or __file__).resolve()
    candidates = [here] + list(here.parents)
    for cand in candidates:
        if (cand / "scripts").exists() and (cand / "src").exists():
            return cand
    return here.parent


def resolve_path(
    raw_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        base = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
        path = (base / path).resolve()
    else:
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def first_existing(paths: Iterable[str | Path]) -> Optional[Path]:
    for p in paths:
        cand = Path(p).expanduser().resolve()
        if cand.exists():
            return cand
    return None
