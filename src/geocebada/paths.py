"""Project-path helpers for GeoCebada.

These helpers let notebooks and package code resolve repository paths without
hard-coding machine-specific absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path


_ROOT_ENV_VAR = "GEOCEBADA_ROOT"


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the GeoCebada repository root.

    Resolution order:

    1. ``GEOCEBADA_ROOT`` environment variable, when defined.
    2. ``start`` (when provided) and its parents.
    3. the current working directory and its parents.
    4. this package file and its parents.

    A valid root must contain both ``pyproject.toml`` and ``src/geocebada``.

    Parameters
    ----------
    start:
        Optional path from which to begin searching.

    Raises
    ------
    FileNotFoundError
        If no repository root can be identified.
    """

    env_root = os.getenv(_ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if _is_project_root(root):
            return root
        raise FileNotFoundError(
            f"{_ROOT_ENV_VAR} points to {root}, but it is not a GeoCebada project root."
        )

    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    visited: set[Path] = set()
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in (current, *current.parents):
            if parent in visited:
                continue
            visited.add(parent)
            if _is_project_root(parent):
                return parent

    raise FileNotFoundError(
        "Could not locate the GeoCebada project root. Run from inside the repository "
        f"or set {_ROOT_ENV_VAR}."
    )


def project_path(
    *parts: str | Path,
    root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Build a path relative to the repository root.

    Parameters
    ----------
    *parts:
        Path components below the repository root.
    root:
        Optional explicit repository root. When omitted, :func:`find_project_root`
        is used.
    must_exist:
        Raise ``FileNotFoundError`` if the resulting path does not exist.
    """

    base = Path(root).expanduser().resolve() if root is not None else find_project_root()
    path = base.joinpath(*map(Path, parts))
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def data_path(
    *parts: str | Path,
    root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Build a path below ``data/``."""

    return project_path("data", *parts, root=root, must_exist=must_exist)


def source_path(
    *parts: str | Path,
    root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Build a path below the immutable ``data/source/`` directory."""

    return data_path("source", *parts, root=root, must_exist=must_exist)


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "geocebada").is_dir()
