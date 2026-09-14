"""Generic file and tabular ingestion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def discover_files(
    directory: str | Path,
    pattern: str = "*",
    *,
    recursive: bool = True,
) -> list[Path]:
    """Return sorted files matching ``pattern`` below ``directory``.

    Parameters
    ----------
    directory:
        Directory to scan.
    pattern:
        Glob pattern such as ``"*.csv"`` or ``"*.tif"``.
    recursive:
        Use recursive globbing when ``True``.
    """

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def concatenate_csvs(
    source: str | Path | Iterable[str | Path],
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    source_column: str | None = None,
    ignore_index: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read and concatenate a collection of CSV files.

    ``source`` can be a directory or an explicit iterable of files. This is the
    standard helper for datasets delivered as many similarly structured CSVs.

    Parameters
    ----------
    source:
        Directory containing CSV files or explicit file paths.
    pattern:
        Glob used when ``source`` is a directory.
    recursive:
        Search subdirectories when ``source`` is a directory.
    source_column:
        Optional column containing the source filename for provenance.
    ignore_index:
        Forwarded to :func:`pandas.concat`.
    **read_csv_kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """

    paths = _resolve_input_files(source, pattern=pattern, recursive=recursive)
    if not paths:
        raise FileNotFoundError("No CSV files matched the requested source/pattern.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, **read_csv_kwargs)
        if source_column is not None:
            if source_column in frame.columns:
                raise ValueError(
                    f"source_column={source_column!r} already exists in {path.name}."
                )
            frame = frame.assign(**{source_column: path.name})
        frames.append(frame)

    return pd.concat(frames, ignore_index=ignore_index)


def _resolve_input_files(
    source: str | Path | Iterable[str | Path],
    *,
    pattern: str,
    recursive: bool,
) -> list[Path]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            return discover_files(path, pattern=pattern, recursive=recursive)
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]

    paths = sorted(Path(item).expanduser().resolve() for item in source)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")
    return paths
