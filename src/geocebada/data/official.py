"""Load the official AgroCebada tabular remote-sensing datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geocebada.paths import source_path

BASIC_FILENAME = "Conjunto_datos_BASICO_AgroCebada2026.csv"
PRO_FILENAME = "Conjunto_datos_PRO_AgroCebada.csv"


def load_official_tabular(
    kind: str,
    *,
    root: str | Path | None = None,
    attach_split: bool = False,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load the official BASIC or PRO table.

    Parameters
    ----------
    kind:
        ``"basic"`` or ``"pro"``.
    root:
        Optional repository root override.
    attach_split:
        If ``True``, attach missing area, split and target metadata by
        ``ID_POLIGONO``. Hidden prediction targets remain missing.
    low_memory:
        Passed to :func:`pandas.read_csv`.
    """

    normalized = kind.strip().lower()
    filenames = {"basic": BASIC_FILENAME, "pro": PRO_FILENAME}
    if normalized not in filenames:
        raise ValueError("kind must be 'basic' or 'pro'.")

    path = source_path("tabular", filenames[normalized], root=root, must_exist=True)
    frame = pd.read_csv(path, low_memory=low_memory)

    if attach_split:
        from geocebada.data.targets import attach_yield_split_metadata

        frame = attach_yield_split_metadata(frame)
    return frame


def load_basic_data(
    *,
    root: str | Path | None = None,
    attach_split: bool = False,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load the official BASIC remote-sensing table."""

    return load_official_tabular(
        "basic",
        root=root,
        attach_split=attach_split,
        low_memory=low_memory,
    )


def load_pro_data(
    *,
    root: str | Path | None = None,
    attach_split: bool = False,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load the official PRO remote-sensing table."""

    return load_official_tabular(
        "pro",
        root=root,
        attach_split=attach_split,
        low_memory=low_memory,
    )
