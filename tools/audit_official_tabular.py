"""Print a compact schema audit for the official BASIC/PRO CSV files.

This utility is intentionally read-only. It helps developers and CI inspect the
large official CSV files without modifying anything under ``data/source``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABULAR = ROOT / "data" / "source" / "tabular"
FILES = [
    TABULAR / "Conjunto_datos_BASICO_AgroCebada2026.csv",
    TABULAR / "Conjunto_datos_PRO_AgroCebada.csv",
]


def audit_csv(path: Path) -> None:
    """Print columns, dtypes, row count and likely identifier cardinalities."""

    frame = pd.read_csv(path, low_memory=False)
    print(f"=== {path.name} ===")
    print(f"shape={frame.shape}")
    print("columns=")
    for idx, column in enumerate(frame.columns, start=1):
        print(f"  {idx:02d}. {column} [{frame[column].dtype}]")

    id_candidates = [
        column
        for column in frame.columns
        if "ID" in str(column).upper() or "POLIG" in str(column).upper()
    ]
    if id_candidates:
        print("identifier_candidates=")
        for column in id_candidates:
            print(
                f"  {column}: nunique={frame[column].nunique(dropna=True)}, "
                f"missing={int(frame[column].isna().sum())}"
            )

    print("missing_top10=")
    missing = frame.isna().sum().sort_values(ascending=False).head(10)
    for column, count in missing.items():
        print(f"  {column}: {int(count)}")
    print()


if __name__ == "__main__":
    for csv_path in FILES:
        audit_csv(csv_path)
