"""Helpers for the official AgroCebada target/split table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geocebada.paths import source_path

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"
SPLIT_COLUMN = "CONJUNTO"
TRAIN_VALUE = "ENTRENAMIENTO"
PREDICTION_VALUE = "PREDICCION"
EXPECTED_TOTAL = 197
EXPECTED_TRAIN = 138
EXPECTED_PREDICTION = 59


def load_yield_split(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the official parcel target/split table.

    Empty trailing ``Unnamed`` columns introduced by the source CSV's final
    delimiter are removed when they contain no information.
    """

    csv_path = (
        Path(path).expanduser()
        if path is not None
        else source_path(
            "tabular",
            "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
            root=root,
            must_exist=True,
        )
    )
    if path is not None and not csv_path.is_absolute():
        from geocebada.paths import project_path

        csv_path = project_path(csv_path, root=root, must_exist=True)

    frame = pd.read_csv(csv_path)
    frame = _drop_empty_unnamed_columns(frame)
    if validate:
        validate_yield_split(frame)
    return frame


def validate_yield_split(
    frame: pd.DataFrame,
    *,
    expected_total: int = EXPECTED_TOTAL,
    expected_train: int = EXPECTED_TRAIN,
    expected_prediction: int = EXPECTED_PREDICTION,
) -> None:
    """Validate the known invariants of the official 70/30 split.

    Raises ``ValueError`` when required columns, row counts, split values,
    identifier uniqueness or target missingness do not match the confirmed
    source contract.
    """

    required = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    if len(frame) != expected_total:
        raise ValueError(f"Expected {expected_total} parcels, found {len(frame)}.")
    if frame[ID_COLUMN].isna().any() or not frame[ID_COLUMN].is_unique:
        raise ValueError(f"{ID_COLUMN} must be non-null and unique.")

    unknown_splits = set(frame[SPLIT_COLUMN].dropna().unique()) - {
        TRAIN_VALUE,
        PREDICTION_VALUE,
    }
    if unknown_splits:
        raise ValueError(f"Unexpected {SPLIT_COLUMN} values: {sorted(unknown_splits)}")

    train = frame[SPLIT_COLUMN].eq(TRAIN_VALUE)
    prediction = frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)
    if int(train.sum()) != expected_train:
        raise ValueError(f"Expected {expected_train} training parcels, found {int(train.sum())}.")
    if int(prediction.sum()) != expected_prediction:
        raise ValueError(
            f"Expected {expected_prediction} prediction parcels, found {int(prediction.sum())}."
        )
    if frame.loc[train, TARGET_COLUMN].isna().any():
        raise ValueError("Training parcels must have observed yield targets.")
    if frame.loc[prediction, TARGET_COLUMN].notna().any():
        raise ValueError("Prediction parcels must not expose hidden yield targets.")


def partition_yield_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent training and prediction DataFrames."""

    validate_yield_split(frame)
    train = frame.loc[frame[SPLIT_COLUMN].eq(TRAIN_VALUE)].copy()
    prediction = frame.loc[frame[SPLIT_COLUMN].eq(PREDICTION_VALUE)].copy()
    return train, prediction


def _drop_empty_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    empty_unnamed = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:") and frame[column].isna().all()
    ]
    return frame.drop(columns=empty_unnamed)
