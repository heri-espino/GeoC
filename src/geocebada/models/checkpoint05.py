"""Persistence and fixed-target inference helpers for Checkpoint 05."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ID_COLUMN = "ID_POLIGONO"
TARGET_COLUMN = "RENDIMIENTO_T_HA"


def load_checkpoint05_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a serialized Checkpoint 05 model bundle."""

    bundle_path = Path(path).expanduser().resolve()
    payload = joblib.load(bundle_path)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint 05 bundle must deserialize to a dictionary.")

    required = {
        "manifest",
        "final_method",
        "final_predictions",
        "actual_candidates",
        "final_diagnostics",
        "target_ids",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Checkpoint 05 bundle is missing keys: {missing}")

    manifest = payload["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("Checkpoint 05 bundle manifest must be a mapping.")
    if str(manifest.get("checkpoint")) != "05":
        raise ValueError("Serialized bundle is not a Checkpoint 05 artifact.")
    if str(manifest.get("inference_scope")) != "fixed_59_competition_targets":
        raise ValueError("Unsupported Checkpoint 05 inference scope.")
    return payload


def fixed_target_predictions(bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Return the canonical fixed-59 prediction table from a loaded bundle."""

    frame = pd.DataFrame(bundle["final_predictions"]).copy()
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Final prediction table is missing columns: {missing}")
    if len(frame) != 59:
        raise ValueError(f"Expected 59 fixed-target predictions, found {len(frame)}.")
    if frame[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Fixed-target prediction IDs are not unique.")
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Fixed-target predictions contain non-finite values.")

    expected_ids = set(map(str, bundle["target_ids"]))
    actual_ids = set(frame[ID_COLUMN].astype(str))
    if actual_ids != expected_ids:
        raise ValueError("Bundle target IDs do not match the final prediction table.")

    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame[TARGET_COLUMN] = values.astype(float)
    return frame.sort_values(ID_COLUMN).reset_index(drop=True)


def fixed_target_diagnostics(bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Return app-facing diagnostics for the fixed competition targets."""

    frame = pd.DataFrame(bundle["final_diagnostics"]).copy()
    if ID_COLUMN not in frame.columns:
        raise KeyError(f"Diagnostics are missing {ID_COLUMN}.")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Fixed-target diagnostic IDs are not unique.")
    if set(frame[ID_COLUMN]) != set(map(str, bundle["target_ids"])):
        raise ValueError("Diagnostic IDs do not match bundle target IDs.")
    return frame.sort_values(ID_COLUMN).reset_index(drop=True)


def verify_fixed_target_bundle(
    bundle: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Verify stored final values against the selected actual-candidate column."""

    final = fixed_target_predictions(bundle)
    candidates = pd.DataFrame(bundle["actual_candidates"]).copy()
    required = {ID_COLUMN, "method", "predicted"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise KeyError(f"Candidate table is missing columns: {missing}")

    method = str(bundle["final_method"])
    selected = candidates.loc[
        candidates["method"].astype(str).eq(method),
        [ID_COLUMN, "predicted"],
    ].copy()
    selected[ID_COLUMN] = selected[ID_COLUMN].astype(str)
    merged = final.merge(
        selected,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 59:
        raise ValueError("Selected candidate does not cover all 59 fixed targets.")

    difference = np.abs(
        merged[TARGET_COLUMN].to_numpy(float)
        - merged["predicted"].to_numpy(float)
    )
    max_difference = float(np.max(difference))
    if max_difference > float(tolerance):
        raise ValueError(
            "Bundle final predictions do not reproduce the selected candidate: "
            f"max abs diff={max_difference:g}."
        )

    return {
        "checkpoint": "05",
        "final_method": method,
        "prediction_rows": int(len(merged)),
        "max_abs_difference": max_difference,
        "verified": True,
    }
