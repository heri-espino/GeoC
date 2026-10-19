"""Reuse prior first-46 DBSCAN outer predictions with strict protocol matching.

The previous 46-way ablation evaluated exactly the first 46 bundle rows.
This importer refuses files produced by different bundle/model/inner-fold
code and checks all 46 held-out IDs and observed targets against the live
138-row training bundle. No estimated labels or private data are published.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def reuse_verified_loo46(
    output: Path,
    previous: Path,
    frame: pd.DataFrame,
    ids: list[str],
    target: str,
    id_column: str,
    model: str,
    full_spec: dict,
) -> int:
    """Reuse prior first-46 DBSCAN outer predictions with strict protocol matching."""
    old_path = previous / "run_spec.json"
    if not old_path.is_file():
        raise FileNotFoundError(f"LOO46 manifest missing: {old_path}")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    pairs = (
        ("bundle_manifest_sha256", "bundle_manifest_sha256"),
        ("peer_source_sha256", "peer_code_sha256"),
        ("gate_implementation_sha256", "gate_code_sha256"),
        ("dbscan_implementation_sha256", "dbscan_code_sha256"),
        ("fold_protocol_sha256", "fold_code_sha256"),
    )
    for previous_key, new_key in pairs:
        if old.get(previous_key) != full_spec.get(new_key):
            raise ValueError(
                f"Cannot reuse first 46: changed input or method ({previous_key})."
            )
    if (
        old.get("outer_positions_zero_based") != list(range(46))
        or "dbscan" not in old.get("baseline_and_variants", [])
        or "Reference26_DBSCAN_geology" not in old.get("model_columns", [])
        or old.get("inner") != (
            "same four-fold train-only KFold(seed=20261018) for every model"
        )
    ):
        raise ValueError("LOO46 grid, rows or inner CV protocol differs.")
    if len(ids) != 138 or len(frame) != 138:
        raise ValueError("Expected complete 138-row original teammate bundle.")

    prepared: list[tuple[Path, pd.DataFrame]] = []
    for position in range(46):
        source = previous / f"fold_{position + 1:03d}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Cannot reuse: missing {source}")
        old_row = pd.read_csv(source, dtype={id_column: str})
        required = {id_column, "observed", "Reference26_DBSCAN_geology"}
        if len(old_row) != 1 or not required.issubset(old_row.columns):
            raise ValueError(f"Invalid first-46 fold {position + 1}.")
        record = old_row.iloc[0]
        if (
            str(record[id_column]) != ids[position]
            or not np.isfinite(
                old_row[["observed", "Reference26_DBSCAN_geology"]].to_numpy(float)
            ).all()
            or abs(float(record["observed"])
                   - float(frame.iloc[position][target])) > 1e-9
        ):
            raise ValueError(f"LOO46 fold {position + 1} does not match bundle.")
        converted = pd.DataFrame([{
            id_column: ids[position],
            "observed": float(record["observed"]),
            model: float(record["Reference26_DBSCAN_geology"]),
        }])
        destination = output / f"fold_{position + 1:03d}.csv"
        if destination.exists():
            saved = pd.read_csv(destination, dtype={id_column: str})
            if set(saved.columns) != set(converted.columns) or len(saved) != 1:
                raise ValueError(f"Existing fold {position + 1} uses wrong schema.")
            if saved.iloc[0][id_column] != ids[position] or not np.allclose(
                saved[["observed", model]].to_numpy(float),
                converted[["observed", model]].to_numpy(float),
                rtol=0, atol=1e-9,
            ):
                raise ValueError(f"Existing full LOOCV fold {position + 1} disagrees.")
        prepared.append((destination, converted))

    # Validate every first-46 row before writing any new output.
    added = 0
    output.mkdir(parents=True, exist_ok=True)
    for destination, converted in prepared:
        if not destination.exists():
            tmp = destination.with_suffix(".csv.tmp")
            converted.to_csv(tmp, index=False, float_format="%.16g")
            tmp.replace(destination)
            added += 1
    (output / "loo46_reuse_audit.json").write_text(
        json.dumps({
            "imported_new_folds": added,
            "verified_previous_folder": str(previous),
            "previous_manifest": str(old_path),
            "first_positions_zero_based": list(range(46)),
            "method": "Reference26 geological DBSCAN",
            "verification": "input/method/fold hashes, IDs and held-out yields",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return added
