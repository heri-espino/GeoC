"""Load GeoC's *original saved PCA20* representation for frozen Local04D.

Checkpoint 04A stored the transductive PCA20 in a versioned CSV. This is
the exact X used by the historical Local04D forecasts, unlike a newly fitted
PCA whose component signs/conventions may differ across environments.

Never bypass a parity check: validate the frozen 197-ID X representation and
then reproduce 41 previously recorded Local04D predictions on the historical
97/41 split to within 1e-7 t/ha. No held-out future/hidden y is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import haversine_distance_matrix
from geocebada.evaluation.checkpoint04b import normalized_distance_matrix
from geocebada.evaluation.checkpoint08_local import (
    ID,
    Inputs,
    Y,
    input_paths,
    validate_memberships,
    verify_original_control,
)

REFERENCE = "C4_all_deterministic"
EXPECTED_X_FEATURES = 2082
COMPONENTS = tuple(f"pc{i:02d}" for i in range(1, 21))


def load_frozen_local04d(root: Path, membership_path: Path) -> Inputs:
    """Read canonical PCA20 (not refit), validate it via original 04D predictions."""
    paths = input_paths(root)
    manifest_path = root / "reports/checkpoint_04a/embedding_manifest.json"
    required = {
        "parcel_metadata": paths["base"],
        "official_targets": paths["targets"],
        "frozen_PCA20": paths["embeddings"],
        "PCA20_manifest": manifest_path,
        "historical_splits": paths["original_splits"],
        "historical_forecasts": paths["reference"],
        "confirmation_membership": membership_path,
    }
    for key, location in required.items():
        if not location.is_file():
            raise FileNotFoundError(
                f"Missing {key} for exact historical Local04D validation: {location}"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [r for r in manifest if r.get("representation") == REFERENCE]
    if len(matches) != 1:
        raise ValueError("Frozen PCA20 manifest has no unique C4 representation.")
    record = matches[0]
    if (record["n_components"] != 20 or
            record["n_features_used"] != EXPECTED_X_FEATURES):
        raise ValueError("Frozen C4 feature dimension or PCA rank has changed.")

    meta = pd.read_csv(
        paths["base"],
        usecols=[
            ID, "CONJUNTO", "base_centroid_lat", "base_centroid_lon",
            "meta_estado", "meta_municipio",
        ],
    )
    meta[ID] = meta[ID].astype(str)
    if len(meta) != 197 or not meta[ID].is_unique:
        raise ValueError("GeoC base metadata must have exactly 197 unique parcels.")
    if not meta["CONJUNTO"].isin(["ENTRENAMIENTO", "PREDICCION"]).all():
        raise ValueError("Unexpected training/prediction role in GeoC metadata.")
    if meta[["base_centroid_lat", "base_centroid_lon"]].isna().any().any():
        raise ValueError("Missing parcel coordinates for Local04D distances.")
    memberships = pd.read_csv(membership_path)
    folds = validate_memberships(meta, memberships)

    targets = pd.read_csv(paths["targets"], usecols=[ID, "CONJUNTO", Y])
    targets[ID] = targets[ID].astype(str)
    if (len(targets) != 197 or not targets[ID].is_unique
            or set(targets[ID]) != set(meta[ID])):
        raise ValueError("Official yield table does not match all 197 parcel IDs.")
    same_roles = (
        targets.set_index(ID).loc[meta[ID], "CONJUNTO"].to_numpy()
        == meta["CONJUNTO"].to_numpy()
    )
    if not bool(np.all(same_roles)):
        raise ValueError("Official training/prediction roles differ from metadata.")
    y = targets.loc[targets["CONJUNTO"].eq("ENTRENAMIENTO")].set_index(ID)[Y]
    hidden = targets.loc[targets["CONJUNTO"].eq("PREDICCION"), Y]
    if (len(y) != 138 or not np.isfinite(y.to_numpy(float)).all()
            or len(hidden) != 59 or hidden.notna().any()):
        raise ValueError("Expected 138 finite observed yields and 59 hidden yields.")

    saved = pd.read_csv(paths["embeddings"], dtype={ID: str})
    c4 = saved.loc[saved["representation"].eq(REFERENCE)].copy()
    if (len(c4) != 197 or not c4[ID].is_unique
            or set(c4[ID]) != set(meta[ID])):
        raise ValueError("Saved C4 PCA20 does not cover exactly 197 unique parcels.")
    if not set(COMPONENTS).issubset(c4.columns):
        raise ValueError("Saved C4 PCA is missing one or more of its 20 axes.")
    if (c4.set_index(ID).loc[meta[ID], "CONJUNTO"].to_numpy()
            != meta["CONJUNTO"].to_numpy()).any():
        raise ValueError("Frozen PCA training/prediction membership changed.")
    original20 = c4.set_index(ID).loc[meta[ID], list(COMPONENTS)].to_numpy(float)
    if original20.shape != (197, 20) or not np.isfinite(original20).all():
        raise ValueError("Invalid or nonfinite frozen Local04D PCA20 array.")
    x = StandardScaler().fit_transform(original20)

    distance = haversine_distance_matrix(
        meta["base_centroid_lat"], meta["base_centroid_lon"]
    )
    geographic, _scale = normalized_distance_matrix(
        distance,
        reference_positions=np.flatnonzero(
            meta["CONJUNTO"].eq("ENTRENAMIENTO").to_numpy()
        ),
    )
    if geographic.shape != (197, 197) or not np.isfinite(geographic).all():
        raise ValueError("Original Local04D geographic distances are invalid.")
    data = Inputs(
        x07=pd.DataFrame(index=meta[ID].astype(str)),
        old=x,
        geographic=geographic,
        index={pid: idx for idx, pid in enumerate(meta[ID])},
        y=y.astype(float),
        meta=meta,
        folds=folds,
    )
    # Validate against actual saved historical 04D predictions, not against
    # another independent SVD that could have arbitrary component signs.
    max_error = verify_original_control(root, data, atol=1e-7)
    print(
        "[LOCAL04D] Original checkpoint04a frozen PCA20, "
        f"197 IDs, 138 labeled / 59 hidden; "
        f"41-prediction historical parity max diff={max_error:.3g} t/ha.",
        flush=True,
    )
    return data
