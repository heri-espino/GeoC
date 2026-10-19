"""Target-free original C4 PCA20 loader and genuine old Local04D parity tests."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint08_local import (
    FROZEN,
    ID,
    Y,
    predict_old,
    verify_original_control,
)
from geocebada.evaluation.peer_frozen_local04d import load_frozen_local04d


def _historical_files(tmp_path):
    root = tmp_path
    (root / "data/processed/features_v1").mkdir(parents=True)
    (root / "data/source/tabular").mkdir(parents=True)
    (root / "reports/checkpoint_04a").mkdir(parents=True)
    (root / "reports/checkpoint_04b").mkdir(parents=True)
    (root / "reports/checkpoint_06").mkdir(parents=True)

    ids = [f"AGC_{i:03}" for i in range(1, 198)]
    roles = ["ENTRENAMIENTO" if i < 138 else "PREDICCION" for i in range(197)]
    rng = np.random.default_rng(12345)
    meta = pd.DataFrame({
        ID: ids,
        "CONJUNTO": roles,
        "base_centroid_lat": rng.normal(19.5, .18, 197),
        "base_centroid_lon": rng.normal(-98.4, .18, 197),
        "meta_estado": "Puebla",
        "meta_municipio": "A",
    })
    meta.to_csv(root / "data/processed/features_v1/parcel_features_all.csv", index=False)
    targets = meta[[ID, "CONJUNTO"]].copy()
    targets[Y] = list(np.linspace(2, 6, 138)) + [np.nan] * 59
    targets.to_csv(
        root / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        index=False,
    )
    z = rng.normal(size=(197, 20))
    embedding = meta[[ID, "CONJUNTO"]].copy()
    embedding.insert(0, "representation", "C4_all_deterministic")
    for j in range(20):
        embedding[f"pc{j + 1:02}"] = z[:, j]
    embedding.to_csv(
        root / "reports/checkpoint_04a/transductive_embeddings.csv",
        index=False,
    )
    (root / "reports/checkpoint_04a/embedding_manifest.json").write_text(
        json.dumps([{
            "representation": "C4_all_deterministic",
            "n_features_used": 2082,
            "n_components": 20,
            "explained_variance_ratio_sum": .7,
        }]),
        encoding="utf-8",
    )

    membership = root / "reports/checkpoint_08/confirmation_membership.csv"
    membership.parent.mkdir(parents=True)
    rows = []
    for k in range(16):
        held = set(ids[(k + j) % 138] for j in range(41))
        rows.extend({
            "split_id": f"confirm_{k + 1:02}",
            ID: pid,
            "role": "pseudo_target" if pid in held else "pseudo_train",
        } for pid in ids[:138])
    pd.DataFrame(rows).to_csv(membership, index=False)
    history = []
    held = set(ids[:41])
    for pid in ids[:138]:
        history.append({
            "split_id": "target_matched_01",
            "family": "target_matched",
            ID: pid,
            "role": "pseudo_target" if pid in held else "pseudo_train",
        })
    pd.DataFrame(history).to_csv(
        root / "reports/checkpoint_04b/pseudo_competition_splits.csv",
        index=False,
    )
    reference = root / "reports/checkpoint_06/parcel_oof_residuals.csv"
    pd.DataFrame({
        "split_id": "target_matched_01",
        "family": "target_matched",
        "method": "Local04D",
        ID: ids[:41],
        "predicted": 0.0,
    }).to_csv(reference, index=False)
    return root, membership, z, reference, ids


def test_exact_saved_pca20_and_original_41_predictions(tmp_path, monkeypatch):
    root, membership, frozen, reference, ids = _historical_files(tmp_path)
    # Build the original reference forecasts before switching on the strict
    # historical check. The production loader always runs the check.
    monkeypatch.setattr(
        "geocebada.evaluation.peer_frozen_local04d.verify_original_control",
        lambda *_args, **_kwargs: 0.0,
    )
    seeded = load_frozen_local04d(root, membership)
    assert seeded.old.shape == (197, 20)
    np.testing.assert_allclose(seeded.old, StandardScaler().fit_transform(frozen))

    training = ids[41:138]
    hidden = ids[:41]
    forecasts = predict_old(seeded, training, hidden, FROZEN)
    table = pd.read_csv(reference)
    table["predicted"] = forecasts
    table.to_csv(reference, index=False)

    monkeypatch.undo()
    validated = load_frozen_local04d(root, membership)
    verified_error = verify_original_control(root, validated, atol=1e-7)
    assert verified_error < 1e-7
    np.testing.assert_array_equal(validated.old, seeded.old)

    # A corrupt historical Local04D forecast must block the experiment.
    table.loc[0, "predicted"] += .1
    table.to_csv(reference, index=False)
    with pytest.raises(RuntimeError, match="Original Local04D mismatch"):
        load_frozen_local04d(root, membership)
