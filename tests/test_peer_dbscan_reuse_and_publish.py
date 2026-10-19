"""Tests for publishing safe aggregates and recycling verified 46-fold DBSCAN."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_dbscan_reuse import reuse_verified_loo46

ROOT = Path(__file__).resolve().parents[1]


def _exporter():
    import importlib.util

    path = ROOT / "tools/publish_peer_results.py"
    spec = importlib.util.spec_from_file_location("publish_peer_results_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_export_whitelist_excludes_identifiers_and_yields(tmp_path):
    export = _exporter()
    private = tmp_path / "private.csv"
    pd.DataFrame({
        "model": ["Reference26", "Compact14"],
        "held_out_parcels": [138, 138],
        "train_labels_per_fold": [137, 137],
        "rmse": [.4682, .4697],
        "r2": [.6952, .6933],
        "mae": [.333, .334],
        "ID_POLIGONO": ["SECRET1", "SECRET2"],
        "observed": [2.5, 2.7],
        "predicted": [2.1, 2.2],
    }).to_csv(private, index=False)
    safe = export.audited_scores(
        private, count=138, allowed={"Reference26", "Compact14"}
    )
    assert set(safe.columns) == {
        "model", "held_out_parcels", "train_labels_per_fold",
        "rmse", "r2", "mae",
    }
    assert "SECRET1" not in safe.to_csv(index=False)
    with pytest.raises(ValueError, match="Wrong fold count"):
        export.audited_scores(
            private, count=46, allowed={"Reference26", "Compact14"}
        )


def test_export_public_aggregates_only_when_summary_complete(tmp_path):
    export = _exporter()
    private = tmp_path / "reports/selected_champions_loo/loo_summary.csv"
    private.parent.mkdir(parents=True)
    pd.DataFrame({
        "model": ["Reference26", "Compact14"],
        "held_out_parcels": [138, 138],
        "train_labels_per_fold": [137, 137],
        "rmse": [.4682, .4697],
        "mae": [.333, .334],
        "r2": [.6952, .6933],
        "observed": [99.0, 99.0],
    }).to_csv(private, index=False)
    destination = tmp_path / "published"
    records = export.export(root=tmp_path, destination=destination)
    assert len(records) == 1
    expected = destination / "reference_compact_loo138.csv"
    assert expected.is_file()
    public = pd.read_csv(expected)
    assert "observed" not in public
    assert len(public) == 2
    assert "SECRET" not in (destination / "sources.json").read_text()


def _previous_fold_bank(tmp_path):
    former = tmp_path / "loo46"
    current = tmp_path / "full138"
    former.mkdir()
    ids = [f"P{i:03d}" for i in range(138)]
    frame = pd.DataFrame({
        "ID_POLIGONO": ids, "RENDIMIENTO_T_HA": np.linspace(2, 5, 138)
    })
    full_spec = {
        "bundle_manifest_sha256": "manifest1",
        "peer_code_sha256": "peer1",
        "gate_code_sha256": "gate1",
        "dbscan_code_sha256": "dbscan1",
        "fold_code_sha256": "fold1",
    }
    previous = {
        "bundle_manifest_sha256": "manifest1",
        "peer_source_sha256": "peer1",
        "gate_implementation_sha256": "gate1",
        "dbscan_implementation_sha256": "dbscan1",
        "fold_protocol_sha256": "fold1",
        "outer_positions_zero_based": list(range(46)),
        "baseline_and_variants": [
            "original", "kmeans_expanded", "dbscan", "support", "global"
        ],
        "model_columns": [
            "Compact14_original", "Reference26_original_KMeans",
            "Reference26_DBSCAN_geology",
        ],
        "inner": "same four-fold train-only KFold(seed=20261018) for every model",
    }
    (former / "run_spec.json").write_text(json.dumps(previous))
    for pos in range(46):
        pd.DataFrame([{
            "ID_POLIGONO": ids[pos],
            "observed": float(frame.iloc[pos]["RENDIMIENTO_T_HA"]),
            "Reference26_DBSCAN_geology": 3.0,
            "Compact14_original": 3.1,
        }]).to_csv(former / f"fold_{pos+1:03d}.csv", index=False)
    return former, current, frame, ids, full_spec


def _reuse(former, current, frame, ids, spec):
    return reuse_verified_loo46(
        current, former, frame, ids, "RENDIMIENTO_T_HA",
        "ID_POLIGONO", "Reference26_DBSCAN", spec,
    )


def test_reuse_exact_first_46_with_no_retraining(tmp_path):
    former, current, frame, ids, spec = _previous_fold_bank(tmp_path)
    assert _reuse(former, current, frame, ids, spec) == 46
    assert _reuse(former, current, frame, ids, spec) == 0
    results = pd.read_csv(current / "fold_046.csv")
    assert list(results.columns) == [
        "ID_POLIGONO", "observed", "Reference26_DBSCAN"
    ]


def test_reuse_rejects_changed_protocol_and_yields_before_copying(tmp_path):
    former, current, frame, ids, spec = _previous_fold_bank(tmp_path)
    incompatible = {**spec, "peer_code_sha256": "different"}
    with pytest.raises(ValueError, match="changed input"):
        _reuse(former, current, frame, ids, incompatible)
    assert not list(current.glob("fold_*.csv"))

    changed = frame.copy()
    changed.loc[20, "RENDIMIENTO_T_HA"] += .4
    with pytest.raises(ValueError, match="does not match bundle"):
        _reuse(former, current, changed, ids, spec)
    assert not list(current.glob("fold_*.csv"))
