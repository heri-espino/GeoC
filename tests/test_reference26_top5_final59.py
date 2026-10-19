"""Synthetic QA for private final 138-trained corrected Reference26 59 export."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def loaded(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "run_reference26_top5_final59",
        folder / "run_reference26_top5_final59.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial(m):
    rng = np.random.default_rng(20261019)
    z = rng.normal(size=(197, 26))
    pca = rng.normal(size=(197, 8))
    y = rng.normal(loc=4, scale=.4, size=138)
    spaces = {"robust26": z, "pca8": pca}
    ids = [f"UNKNOWN_{i:03d}" for i in range(59)]
    baseline = np.linspace(2.8, 5.1, 59)
    return y, spaces, ids, baseline


def test_final_all_five_match_original_fold_safe_corrections(monkeypatch):
    m = loaded(monkeypatch)
    y, spaces, ids, base = artificial(m)
    output = m.corrected_targets(
        y, spaces, ids, base, m.nested.NAMES[0], .5
    )
    assert len(output) == 59
    assert output[m.ID].is_unique
    np.testing.assert_array_equal(output["Reference26_mean_t_ha"], base)
    for cfg in m.nested.TOP5:
        name = m.analog.signature(cfg)
        corrected = m.nested.apply_candidate(
            cfg, spaces[cfg["geometry"]], y, np.arange(138),
            np.arange(138, 197), base, .5
        )
        np.testing.assert_array_equal(output[name+"_t_ha"], corrected)
    np.testing.assert_array_equal(
        output["NestedSelected_final_t_ha"],
        output[m.nested.NAMES[0]+"_t_ha"],
    )
    assert output["selected_model"].nunique() == 1


def test_native_fallback_returns_exact_original_predictions(monkeypatch):
    m = loaded(monkeypatch)
    y, spaces, ids, base = artificial(m)
    output = m.corrected_targets(y, spaces, ids, base, m.nested.BASE, .5)
    np.testing.assert_array_equal(output["NestedSelected_final_t_ha"], base)
    np.testing.assert_array_equal(output["delta_vs_reference26_t_ha"], 0)


def test_nonexistent_hidden_y_cannot_be_used_for_59(monkeypatch):
    m = loaded(monkeypatch)
    y, spaces, ids, base = artificial(m)
    first = m.corrected_targets(
        y, spaces, ids, base, m.nested.NAMES[1], .5
    )
    # Only original known y[0:138] is an argument: changing the 59 unknown
    # target feature vectors changes analog predictions but no hidden labels
    # exist to be read.
    y_altered = y.copy()
    y_altered += .1
    second = m.corrected_targets(
        y_altered, spaces, ids, base, m.nested.NAMES[1], .5
    )
    assert first.shape == second.shape
    assert (first["NestedSelected_final_t_ha"].to_numpy() !=
            second["NestedSelected_final_t_ha"].to_numpy()).any()


def test_reject_illegal_selection_or_size(monkeypatch):
    m = loaded(monkeypatch)
    y, spaces, ids, base = artificial(m)
    with pytest.raises(ValueError, match="Invalid"):
        m.corrected_targets(y, spaces, ids, base, "invented", .5)
    with pytest.raises(ValueError, match="Invalid"):
        m.corrected_targets(y, spaces, ids[:-1], base, m.nested.BASE, .5)


def test_inner_fold_selection_uses_training_y_exclusively(monkeypatch, tmp_path):
    m = loaded(monkeypatch)
    y, spaces, ids, base = artificial(m)
    known = pd.DataFrame({
        m.ID: [f"REAL_{i:03d}" for i in range(138)],
        m.TARGET: y
    })
    seen = []

    def fake_fit(frame, groups, distance, manifest, train, inner,
                 expert, folder, *, mean):
        train = np.asarray(train, int)
        assert mean
        validation = np.setdiff1d(np.arange(138), train)
        assert len(train) in (103, 104)
        assert len(validation) in (34, 35)
        masked = frame.copy(deep=True)
        masked.loc[validation, m.TARGET] = np.nan
        seen.append(tuple(train))
        return {"train_positions": train}, masked

    def fake_predict(predict, pack, query, masked, distances, key):
        assert key == "Reference26"
        train = pack["train_positions"]
        assert not np.intersect1d(train, query).size
        assert masked.iloc[query][m.TARGET].isna().all()
        return np.full(len(query), masked.iloc[train][m.TARGET].mean())

    monkeypatch.setattr(m.native, "build_fresh_fit", fake_fit)
    monkeypatch.setattr(m.native, "predict_model", fake_predict)
    one = m.inner_fold_predictions(
        0, known, None, np.zeros((138, 138)), {}, None,
        None, spaces, y, tmp_path, max_delta=.5
    )
    m.validate_inner_fold(one, 0, known, y)
    assert len(seen) == 1
    assert len(one) in (34, 35)
    y_changed = y.copy()
    y_changed[m.inner_splits()[0][1]] += 999.
    known2 = known.copy()
    known2[m.TARGET] = y_changed
    two = m.inner_fold_predictions(
        0, known2, None, np.zeros((138, 138)), {}, None,
        None, spaces, y_changed, tmp_path, max_delta=.5
    )
    for name in ("Reference26_mean", *m.nested.NAMES):
        np.testing.assert_array_equal(one[name], two[name])


def test_final_selector_chooses_based_only_on_four_inner_oof(monkeypatch):
    m = loaded(monkeypatch)
    rng = np.random.default_rng(45)
    data = []
    for _fit, val in m.inner_splits():
        truth = rng.normal(size=len(val))
        row = {
            m.ID: [f"T_{i:03d}" for i in val],
            "observed": truth,
            "Reference26_mean": truth+.5,
        }
        for name in m.nested.NAMES:
            row[name] = truth + (.2 if name == m.nested.NAMES[0] else .6)
        data.append(pd.DataFrame(row))
    selected, losses, combined = m.combine_inner_selection(
        None, data, min_inner_gain=.01
    )
    assert selected == m.nested.NAMES[0]
    assert len(combined) == 138
    assert losses[selected] < losses[m.nested.BASE]


def test_nested_manifest_allows_only_code_sha_drift(monkeypatch):
    m = loaded(monkeypatch)
    expected = {
        "schema": 1,
        "source_sha": "new-script",
        "screen_source_sha": "new-screen",
        "screen_private_sha": "same-private-screen",
        "native_run_spec_sha": "same-model-fits",
        "peer_source_sha": "same-native-source",
        "bundle_sha": "same-training-bundle",
        "minimum_relative_inner_mse_gain": 0.01,
        "max_delta": 0.5,
        "top5": list(m.nested.TOP5),
    }
    assert m.validate_nested_provenance(expected, expected) == ()
    # The nested runner writes tuple-valued TOP5 as a JSON list on disk.
    # Reproduce the actual saved-manifest versus freshly built situation.
    from_disk = json.loads(json.dumps({
        **expected, "top5": m.nested.TOP5,
    }))
    freshly_computed = {**expected, "top5": m.nested.TOP5}
    assert isinstance(from_disk["top5"], list)
    assert isinstance(freshly_computed["top5"], tuple)
    assert m.validate_nested_provenance(
        from_disk, freshly_computed
    ) == ()
    mutated = dict(from_disk)
    mutated["top5"] = [dict(value) for value in from_disk["top5"]]
    mutated["top5"][0]["blend"] = 0.75
    with pytest.raises(ValueError, match="top5"):
        m.validate_nested_provenance(mutated, freshly_computed)
    reverse_order = dict(from_disk)
    reverse_order["top5"] = list(reversed(from_disk["top5"]))
    with pytest.raises(ValueError, match="top5"):
        m.validate_nested_provenance(reverse_order, freshly_computed)
    old = dict(expected)
    old["source_sha"] = "original-nested-run"
    old["screen_source_sha"] = "original-analog-screen"
    assert m.validate_nested_provenance(old, expected) == (
        "screen_source_sha", "source_sha"
    )
    assert m.validate_nested_provenance(
        {**old, "source_sha": "a"}, expected
    )
    for key, replacement in (
        ("bundle_sha", "wrong-bundle"),
        ("screen_private_sha", "wrong-screen-records"),
        ("peer_source_sha", "different-native-training-code"),
        ("native_run_spec_sha", "different-loocv-baseline"),
        ("minimum_relative_inner_mse_gain", 0.05),
        ("max_delta", 1.0),
        ("top5", []),
    ):
        tampered = dict(old)
        tampered[key] = replacement
        with pytest.raises(ValueError, match=key):
            m.validate_nested_provenance(tampered, expected)


def test_nested_manifest_missing_fields_are_rejected(monkeypatch):
    m = loaded(monkeypatch)
    expected = {"source_sha": "new", "bundle_sha": "real"}
    with pytest.raises(ValueError, match="bundle_sha"):
        m.validate_nested_provenance({"source_sha": "old"}, expected)
    with pytest.raises(ValueError, match="manifest must"):
        m.validate_nested_provenance(None, expected)
