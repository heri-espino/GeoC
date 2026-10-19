"""Synthetic checks of train-only arithmetic-mean overrides and 138-fold CV."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geocebada.evaluation.peer_mean_imputation import (
    ATTRIBUTES,
    missing_numeric_inputs,
    observed_means,
    use_training_means,
)


def _runner(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "reference26_mean_imputation_loo_tests",
        root / "run_reference26_mean_imputation_loo.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_peer_classes(monkeypatch):
    """Provide five representative source imputers without teammate checkout."""
    modules = {}
    for fullname, _name, _fields, _attr in ATTRIBUTES:
        parts = fullname.split(".")
        for i in range(1, len(parts) + 1):
            path = ".".join(parts[:i])
            if path not in modules:
                node = types.ModuleType(path)
                node.__path__ = []
                modules[path] = node
                monkeypatch.setitem(sys.modules, path, node)
                if i > 1:
                    setattr(modules[".".join(parts[:i-1])], parts[-1], node)

    def make_class(name, fields_name, imputed_name):
        class FakeRepresentation:
            def __init__(self, cols):
                setattr(self, fields_name, list(cols))

            def fit(self, X, y=None):
                cols = list(getattr(self, fields_name))
                self.seen_y = y
                numeric = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
                self.training_matrix = X[numeric].to_numpy(float)
                out = []
                for c in numeric:
                    a = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
                    good = a[np.isfinite(a)]
                    out.append(float(np.median(good)) if len(good) else 0.)
                if imputed_name == "fields_":
                    self.fields_ = {
                        c: {"numeric": c in numeric, "median": out[numeric.index(c)]}
                        if c in numeric else {"numeric": False}
                        for c in cols
                    }
                else:
                    setattr(self, imputed_name, np.asarray(out))
                return self

        FakeRepresentation.__name__ = name
        return FakeRepresentation

    classes = {}
    for module_name, class_name, field_name, median_name in ATTRIBUTES:
        cls = make_class(class_name, field_name, median_name)
        setattr(modules[module_name], class_name, cls)
        classes[class_name] = cls
    return classes


def test_observed_means_training_only_and_source_unchanged():
    train = pd.DataFrame({
        "x": [1., 1., 10., np.nan],
        "category": ["a", "b", "a", "a"],
    })
    original = train.copy(deep=True)
    filled, stats = observed_means(train, ["x", "category"])
    assert stats == {"x": 4.}
    assert filled["x"].tolist() == [1., 1., 10., 4.]
    pd.testing.assert_frame_equal(train, original)
    # Excluding the high-valued donor must change the training-only mean.
    subset, stats = observed_means(train.iloc[[0, 1, 3]], ["x"])
    assert subset["x"].iloc[2] == 1.
    assert stats == {"x": 1.}


def test_all_original_preprocessors_switch_to_training_mean(monkeypatch):
    classes = _stub_peer_classes(monkeypatch)
    original_fits = {key: cls.fit for key, cls in classes.items()}
    raw = pd.DataFrame({"x": [1., 1., 10., np.nan], "group": ["a", "b", "a", "b"]})
    y = np.asarray([2., 3., 4., 5.])
    with use_training_means():
        for module_name, class_name, _field, imputed in ATTRIBUTES:
            cls = classes[class_name]
            mod = sys.modules[module_name]
            assert getattr(mod, class_name) is cls
            fitted = cls(["x", "group"] if imputed == "fields_" else ["x"]).fit(raw, y)
            assert fitted.seen_y is y
            assert fitted.training_matrix[-1, 0] == 4.
            if imputed == "fields_":
                assert fitted.fields_["x"]["median"] == 4.
                assert not fitted.fields_["group"]["numeric"]
            else:
                assert getattr(fitted, imputed)[0] == 4.
        with pytest.raises(RuntimeError, match="already active"):
            with use_training_means():
                pass
    for name, cls in classes.items():
        assert cls.fit is original_fits[name]
    # The original median remains the original median after the patch exits.
    original = classes["SignalRepresentation"](["x"]).fit(raw)
    assert original.medians_[0] == 1.


def test_numeric_missingness_audit_no_targets():
    frame = pd.DataFrame({
        "x": [1., 10., np.nan],
        "y": [5., 5., 5.],
        "category": ["a", "b", "c"],
        "RENDIMIENTO_T_HA": [3., 6., 9.],
    })
    audit = missing_numeric_inputs(frame, ["x", "y", "category"])
    assert len(audit) == 1 and audit.iloc[0]["feature"] == "x"
    assert audit.iloc[0]["missing_cells"] == 1
    assert audit.iloc[0]["mean_observed"] == 5.5
    assert audit.iloc[0]["median_observed"] == 5.5


def _data(module):
    ids = [f"P{i:03d}" for i in range(138)]
    y = np.linspace(2., 5., 138)
    raw = pd.DataFrame({
        module.ID: ids,
        module.TARGET: y,
        "x": np.linspace(1., 7., 138),
    })
    return ids, y, raw


def test_saved_baseline_alignment(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids, y, frame = _data(module)
    saved = pd.DataFrame({
        module.ID: ids[::-1],
        "observed": y[::-1],
        "Reference26": y[::-1] + .2,
    })
    path = tmp_path / "original.csv"
    module.write_csv(path, saved)
    aligned = module.load_baseline(path, frame)
    assert aligned[module.ID].tolist() == ids
    assert aligned[module.BASE].to_numpy(float) == pytest.approx(y + .2)
    saved.loc[5, "observed"] += .1
    module.write_csv(path, saved)
    with pytest.raises(ValueError, match="yields differ"):
        module.load_baseline(path, frame)


def test_outer_fold_masks_heldout_and_only_uses_137_labels(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids, y, frame = _data(module)
    counts = []

    class FakeExpert:
        def __init__(self, masked, groups, distance, geology, protocol, tr, inner, cache):
            assert np.isnan(masked.iloc[0][module.TARGET])
            assert len(tr) == 137 and 0 not in tr and len(inner) == 4
            assert np.isfinite(masked.iloc[tr][module.TARGET].to_numpy(float)).all()
            counts.append(len(tr))
            self.tr = tr

        def full_fit(self, tr):
            return {"train_positions": tr}

    def predict(pack, query, masked, distance):
        assert query.tolist() == [0]
        return {"Reference26": [3.14]}

    monkeypatch.setattr(
        module, "_INPUTS",
        (
            frame, pd.DataFrame(index=ids), np.zeros((138, 138)),
            {"geology": {}}, ids, FakeExpert, predict,
        ),
    )
    from contextlib import nullcontext

    monkeypatch.setattr(module, "use_training_means", nullcontext)
    fold, state, elapsed = module.fit_one(0, str(tmp_path))
    assert (fold, state) == (1, "trained")
    assert elapsed >= 0 and counts == [137]
    assert module.validate(
        module.fold_file(tmp_path, 0), 0, ids, y
    ) == 3.14
    fold, state, elapsed = module.fit_one(0, str(tmp_path))
    assert (fold, state, elapsed) == (1, "cached", 0.0)
    assert counts == [137]


def test_full_summary_is_paired_and_mean_does_not_overwrite_original(monkeypatch, tmp_path):
    module = _runner(monkeypatch)
    ids, y, _raw = _data(module)
    original = pd.DataFrame({
        module.ID: ids,
        "observed": y,
        module.BASE: y + .25,
    })
    for i in range(138):
        module.write_csv(
            module.fold_file(tmp_path, i),
            pd.DataFrame([{
                module.ID: ids[i], "observed": y[i], module.NEW: y[i] + .1,
            }]),
        )
    module.summary(tmp_path, original, 138)
    table = pd.read_csv(tmp_path / "loo138_summary.csv")
    assert len(table) == 2
    assert table["n_heldout"].tolist() == [138, 138]
    assert table["rmse"].tolist() == pytest.approx([.25, .1], abs=1e-12)
    assert (tmp_path / "loo138_predictions_private.csv").is_file()
