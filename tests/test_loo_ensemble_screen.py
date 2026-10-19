"""Synthetic, private-data-free QA for fast paired 138-LOOCV ensembles."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "tools/run_loo_ensemble_screen.py"
    spec = importlib.util.spec_from_file_location("run_loo_ensemble_screen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predictions(module):
    ids = [f"PARCEL_{i:03d}" for i in range(138)]
    y = np.linspace(2, 5, 138)
    anchor = pd.DataFrame({
        module.ID: ids, "observed": y, module.ANCHOR: y + 0.2,
    })
    local = pd.DataFrame({
        module.ID: ids[::-1], "observed": y[::-1], "Local04D": y[::-1] - 0.2,
    })
    return anchor, local


def test_strict_audit_rejects_repeated_16x41_instead_of_138_unique_heldouts():
    module = runner()
    anchor, _ = predictions(module)
    repeated = pd.concat([anchor.iloc[:41]] * 4, ignore_index=True)
    with pytest.raises(ValueError, match="138 different"):
        module.audited_model(repeated, module.ANCHOR, module.ANCHOR)
    with pytest.raises(ValueError, match="138 different"):
        module.audited_model(anchor.head(46), module.ANCHOR, module.ANCHOR)


def test_train_137_heldout_one_proof_and_misaligned_target_fail():
    module = runner()
    anchor, local = predictions(module)
    external = local.assign(outer_train=97, outer_heldout=41)
    with pytest.raises(ValueError, match="137-train/1-heldout"):
        module.audited_model(
            external, "Local04D", "Local04D", require_fold_proof=True
        )
    external["outer_train"] = 137
    external["outer_heldout"] = 1
    assert len(module.audited_model(
        external, "Local04D", "Local04D", require_fold_proof=True
    )) == 138
    local.loc[20, "observed"] += 1
    with pytest.raises(ValueError, match="observed yield mismatch"):
        module.join_models(anchor, [local])


def test_order_independent_exact_matching_and_sensible_convex_blend():
    module = runner()
    anchor, local = predictions(module)
    aligned = module.join_models(anchor, [local])
    assert aligned[module.ID].tolist() == anchor[module.ID].tolist()
    ranking, corr, diversity = module.screen(aligned)
    assert len(ranking) >= 7
    assert np.isclose(corr.loc[module.ANCHOR, "Local04D"], -1)
    assert len(diversity) == 1
    perfect = ranking.loc[
        (ranking["method"] == "anchor_plus_one")
        & ranking["weights"].str.contains('"Local04D": 0.5', regex=False)
    ]
    assert len(perfect) == 1
    assert perfect.iloc[0]["rmse"] < 1e-12


def test_mean_anchor_requires_complete_138_and_mean_provenance(tmp_path):
    module = runner()
    path = tmp_path / "mean"
    path.mkdir()
    with pytest.raises(FileNotFoundError, match="provenance"):
        module.load_mean(path)
    (path / "run_spec.json").write_text('{"imputation":"median"}')
    with pytest.raises(ValueError, match="mean-imputed"):
        module.load_mean(path)
    (path / "run_spec.json").write_text('{"imputation":"arithmetic mean"}')
    with pytest.raises(ValueError, match="all 138"):
        module.load_mean(path)


def test_external_models_require_proof_and_exact_names(tmp_path):
    module = runner()
    anchor, _ = predictions(module)
    file = tmp_path / "external.csv"
    anchor[["ID_POLIGONO", "observed"]].assign(
        predicted=anchor["observed"] + .3
    ).to_csv(file, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        module.load_extra(f"CatBoost={file}")
    model = pd.read_csv(file).assign(
        outer_train=137, outer_heldout=1
    )
    model.to_csv(file, index=False)
    name, result = module.load_extra(f"CatBoost={file}")
    assert name == "CatBoost" and result.shape == (138, 3)
    with pytest.raises(ValueError, match="Invalid/conflicting"):
        module.load_extra(f"Reference26_mean={file}")
