from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from geocebada.models.onnx_export import (
    export_regressor_to_onnx,
    verify_onnx_regressor,
)


def test_ridge_onnx_roundtrip(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 4)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + 2.0).astype(np.float32)
    model = Ridge(alpha=1.0).fit(x, y)

    path = export_regressor_to_onnx(
        model,
        kind="ridge",
        n_features=x.shape[1],
        path=tmp_path / "ridge.onnx",
        target_opset=17,
    )
    result = verify_onnx_regressor(
        model,
        x,
        path=path,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert result["verified"] is True
    assert result["rows"] == len(x)



def test_catboost_native_onnx_roundtrip(tmp_path) -> None:
    pytest.importorskip("catboost")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from catboost import CatBoostRegressor

    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    y = (1.0 + 0.4 * x[:, 0] - 0.2 * x[:, 2]).astype(np.float32)
    model = CatBoostRegressor(
        iterations=8,
        depth=3,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        random_seed=7,
    ).fit(x, y)

    path = export_regressor_to_onnx(
        model,
        kind="catboost",
        n_features=x.shape[1],
        path=tmp_path / "catboost.onnx",
        target_opset=17,
    )
    result = verify_onnx_regressor(
        model,
        x,
        path=path,
        atol=5.0e-4,
        rtol=5.0e-4,
    )
    assert result["verified"] is True
    assert result["rows"] == len(x)
