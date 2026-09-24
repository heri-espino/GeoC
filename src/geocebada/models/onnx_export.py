"""ONNX export and round-trip verification for deployable regressors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def export_regressor_to_onnx(
    model: Any,
    *,
    kind: str,
    n_features: int,
    path: str | Path,
    target_opset: int = 17,
) -> Path:
    """Export one fitted regressor to ONNX using the appropriate converter."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_kind = str(kind).casefold()

    if normalized_kind in {"ridge", "pls", "extra_trees", "hist_gradient_boosting"}:
        try:
            from skl2onnx import to_onnx
        except ImportError as exc:
            raise ImportError(
                "skl2onnx is required for ONNX deployment. Install .[deployment]."
            ) from exc

        sample = np.zeros((1, int(n_features)), dtype=np.float32)
        onnx_model = to_onnx(
            model,
            sample,
            target_opset=int(target_opset),
        )

    elif normalized_kind == "catboost":
        if not hasattr(model, "save_model"):
            raise TypeError("CatBoost ONNX export requires a fitted CatBoost model.")
        model.save_model(
            str(destination),
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.geocebada",
                "onnx_model_version": 1,
                "onnx_doc_string": "GeoC Checkpoint 03D regressor",
                "onnx_graph_name": "GeoC_Checkpoint03D_CatBoostRegressor",
            },
        )
        onnx_model = None

    elif normalized_kind in {"xgboost", "lightgbm"}:
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType
            from onnxmltools.convert.common.onnx_ex import (
                get_maximum_opset_supported,
            )
        except ImportError as exc:
            raise ImportError(
                "onnxmltools is required for boosted-tree ONNX export. "
                "Install .[deployment]."
            ) from exc

        initial_types = [("features", FloatTensorType([None, int(n_features)]))]
        effective_opset = min(
            int(target_opset),
            int(get_maximum_opset_supported()),
        )
        if normalized_kind == "xgboost":
            onnx_model = onnxmltools.convert_xgboost(
                model,
                initial_types=initial_types,
                target_opset=effective_opset,
            )
        else:
            onnx_model = onnxmltools.convert_lightgbm(
                model,
                initial_types=initial_types,
                target_opset=effective_opset,
            )

    else:
        raise ValueError(f"Unsupported ONNX regressor kind: {kind}")

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate exported models. Install .[deployment]."
        ) from exc

    if onnx_model is None:
        onnx_model = onnx.load(str(destination))
    else:
        destination.write_bytes(onnx_model.SerializeToString())

    onnx.checker.check_model(onnx_model)
    return destination


def verify_onnx_regressor(
    model: Any,
    x: np.ndarray,
    *,
    path: str | Path,
    atol: float = 5.0e-4,
    rtol: float = 5.0e-4,
) -> dict[str, Any]:
    """Compare Python and ONNX Runtime predictions on the same numeric matrix."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX verification. Install .[deployment]."
        ) from exc

    matrix = np.asarray(x, dtype=np.float32)
    expected = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    model_path = Path(path)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_meta = session.get_outputs()[0]
    output_shape = list(output_meta.shape)
    single_target_schema = (
        len(output_shape) == 1
        or (
            len(output_shape) == 2
            and output_shape[1] == 1
        )
    )
    if not single_target_schema:
        raise RuntimeError(
            "ONNX output schema is not single-target regression: "
            f"{output_shape}."
        )

    outputs = session.run(None, {input_name: matrix})
    if not outputs:
        raise RuntimeError("ONNX Runtime produced no outputs.")
    actual = np.asarray(outputs[0], dtype=float).reshape(-1)
    if len(actual) != len(expected):
        raise RuntimeError(
            f"ONNX output length mismatch: {len(actual)} != {len(expected)}."
        )

    max_abs = float(np.max(np.abs(expected - actual))) if len(actual) else 0.0
    verified = bool(
        np.allclose(
            expected,
            actual,
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    if not verified:
        raise RuntimeError(
            "ONNX round-trip verification failed; "
            f"max_abs_difference={max_abs:.8g}."
        )
    try:
        import onnx

        proto = onnx.load(str(model_path))
        opset_imports = {
            str(item.domain or "ai.onnx"): int(item.version)
            for item in proto.opset_import
        }
    except Exception:
        opset_imports = {}

    return {
        "verified": True,
        "rows": int(len(actual)),
        "max_abs_difference": max_abs,
        "output_shape": output_shape,
        "opset_imports": opset_imports,
    }
