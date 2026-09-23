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

    elif normalized_kind in {"xgboost", "lightgbm", "catboost"}:
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError as exc:
            raise ImportError(
                "onnxmltools is required for boosted-tree ONNX export. "
                "Install .[deployment]."
            ) from exc

        initial_types = [
            ("features", FloatTensorType([None, int(n_features)]))
        ]
        if normalized_kind == "xgboost":
            onnx_model = onnxmltools.convert_xgboost(
                model,
                initial_types=initial_types,
                target_opset=int(target_opset),
            )
        elif normalized_kind == "lightgbm":
            onnx_model = onnxmltools.convert_lightgbm(
                model,
                initial_types=initial_types,
                target_opset=int(target_opset),
            )
        else:
            onnx_model = onnxmltools.convert_catboost(
                model,
                initial_types=initial_types,
                target_opset=int(target_opset),
            )

    else:
        raise ValueError(f"Unsupported ONNX regressor kind: {kind}")

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate exported models. Install .[deployment]."
        ) from exc

    onnx.checker.check_model(onnx_model)
    destination.write_bytes(onnx_model.SerializeToString())
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
    session = ort.InferenceSession(
        str(Path(path)),
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
    return {
        "verified": True,
        "rows": int(len(actual)),
        "max_abs_difference": max_abs,
        "output_shape": output_shape,
    }
