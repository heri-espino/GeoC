import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from geocebada.evaluation import benchmark_regressors, summarize_benchmark
from geocebada.features import FeatureRecipe, apply_feature_recipe
from geocebada.statistics import adjust_pvalues, correlation_test, linear_regression_diagnostics


def test_adjust_pvalues_controls_bounds() -> None:
    adjusted = adjust_pvalues([0.01, 0.04, 0.20], method="fdr_bh")
    assert adjusted.shape == (3,)
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_correlation_and_linear_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(1.0, 11.0),
            "z": np.linspace(0.0, 1.0, 10),
        }
    )
    frame["y"] = 2.0 * frame["x"] + 0.5 * frame["z"]

    association = correlation_test(frame["x"], frame["y"], method="pearson")
    assert association["n"] == 10
    assert association["statistic"] > 0.99

    diagnostics = linear_regression_diagnostics(frame, "y", ["x", "z"])
    assert diagnostics["r2"] > 0.999
    assert len(diagnostics["coefficients"]) == 2


def test_temporal_feature_recipe() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A", "B", "B"],
            "date": ["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"],
            "ndvi": [0.2, 0.6, 0.1, 0.3],
        }
    )
    recipe = FeatureRecipe(
        name="ndvi_mean",
        value_column="ndvi",
        group_column="ID_POLIGONO",
        aggregation="mean",
        date_column="date",
    )
    result = apply_feature_recipe(frame, recipe).set_index("ID_POLIGONO")
    assert result.loc["A", "ndvi_mean"] == 0.4
    assert result.loc["B", "ndvi_mean"] == 0.2


def test_regression_benchmark_with_custom_model() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(20, dtype=float),
            "y": np.arange(20, dtype=float) * 3.0,
        }
    )
    scores = benchmark_regressors(
        frame,
        "y",
        ["x"],
        regressors={"linear": LinearRegression()},
        n_splits=4,
        random_state=42,
    )
    summary = summarize_benchmark(scores)
    assert len(scores) == 4
    assert summary.loc[0, "model"] == "linear"
    assert summary.loc[0, "rmse_mean"] < 1e-10
