import pandas as pd

from geocebada.data import attach_yield_split_metadata, filter_frame
from geocebada.geo import aggregate_to_parcels
from geocebada.visualization import correlation_heatmap, outlier_summary, pairplot_figure


def test_filter_frame_combines_filter_types() -> None:
    frame = pd.DataFrame(
        {
            "id": ["AGC_001", "AGC_002", "AGC_101"],
            "group": ["A", "A", "B"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    result = filter_frame(
        frame,
        categorical={"group": ["A"]},
        numeric_ranges={"value": (1.5, 3.0)},
        text_contains={"id": "AGC"},
    )
    assert result["id"].tolist() == ["AGC_002"]


def test_aggregate_to_parcels_uses_mean_and_first() -> None:
    frame = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "A", "B"],
            "ndvi": [0.2, 0.6, 0.4],
            "group": ["train", "train", "prediction"],
        }
    )
    result = aggregate_to_parcels(frame, "ID_POLIGONO").set_index("ID_POLIGONO")
    assert result.loc["A", "ndvi"] == 0.4
    assert result.loc["A", "group"] == "train"


def test_visualization_helpers_return_figures_and_outlier_summary() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 100.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "group": ["a", "a", "b", "b"],
        }
    )
    pair = pairplot_figure(frame, ["x", "y"], color="group")
    heatmap = correlation_heatmap(frame, ["x", "y"], method="spearman")
    summary = outlier_summary(frame, ["x", "y"])

    assert len(pair.data) >= 1
    assert len(heatmap.data) >= 1
    assert set(summary["feature"]) == {"x", "y"}
    assert summary.loc[summary["feature"].eq("x"), "outliers"].iloc[0] == 1


def test_attach_yield_split_metadata_preserves_hidden_target() -> None:
    frame = pd.DataFrame({"ID_POLIGONO": ["AGC_001", "AGC_003"], "feature": [1.0, 2.0]})
    result = attach_yield_split_metadata(frame)

    training = result.loc[result["ID_POLIGONO"].eq("AGC_001")].iloc[0]
    prediction = result.loc[result["ID_POLIGONO"].eq("AGC_003")].iloc[0]
    assert training["CONJUNTO"] == "ENTRENAMIENTO"
    assert pd.notna(training["RENDIMIENTO_T_HA"])
    assert prediction["CONJUNTO"] == "PREDICCION"
    assert pd.isna(prediction["RENDIMIENTO_T_HA"])
