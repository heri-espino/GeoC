"""Synthetic geometry QA; no observed barley yields or proprietary parcels."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances


def load(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "plot_reference26_error_geometry_197",
        directory/"plot_reference26_error_geometry_197.py"
    )
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classical_mds_reconstructs_euclidean_geography(monkeypatch):
    m=load(monkeypatch)
    rng=np.random.default_rng(5)
    x=rng.normal(size=(35,2))
    d=pairwise_distances(x)
    reconstructed=m.classical_mds(d)
    assert reconstructed.shape==(35,2)
    np.testing.assert_allclose(
        pairwise_distances(reconstructed), d, atol=1e-6
    )


def test_private_top10_and_feature_outlier_metrics(monkeypatch):
    m=load(monkeypatch)
    rng=np.random.default_rng(6)
    z=rng.normal(size=(197,4))
    geographic=pairwise_distances(rng.normal(size=(197,2)))
    errors=rng.normal(size=138)
    errors[0]=9.
    errors[35]=-8.
    ids=[f"X{j:03d}" for j in range(197)]
    table,corr,top=m.neighbor_diagnostics(z,geographic,ids,errors)
    assert table.shape[0]==197
    assert len(top)==10
    assert set(top[:2])=={0,35}
    assert table["top10_reference26_error"].sum()==10
    assert table.loc[138:, "reference26_signed_error_private"].isna().all()
    assert (table["x_nn_distance_scaled"]>=0).all()
    assert (table["feature_rank_of_spatial_nn"]>=1).all()
    assert (table["spatial_rank_of_feature_nn"]>=1).all()
    assert len(corr)==7


def test_coordinate_detection_is_strict(monkeypatch):
    m=load(monkeypatch)
    data=pd.DataFrame({
        "centroid_lat":[19.0,19.3], "centroid_lon":[-98.2,-97.9],
    })
    np.testing.assert_array_equal(
        m.checked_coordinates(data),data[["centroid_lon","centroid_lat"]].to_numpy()
    )
    assert m.checked_coordinates(data.drop(columns="centroid_lat")) is None
    data["centroid_lon"]=120.
    assert m.checked_coordinates(data) is None
    with pytest.raises(ValueError, match="outside valid Mexico"):
        m.checked_coordinates(data,lat_col="centroid_lat",lon_col="centroid_lon")
