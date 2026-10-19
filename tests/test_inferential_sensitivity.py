"""Synthetic contract tests: no access to private parcel observations."""
import numpy as np
import pytest

from geocebada.evaluation.inferential_sensitivity import (
    grouped_bootstrap,
    influential_parcels,
    paired_advantage_fragility,
    spatial_grid_labels,
    validate_predictions,
)


def example():
    rng = np.random.default_rng(731)
    n = 138
    actual = 4 + rng.normal(0, 1, n)
    base = actual + rng.normal(0, .5, n)
    alt = actual + rng.normal(0, .35, n)
    estimates = {
        "Reference26_mean": base,
        "Compact14": alt,
        "Ensemble_final": .7 * base + .3 * alt,
    }
    return actual, estimates


def test_finite_length_and_reference_are_mandatory():
    y, estimates = example()
    with pytest.raises(ValueError, match="Reference"):
        validate_predictions(y, {"Other": estimates["Compact14"]}, "Reference26_mean")
    estimates["Compact14"][0] = np.nan
    with pytest.raises(ValueError, match="Compact14"):
        validate_predictions(y, estimates, "Reference26_mean")


def test_grouped_bootstrap_keeps_paired_identical_models_at_zero():
    y, estimates = example()
    estimates["Compact14"] = estimates["Reference26_mean"].copy()
    labels = np.repeat(np.arange(23), 6).astype(str)
    frame = grouped_bootstrap(
        y, estimates, labels, replicates=250, seed=10, min_groups=8
    )
    same = frame.loc[
        frame.model.eq("Compact14")
        & frame.metric.eq("delta_mse_vs_reference_t2_ha2")
    ].iloc[0]
    assert same["estimate"] == pytest.approx(0)
    assert same["ci_lower_2p5"] == pytest.approx(0)
    assert same["ci_upper_97p5"] == pytest.approx(0)
    assert frame["n_groups"].eq(23).all()
    assert len(frame) == 3 * 6


def test_bootstrap_is_deterministic_and_group_count_checked():
    y, estimates = example()
    labels = np.repeat(np.arange(23), 6)
    one = grouped_bootstrap(y, estimates, labels, replicates=100, seed=23)
    two = grouped_bootstrap(y, estimates, labels, replicates=100, seed=23)
    assert one.equals(two)
    with pytest.raises(ValueError, match="Too few groups"):
        grouped_bootstrap(y, estimates, np.zeros(138), replicates=100)
    with pytest.raises(ValueError, match="replicates"):
        grouped_bootstrap(y, estimates, labels, replicates=3)


def test_influence_report_is_private_and_deleting_worst_reduces_rmse():
    y, estimates = example()
    ids = np.array([f"PRIVATE_{i}" for i in range(138)])
    per_parcel, grouped = influential_parcels(y, estimates, ids)
    assert "ID_POLIGONO" in per_parcel
    assert len(per_parcel) == 3 * 138
    assert len(grouped) == 3
    assert grouped["share_total_sse_largest_two"].ge(
        grouped["share_total_sse_largest"]
    ).all()
    assert grouped["rmse_delete_largest_t_ha"].le(
        grouped["rmse_full_t_ha"] + 1e-12
    ).all()
    with pytest.raises(ValueError, match="unique"):
        influential_parcels(y, estimates, np.array(["repeat"] * 138))


def test_relative_loss_sensitivity_removes_best_advantage():
    y, estimates = example()
    estimates["Compact14"] = estimates["Reference26_mean"].copy()
    report = paired_advantage_fragility(y, estimates)
    exact = report.loc[report.model.eq("Compact14")]
    assert np.allclose(exact["delta_mse_t2_ha2"], 0)
    assert list(exact["number_most_favorable_removed"]) == [0, 1, 2, 5, 10]


def test_spatial_partition_validates_coordinates_and_shifts():
    lat = 19. + np.linspace(0, .9, 138)
    lon = -98. + np.linspace(0, .8, 138)
    a = spatial_grid_labels(lat, lon, width_km=15, offset_fraction=0)
    b = spatial_grid_labels(lat, lon, width_km=15, offset_fraction=.5)
    assert a.shape == b.shape == (138,)
    assert len(np.unique(a)) > 1
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError, match="latitude/longitude"):
        spatial_grid_labels(lat, lon + np.nan, width_km=15)
    with pytest.raises(ValueError, match="width/offset"):
        spatial_grid_labels(lat, lon, width_km=-3)
