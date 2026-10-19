"""Sensitivity analyses of paired LOOCV errors with spatially grouped resampling.

These are *conditional diagnostics on already-fitted, overlapping LOOCV forecasts*.
They do not refit any model, validate adaptively selected weights, or supply
coverage guarantees for the 59 unknown parcels. No parcel-level data is public.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_predictions(
    observed: np.ndarray, estimates: dict[str, np.ndarray], reference: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reject incomplete, unpaired, nonfinite predictions."""
    y = np.asarray(observed, dtype=float)
    if y.ndim != 1 or y.size < 3 or not np.isfinite(y).all():
        raise ValueError("Observed yields must be a finite one-dimensional array.")
    if reference not in estimates or not estimates:
        raise ValueError("Reference prediction missing.")
    arrays = {}
    for name, values in estimates.items():
        v = np.asarray(values, dtype=float)
        if v.shape != y.shape or not np.isfinite(v).all():
            raise ValueError(f"Invalid prediction vector for {name}.")
        arrays[name] = v
    return y, arrays


def spatial_grid_labels(
    latitude: np.ndarray, longitude: np.ndarray, *, width_km: float,
    offset_fraction: float = 0.0,
) -> np.ndarray:
    """Fixed square bins in a local equirectangular projection, not a CRS substitute.

    Bin origin uses the observed bounding box, so grid shifts are reported as
    sensitivity checks rather than treating a single partition as canonical.
    """
    lat, lon = np.asarray(latitude, float), np.asarray(longitude, float)
    if (lat.ndim != 1 or lon.shape != lat.shape
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180)):
        raise ValueError("Valid latitude/longitude vectors are required.")
    if (not np.isfinite(width_km) or width_km <= 0
            or not 0 <= offset_fraction < 1):
        raise ValueError("Invalid spatial grid width/offset.")
    cos_lat = float(np.cos(np.deg2rad(np.mean(lat))))
    if cos_lat < 0.01:
        raise ValueError("Local equirectangular grid is unsuitable near poles.")
    km_per_degree = 111.195
    x = lon * km_per_degree * cos_lat
    y = lat * km_per_degree
    ix = np.floor((x - x.min()) / width_km + offset_fraction).astype(int)
    iy = np.floor((y - y.min()) / width_km + offset_fraction).astype(int)
    return np.array([f"{i}:{j}" for i, j in zip(ix, iy, strict=True)])


def grouped_bootstrap(
    observed: np.ndarray, estimates: dict[str, np.ndarray], labels: np.ndarray,
    *, reference: str = "Reference26_mean", replicates: int = 3000,
    seed: int = 20261019, min_groups: int = 8,
    grouping: str = "spatial_blocks",
) -> pd.DataFrame:
    """Cluster bootstrap of *paired* mean losses, holding predictions fixed.

    Resample spatial groups with replacement (not single parcels). Every model
    receives the exact same sampled group multiplicities. CI is a descriptive
    percentile interval conditional on group choices; not an exact p-value.
    """
    y, predictions = validate_predictions(observed, estimates, reference)
    group = np.asarray(labels).astype(str)
    if group.shape != y.shape or np.any(group == ""):
        raise ValueError("One group label is required per held-out parcel.")
    if replicates < 100 or min_groups < 2:
        raise ValueError("Require >=100 bootstrap replicates and >=2 groups.")
    names, inverse = np.unique(group, return_inverse=True)
    g = len(names)
    if g < min_groups:
        raise ValueError(
            f"Too few groups ({g}) for bootstrap; require {min_groups}. "
            "Choose a finer grid or do not report a group interval."
        )
    group_sizes = np.bincount(inverse)
    errors = {name: pred - y for name, pred in predictions.items()}
    baseline = errors[reference]
    rows = []
    rng = np.random.default_rng(seed)
    # All estimates use the same resampled group draws for paired comparisons.
    samples = np.empty((replicates, len(predictions), 6), dtype=float)
    for b in range(replicates):
        sampled_groups = rng.integers(0, g, size=g)
        multiplicities = np.bincount(sampled_groups, minlength=g)
        w = multiplicities[inverse].astype(float)
        denominator = np.sum(w)
        mse_ref = np.dot(w, baseline**2) / denominator
        mae_ref = np.dot(w, np.abs(baseline)) / denominator
        for j, name in enumerate(predictions):
            e = errors[name]
            mse = np.dot(w, e**2) / denominator
            mae = np.dot(w, np.abs(e)) / denominator
            bias = np.dot(w, e) / denominator
            samples[b, j] = (
                np.sqrt(mse), mae, bias,
                mse - mse_ref, np.sqrt(mse) - np.sqrt(mse_ref), mae - mae_ref,
            )
    rmse_ref = np.sqrt(np.mean(baseline**2))
    mae_ref = np.mean(np.abs(baseline))
    for j, name in enumerate(predictions):
        e = errors[name]
        point = np.array([
            np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), np.mean(e),
            np.mean(e**2) - np.mean(baseline**2),
            np.sqrt(np.mean(e**2)) - rmse_ref,
            np.mean(np.abs(e)) - mae_ref,
        ])
        for col, metric in enumerate((
            "rmse_t_ha", "mae_t_ha", "bias_t_ha",
            "delta_mse_vs_reference_t2_ha2",
            "delta_rmse_vs_reference_t_ha",
            "delta_mae_vs_reference_t_ha",
        )):
            ci = np.quantile(samples[:, j, col], [.025, .975])
            rows.append({
                "model": name, "metric": metric, "estimate": float(point[col]),
                "ci_lower_2p5": float(ci[0]),
                "ci_upper_97p5": float(ci[1]),
                "fraction_bootstrap_below_zero": float(
                    np.mean(samples[:, j, col] < 0)
                ),
                "grouping": grouping, "n_groups": int(g),
                "min_group_size": int(group_sizes.min()),
                "max_group_size": int(group_sizes.max()),
                "n_resamples": int(replicates), "seed": int(seed),
                "interpretation": (
                    "Exploratory conditional cluster bootstrap; dependent "
                    "LOOCV fits and outcome-selected models invalidate "
                    "confirmatory confidence/selection claims."
                ),
            })
    return pd.DataFrame(rows)


def influential_parcels(
    observed: np.ndarray, estimates: dict[str, np.ndarray],
    ids: np.ndarray, *, reference: str = "Reference26_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deletion diagnostics on frozen predictions; NEVER interpreted as new CV fits.

    Per-parcel output with IDs MUST stay gitignored; summary is aggregate only.
    """
    y, predictions = validate_predictions(observed, estimates, reference)
    ids = np.asarray(ids).astype(str)
    if ids.shape != y.shape or len(np.unique(ids)) != len(y):
        raise ValueError("Parcel identifiers must be unique and aligned.")
    n = len(y)
    records = []
    summaries = []
    baseline = predictions[reference] - y
    for name, pred in predictions.items():
        e = pred - y
        sq = e**2
        total = float(sq.sum())
        rank = np.argsort(-sq, kind="stable")
        loo_rmse = np.sqrt((total - sq) / (n - 1))
        for i in range(n):
            records.append({
                "ID_POLIGONO": ids[i], "model": name,
                "signed_error_t_ha": float(e[i]),
                "absolute_error_t_ha": float(abs(e[i])),
                "squared_error_t2_ha2": float(sq[i]),
                "rmse_after_deleting_row_t_ha": float(loo_rmse[i]),
                "delta_sq_loss_vs_reference_t2_ha2":
                    float(sq[i] - baseline[i]**2),
            })
        summaries.append({
            "model": name, "n": n,
            "rmse_full_t_ha": float(np.sqrt(total / n)),
            "rmse_delete_largest_t_ha": float(
                np.sqrt((total - sq[rank[0]]) / (n - 1))
            ),
            "rmse_delete_largest_two_t_ha": float(
                np.sqrt((total - sq[rank[:2]].sum()) / (n - 2))
            ),
            "share_total_sse_largest": float(sq[rank[0]] / total)
                if total else 0.0,
            "share_total_sse_largest_two": float(sq[rank[:2]].sum() / total)
                if total else 0.0,
            "note": "Deletion on saved predictions, not model retraining.",
        })
    private = pd.DataFrame(records)
    summary = pd.DataFrame(summaries)
    return private, summary


def paired_advantage_fragility(
    observed: np.ndarray, estimates: dict[str, np.ndarray],
    *, reference: str = "Reference26_mean",
) -> pd.DataFrame:
    """Check whether loss advantage survives deleting its most favorable rows."""
    y, predictions = validate_predictions(observed, estimates, reference)
    n = len(y)
    base_sq = (predictions[reference] - y)**2
    rows = []
    for name, pred in predictions.items():
        if name == reference:
            continue
        diff = (pred - y)**2 - base_sq
        rank = np.argsort(diff, kind="stable")
        for q in (0, 1, 2, 5, 10):
            if q >= n:
                continue
            surviving = np.ones(n, dtype=bool)
            surviving[rank[:q]] = False
            rows.append({
                "model": name, "reference": reference,
                "number_most_favorable_removed": q,
                "remaining_parcels": int(surviving.sum()),
                "delta_mse_t2_ha2": float(diff[surviving].mean()),
                "note": (
                    "Diagnostic removal chosen after inspecting losses; "
                    "not a valid independent test or a new LOOCV fit."
                ),
            })
    return pd.DataFrame(rows)
