"""Aggregate metrics for an audited set of genuine 138-fold LOOCV predictions.

Signed prediction error: e_i = yhat_i - y_i. All yield errors use t/ha.

The cross-validated predictive R² = 1 - SSE / sum((y-mean(y))²) is
reported. "Adjusted R²" is deliberately NOT calculated for the composite,
regularized and selected models: it has no well-defined number of fitted
parameters p, and inserting the number of input features would mislead.

No individual labels, IDs or prediction vectors are exported from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N = 138
ID = "ID_POLIGONO"
TARGET = "observed"
BOOTSTRAP_SEED = 20261019

MAIN_FIELDS = (
    "model", "n_holdouts", "train_labels_per_fold", "rmse_t_ha",
    "mae_t_ha", "median_ae_t_ha", "p90_ae_t_ha", "p95_ae_t_ha",
    "max_ae_t_ha", "bias_t_ha", "r2_cv",
)
OTHER_FIELDS = (
    "l1_t_ha", "l2_t_ha", "linf_t_ha", "sse_t_ha2", "mse_t_ha2",
    "error_sd_t_ha", "max_overprediction_t_ha",
    "max_underprediction_t_ha", "explained_variance",
    "pearson_r", "spearman_rho", "wape_pct", "nrmse_mean_pct",
    "nrmse_sd_pct", "mae_skill_vs_mean",
    "adjusted_r2", "adjusted_r2_status", "source_kind",
)


def validate_predictions(table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if ID not in table or TARGET not in table:
        raise ValueError("Missing unique parcel ID or observed-y column.")
    if len(table) != N or table[ID].isna().any() or table[ID].duplicated().any():
        raise ValueError("Require exactly 138 distinct LOOCV held-out IDs.")
    models = [c for c in table if c not in (ID, TARGET)]
    if not models:
        raise ValueError("No model columns in LOOCV table.")
    values = table[[TARGET, *models]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Observed yields and LOOCV predictions must be finite.")
    y = values[:, 0]
    if not np.isfinite(y).all() or np.var(y) <= 0:
        raise ValueError("Observed 138 yields must have positive variance.")
    return y, models


def metrics(y: np.ndarray, pred: np.ndarray, model: str, *, source: str) -> dict:
    """Report mathematical norms as TOTAL norms, alongside normalized errors."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 1 or y.size != N:
        raise ValueError("Metrics require one prediction per 138 held-out parcels.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite observed or predicted values.")
    e = pred - y
    ae = abs(e)
    l1 = float(np.sum(ae))
    sse = float(np.sum(e**2))
    l2 = float(np.sqrt(sse))
    tss = float(np.sum((y-y.mean())**2))
    if tss <= 0:
        raise ValueError("Predictive R² undefined for constant observed yields.")
    error_var = float(np.mean((e-e.mean())**2))
    y_var = float(np.mean((y-y.mean())**2))
    denominator = float(np.sum(abs(y)))
    mean_abs_baseline = float(np.mean(abs(y-y.mean())))
    result = {
        "model": model,
        "n_holdouts": N, "train_labels_per_fold": N-1,
        "rmse_t_ha": l2 / np.sqrt(N),
        "mae_t_ha": l1/N,
        "median_ae_t_ha": float(np.median(ae)),
        "p90_ae_t_ha": float(np.quantile(ae, .90)),
        "p95_ae_t_ha": float(np.quantile(ae, .95)),
        "max_ae_t_ha": float(np.max(ae)),
        "bias_t_ha": float(np.mean(e)),
        "r2_cv": 1-sse/tss,
        "l1_t_ha": l1,
        "l2_t_ha": l2,
        "linf_t_ha": float(np.max(ae)),
        "sse_t_ha2": sse,
        "mse_t_ha2": sse/N,
        "error_sd_t_ha": float(np.sqrt(error_var)),
        "max_overprediction_t_ha": float(max(0.0, np.max(e))),
        "max_underprediction_t_ha": float(max(0.0, -np.min(e))),
        "explained_variance": 1-error_var/y_var,
        "pearson_r": float(np.corrcoef(y, pred)[0, 1])
        if np.std(pred) > 0 else np.nan,
        "spearman_rho": float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
        if np.std(pred) > 0 else np.nan,
        "wape_pct": 100*l1/denominator if denominator > 0 else np.nan,
        "nrmse_mean_pct": (100*l2/np.sqrt(N)/abs(float(y.mean()))
                            if abs(float(y.mean())) > 0 else np.nan),
        "nrmse_sd_pct": 100*l2/np.sqrt(N*y_var),
        "mae_skill_vs_mean": 1-(l1/N)/mean_abs_baseline
        if mean_abs_baseline > 0 else np.nan,
        "adjusted_r2": np.nan,
        "adjusted_r2_status": (
            "not_applicable: nonlinear/regularized composite LOOCV; "
            "no defensible fixed p"
        ),
        "source_kind": source,
    }
    return result


def all_metrics(table: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    y, names = validate_predictions(table)
    if set(sources) != set(names):
        raise ValueError("Metrics provenance must include exactly the input models.")
    records = [
        metrics(y, table[name].to_numpy(float), name, source=sources[name])
        for name in names
    ]
    result = pd.DataFrame(records).sort_values(
        ["rmse_t_ha", "model"], kind="stable"
    ).reset_index(drop=True)
    return result[[*MAIN_FIELDS, *OTHER_FIELDS]]


def paired_metrics(
    table: pd.DataFrame, *, baseline: str, draws: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Pairs share all 138 observations; naive parcel bootstrap is descriptive.

    R² and RMSE differences are descriptive, not independent test performance
    after feature/model/hyperparameter selection on these same 138 labels.
    """
    y, models = validate_predictions(table)
    if baseline not in models:
        raise ValueError("Paired baseline missing from the audited table.")
    if draws < 100:
        raise ValueError("At least 100 bootstrap samples are needed.")
    ref = table[baseline].to_numpy(float) - y
    ref_sq = ref**2
    ref_rmse = float(np.sqrt(np.mean(ref_sq)))
    ref_mae = float(np.mean(abs(ref)))
    boot = np.random.default_rng(seed).integers(0, N, size=(draws, N))
    ref_boot_rmse = np.sqrt(np.mean(ref_sq[boot], axis=1))
    rows = []
    for name in models:
        if name == baseline:
            continue
        e = table[name].to_numpy(float)-y
        sq = e**2
        delta_mse = sq-ref_sq
        diff_boot_rmse = np.sqrt(np.mean(sq[boot], axis=1))-ref_boot_rmse
        rows.append({
            "reference": baseline, "model": name,
            "n_common_holdouts": N,
            "delta_rmse_t_ha": float(np.sqrt(np.mean(sq))-ref_rmse),
            "delta_mae_t_ha": float(np.mean(abs(e))-ref_mae),
            "delta_mse_t_ha2": float(np.mean(delta_mse)),
            "fraction_lower_ae": float(np.mean(abs(e)<abs(ref))),
            "fraction_equal_ae": float(np.mean(abs(e)==abs(ref))),
            "error_pearson_corr": float(np.corrcoef(e, ref)[0, 1])
            if np.std(e) > 0 and np.std(ref) > 0 else np.nan,
            "naive_boot_delta_rmse_ci95_lo": float(np.quantile(diff_boot_rmse, .025)),
            "naive_boot_delta_rmse_ci95_hi": float(np.quantile(diff_boot_rmse, .975)),
            "bootstrap_draws": draws,
            "bootstrap_caveat": "parcel bootstrap: spatial dependence uncorrected",
        })
    return pd.DataFrame(rows).sort_values(
        "delta_rmse_t_ha", kind="stable"
    ).reset_index(drop=True)
