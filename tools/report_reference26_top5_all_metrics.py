#!/usr/bin/env python3
"""Comprehensive post-hoc error audit: seven genuine Reference26 138-LOOCV predictors.

NO fitting, no new predictions, no synthetic y, no 59-target error metrics.
Uses existing validated fold CSVs or their merged private nested predictions.
Writes only AGGREGATED metrics, tests and plots to gitignored reports/.

  python tools/report_reference26_top5_all_metrics.py --check
  python tools/report_reference26_top5_all_metrics.py --run
  python tools/report_reference26_top5_all_metrics.py --summary
  python tools/report_reference26_top5_all_metrics.py --run --bootstrap 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.cluster import AgglomerativeClustering

from geocebada.evaluation.loocv_comprehensive_metrics import all_metrics

# Force a noninteractive backend on Windows VMs and headless CI workers.
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
ID = "ID_POLIGONO"
N = 138
BASE = "Reference26_mean_unchanged"
NESTED = "NestedSelection"
TOP5 = (
    "analog__robust26__k3__blend0.25__supported",
    "analog__pca8__k3__blend0.25__supported",
    "analog__pca8__k8__blend0.25__supported",
    "analog__robust26__k3__blend0.1__supported",
    "analog__robust26__k3__blend0.1__none",
)
MODELS = (BASE, *TOP5, NESTED)
ALIASES = {
    BASE: "Reference26 original",
    TOP5[0]: "Analog26 · k3 · 0.25 · gated",
    TOP5[1]: "PCA8 · k3 · 0.25 · gated",
    TOP5[2]: "PCA8 · k8 · 0.25 · gated",
    TOP5[3]: "Analog26 · k3 · 0.10 · gated",
    TOP5[4]: "Analog26 · k3 · 0.10 · plain",
    NESTED: "NestedSelection",
}
DEFAULT_INPUT = ROOT / "reports/reference26_top5_nested/private_nested_predictions.csv"
OUT = ROOT / "reports/reference26_top5_full_metrics"
SEED = 20261019
EPS = 0.1     # t/ha, explicitly documented for relative error safeguards.
HUBER = (0.25, 0.5, 1.0)
THRESHOLDS = (0.25, 0.5, 1.0)
AGG_FILES = (
    "metrics_complete_7.csv", "paired_tests_vs_reference26.csv",
    "bootstrap_95ci_7.csv", "stratified_yield_tertiles.csv",
    "influence_aggregate_7.csv", "spatial_sensitivity_7.csv",
    "metric_definitions.json",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified(path: Path) -> pd.DataFrame:
    """Reject incomplete, mismatched, or non-LOOCV datasets."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run "
            "python tools/run_reference26_top5_nested.py --summary first."
        )
    raw = pd.read_csv(path, dtype={ID: str})
    expected = {ID, "observed", "Reference26_mean", "chosen_model",
                "NestedSelection", "outer_train_n", "inner_oof_n", *TOP5}
    if len(raw) != N or not expected.issubset(raw.columns):
        raise ValueError("Need the complete 138-row seven-model nested LOOCV CSV.")
    if raw[ID].isna().any() or raw[ID].duplicated().any():
        raise ValueError("Expected 138 distinct parcel IDs.")
    if (raw["outer_train_n"].ne(N-1).any()
            or raw["inner_oof_n"].ne(N-1).any()):
        raise ValueError("Not authentic 137-train, 1-heldout nested folds.")
    if raw["chosen_model"].isna().any() or not set(
        raw["chosen_model"].unique()
    ).issubset({BASE, *TOP5}):
        raise ValueError("Unrecognized fold-specific nested choice.")
    numeric = raw[["observed", "Reference26_mean", *TOP5, NESTED]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite observed or heldout predictions.")
    y = raw["observed"].to_numpy(float)
    base = raw["Reference26_mean"].to_numpy(float)
    if not np.isclose(np.sqrt(np.mean((y-base)**2)), 0.4681381271,
                      atol=0.000002, rtol=0):
        raise ValueError("Original native Reference26 mean RMSE differs.")
    nested = raw[NESTED].to_numpy(float)
    if not np.isclose(np.sqrt(np.mean((y-nested)**2)), 0.462892,
                      atol=0.000002, rtol=0):
        raise ValueError("NestedSelection RMSE does not match completed 138 CV.")
    for i, row in raw.iterrows():
        choice = row["chosen_model"]
        predictor = "Reference26_mean" if choice == BASE else choice
        if not np.isclose(row[NESTED], row[predictor], atol=1e-10, rtol=0):
            raise ValueError(f"Nested route mismatch at original fold {i}.")
    data = raw[[ID, "observed", "Reference26_mean", *TOP5, NESTED]].copy()
    data = data.rename(columns={"Reference26_mean": BASE})
    # Check inherited baseline values without exposing labels/IDs in outputs.
    if np.var(y) <= 0:
        raise ValueError("All observed yields identical.")
    return data


def robust_and_relative(y: np.ndarray, p: np.ndarray) -> dict:
    e = p-y
    ae = np.abs(e)
    n = len(y)
    ranks = np.sort(ae)
    result = {
        "p25_ae_t_ha": float(np.quantile(ae, .25)),
        "p50_ae_t_ha": float(np.median(ae)),
        "p75_ae_t_ha": float(np.quantile(ae, .75)),
        "p99_ae_t_ha": float(np.quantile(ae, .99)),
        "iqr_ae_t_ha": float(np.quantile(ae, .75)-np.quantile(ae, .25)),
        "median_signed_error_t_ha": float(np.median(e)),
        "p05_signed_error_t_ha": float(np.quantile(e, .05)),
        "p95_signed_error_t_ha": float(np.quantile(e, .95)),
        "signed_error_sd_sample_t_ha": float(np.std(e, ddof=1)),
        "error_mad_about_median_t_ha": float(np.median(np.abs(e-np.median(e)))),
        "trimmed_mae_5pct_t_ha": float(np.mean(ranks[:n-int(np.floor(.05*n))])),
        "trimmed_mae_10pct_t_ha": float(np.mean(ranks[:n-int(np.floor(.10*n))])),
        "trimmed_rmse_10pct_t_ha": float(np.sqrt(
            np.mean(ranks[:n-int(np.floor(.10*n))]**2)
        )),
        "log_cosh_mean": float(np.mean(np.logaddexp(e, -e)-np.log(2.))),
        "mape_y_gt_0p1_pct": float(
            100*np.mean(ae[np.abs(y) >= EPS]/np.abs(y[np.abs(y) >= EPS]))
        ) if (np.abs(y) >= EPS).any() else np.nan,
        "mape_excluded_near_zero_count": int(np.sum(np.abs(y) < EPS)),
        "mape_floor_0p1_pct": float(
            100*np.mean(ae/np.maximum(np.abs(y), EPS))
        ),
        "smape_floor_0p1_pct": float(
            200*np.mean(ae/np.maximum(np.abs(y)+np.abs(p), 2*EPS))
        ),
        "rmspe_floor_0p1_pct": float(
            100*np.sqrt(np.mean((e/np.maximum(np.abs(y), EPS))**2))
        ),
        "median_ape_floor_0p1_pct": float(
            100*np.median(ae/np.maximum(np.abs(y), EPS))
        ),
        "nrmse_observed_range_pct": float(
            100*np.sqrt(np.mean(e**2))/np.ptp(y)
        ) if np.ptp(y) else np.nan,
        "calibration_slope_y_on_pred": float(
            np.cov(p, y, ddof=0)[0, 1]/np.var(p)
        ) if np.var(p) > 0 else np.nan,
        "calibration_intercept_y_on_pred": float(
            np.mean(y)-np.mean(p)*np.cov(p, y, ddof=0)[0, 1]/np.var(p)
        ) if np.var(p) > 0 else np.nan,
        "prediction_sd_t_ha": float(np.std(p, ddof=0)),
        "observed_sd_t_ha": float(np.std(y, ddof=0)),
    }
    for delta in HUBER:
        loss = np.where(ae <= delta, .5*e*e,
                        delta*(ae-.5*delta))
        result[f"huber_mean_delta_{delta:g}"] = float(np.mean(loss))
    for t in THRESHOLDS:
        result[f"coverage_abs_error_le_{t:g}_t_ha_pct"] = float(
            100*np.mean(ae <= t)
        )
    return result


def full_metrics(table: pd.DataFrame) -> pd.DataFrame:
    sources = {k: "genuine_native_138_LOOCV_conditional_posthoc"
               for k in MODELS}
    result = all_metrics(table, sources)
    y = table["observed"].to_numpy(float)
    additions = pd.DataFrame([
        {"model": name, **robust_and_relative(y, table[name].to_numpy(float))}
        for name in MODELS
    ])
    result = result.merge(additions, on="model", validate="one_to_one")
    return result.sort_values("rmse_t_ha", kind="stable").reset_index(drop=True)


def bootstrap_metrics(y, predictions, *, draws, seed):
    """Common resampled original parcel IDs preserve paired dependence."""
    rng = np.random.default_rng(seed)
    n, m = predictions.shape
    records = {name: {"rmse": [], "mae": [], "bias": [], "r2": []}
               for name in MODELS}
    errs = predictions-y[:, None]
    for first in range(0, draws, 150):
        size = min(150, draws-first)
        idx = rng.integers(0, n, size=(size, n))
        yb = y[idx]
        e = errs[idx, :]
        rmse = np.sqrt(np.mean(e**2, axis=1))
        mae = np.mean(np.abs(e), axis=1)
        bias = np.mean(e, axis=1)
        tss = np.sum((yb-np.mean(yb, axis=1, keepdims=True))**2, axis=1)
        r2 = 1.-np.sum(e**2, axis=1)/np.maximum(tss[:, None], 1e-12)
        for j, name in enumerate(MODELS):
            rec = records[name]
            for metric, values in (
                ("rmse", rmse), ("mae", mae),
                ("bias", bias), ("r2", r2)
            ):
                rec[metric].extend(values[:, j].tolist())
    out = []
    for name, distributions in records.items():
        entry = {"model": name, "n_original_heldouts": n,
                 "bootstrap_draws": draws,
                 "method": "iid_paired_original_parcel_bootstrap"}
        for metric, values in distributions.items():
            entry[f"{metric}_ci95_low"] = float(np.quantile(values, .025))
            entry[f"{metric}_ci95_high"] = float(np.quantile(values, .975))
        out.append(entry)
    return pd.DataFrame(out)


def paired_tests(table, *, draws, seed):
    base = table[BASE].to_numpy(float)
    y = table["observed"].to_numpy(float)
    candidates = np.column_stack(
        [table[name].to_numpy(float) for name in MODELS[1:]]
    )
    e_ref = base-y
    e = candidates-y[:, None]
    # Positive delta means alternative is better; MSE paired gain.
    gains = e_ref[:, None]**2-e**2
    abs_gains = np.abs(e_ref[:, None])-np.abs(e)
    reference_rmse = np.sqrt(np.mean(e_ref**2))
    reference_mae = np.mean(np.abs(e_ref))
    rng = np.random.default_rng(seed)
    observed_mse = np.mean(gains, axis=0)
    observed_mae = np.mean(abs_gains, axis=0)
    perm_mse = np.zeros((draws, len(MODELS)-1))
    perm_mae = np.zeros((draws, len(MODELS)-1))
    boot_delta_rmse = {name: [] for name in MODELS[1:]}
    boot_delta_mae = {name: [] for name in MODELS[1:]}
    for first in range(0, draws, 150):
        size = min(150, draws-first)
        signs = 2*rng.integers(0, 2, size=(size, len(y), 1))-1
        perm_mse[first:first+size] = np.mean(
            signs*gains[None, :, :], axis=1
        )
        perm_mae[first:first+size] = np.mean(
            signs*abs_gains[None, :, :], axis=1
        )
        idx = rng.integers(0, len(y), size=(size, len(y)))
        base_sq = np.mean(e_ref[idx]**2, axis=1)
        base_mae = np.mean(np.abs(e_ref[idx]), axis=1)
        for j, name in enumerate(MODELS[1:]):
            boot_delta_rmse[name].extend(
                (np.sqrt(np.mean(e[idx, j]**2, axis=1)) -
                 np.sqrt(base_sq)).tolist()
            )
            boot_delta_mae[name].extend(
                (np.mean(np.abs(e[idx, j]), axis=1)-base_mae).tolist()
            )
    max_mse = np.max(perm_mse, axis=1)
    max_mae = np.max(perm_mae, axis=1)
    rows = []
    for j, name in enumerate(MODELS[1:]):
        ab = abs_gains[:, j]
        n_wins = int(np.sum(ab > 1e-12))
        n_losses = int(np.sum(ab < -1e-12))
        non_ties = n_wins+n_losses
        try:
            # Two-sided Wilcoxon on paired absolute errors.
            w_p = float(wilcoxon(
                ab, zero_method="wilcox", alternative="two-sided",
                method="auto"
            ).pvalue) if np.any(np.abs(ab) > 1e-12) else 1.
        except ValueError:
            w_p = np.nan
        rows.append({
            "model": name, "baseline": BASE,
            "n_common_heldouts": len(y),
            "delta_rmse_candidate_minus_baseline_t_ha": float(
                np.sqrt(np.mean(e[:, j]**2))-reference_rmse
            ),
            "delta_mae_candidate_minus_baseline_t_ha": float(
                np.mean(np.abs(e[:, j]))-reference_mae
            ),
            "mse_reduction_baseline_minus_candidate_t_ha2": float(
                observed_mse[j]
            ),
            "mae_reduction_baseline_minus_candidate_t_ha": float(
                observed_mae[j]
            ),
            "n_abs_error_improved": n_wins,
            "n_abs_error_worsened": n_losses,
            "n_abs_error_tied": len(y)-non_ties,
            "sign_test_two_sided_p": float(
                binomtest(n_wins, non_ties, .5).pvalue
            ) if non_ties else 1.,
            "wilcoxon_abs_error_two_sided_p_exploratory": w_p,
            "signflip_mse_one_sided_p": float(
                (1+np.sum(perm_mse[:, j] >= observed_mse[j]))/(draws+1)
            ),
            "signflip_mse_maxT_six_one_sided_p": float(
                (1+np.sum(max_mse >= observed_mse[j]))/(draws+1)
            ),
            "signflip_mae_one_sided_p": float(
                (1+np.sum(perm_mae[:, j] >= observed_mae[j]))/(draws+1)
            ),
            "signflip_mae_maxT_six_one_sided_p": float(
                (1+np.sum(max_mae >= observed_mae[j]))/(draws+1)
            ),
            "bootstrap_delta_rmse_ci95_low": float(
                np.quantile(boot_delta_rmse[name], .025)
            ),
            "bootstrap_delta_rmse_ci95_high": float(
                np.quantile(boot_delta_rmse[name], .975)
            ),
            "bootstrap_delta_mae_ci95_low": float(
                np.quantile(boot_delta_mae[name], .025)
            ),
            "bootstrap_delta_mae_ci95_high": float(
                np.quantile(boot_delta_mae[name], .975)
            ),
            "permutation_draws": draws,
            "test_caveat": (
                "Exploratory: same 138 data used to preselect top-5; "
                "IID/sign-exchangeability unverified; spatial dependence possible"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "delta_rmse_candidate_minus_baseline_t_ha"
    ).reset_index(drop=True)


def influence(table):
    y = table["observed"].to_numpy(float)
    err2 = (table[list(MODELS)].to_numpy(float)-y[:, None])**2
    b2 = err2[:, 0]
    rows = []
    n = len(y)
    full = np.sqrt(np.mean(err2, axis=0))
    for j, name in enumerate(MODELS):
        no_one = np.sqrt((np.sum(err2[:, j])-err2[:, j])/(n-1))
        result = {
            "model": name,
            "rmse_full": float(full[j]),
            "rmse_deletion_min": float(no_one.min()),
            "rmse_deletion_max": float(no_one.max()),
            "max_abs_rmse_shift_by_deleting_one": float(
                np.max(np.abs(no_one-full[j]))
            ),
            "largest_single_sse_share_pct": float(
                100*np.max(err2[:, j])/np.sum(err2[:, j])
            ),
            "top_two_sse_share_pct": float(
                100*np.sum(np.sort(err2[:, j])[-2:])/np.sum(err2[:, j])
            ),
        }
        if j != 0:
            improvement = np.sqrt(
                (np.sum(b2)-b2)/(n-1)
            )-no_one
            result.update({
                "n_deletions_with_candidate_better_than_baseline": int(
                    np.sum(improvement > 0)
                ),
                "deletion_gain_min_t_ha": float(np.min(improvement)),
                "deletion_gain_max_t_ha": float(np.max(improvement)),
                "deletion_gain_median_t_ha": float(np.median(improvement)),
            })
        rows.append(result)
    return pd.DataFrame(rows)


def strata(table):
    y = table["observed"].to_numpy(float)
    # 46 real heldouts per tercile, deterministic tie-breaking by index.
    sorted_indices = np.argsort(y, kind="stable")
    bins = np.empty(len(y), dtype=int)
    for k, ids in enumerate(np.array_split(sorted_indices, 3)):
        bins[ids] = k
    out = []
    for name in MODELS:
        e = table[name].to_numpy(float)-y
        for k, label in enumerate(("low_observed_third",
                                   "middle_observed_third",
                                   "high_observed_third")):
            part = e[bins == k]
            out.append({
                "model": name, "observed_yield_group": label,
                "n_heldouts": len(part),
                "rmse_t_ha": float(np.sqrt(np.mean(part**2))),
                "mae_t_ha": float(np.mean(np.abs(part))),
                "signed_bias_t_ha": float(np.mean(part)),
                "p95_absolute_error_t_ha": float(
                    np.quantile(np.abs(part), .95)
                ),
            })
    return pd.DataFrame(out)


def spatial_sensitivity(table, distance, *, draws, blocks, seed):
    """Descriptive bootstrap by X-ordered geographic-distance clusters."""
    if distance is None:
        return pd.DataFrame([{
            "model": name, "status": "SKIPPED: no trusted distance197 order"
        } for name in MODELS])
    d = np.load(distance, allow_pickle=False)
    if d.shape == (197, 197):
        d = d[:138, :138]
    if (d.shape != (N, N) or not np.isfinite(d).all()
            or not np.allclose(d, d.T, atol=1e-8)
            or np.any(d < -1e-8)
            or not np.allclose(np.diag(d), 0, atol=1e-8)):
        raise ValueError("Spatial distance matrix must align to 138 rows.")
    if not 2 <= blocks <= N//2:
        raise ValueError("Invalid geographic block count.")
    cluster = AgglomerativeClustering(
        n_clusters=blocks, metric="precomputed", linkage="average"
    ).fit_predict(np.maximum(d, 0))
    ys = table["observed"].to_numpy(float)
    err2 = (table[list(MODELS)].to_numpy(float)-ys[:, None])**2
    random = np.random.default_rng(seed)
    values = np.empty((draws, len(MODELS)))
    deltas = np.empty((draws, len(MODELS)))
    for pos in range(0, draws, 100):
        size = min(100, draws-pos)
        picks = random.integers(0, blocks, size=(size, blocks))
        count = (picks[:, :, None] == np.arange(blocks)).sum(axis=1)
        weights = count[:, cluster]
        denom = np.sum(weights, axis=1)
        mse = weights @ err2 / denom[:, None]
        block_rmse = np.sqrt(mse)
        values[pos:pos+size] = block_rmse
        deltas[pos:pos+size] = block_rmse-block_rmse[:, :1]
    result = []
    for j, name in enumerate(MODELS):
        result.append({
            "model": name, "status": "EXPLORATORY_BLOCK_SENSITIVITY",
            "n_geo_blocks": blocks,
            "smallest_block": int(np.bincount(cluster).min()),
            "largest_block": int(np.bincount(cluster).max()),
            "block_rmse_ci95_low": float(np.quantile(values[:, j], .025)),
            "block_rmse_ci95_high": float(np.quantile(values[:, j], .975)),
            "block_delta_rmse_ci95_low": float(
                np.quantile(deltas[:, j], .025)
            ),
            "block_delta_rmse_ci95_high": float(
                np.quantile(deltas[:, j], .975)
            ),
            "block_resample_fraction_candidate_better": float(
                np.mean(deltas[:, j] < 0)
            ) if j else None,
            "block_draws": draws,
        })
    return pd.DataFrame(result)


def make_plots(table, metrics, paired, ci, subgroups, out):
    """All images are AGGREGATE-only; no labeled per-parcel scatter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODELS:
        errors = np.sort(np.abs(
            table[name].to_numpy(float)-table["observed"].to_numpy(float)
        ))
        ax.step(errors, np.arange(1, N+1)/N, where="post",
                label=ALIASES[name], linewidth=1.5)
    ax.set(xlabel="Absolute LOOCV error (t/ha)",
           ylabel="Fraction of original parcels with error ≤ x",
           title="Absolute error ECDF, 138 true heldouts")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out/"absolute_error_ecdf_7.png", dpi=170)
    plt.close(fig)

    order = list(metrics["model"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, name in enumerate(order):
        m = metrics.set_index("model").loc[name]
        row = ci.set_index("model").loc[name]
        ax.errorbar(m["rmse_t_ha"], j, xerr=[
            [max(0., m["rmse_t_ha"]-row["rmse_ci95_low"])],
            [max(0., row["rmse_ci95_high"]-m["rmse_t_ha"])]
        ], fmt="o", capsize=3)
    ax.set_yticks(range(len(order)), [ALIASES[n] for n in order])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE ± naive parcel bootstrap 95% interval (t/ha)")
    ax.set_title("Observed RMSE and conditional resampling sensitivity")
    fig.tight_layout()
    fig.savefig(out/"bootstrap_rmse_7.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    group_names = ["low_observed_third", "middle_observed_third",
                   "high_observed_third"]
    x = np.arange(3)
    for name in (BASE, NESTED, TOP5[0]):
        subset = subgroups.set_index(
            ["model", "observed_yield_group"]
        ).loc[name]
        ax.plot(x, [subset.loc[k, "rmse_t_ha"] for k in group_names],
                marker="o", label=ALIASES[name])
    ax.set_xticks(x, ["Low yields (n=46)", "Middle (n=46)",
                      "High yields (n=46)"])
    ax.set_ylabel("RMSE (t/ha)")
    ax.set_title("Descriptive error by observed-yield third")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"rmse_by_yield_tertile.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    comparators = paired.sort_values(
        "delta_rmse_candidate_minus_baseline_t_ha"
    )
    x = comparators["delta_rmse_candidate_minus_baseline_t_ha"].to_numpy()
    lows = comparators["bootstrap_delta_rmse_ci95_low"].to_numpy()
    highs = comparators["bootstrap_delta_rmse_ci95_high"].to_numpy()
    ys = np.arange(len(x))
    ax.errorbar(x, ys, xerr=[np.maximum(x-lows, 0), np.maximum(highs-x, 0)], fmt="o", capsize=3)
    ax.axvline(0, linewidth=1, linestyle="--", color="grey")
    ax.set_yticks(ys, [ALIASES[k] for k in comparators["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Candidate − Reference26 RMSE (t/ha)")
    ax.set_title("Paired RMSE differences · naive 95% bootstrap interval")
    fig.tight_layout()
    fig.savefig(out/"paired_rmse_delta.png", dpi=170)
    plt.close(fig)


def definitions(args, source):
    return {
        "dataset": "138 genuine original heldouts; 137 real training labels each",
        "models": list(MODELS), "n_models": 7, "source_sha256": sha(source),
        "n_target59_with_known_y": 0,
        "residual_sign": "predicted_minus_observed",
        "units": "t/ha, squared loss (t/ha)^2, percent for relative losses",
        "core_module": "geocebada.evaluation.loocv_comprehensive_metrics",
        "norms": "L1=sum|e|; L2=sqrt(sum e^2); Linf=max|e|",
        "r2": "1-SSE/sum[(y-y.mean)^2], predictive LOOCV",
        "adjusted_r2": "NA; regularized ensemble + model selection no fixed p",
        "percentiles": "NumPy linear quantiles of absolute residual",
        "trimmed_losses": (
            "sort n absolute errors, discard floor(frac*n) largest; "
            "report average absolute or sqrt average squared remaining"
        ),
        "huber_loss": "mean 0.5 e^2 if |e|<=delta else delta(|e|-delta/2)",
        "mape": "exclude |y|<0.1 t/ha; report number of exclusions",
        "mape_floor": "replace |y| denominator with max(|y|,0.1 t/ha)",
        "smape_floor": "200*mean(|e|/max(|y|+|pred|,0.2 t/ha))",
        "rmspe_floor": "100*sqrt(mean[(e/max(|y|,0.1))^2])",
        "coverage_abs_error": (
            "percent |prediction error| <= fixed t/ha threshold; "
            "NOT predictive interval coverage"
        ),
        "calibration": (
            "OLS descriptive slope/intercept observed on predictions; "
            "NOT independently calibrated reliability"
        ),
        "iid_bootstrap": (
            f"paired same-138 original parcels resampled {args.bootstrap} "
            "times; conditional descriptive, spatial correlations ignored"
        ),
        "paired_signflip": (
            "within original parcel, same sign across six alternative-vs-"
            "baseline differences; one-sided improvement, maxT family of 6 "
            "only; not prior historical searches"
        ),
        "spatial": (
            "optional descriptive block resampling; distance197 must be "
            "confirmed to have first 138 rows aligned to heldout CSV order"
        ),
        "posthoc_warning": (
            "Top five corrections selected on the SAME 138 yields; "
            "nested selector is conditional on already selected candidates. "
            "No fully independent test or valid confirmatory p-values."
        ),
        "unavailable": (
            "Adjusted R2, coverage of prediction intervals, "
            "accuracy for hidden 59, new independent holdout accuracy"
        ),
        "params": {"bootstrap": args.bootstrap, "seed": args.seed,
                   "eps_t_ha": EPS, "geo_blocks": args.geo_blocks,
                   "spatial_enabled": args.distance197 is not None},
    }


def run(args):
    table = load_verified(args.input.resolve())
    basic = full_metrics(table)
    y = table["observed"].to_numpy(float)
    p = table[list(MODELS)].to_numpy(float)
    ci = bootstrap_metrics(y, p, draws=args.bootstrap, seed=args.seed)
    paired = paired_tests(table, draws=args.bootstrap, seed=args.seed+13)
    infl = influence(table)
    groups = strata(table)
    spatial = spatial_sensitivity(
        table, args.distance197, draws=args.bootstrap,
        blocks=args.geo_blocks, seed=args.seed+29
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, record in (
        ("metrics_complete_7.csv", basic),
        ("paired_tests_vs_reference26.csv", paired),
        ("bootstrap_95ci_7.csv", ci),
        ("stratified_yield_tertiles.csv", groups),
        ("influence_aggregate_7.csv", infl),
        ("spatial_sensitivity_7.csv", spatial),
    ):
        record.to_csv(out/name, index=False, float_format="%.14g")
    (out/"metric_definitions.json").write_text(
        json.dumps(definitions(args, args.input.resolve()),
                   indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    make_plots(table, basic, paired, ci, groups, out)
    summary(args)
    print("[DONE] 7 predictors x 138 genuine heldouts, no new training.")
    print("[PRIVACY] Aggregates and figures only; original data untouched.")
    print(f"[REPORT] {out}")


def summary(args):
    out = args.output.resolve()
    path = out/"metrics_complete_7.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No full report: {path}; run --run.")
    data = pd.read_csv(path)
    selected = ["model", "n_holdouts", "l1_t_ha", "l2_t_ha",
                "linf_t_ha", "mse_t_ha2", "rmse_t_ha", "mae_t_ha",
                "p95_ae_t_ha", "p99_ae_t_ha", "r2_cv", "bias_t_ha",
                "wape_pct", "huber_mean_delta_0.5"]
    print("\n[ALL 7 MODELS / MAIN SCORES]")
    print(data[selected].to_string(index=False))
    comparisons = pd.read_csv(out/"paired_tests_vs_reference26.csv")
    cols = ["model", "delta_rmse_candidate_minus_baseline_t_ha",
            "signflip_mse_maxT_six_one_sided_p",
            "bootstrap_delta_rmse_ci95_low",
            "bootstrap_delta_rmse_ci95_high"]
    print("\n[PAIRED COMPARISON / CONDITIONAL SENSITIVITY]")
    print(comparisons[cols].to_string(index=False))
    print("\n[CAUTION] p-values/CI are exploratory: top-5 selected "
          "post-hoc; IID assumptions may fail under spatial dependence.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    for name in ("check", "run", "summary"):
        action.add_argument("--"+name, action="store_true")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--geo-blocks", type=int, default=10)
    p.add_argument("--distance197", type=Path, default=None,
                   help=("Optional 197x197 distance file. Its first 138 row "
                         "IDs MUST match the saved private nested CSV order."))
    args = p.parse_args()
    if args.bootstrap < 200 or args.geo_blocks < 2:
        p.error("Need >=200 bootstrap draws, at least 2 geographic blocks.")
    if args.check:
        data = load_verified(args.input.resolve())
        ref = data[BASE].to_numpy(float)
        y = data["observed"].to_numpy(float)
        sel = data[NESTED].to_numpy(float)
        print("[CHECK] Verified seven distinct genuine 138-LOOCV predictors.")
        print("[CHECK] 138/138 original label folds, each 137 train + 1 real test.")
        print(f"[CHECK] Original RMSE={np.sqrt(np.mean((ref-y)**2)):.9f}; "
              f"nested RMSE={np.sqrt(np.mean((sel-y)**2)):.9f}.")
        print("[CHECK] No 59-target labels or prediction values loaded.")
    elif args.run:
        run(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
