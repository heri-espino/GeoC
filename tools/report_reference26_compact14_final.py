#!/usr/bin/env python3
"""Full final-model jury metrics, spatial error diagnostics and public JSON.

Read ONLY newly recomputed 138-fold pair of genuine 137/1 predictions.
Moran's I / Geary's C computed with KNN row-standardized spatial weights and
Monte Carlo pseudo-p values. No new fits, no raw parcel data publicly exported.

  python tools/report_reference26_compact14_final.py --run
  python tools/report_reference26_compact14_final.py --summary
  python tools/report_reference26_compact14_final.py --publish

The publication command stages aggregate-only site/data JSON after explicit
validation; it does not git-add or publish any confidential inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_reference26_compact14_final as runner

from geocebada.evaluation.jury_residual_diagnostics import (
    adjust_spatial_tests,
    conditional_error_diagnostics,
    interpret_tests,
    normality_and_distribution,
    paired_loss_diagnostics,
)
from geocebada.evaluation.loocv_comprehensive_metrics import (
    all_metrics,
)

ROOT = runner.ROOT
ID = runner.ID
TARGET = runner.TARGET
N = runner.N
FOLDER = runner.OUTPUT
OUT = FOLDER / "analysis"
SOURCE = runner.BUNDLE
JSON_STAGING = OUT / "jury_metrics_site_ready.json"
NAMES = ("Reference26_mean", "Compact14", "Ensemble_final")
K_VALUES = (4, 8, 12, 24)
DEFAULT_PERMUTATIONS = 1999
SEED = 20261019


def geo_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Haversine distance in kilometres, WGS84 approximate mean Earth radius."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if (lat.shape != (N,) or lon.shape != (N,)
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or np.any(abs(lat) > 90) or np.any(abs(lon) > 180)):
        raise ValueError("138 valid centroid coordinates in degrees required.")
    a, b = np.radians(lat), np.radians(lon)
    dlat = a[:, None]-a[None, :]
    dlon = b[:, None]-b[None, :]
    h = (np.sin(dlat/2)**2
         + np.cos(a[:, None])*np.cos(a[None, :])
         * np.sin(dlon/2)**2)
    distance = 6371.0088*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    np.fill_diagonal(distance, 0.)
    return distance


def neighbors(distance: np.ndarray, k: int) -> np.ndarray:
    """KNN directed/row-standardized, never includes own parcel."""
    if distance.shape != (N, N) or k < 1 or k >= N:
        raise ValueError("Require N×N distances and 1 <= k < N.")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Invalid geographic distances.")
    ranked = np.argsort(
        distance + np.diag(np.full(N, np.inf)), axis=1,
        kind="stable",
    )[:, :k]
    if np.any(ranked == np.arange(N)[:, None]):
        raise AssertionError("A spatial neighbor list includes self.")
    return ranked


def moran(values: np.ndarray, knn: np.ndarray) -> float:
    """Global Moran's I, S0=n for row-standardized directed KNN weights."""
    v = np.asarray(values, float)
    z = v-v.mean()
    den = float(z @ z)
    if den <= 1e-15:
        return np.nan
    return float(np.sum(z * z[knn].mean(axis=1)) / den)


def geary(values: np.ndarray, knn: np.ndarray) -> float:
    """Global Geary C: C≈1 null, C<1 locally similar values."""
    v = np.asarray(values, float)
    den = float(np.sum((v-v.mean())**2))
    if den <= 1e-15:
        return np.nan
    numerator = float(np.sum(
        (v[:, None]-v[knn])**2
    ) / knn.shape[1])
    return (N-1)*numerator/(2*N*den)


def permutation_test(
    values: np.ndarray, knn: np.ndarray, statistic: str,
    *, permutations: int, seed: int,
):
    """Two-sided p centers at permutation mean, plus directional p-values."""
    if permutations < 99:
        raise ValueError("At least 99 permutations required.")
    method = {"moran": moran, "geary": geary}[statistic]
    observed = method(values, knn)
    if not np.isfinite(observed):
        return {
            "estimate": np.nan, "expected_permutation": np.nan,
            "p_two_sided": np.nan, "p_right": np.nan, "p_left": np.nan,
            "permutations": permutations,
        }
    rng = np.random.default_rng(seed)
    sims = np.asarray([
        method(rng.permutation(values), knn) for _ in range(permutations)
    ], dtype=float)
    if not np.isfinite(sims).all():
        raise ValueError("Nonfinite spatial permutation statistic.")
    center = float(np.mean(sims))
    p_two = (1+int(np.sum(
        abs(sims-center) >= abs(observed-center)
    )))/(permutations+1)
    p_right = (1+int(np.sum(sims >= observed)))/(permutations+1)
    p_left = (1+int(np.sum(sims <= observed)))/(permutations+1)
    return {
        "estimate": float(observed),
        "expected_permutation": center,
        "p_two_sided": float(p_two),
        "p_right": float(p_right),
        "p_left": float(p_left),
        "permutations": permutations,
    }


def spatial_metrics(
    actual: np.ndarray, estimates: dict[str, np.ndarray],
    distance: np.ndarray, *, permutations: int, seed: int,
) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        knn = neighbors(distance, k)
        for model_index, (name, pred) in enumerate(estimates.items()):
            error = pred-actual
            variables = (
                ("signed_residual", error),
                ("absolute_error", np.abs(error)),
                ("squared_error", error**2),
            )
            for var_index, (metric, vector) in enumerate(variables):
                for stat_index, stat in enumerate(("moran", "geary")):
                    test = permutation_test(
                        vector, knn, stat, permutations=permutations,
                        seed=seed + 100000*k + 10000*model_index
                        + 100*var_index + stat_index,
                    )
                    rows.append({
                        "model": name, "knn": k,
                        "spatial_variable": metric,
                        "statistic": ("Moran_I" if stat == "moran"
                                      else "Geary_C"),
                        **test,
                        "distance_units": "kilometers",
                        "weights": "directed_KNN_equal_1/k_row_standardized",
                        "two_sided_definition":
                        "abs(statistic-E_perm)>=abs(observed-E_perm)",
                    })
    return pd.DataFrame(rows)


def distance_semivariogram(
    actual: np.ndarray, estimates: dict, distance: np.ndarray,
) -> pd.DataFrame:
    """Descriptive pooled semivariogram of holdout residual differences."""
    iu = np.triu_indices(N, 1)
    separations = distance[iu]
    if np.unique(separations).size < 2:
        raise ValueError("Spatial distances have insufficient variation.")
    cut = np.unique(np.quantile(
        separations, [0, .1, .2, .4, .6, .8, 1.]
    ))
    rows = []
    for model, pred in estimates.items():
        err = np.asarray(pred, float)-actual
        half_sq = (err[iu[0]]-err[iu[1]])**2/2
        for i in range(len(cut)-1):
            lo, hi = float(cut[i]), float(cut[i+1])
            chosen = ((separations >= lo) &
                      (separations <= hi if i == len(cut)-2
                       else separations < hi))
            if not np.any(chosen):
                continue
            rows.append({
                "model": model, "distance_bin": i+1,
                "distance_min_km": lo, "distance_max_km": hi,
                "n_parcel_pairs": int(chosen.sum()),
                "semivariance_t2_ha2": float(np.mean(half_sq[chosen])),
                "median_distance_km": float(np.median(separations[chosen])),
                "warning": (
                    "Pairs are correlated; descriptive, not fitted kriging."
                ),
            })
    return pd.DataFrame(rows)


def extended_metrics(
    actual: np.ndarray, estimates: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Auxiliary scale, tail, calibration and robust losses for the jury."""
    rows = []
    mu = float(actual.mean())
    variance = float(np.mean((actual-mu)**2))
    for name, pred in estimates.items():
        e = pred-actual
        ae = abs(e)
        sq = e**2
        entry = {
            "model": name,
            "n_holdouts": N,
            "l0_nonzero_errors": int(np.count_nonzero(ae > 1e-12)),
            "p25_absolute_error_t_ha": float(np.quantile(ae, .25)),
            "p75_absolute_error_t_ha": float(np.quantile(ae, .75)),
            "p99_absolute_error_t_ha": float(np.quantile(ae, .99)),
            "median_signed_error_t_ha": float(np.median(e)),
            "median_squared_error_t2_ha2": float(np.median(sq)),
            "mean_cubic_absolute_error_t3_ha3": float(np.mean(ae**3)),
            "root_mean_cubic_absolute_error_t_ha": float(
                np.cbrt(np.mean(ae**3))
            ),
            "log_cosh_error": float(
                np.mean(np.logaddexp(e, -e)-np.log(2.))
            ),
            "huber_delta_0p5_t2_ha2": float(np.mean(
                np.where(ae <= .5, .5*sq, .5*ae-.5*.5*.5)
            )),
            "quantile_loss_q10": float(np.mean(np.maximum(
                .10*(actual-pred), -.90*(actual-pred)
            ))),
            "quantile_loss_q50": float(np.mean(np.maximum(
                .50*(actual-pred), -.50*(actual-pred)
            ))),
            "quantile_loss_q90": float(np.mean(np.maximum(
                .90*(actual-pred), -.10*(actual-pred)
            ))),
            "fraction_ae_le_0p25": float(np.mean(ae <= .25)),
            "fraction_ae_le_0p50": float(np.mean(ae <= .5)),
            "fraction_ae_le_1p0": float(np.mean(ae <= 1.)),
            "concordance_ccc": float(
                2*np.mean((actual-mu)*(pred-pred.mean()))
                / (variance+np.var(pred)+(mu-pred.mean())**2)
            ),
            "calibration_intercept_t_ha": np.nan,
            "calibration_slope": np.nan,
            "smape_pct": float(100*np.mean(
                2*ae/np.maximum(abs(actual)+abs(pred), 1e-12)
            )),
            "mape_pct": (
                float(100*np.mean(ae/abs(actual)))
                if np.all(abs(actual) > 1e-10)
                else np.nan
            ),
            "rmsle": (
                float(np.sqrt(np.mean(
                    (np.log1p(pred)-np.log1p(actual))**2
                ))) if np.all(actual >= 0) and np.all(pred >= 0)
                else np.nan
            ),
            "rmse_kg_ha": float(1000*np.sqrt(np.mean(sq))),
            "mean_yield_t_ha": mu,
            "mean_predicted_t_ha": float(np.mean(pred)),
            "error_to_mean_yield_pct": (
                float(100*np.sqrt(np.mean(sq))/abs(mu))
                if abs(mu) > 1e-10 else np.nan
            ),
        }
        if np.var(pred) > 1e-15:
            entry["calibration_slope"] = float(
                np.mean((actual-mu)*(pred-pred.mean()))/np.var(pred)
            )
            entry["calibration_intercept_t_ha"] = float(
                mu-entry["calibration_slope"]*pred.mean()
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def by_region(
    frame: pd.DataFrame, completed: pd.DataFrame,
    *, min_count: int = 5,
) -> pd.DataFrame:
    """Private descriptive regional bias/error; suppress n<5 for privacy."""
    if ID not in frame or ID not in completed:
        raise ValueError("Unknown parcel IDs.")
    for col in ("Estado", "Municipio"):
        if col not in frame:
            raise ValueError(f"Missing geography group {col}.")
    model_cols = list(NAMES)
    info = frame[[ID, "Estado", "Municipio"]].copy()
    info[ID] = info[ID].astype(str)
    joined = completed.merge(info, on=ID, validate="one_to_one")
    if len(joined) != N:
        raise ValueError("Regional analysis must retain all 138 labeled rows.")
    rows = []
    for group in ("Estado", "Municipio"):
        for value, part in joined.groupby(group, dropna=False):
            if len(part) < min_count:
                continue
            for name in model_cols:
                residual = part[name].to_numpy(float)-part["observed"].to_numpy(float)
                rows.append({
                    "group": group, "label": str(value),
                    "model": name, "n": len(part),
                    "rmse_t_ha": float(np.sqrt(np.mean(residual**2))),
                    "mae_t_ha": float(np.mean(abs(residual))),
                    "bias_t_ha": float(residual.mean()),
                    "limit": "descriptive local subset, not external group-CV",
                })
    return pd.DataFrame(rows)


def interpret_metric_family():
    """Spanish UI metadata: explicitly distinguish dimensions and limitations."""
    return {
        "prediction_error": [
            "rmse_t_ha", "mae_t_ha", "mse_t_ha2", "median_ae_t_ha",
            "l1_t_ha", "l2_t_ha", "linf_t_ha", "sse_t_ha2",
            "p90_ae_t_ha", "p95_ae_t_ha", "max_ae_t_ha",
            "bias_t_ha", "error_sd_t_ha",
        ],
        "association_and_skill": [
            "r2_cv", "explained_variance", "pearson_r",
            "spearman_rho", "concordance_ccc", "mae_skill_vs_mean",
        ],
        "relative_and_specialized": [
            "wape_pct", "nrmse_mean_pct", "nrmse_sd_pct",
            "smape_pct", "mape_pct", "rmsle",
            "huber_delta_0p5_t2_ha2", "log_cosh_error",
            "fraction_ae_le_0p25", "fraction_ae_le_0p50",
            "fraction_ae_le_1p0", "calibration_slope",
            "calibration_intercept_t_ha",
        ],
        "spatial": [
            "Global Moran I", "Global Geary C",
            "semi-variogram of CV residuals",
        ],
        "undefined_on_this_model": {
            "adjusted_r2": (
                "No meaningful fixed effective degrees of freedom for a "
                "nested, nonlinear, regularized mixture."
            ),
        },
        "warnings": [
            "L1 and L2 are total vector norms and depend on sample size.",
            "Linfty is max absolute error and is sensitive to one parcel.",
            "R2_cv is 1 - pooled SSE / pooled TSS, not percentage correct.",
            "RMSE/MSE, MAE and Lp errors use t/ha or squared t/ha.",
            "MAPE, sMAPE and RMSLE have domain/near-zero limitations.",
            "Global Moran I is descriptive residual spatial autocorrelation, "
            "not causal geographic effect or out-of-region performance.",
            "Multiple Moran neighborhood and residual-target tests are "
            "exploratory: adjust for multiple comparisons before claiming discoveries.",
            "Weights were selected using the same 138 observed outcomes; "
            "a fixed-weight blend LOOCV does not validate their selection.",
        ],
    }


def sanitize_number(value):
    if isinstance(value, (int, float, np.number)):
        x = float(value)
        return x if np.isfinite(x) else None
    return value


def export_public(
    primary: pd.DataFrame, extended: pd.DataFrame,
    spatial: pd.DataFrame, weights: tuple[float, float],
    out: Path,
):
    """Only model-level aggregate allowlisted values; never IDs, y or coords."""
    permitted = set(NAMES)
    if set(primary.model) != permitted or set(extended.model) != permitted:
        raise ValueError("All 3 verified model metric rows required.")
    if (len(primary) != 3 or len(extended) != 3
            or not set(spatial.model).issubset(permitted)):
        raise ValueError("Unexpected metric output schemas.")
    if any(col in primary or col in extended or col in spatial
           for col in (ID, TARGET, "RENDIMIENTO_T_HA",
                       "base_centroid_lat", "base_centroid_lon")):
        raise ValueError("Sensitive fields cannot enter publication.")
    safe_stats = []
    for name in NAMES:
        a = primary.loc[primary.model.eq(name)].iloc[0].to_dict()
        b = extended.loc[extended.model.eq(name)].iloc[0].to_dict()
        if int(a["n_holdouts"]) != N or int(b["n_holdouts"]) != N:
            raise ValueError("Incomplete metrics cannot be exported.")
        merged = {**a, **b}
        safe_stats.append({
            "model": name,
            "metrics": {key: sanitize_number(value)
                        for key, value in merged.items() if key != "model"},
        })
    geo = []
    for row in spatial.to_dict("records"):
        if (row["knn"] not in K_VALUES
                or row["spatial_variable"] not in
                ("signed_residual", "absolute_error", "squared_error")
                or row["statistic"] not in ("Moran_I", "Geary_C")):
            raise ValueError("Unexpected spatial group.")
        geo.append({
            key: sanitize_number(row[key]) for key in (
                "model", "knn", "spatial_variable", "statistic",
                "estimate", "expected_permutation", "p_two_sided",
                "p_right", "p_left", "permutations",
            )
        })
    data = {
        "schema_version": 1,
        "source": "fresh_138_outer_LOOCV_137_train_fixed_historical_weights",
        "exploratory": True,
        "n_labeled": N, "n_unknown": 59,
        "not_independent_weight_validation": True,
        "models": safe_stats,
        "spatial": geo,
        "weights": dict(zip(NAMES[:2], weights, strict=True)),
        "interpretation": {
            **interpret_metric_family(),
            "statistical_tests": interpret_tests(),
        },
    }
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    return data


def run_report(args):
    peer = args.peer_repo.resolve()
    bundle = args.bundle.resolve()
    frame, groups, dist, manifest, ids, c, pred = runner.load_inputs(
        bundle, peer
    )
    weights = runner.frozen_weights(args.config.resolve())
    folder = args.input.resolve()
    source = folder / "run_spec.json"
    expected = runner.provenance(bundle, peer, args.config.resolve())
    if not source.is_file() or json.loads(source.read_text()) != expected:
        raise ValueError("LOOCV run_spec missing or not matching current sources.")
    completed = runner.read_completed(
        folder, ids, frame[TARGET].to_numpy(float), weights
    )
    y = completed["observed"].to_numpy(float)
    arrays = {name: completed[name].to_numpy(float) for name in NAMES}
    primary = all_metrics(
        completed[[ID, "observed", *NAMES]],
        {name: "retrained_fresh_138_outer_cv" for name in NAMES},
    )
    aux = extended_metrics(y, arrays)
    distance = geo_km(
        frame["base_centroid_lat"].to_numpy(float),
        frame["base_centroid_lon"].to_numpy(float),
    )
    spatial = adjust_spatial_tests(spatial_metrics(
        y, arrays, distance,
        permutations=args.permutations, seed=args.seed,
    ))
    normality = normality_and_distribution(y, arrays)
    heteroscedasticity = conditional_error_diagnostics(y, arrays)
    paired = paired_loss_diagnostics(y, arrays)
    semivariogram = distance_semivariogram(y, arrays, distance)
    regions = by_region(frame, completed)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    primary.to_csv(out / "primary_metrics.csv", index=False,
                   float_format="%.16g")
    aux.to_csv(out / "specialized_metrics.csv", index=False,
               float_format="%.16g")
    spatial.to_csv(out / "global_spatial_autocorrelation.csv", index=False,
                   float_format="%.16g")
    normality.to_csv(out / "normality_and_distribution.csv", index=False,
                     float_format="%.16g")
    heteroscedasticity.to_csv(
        out / "heteroscedasticity_and_calibration.csv", index=False,
        float_format="%.16g",
    )
    paired.to_csv(out / "paired_loss_tests.csv", index=False,
                  float_format="%.16g")
    semivariogram.to_csv(out / "residual_semivariogram.csv", index=False,
                        float_format="%.16g")
    regions.to_csv(out / "private_regional_errors.csv", index=False,
                   float_format="%.16g")
    export_public(primary, aux, spatial, weights, JSON_STAGING if out == OUT
                  else out / JSON_STAGING.name)
    print("[JURY] Fresh 138 LOOCV extended model metrics:\n"
          + primary[["model", "rmse_t_ha", "mae_t_ha",
                     "l1_t_ha", "l2_t_ha", "linf_t_ha",
                     "r2_cv", "adjusted_r2"]].to_string(index=False),
          flush=True)
    moran_rows = spatial.loc[
        spatial.statistic.eq("Moran_I")
        & spatial.spatial_variable.eq("signed_residual")
    ]
    print("[GEO] KNN global Moran residual I + permutations:\n"
          + moran_rows[["model", "knn", "estimate",
                        "p_two_sided", "p_right"]].to_string(index=False),
          flush=True)
    print("[NORMALITY] Exploratory 138 residual diagnostics:\n"
          + normality[["model", "shapiro_w", "shapiro_p_exploratory",
                       "dagostino_pearson_p_exploratory",
                       "anderson_reject_at_5pct"]].to_string(index=False),
          flush=True)
    print("[PAIRED LOSS] Difference vs Reference26 with media:\n"
          + paired[["model", "loss", "mean_loss_difference",
                    "fraction_improved", "wilcoxon_p_exploratory",
                    "wilcoxon_p_exploratory_holm_all"]].to_string(index=False),
          flush=True)
    print("[GEO] n<5 regional groups suppressed; "
          "not a geographically blocked cross-validation.")
    print("[CAUTION] No independent validation of selected mixture weights; "
          "geographic p-values descriptive and subject to multiplicity.")
    return 0


def summary(args):
    directory = args.output.resolve()
    for file in ("primary_metrics.csv", "specialized_metrics.csv",
                 "normality_and_distribution.csv",
                 "heteroscedasticity_and_calibration.csv",
                 "paired_loss_tests.csv",
                 "global_spatial_autocorrelation.csv"):
        path = directory / file
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run --run.")
        data = pd.read_csv(path)
        if file.startswith("global"):
            data = data.loc[
                data.statistic.eq("Moran_I")
                & data.spatial_variable.eq("signed_residual")
            ]
        print(f"[{file}]\n" + data.to_string(index=False), flush=True)
    return 0


def publish(args):
    """Explicit aggregate-only transfer into local sibling todoseboso site."""
    source = args.output.resolve() / JSON_STAGING.name
    if not source.is_file():
        raise FileNotFoundError(f"First execute --run; missing {source}.")
    data = json.loads(source.read_text(encoding="utf-8"))
    if (data.get("n_labeled") != N or
            not data.get("not_independent_weight_validation") or
            set(row["model"] for row in data["models"]) != set(NAMES)):
        raise ValueError("Refuse incomplete/incorrect publishing payload.")
    dest = args.site_output.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(
            f"Site metrics file already exists: {dest}. "
            "Review/remove explicitly; do not silently replace."
        )
    dest.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PUBLISH] Staged AGGREGATES ONLY to {dest}. "
          "No HTML/JS was changed, so existing site doesn't display "
          "the new metrics until a separate reviewed UI integration.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for verb in ("run", "summary", "publish"):
        actions.add_argument("--"+verb, action="store_true")
    parser.add_argument("--input", type=Path, default=FOLDER)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bundle", type=Path, default=SOURCE)
    parser.add_argument("--peer-repo", type=Path, default=runner.PEER_REPO)
    parser.add_argument("--config", type=Path, default=runner.CONFIG)
    parser.add_argument("--permutations", type=int,
                        default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--site-output", type=Path,
        default=ROOT.parent / "todoseboso/site/data/ensemble_evaluation.json",
    )
    args = parser.parse_args()
    if args.run:
        return run_report(args)
    if args.summary:
        return summary(args)
    return publish(args)


if __name__ == "__main__":
    raise SystemExit(main())
