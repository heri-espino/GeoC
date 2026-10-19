"""Academic diagnostics on 138 genuine outer-LOOCV *prediction errors*.

All p-values are labelled exploratory: LOOCV residuals share overlapping
training sets and model/weight selection has reused the same labels.
Do not treat residual-normality as a requirement of predictive RMSE or as
evidence for independent errors. No new fits or model selection here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N = 138


def multiplicity(pvalues: np.ndarray, method: str):
    """Benjamini-Hochberg FDR or Holm FWER, finite p-values only."""
    p = np.asarray(pvalues, float)
    valid = np.isfinite(p)
    n = int(valid.sum())
    result = np.full(len(p), np.nan)
    if not n:
        return result
    order = np.argsort(p[valid], kind="stable")
    sorted_p = p[valid][order]
    if method == "fdr_bh":
        adjusted = np.minimum.accumulate(
            (sorted_p * n / np.arange(1, n+1))[::-1]
        )[::-1]
    elif method == "holm":
        adjusted = np.maximum.accumulate(
            sorted_p*(n-np.arange(n))
        )
    else:
        raise ValueError("Only fdr_bh or holm corrections supported.")
    restore = np.empty(n)
    restore[order] = np.minimum(1., adjusted)
    result[valid] = restore
    return result


def normality_and_distribution(
    actual: np.ndarray, estimates: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Shapiro-Wilk, D'Agostino K2, Anderson-Darling, moments and bias."""
    y = np.asarray(actual, float)
    if y.shape != (N,) or not np.isfinite(y).all():
        raise ValueError("Require 138 observed labels.")
    rows = []
    for name, pred in estimates.items():
        p = np.asarray(pred, float)
        if p.shape != y.shape or not np.isfinite(p).all():
            raise ValueError(f"{name}: Invalid predictions.")
        e = p-y
        if np.std(e) <= 1e-14:
            raise ValueError("Degenerate all-zero residuals cannot be tested.")
        sh = stats.shapiro(e)
        dag = stats.normaltest(e)
        ad = stats.anderson(e, dist="norm")
        percent = np.where(np.isclose(ad.significance_level, 5.), True, False)
        if percent.sum() != 1:
            raise RuntimeError("Anderson 5%-critical-value threshold missing.")
        ad_limit = float(ad.critical_values[percent][0])
        median = float(np.median(e))
        mad = float(stats.median_abs_deviation(e, scale="normal"))
        neg, pos = int(np.sum(e < 0)), int(np.sum(e > 0))
        signed = stats.binomtest(pos, neg+pos, p=.5)
        mean_bias = stats.ttest_1samp(e, popmean=0.)
        q = np.quantile(e, [.01, .05, .10, .25, .50, .75, .90, .95, .99])
        # SciPy: asymptotic Jarque-Bera chi-square p needs >2000 samples;
        # reporting that p with 138 samples would be misleading.
        jb = stats.jarque_bera(e)
        rows.append({
            "model": name, "n": N,
            "skewness_bias_corrected": float(stats.skew(e, bias=False)),
            "excess_kurtosis_bias_corrected": float(
                stats.kurtosis(e, fisher=True, bias=False)
            ),
            "residual_median": median,
            "residual_mad_normalized": mad,
            "residual_iqr": float(q[5]-q[3]),
            "residual_q01": q[0], "residual_q05": q[1],
            "residual_q10": q[2], "residual_q25": q[3],
            "residual_q50": q[4], "residual_q75": q[5],
            "residual_q90": q[6], "residual_q95": q[7],
            "residual_q99": q[8],
            "residuals_positive": pos, "residuals_negative": neg,
            "residuals_equal_zero": int(np.sum(e == 0)),
            "shapiro_w": float(sh.statistic),
            "shapiro_p_exploratory": float(sh.pvalue),
            "dagostino_pearson_k2": float(dag.statistic),
            "dagostino_pearson_p_exploratory": float(dag.pvalue),
            "anderson_darling_statistic": float(ad.statistic),
            "anderson_darling_critical_at_5pct": ad_limit,
            "anderson_reject_at_5pct": bool(ad.statistic > ad_limit),
            "jarque_bera_statistic": float(jb.statistic),
            "jarque_bera_p_not_reported": (
                "chi_square_asymptotic_inaccurate_at_n138"
            ),
            "positive_negative_sign_test_p_exploratory": float(signed.pvalue),
            "mean_zero_t_statistic_exploratory": float(mean_bias.statistic),
            "mean_zero_t_p_exploratory": float(mean_bias.pvalue),
            "normality_warning": (
                "Residuals are cross-validated and not independent "
                "identically distributed model innovations; normality "
                "is not required for the predictive metrics."
            ),
        })
    result = pd.DataFrame(rows)
    for key in ("shapiro_p_exploratory",
                "dagostino_pearson_p_exploratory",
                "positive_negative_sign_test_p_exploratory",
                "mean_zero_t_p_exploratory"):
        result[key+"_holm"] = multiplicity(result[key].to_numpy(float), "holm")
    return result


def conditional_error_diagnostics(
    actual: np.ndarray, estimates: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Prediction-dependent error scatter and robust Brown-Forsythe test.

    Koenker BP n*R² is computed in an auxiliary OLS of e² on intercept
    plus predicted yield. It is diagnostic under iid assumptions and
    must not be interpreted as a formal inference test for OOF residuals.
    """
    y = np.asarray(actual, float)
    rows = []
    for name, pred in estimates.items():
        p = np.asarray(pred, float)
        if p.shape != y.shape or np.std(p) <= 1e-14:
            raise ValueError("A predictor has no variance.")
        error = p-y
        absolute = abs(error)
        squared = error**2
        spearman = stats.spearmanr(p, absolute)
        pearson_signed = stats.pearsonr(p, error)
        pearson_sq = stats.pearsonr(p, squared)
        p_center = p - p.mean()
        sq_center = squared-squared.mean()
        # OLS with intercept; LM=n*R², chi2 with df=1, asymptotic.
        r2 = float(
            (p_center @ sq_center)**2
            / ((p_center @ p_center)*(sq_center @ sq_center))
        ) if sq_center @ sq_center > 1e-14 else 0.
        lm = N*r2
        cuts = np.quantile(p, (1/3, 2/3))
        if cuts[0] >= cuts[1]:
            group_p = np.nan
            bf = np.nan
        else:
            groups = (
                error[p <= cuts[0]],
                error[(p > cuts[0]) & (p <= cuts[1])],
                error[p > cuts[1]],
            )
            test = stats.levene(*groups, center="median")
            group_p = float(test.pvalue)
            bf = float(test.statistic)
        rows.append({
            "model": name,
            "residual_vs_predicted_pearson_r": float(pearson_signed.statistic),
            "residual_vs_predicted_p_exploratory": float(
                pearson_signed.pvalue
            ),
            "absolute_error_vs_predicted_spearman_rho": float(
                spearman.statistic
            ),
            "absolute_error_vs_predicted_p_exploratory": float(
                spearman.pvalue
            ),
            "squared_error_vs_predicted_pearson_r": float(
                pearson_sq.statistic
            ),
            "squared_error_vs_predicted_p_exploratory": float(
                pearson_sq.pvalue
            ),
            "koenker_breusch_pagan_lm_diagnostic": lm,
            "koenker_breusch_pagan_asymptotic_p_exploratory": float(
                stats.chi2.sf(lm, df=1)
            ),
            "brown_forsythe_3_predicted_terciles_statistic": bf,
            "brown_forsythe_3_predicted_terciles_p_exploratory": group_p,
            "conditional_warning": (
                "Predicted and residual values are algebraically "
                "coupled; OOF rows share overlapping training sets. "
                "All reported p-values are exploratory only."
            ),
        })
    out = pd.DataFrame(rows)
    for col in out.columns:
        if col.endswith("_p_exploratory"):
            out[col+"_holm"] = multiplicity(
                out[col].to_numpy(float), "holm"
            )
    return out


def paired_loss_diagnostics(
    actual: np.ndarray,
    estimates: dict[str, np.ndarray],
    *, reference: str = "Reference26_mean",
) -> pd.DataFrame:
    """Per-model paired loss comparisons with multiplicity correction.

    138 OOF errors per model are paired by parcel. Sign/Wilcoxon p-values
    are descriptive, not independent meta-hyperparameter validation.
    """
    if reference not in estimates:
        raise ValueError("Reference predictor missing.")
    y = np.asarray(actual, float)
    anchor = np.asarray(estimates[reference], float)-y
    rows = []
    for name, pred in estimates.items():
        if name == reference:
            continue
        e = np.asarray(pred, float)-y
        for loss_name, loss in (
            ("absolute_error", abs(e)-abs(anchor)),
            ("squared_error", e**2-anchor**2),
        ):
            informative = loss[~np.isclose(loss, 0., rtol=0, atol=1e-12)]
            if not len(informative):
                p_sign, p_wilcox, stat = 1., 1., 0.
            else:
                sign = stats.binomtest(
                    int(np.sum(informative < 0)),
                    n=len(informative), p=.5,
                )
                p_sign = float(sign.pvalue)
                test = stats.wilcoxon(
                    informative, alternative="two-sided", method="auto"
                )
                p_wilcox, stat = float(test.pvalue), float(test.statistic)
            rows.append({
                "reference": reference, "model": name,
                "loss": loss_name, "n_heldouts": N,
                "mean_loss_difference": float(np.mean(loss)),
                "median_loss_difference": float(np.median(loss)),
                "fraction_improved": float(np.mean(loss < -1e-12)),
                "fraction_worse": float(np.mean(loss > 1e-12)),
                "n_non_tied": len(informative),
                "sign_test_p_exploratory": p_sign,
                "wilcoxon_p_exploratory": p_wilcox,
                "wilcoxon_statistic": stat,
                "warning": (
                    "Same 138 parcels reused to select blend weights; "
                    "cross-validated model errors are not independent."
                ),
            })
    result = pd.DataFrame(rows)
    for name in ("sign_test_p_exploratory", "wilcoxon_p_exploratory"):
        result[name+"_holm_all"] = multiplicity(
            result[name].to_numpy(float), "holm"
        )
    return result


def adjust_spatial_tests(spatial: pd.DataFrame):
    """Across ALL tested k, variables, models and spatial statistics."""
    frame = spatial.copy()
    p = frame["p_two_sided"].to_numpy(float)
    frame["p_two_sided_holm_all"] = multiplicity(p, "holm")
    frame["p_two_sided_fdr_bh_all"] = multiplicity(p, "fdr_bh")
    frame["multiple_testing_family"] = (
        "all tested model × K × residual definition × statistic"
    )
    return frame


def interpret_tests():
    return {
        "normality": (
            "Shapiro-Wilk, D'Agostino-Pearson K2 and Anderson-Darling "
            "test compatible normal-distribution patterns for pooled "
            "cross-validated errors. Rejection diagnoses tails/skewness, "
            "not invalidation of RMSE; non-rejection does not prove normality."
        ),
        "heteroscedasticity": (
            "Auxiliary squared-error-versus-predicted Koenker LM and "
            "Brown-Forsythe across predicted-yield terciles probe "
            "conditional dispersion; overlapping OOF training sets "
            "make textbook iid p-values exploratory."
        ),
        "paired": (
            "Sign and Wilcoxon compare matched parcel losses; "
            "historical weight selection on the same outcomes introduces "
            "optimism; no p-value validates the chosen ensemble."
        ),
        "moran": (
            "Permutation Moran I and Geary C diagnose geographical "
            "pattern in residuals, not a causal role of geography. "
            "Holm/BH corrections cover every reported geographic test."
        ),
        "limitations": (
            "At n=138, SciPy specifically warns that the Jarque-Bera "
            "asymptotic chi-square p-value requires much larger samples "
            "(>2000); only its statistic is reported. "
            "Spatial dependence compromises naive iid tests and "
            "pair bootstrap intervals. Adjusted R² has no defensible "
            "single number of effective fitted parameters."
        ),
    }
