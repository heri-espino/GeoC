#!/usr/bin/env python3
"""Full statistical diagnostics for the selected 59-target analog correction.

Reuse validated true 138-fold 137/1 Reference26, screened top-five, nested
outer folds and final-selection manifest. NO model refitting or publication.
Outputs including parcel residuals/maps stay in ignored reports/.
Usage:
  python tools/report_reference26_top5_statistics.py --check
  python tools/report_reference26_top5_statistics.py --run
  python tools/report_reference26_top5_statistics.py --summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import report_reference26_compact14_final as jury
import run_reference26_analog_tail_screen as analog
import run_reference26_compact14_final as native
import run_reference26_top5_final59 as final59
import run_reference26_top5_nested as nested
from scipy import stats

from geocebada.evaluation.inferential_sensitivity import (
    grouped_bootstrap,
    influential_parcels,
    paired_advantage_fragility,
    spatial_grid_labels,
)
from geocebada.evaluation.jury_residual_diagnostics import (
    adjust_spatial_tests,
    conditional_error_diagnostics,
    normality_and_distribution,
    paired_loss_diagnostics,
)
from geocebada.evaluation.jury_statistical_report import render_figures
from geocebada.evaluation.loocv_comprehensive_metrics import all_metrics

ID, N = native.ID, native.N
BASE, FIXED, POLICY = "Reference26_mean", "SelectedFinalRule", "NestedSelection"
OUT = native.ROOT / "reports/reference26_top5_statistics"


def read_verified(args):
    frame, _, dist, features, ids, _, _ = native.load_inputs(
        args.bundle.resolve(), args.peer_repo.resolve()
    )
    ids = np.asarray(ids, str)
    y = frame[native.TARGET].to_numpy(float)
    if (len(frame) != N or len(np.unique(ids)) != N
            or not np.isfinite(y).all()
            or not np.array_equal(ids, frame[ID].astype(str).to_numpy())):
        raise ValueError("Incorrect 138 labeled parcel ordering.")
    loo = native.read_completed(
        args.fresh_fold_dir.resolve(), ids, y,
        native.frozen_weights(args.config.resolve())
    )
    original = loo[BASE].to_numpy(float)
    screen = nested.read_screen(args.screen.resolve(), ids, y, original)
    specpath = args.nested_output.resolve() / "run_spec.json"
    if not specpath.is_file():
        raise FileNotFoundError("Original nested experiment run_spec.json missing.")
    # Allow only code-formatting SHA drift; verify all 138 fold predictions next.
    # Same strict provenance policy as the final59 runner itself.
    final59.validate_nested_provenance(
        json.loads(specpath.read_text(encoding="utf-8")),
        nested.provenance(args),
    )
    scores, counts = nested.summary_from_folds(
        args.nested_output.resolve(), ids, y, original, screen, save=False
    )
    folded = pd.concat([
        pd.read_csv(nested.record_path(args.nested_output.resolve(), i),
                    dtype={ID: str})
        for i in range(N)
    ], ignore_index=True)
    if not np.array_equal(folded[ID].astype(str).to_numpy(), ids):
        raise ValueError("138 verified nested predictions lost ID alignment.")
    final_path = args.final_output.resolve() / "final_selection_manifest.json"
    if not final_path.is_file():
        raise FileNotFoundError("Run final59 --run first; missing selection manifest.")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    winner = final.get("selected_final_model")
    if (winner not in nested.CHOICES
            or final.get("source_nested_spec_sha256") != native.sha(specpath)
            or final.get("source_screen_sha256") != native.sha(args.screen.resolve())
            or final.get("training_labels") != N
            or final.get("final_prediction_count") != 59
            or not np.isclose(final.get("max_delta_t_ha", -1), args.max_delta)
            or not np.isclose(final.get("minimum_relative_mse_gain", -1),
                              args.min_inner_gain)):
        raise ValueError("Final selected recipe does not match this CV study.")
    fixed_column = BASE if winner == nested.BASE else winner
    estimates = {
        BASE: original,
        FIXED: folded[fixed_column].to_numpy(float),
        POLICY: folded[POLICY].to_numpy(float),
    }
    if not all(v.shape == (N,) and np.isfinite(v).all()
               for v in estimates.values()):
        raise ValueError("A heldout prediction vector is invalid.")
    full, _ = native.prepare_197(args.peer_repo.resolve(), frame, dist)
    spaces = analog.feature_spaces(full, features)
    return frame, ids, y, estimates, winner, final, spaces, scores, counts


def correction_details(y, base, corrected, winner, spaces, cap):
    """Reconstruct fixed chosen rule using 137 neighbor labels per fold."""
    if winner == nested.BASE:
        if not np.allclose(base, corrected, atol=1e-10, rtol=0):
            raise ValueError("Unchanged final rule mismatches base.")
        return pd.DataFrame({
            "outer_fold": np.arange(1, N+1), "nearest_distance": np.nan,
            "neighbor_sd": np.nan, "support_gate": 0.,
            "correction_t_ha": corrected-base, "capped": False,
        })
    cfg = next(c for c in nested.TOP5 if analog.signature(c) == winner)
    z = spaces[cfg["geometry"]]
    rows = []
    for i in range(N):
        train = np.delete(np.arange(N), i)
        neighbor_y, nearest, spread = analog.neighborhood(
            z, y, train, [i], k=cfg["k"]
        )
        estimate, gate = analog.analog_correction(
            np.array([base[i]]), neighbor_y, nearest,
            analog.training_spacing(z, train), float(np.std(y[train])),
            spread, blend=cfg["blend"], gate=cfg["gate"], max_delta=cap
        )
        if not np.isclose(estimate[0], corrected[i], atol=1e-8, rtol=0):
            raise ValueError(f"Correction parity failure on real heldout {i+1}")
        raw_shift = cfg["blend"]*gate[0]*(neighbor_y[0]-base[i])
        rows.append({
            "outer_fold": i+1, "nearest_distance": float(nearest[0]),
            "neighbor_sd": float(spread[0]), "support_gate": float(gate[0]),
            "correction_t_ha": float(corrected[i]-base[i]),
            "capped": bool(abs(raw_shift) > cap + 1e-12),
        })
    return pd.DataFrame(rows)


def calibration_ols_hc3(y, predictions):
    """OLS y on OOF yhat; conditional HC3, NOT model coefficient inference."""
    rows = []
    for name, pred in predictions.items():
        x = np.column_stack((np.ones(N), pred))
        if np.linalg.matrix_rank(x) != 2:
            raise ValueError("Calibration predictor has zero variance.")
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y-x@b
        bread = np.linalg.inv(x.T@x)
        leverage = np.sum((x@bread)*x, axis=1)
        omega = (resid/np.maximum(1-leverage, 1e-9))**2
        se = np.sqrt(np.maximum(np.diag(
            bread@(x.T@(omega[:, None]*x))@bread
        ), 0))
        half = stats.t.ppf(.975, N-2)*se
        for j, param in enumerate(("intercept_t_ha", "slope")):
            rows.append({
                "model": name, "diagnostic_parameter": param,
                "estimate": float(b[j]), "conditional_HC3_se": float(se[j]),
                "conditional_ci95_lower": float(b[j]-half[j]),
                "conditional_ci95_upper": float(b[j]+half[j]),
                "warning": "Conditional calibration only; overlapping OOF models "
                           "and selected predictors invalidate naive inferential coverage.",
            })
    return pd.DataFrame(rows)


def supplemental_spatial(y, predictions, geo, permutations, seed):
    signals = [("ObservedYield", "observed_yield", y)]
    signals += [(name, "predicted_yield", p) for name,p in predictions.items()]
    signals.append((FIXED, "correction_delta", predictions[FIXED]-predictions[BASE]))
    rows = []
    for k in jury.K_VALUES:
        near = jury.neighbors(geo, k)
        for j, (name, variable, data) in enumerate(signals):
            for s, test in enumerate(("moran", "geary")):
                result = jury.permutation_test(
                    data, near, test, permutations=permutations,
                    seed=seed+5000000+k*100000+j*100+s
                )
                rows.append({
                    "model": name, "knn": k, "spatial_variable": variable,
                    "statistic": "Moran_I" if test=="moran" else "Geary_C",
                    **result, "distance_units": "kilometers",
                    "weights": "directed_KNN_equal_1/k_row_standardized",
                })
    return pd.DataFrame(rows)


def tails(y, predictions):
    records = []
    for definition, anchor in (
        ("observed_yield_posthoc", y),
        ("reference_predicted_yield", predictions[BASE]),
    ):
        low, high = np.quantile(anchor, [.1, .9])
        for region, selected in {
            "lower_10pct": anchor<=low,
            "middle_80pct": (anchor>low)&(anchor<high),
            "upper_10pct": anchor>=high,
        }.items():
            baseline_error = predictions[BASE][selected]-y[selected]
            for name, pred in predictions.items():
                e = pred[selected]-y[selected]
                records.append({
                    "definition": definition, "region": region,
                    "model": name, "n": int(selected.sum()),
                    "rmse_t_ha": float(np.sqrt(np.mean(e**2))),
                    "mae_t_ha": float(np.mean(abs(e))),
                    "bias_t_ha": float(e.mean()),
                    "delta_mse_vs_base_t2_ha2":
                        float(np.mean(e**2-baseline_error**2)),
                    "warning": "Post-hoc subgroup; observed yield unavailable for 59.",
                })
    return pd.DataFrame(records)


def grouped_regions(frame, complete):
    info = frame[[ID, "Estado", "Municipio"]].copy()
    info[ID] = info[ID].astype(str)
    joined = complete.merge(info, on=ID, how="inner", validate="one_to_one")
    if len(joined) != N:
        raise ValueError("Regional data are incomplete.")
    rows=[]
    for by in ("Estado", "Municipio"):
        for region, part in joined.groupby(by, dropna=False):
            if len(part)<5:
                continue
            for name in (BASE,FIXED,POLICY):
                e=part[name].to_numpy(float)-part["observed"].to_numpy(float)
                rows.append({
                    "group":by, "region":str(region), "model":name, "n":len(part),
                    "rmse_t_ha":float(np.sqrt(np.mean(e**2))),
                    "mae_t_ha":float(abs(e).mean()), "bias_t_ha":float(e.mean()),
                })
    return pd.DataFrame(rows)


def run(args):
    frame, ids, y, estimates, winner, final, spaces, scores, counts = read_verified(args)
    print(f"[VERIFIED] 138 authentic outer LOOCV folds; final recipe={winner}")
    if args.check:
        print(scores.to_string(index=False))
        print("[SELECTION COUNTS] "+str(counts.to_dict()))
        return
    out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    complete=pd.DataFrame({ID:ids, "observed":y, **estimates})
    params=correction_details(
        y, estimates[BASE], estimates[FIXED], winner, spaces, args.max_delta
    )
    params.insert(0, ID, ids)
    primary=all_metrics(complete, {
        BASE:"native_137_train_outer_LOOCV",
        FIXED:"fixed_59_final_recipe_selected_after_historical_138_screen",
        POLICY:"fold_nested_selection_from_historically_preselected_5",
    })
    specialized=jury.extended_metrics(y, estimates)
    normality=normality_and_distribution(y, estimates)
    conditional=conditional_error_diagnostics(y, estimates)
    paired=paired_loss_diagnostics(y, estimates, reference=BASE)
    location=frame[["base_centroid_lat","base_centroid_lon"]].to_numpy(float)
    geo=jury.geo_km(location[:,0], location[:,1])
    spatial=adjust_spatial_tests(pd.concat([
        jury.spatial_metrics(y, estimates, geo,
                             permutations=args.permutations, seed=args.seed),
        supplemental_spatial(y, estimates, geo,
                             args.permutations, args.seed)
    ], ignore_index=True))
    variogram=jury.distance_semivariogram(y, estimates, geo)
    influence_private, influence=influential_parcels(y, estimates, ids, reference=BASE)
    fragility=paired_advantage_fragility(y, estimates, reference=BASE)
    region=grouped_regions(frame, complete)
    labels=("state="+frame["Estado"].astype(str)
            +"|municipality="+frame["Municipio"].astype(str)).to_numpy()
    partitions=[("municipality",labels)]
    for width in (15.,30.,50.):
        for shift in (0.,.5):
            partitions.append((
                f"grid_{int(width)}km_shift_{shift}",
                spatial_grid_labels(location[:,0],location[:,1],
                                    width_km=width,offset_fraction=shift)
            ))
    bootstrap=[]
    audit=[]
    for i,(name,group) in enumerate(partitions):
        n=len(np.unique(group))
        if n<args.min_groups:
            audit.append({"grouping":name,"n_groups":n,"status":"SKIPPED"})
            continue
        bootstrap.append(grouped_bootstrap(
            y,estimates,group,reference=BASE,grouping=name,
            replicates=args.replicates,seed=args.seed+i,min_groups=args.min_groups
        ))
        audit.append({"grouping":name,"n_groups":n,"status":"COMPUTED"})
    if not bootstrap:
        raise ValueError("No valid spatial partitions for grouped bootstrap.")
    tables={
        "primary_metrics.csv":primary,
        "specialized_metrics.csv":specialized,
        "normality_and_distribution.csv":normality,
        "heteroscedasticity_and_calibration.csv":conditional,
        "calibration_parameters_conditional.csv":calibration_ols_hc3(y,estimates),
        "paired_loss_tests.csv":paired,
        "global_spatial_autocorrelation.csv":spatial,
        "residual_semivariogram.csv":variogram,
        "tail_conditional_metrics.csv":tails(y,estimates),
        "private_regional_errors.csv":region,
        "leave_one_out_deletion_summary.csv":influence,
        "paired_advantage_fragility.csv":fragility,
        "grouped_bootstrap_intervals.csv":pd.concat(bootstrap,ignore_index=True),
        "spatial_partition_audit.csv":pd.DataFrame(audit),
    }
    for filename,table in tables.items():
        table.to_csv(out/filename,index=False,float_format="%.16g")
    complete.to_csv(out/"PRIVATE_oof_138.csv",index=False,float_format="%.16g")
    params.to_csv(out/"PRIVATE_selected_gate_by_fold.csv",
                  index=False,float_format="%.16g")
    influence_private.to_csv(out/"PRIVATE_parcel_influence.csv",
                             index=False,float_format="%.16g")
    figures=render_figures(y,estimates,location,spatial,variogram,out)
    cfg=next((c for c in nested.TOP5 if analog.signature(c)==winner),None)
    parameters={
        "winner_selected_for_59":winner,
        "fixed_correction":cfg or {"fallback":"Reference26_mean_unchanged"},
        "max_abs_correction_t_ha":args.max_delta,
        "robust_feature_count":26 if cfg and cfg["geometry"]=="robust26" else None,
        "distance_power":1.0 if cfg else None,
        "median_support_gate":float(params.support_gate.median()),
        "mean_support_gate":float(params.support_gate.mean()),
        "mean_correction_t_ha":float(params.correction_t_ha.mean()),
        "max_abs_correction_138_t_ha":float(abs(params.correction_t_ha).max()),
        "n_capped_138":int(params.capped.sum()),
        "model_coefficient_pvalues":"not_applicable",
        "adjusted_r2":"not_applicable_no_stable_effective_df",
    }
    (out/"selected_algorithm_parameters.json").write_text(
        json.dumps(parameters,indent=2)+"\n",encoding="utf-8")
    (out/"provenance.json").write_text(json.dumps({
        "n_heldouts":N, "outer_train_per_fold":N-1,
        "selected_manifest_sha256":native.sha(
            args.final_output.resolve()/"final_selection_manifest.json"),
        "nested_spec_sha256":native.sha(
            args.nested_output.resolve()/"run_spec.json"),
        "selection_counts":{str(k):int(v) for k,v in counts.items()},
        "permutations":args.permutations,
        "bootstrap_replicates":args.replicates,
        "selection_bias_removed":False,
        "model_refitted":False,
        "individual_results_are_private":True,
    },indent=2)+"\n",encoding="utf-8")
    shown=primary[["model","rmse_t_ha","mae_t_ha","r2_cv",
                   "l1_t_ha","l2_t_ha","linf_t_ha"]]
    moran=spatial.loc[
        spatial.spatial_variable.eq("signed_residual")&
        spatial.statistic.eq("Moran_I"),
        ["model","knn","estimate","p_two_sided",
         "p_two_sided_holm_all"],
    ]
    lines=[
        "# Statistical analysis of selected tail-corrected Reference26",
        "",
        "Selected fixed recipe for 59 targets: "+winner,
        "",
        "This is a conditional exploratory analysis of 138 observed yields.",
        "Reference26 native LOOCV, fixed final correction, and fold-wise",
        "nested policy are separate columns. The top five were originally",
        "selected on these SAME labels, so neither metric validates the",
        "study-wide model search or guarantees error on the 59 hidden yields.",
        "",
        "## All common metrics: RMSE, MAE, R2, L1, L2, Linfinity",
        "",
        shown.to_string(index=False), "",
        "## Signed-residual Moran I across spatial k",
        "",
        moran.to_string(index=False), "",
        "## Selected algorithmic parameters",
        "",
        json.dumps(parameters,indent=2), "",
        "## Other complete analyses",
    ]
    lines += ["- "+key for key in tables]
    lines += [
        "",
        "## Private figures",
    ]
    lines += ["- figures/"+p.name for p in figures]
    lines += [
        "",
        "## Inference limitations",
        "- Moran I/Geary C, normality tests and heteroscedasticity tests",
        "  use correlated OOF errors and cannot prove error independence.",
        "- Holm and BH correct the complete set of displayed spatial tests.",
        "- Cluster bootstrap is conditional on fixed predictions and groups;",
        "  it does not account for retraining or adaptive model selection.",
        "- Observed-y tail groups are post-hoc descriptive and cannot be",
        "  used as decision rules on 59 unknown-y target parcels.",
        "- Conditional HC3 OLS of y on yhat estimates calibration, NOT",
        "  the fitted algorithm's regression parameters or valid coverage.",
        "- Adjusted R2 and coefficient significance have no defensible",
        "  fixed degrees of freedom for this adaptive composite model.",
        "- Never publish any PRIVATE CSV, parcel map or full report",
        "  without checking confidentiality and source approval.",
    ]
    (out/"STATISTICAL_REPORT.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    print("[METRICS]\n"+shown.to_string(index=False))
    print("[MORAN]\n"+moran.to_string(index=False))
    print("[PARAMETERS]\n"+json.dumps(parameters,indent=2))
    print("[PRIVATE] "+str(out/"STATISTICAL_REPORT.md"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ("check","run","summary"):
        actions.add_argument("--"+action,action="store_true")
    parser.add_argument("--bundle",type=Path,default=native.BUNDLE)
    parser.add_argument("--peer-repo",type=Path,default=native.PEER_REPO)
    parser.add_argument("--config",type=Path,default=native.CONFIG)
    parser.add_argument("--fresh-fold-dir",type=Path,default=native.OUTPUT)
    parser.add_argument("--screen",type=Path,default=nested.SCREEN)
    parser.add_argument("--nested-output",type=Path,default=nested.OUT)
    parser.add_argument("--final-output",type=Path,default=native.ROOT/
                        "reports/reference26_top5_final59")
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--max-delta",type=float,default=.5)
    parser.add_argument("--min-inner-gain",type=float,default=.01)
    parser.add_argument("--permutations",type=int,default=1999)
    parser.add_argument("--replicates",type=int,default=3000)
    parser.add_argument("--min-groups",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20261019)
    args=parser.parse_args()
    if (args.permutations<99 or args.replicates<100 or
        args.min_groups<2 or args.max_delta<=0 or
        not 0<=args.min_inner_gain<1):
        parser.error("Invalid experiment settings.")
    if args.summary:
        p=pd.read_csv(args.output/"primary_metrics.csv")
        s=pd.read_csv(args.output/"global_spatial_autocorrelation.csv")
        print("[METRICS]\n"+p.to_string(index=False))
        print("[MORAN]\n"+s.loc[
            s.spatial_variable.eq("signed_residual")&
            s.statistic.eq("Moran_I")
        ].to_string(index=False))
        return
    run(args)


if __name__=="__main__":
    main()
