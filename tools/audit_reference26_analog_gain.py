#!/usr/bin/env python3
"""PRIVATE paired stability audit of the Reference26 + 3-real-analog gain.

Reads previously generated genuine heldout predictions; does NOT fit or
modify models, does NOT export any parcel IDs, labels, or 59 predictions.

  python tools/audit_reference26_analog_gain.py --check
  python tools/audit_reference26_analog_gain.py --run
  python tools/audit_reference26_analog_gain.py --summary

Caveat: choosing the winner after inspecting these same 138 heldouts is
post-hoc. Even paired bootstrap, geographic sensitivity and a maxT
sign-flip diagnostic cannot eliminate all historical selection bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_reference26_analog_tail_screen as analog
from sklearn.cluster import AgglomerativeClustering

OUT = analog.ROOT / "reports/reference26_analog_tail_screen"
WINNER = "analog__robust26__k3__blend0.25__supported"
BASELINE = "Reference26_mean_unchanged"
SEED = 20261019


def validate(frame: pd.DataFrame):
    if len(frame) != 138 or frame[analog.ID].duplicated().any():
        raise ValueError("Exactly 138 unique original heldout parcels required.")
    expected = {"observed", "Reference26_mean", BASELINE, WINNER}
    if not expected.issubset(frame):
        raise ValueError(f"Missing required heldout columns: {expected-set(frame.columns)}")
    data = frame.drop(columns=[analog.ID]).apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite private heldout columns.")
    y = frame["observed"].to_numpy(float)
    base = frame[BASELINE].to_numpy(float)
    other = frame["Reference26_mean"].to_numpy(float)
    if not np.allclose(base, other, rtol=0, atol=1e-12):
        raise ValueError("Frozen native predictions differ across columns.")
    rmse = float(np.sqrt(np.mean((base-y)**2)))
    if not np.isclose(rmse, .4681381271, atol=2e-6, rtol=0):
        raise ValueError(f"Reference26 control RMSE mismatch: {rmse:.9f}")
    candidate_cols = [
        c for c in frame.columns
        if c not in {analog.ID, "observed", "Reference26_mean", BASELINE}
    ]
    if len(candidate_cols) != 30:
        raise ValueError(
            f"Expected all 30 predeclared corrections, found {len(candidate_cols)}."
        )
    return y, base, candidate_cols


def statistics(y, base, alternative):
    be, ae = base-y, alternative-y
    bs, cs = be**2, ae**2
    return {
        "rmse_reference26": float(np.sqrt(np.mean(bs))),
        "rmse_corrected": float(np.sqrt(np.mean(cs))),
        "rmse_improvement_t_ha": float(
            np.sqrt(np.mean(bs))-np.sqrt(np.mean(cs))
        ),
        "relative_rmse_improvement_pct": float(100.*(
            1.-np.sqrt(np.mean(cs))/np.sqrt(np.mean(bs))
        )),
        "mae_improvement_t_ha": float(np.mean(np.abs(be)-np.abs(ae))),
        "mean_paired_mse_improvement": float(np.mean(bs-cs)),
        "n_improved_squared_error": int(np.sum(cs < bs-1e-12)),
        "n_worsened_squared_error": int(np.sum(cs > bs+1e-12)),
        "max_absolute_prediction_change": float(np.max(np.abs(alternative-base))),
    }


def iid_bootstrap(y, base, candidate, *, draws, rng):
    bs=(base-y)**2
    cs=(candidate-y)**2
    gains=np.empty(draws)
    for begin in range(0,draws,250):
        size=min(250,draws-begin)
        index=rng.integers(0,len(y),size=(size,len(y)))
        gains[begin:begin+size]=(
            np.sqrt(bs[index].mean(axis=1)) -
            np.sqrt(cs[index].mean(axis=1))
        )
    return {
        "iid_bootstrap_rmse_gain_ci95": np.quantile(gains,[.025,.975]).tolist(),
        "iid_bootstrap_fraction_positive": float(np.mean(gains > 0)),
        "iid_bootstrap_draws": draws,
    }


def leave_one_out_stability(y,base,candidate):
    bs=(base-y)**2
    cs=(candidate-y)**2
    n=len(y)
    without_i = np.sqrt((bs.sum()-bs)/(n-1))-np.sqrt(
        (cs.sum()-cs)/(n-1)
    )
    current=float(np.sqrt(np.mean(bs))-np.sqrt(np.mean(cs)))
    return {
        "leave_one_parcel_out_gain_min":float(np.min(without_i)),
        "leave_one_parcel_out_gain_max":float(np.max(without_i)),
        "leave_one_parcel_out_gain_median":float(np.median(without_i)),
        "n_of_138_deletions_still_positive":int(np.sum(without_i>0)),
        "max_change_in_gain_after_deleting_one":float(
            np.max(np.abs(without_i-current))
        ),
    }


def familywise_signflip(y,baseline,all_candidates, *, draws, rng):
    """Approximate maxT paired sign-flip for 30 ALREADY-SPECIFIED variants.

    Simultaneously flip all methods' within-parcel paired MSE benefits.
    This retains cross-method covariance but assumes approximate symmetry
    of each parcel's gain under the null. It does not adjust for the many
    previous studies, adaptation or spatial dependence. Exploratory.
    """
    sq=(baseline-y)**2
    gains=sq[:,None]-(all_candidates-y[:,None])**2
    observed=gains.mean(axis=0)
    max_stat=float(observed.max())
    maxima=np.empty(draws)
    for begin in range(0,draws,100):
        size=min(100,draws-begin)
        signs=(rng.integers(0,2,size=(size,len(y),1))*2-1)
        perm=(signs*gains[None,:,:]).mean(axis=1)
        maxima[begin:begin+size]=perm.max(axis=1)
    return {
        "best_improvement_mse_observed":max_stat,
        "maxT_signflip_approx_p":float((1+np.sum(maxima>=max_stat))/(draws+1)),
        "family_size":int(all_candidates.shape[1]),
        "draws":draws,
        "assumptions":(
            "paired parcel sign symmetry; fixed family of 30 corrections; "
            "does not account for earlier modeling iterations, spatial correlation, "
            "or posthoc Reference26 selection"
        ),
    }


def block_bootstrap(y,base,candidate,distance, *, groups,draws,rng):
    distance=np.asarray(distance,float)
    n=len(y)
    if (distance.shape!=(n,n) or not np.isfinite(distance).all() or
            np.any(distance < -1e-9) or
            not np.allclose(distance,distance.T,atol=1e-8) or
            not np.allclose(np.diag(distance),0,atol=1e-8)):
        raise ValueError("Invalid 138x138 geographic distance matrix.")
    if not 2<=groups<=n//2:
        raise ValueError("Invalid number of spatial blocks.")
    clusters=AgglomerativeClustering(
        n_clusters=groups,linkage="average",metric="precomputed"
    ).fit_predict(np.maximum(distance,0))
    b=(base-y)**2
    c=(candidate-y)**2
    # Instead of treating 138 correlated parcels as independent,
    # resample the groups, then all real member parcels of each group.
    gains=np.empty(draws)
    for begin in range(0,draws,100):
        size=min(100,draws-begin)
        selected=rng.integers(0,groups,size=(size,groups))
        counts=(selected[:,:,None]==np.arange(groups)).sum(axis=1)
        w=counts[:,clusters]
        total=w.sum(axis=1)
        gains[begin:begin+size]=(
            np.sqrt((w*b).sum(axis=1)/total) -
            np.sqrt((w*c).sum(axis=1)/total)
        )
    return {
        "geo_blocks":groups,
        "geo_block_size_min":int(np.bincount(clusters).min()),
        "geo_block_size_max":int(np.bincount(clusters).max()),
        "geo_cluster_bootstrap_rmse_gain_ci95":np.quantile(
            gains,[.025,.975]
        ).tolist(),
        "geo_cluster_bootstrap_fraction_positive":float(np.mean(gains>0)),
        "geo_cluster_bootstrap_draws":draws,
        "geo_cluster_method":"precomputed geographic distance, average linkage",
        "spatial_block_result_is_sensitivity_not_calibrated_CI":True,
    }


def report(path, distance_path, draws, blocks):
    frame=pd.read_csv(path,dtype={analog.ID:str})
    y,base,columns=validate(frame)
    candidate=frame[WINNER].to_numpy(float)
    rng=np.random.default_rng(SEED)
    result={
        "schema":1,
        "winner":WINNER,
        "n_original_heldout":138,
        "prior_selection_from_same_138":True,
        "official_model_not_replaced":True,
        "statistics":statistics(y,base,candidate),
        "iid_parcel_bootstrap":iid_bootstrap(
            y,base,candidate,draws=draws,rng=rng
        ),
        "delete_one_original_parcel":leave_one_out_stability(
            y,base,candidate
        ),
        "all_30_recipe_maxT_signflip":familywise_signflip(
            y,base,frame[columns].to_numpy(float),draws=draws,rng=rng
        ),
    }
    if distance_path is not None:
        matrix=np.load(distance_path,allow_pickle=False)
        if matrix.shape == (197,197):
            matrix=matrix[:138,:138]
        result["spatial_blocks"]=block_bootstrap(
            y,base,candidate,matrix,groups=blocks,draws=draws,rng=rng
        )
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    for name in ("check","run","summary"):
        group.add_argument("--"+name,action="store_true")
    p.add_argument("--input",type=Path,default=OUT/"private_heldout_predictions.csv")
    p.add_argument("--output",type=Path,default=OUT/"stability_aggregate.json")
    p.add_argument("--distance197",type=Path,default=analog.native.PEER_REPO /
                   "deployment/model_artifacts/distance197.npy")
    p.add_argument("--draws",type=int,default=5000)
    p.add_argument("--geo-blocks",type=int,default=10)
    args=p.parse_args()
    if args.summary:
        if not args.output.is_file():
            raise FileNotFoundError("Run the stability audit first.")
        print(json.dumps(json.loads(args.output.read_text(encoding="utf-8")),
                         ensure_ascii=False,indent=2))
        return
    data=pd.read_csv(args.input,dtype={analog.ID:str})
    y,base,cols=validate(data)
    if args.check:
        print(f"[AUDIT CHECK] 138 real heldouts; native RMSE "
              f"{np.sqrt(np.mean((base-y)**2)):.9f}; 30 corrections verified.")
        print(f"[AUDIT CHECK] Distance asset present: {args.distance197.is_file()}")
        return
    if args.draws<100 or args.geo_blocks<2:
        raise ValueError("Insufficient bootstrap draws / spatial blocks")
    # Without geographic distances, run IID / maxT but report omission.
    distance=args.distance197 if args.distance197.is_file() else None
    payload=report(args.input,distance,args.draws,args.geo_blocks)
    if distance is None:
        payload["spatial_blocks"]={
            "omitted":"verified geographic distance197 asset not found"
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    print("[AUDIT] Paired bootstrap, 138 delete-one, 30-recipe maxT "
          "and geographic block sensitivity calculated.")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("[CAUTION] These diagnostics cannot remove the previously "
          "observed selection bias. No official 59 target changes.")


if __name__=="__main__":
    main()
