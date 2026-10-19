#!/usr/bin/env python3
"""Direct Reference26 mean improvement screen: X-analog blending + tail de-shrink.

No synthetic observations, unrelated global regressors, or three-way routers.
Starts from 138 AUTHENTIC leave-one-parcel-out Reference26 mean predictions.
For held-out parcel i, ONLY y of other 137 labeled parcels informs its
neighbor-analog correction. The observed response of i is scoring-only.
Observable X of all 197 parcels may be used transductively.

Modes:
- analog: interpolate between native Reference26 and locally averaged REAL
  yields from X-neighbor parcels; optionally attenuate if unsupported in X.
- tail: expand predictions away from the TRAINING-ONLY mean if they deviate
  past a prespecified central band (tests shrinkage hypothesis).
- Reference26 unchanged as an exact zero-correction control.

This screen is exploratory: the candidate grid is examined on the same 138
historically studied heldouts, so the best observed RMSE is NOT independent
confirmation. The frozen original final 59 are NEVER overwritten.

  python tools/run_reference26_analog_tail_screen.py --check
  python tools/run_reference26_analog_tail_screen.py --smoke
  python tools/run_reference26_analog_tail_screen.py --run
  python tools/run_reference26_analog_tail_screen.py --summary
  python tools/run_reference26_analog_tail_screen.py --predict59 --method analog --geometry robust26 --k 8 --blend 0.1 --gate supported
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import run_reference26_compact14_final as native

ROOT, N, ID, TARGET = native.ROOT, native.N, native.ID, native.TARGET
OUT = ROOT / "reports/reference26_analog_tail_screen"
SEED = 20261019


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_spaces(full: pd.DataFrame, manifest: dict):
    names = list(manifest["features"])
    if len(full) != 197 or len(names) != 26:
        raise ValueError("Expected 197 full X rows and original 26 inputs.")
    raw = full[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("Infinity in model features.")
    med = np.nanmedian(raw, axis=0)
    if not np.isfinite(med).all():
        raise ValueError("All-missing feature column.")
    raw = np.where(np.isnan(raw), med, raw)
    med = np.median(raw, axis=0)
    mad = 1.4826*np.median(np.abs(raw-med), axis=0)
    std = raw.std(axis=0)
    scale = np.where(mad > 1e-10, mad, np.maximum(std, 1e-8))
    x = np.clip((raw-med)/scale, -8, 8)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite X-only transductive representation.")
    pca = PCA(n_components=8, random_state=SEED).fit_transform(x)
    # Avoid giving PC1 disproportionate Euclidean influence solely from units.
    pca_std = np.maximum(pca.std(axis=0), 1e-9)
    return {"robust26": x, "pca8": pca/pca_std}


def neighborhood(z, y, train, query, *, k: int, power: float = 1.):
    train = np.asarray(train, dtype=int)
    query = np.asarray(query, dtype=int)
    if (not len(train) or not len(query) or k < 1
            or np.intersect1d(train, query).size):
        raise ValueError("Training and queries must be disjoint and nonempty.")
    if not np.isfinite(y[train]).all():
        raise ValueError("Training labels must be finite.")
    distances = pairwise_distances(z[query], z[train]) / np.sqrt(z.shape[1])
    selected = np.argsort(distances, axis=1, kind="stable")[:, :min(k,len(train))]
    nearest_dist = np.take_along_axis(distances, selected, axis=1)
    nearest_targets = y[train][selected]
    # Equal weighting when power=0; inverse-distance otherwise.
    weight = np.power(np.maximum(nearest_dist, 1e-5), -power)
    weight /= weight.sum(axis=1, keepdims=True)
    predicted = np.sum(weight*nearest_targets, axis=1)
    dispersion = np.sqrt(np.sum(weight*(nearest_targets-predicted[:, None])**2, axis=1))
    return predicted, nearest_dist[:, 0], dispersion


def training_spacing(z, train):
    if len(train) < 3:
        raise ValueError("At least three real neighbor references needed.")
    d = pairwise_distances(z[train]) / np.sqrt(z.shape[1])
    np.fill_diagonal(d, np.inf)
    return max(float(np.median(np.min(d, axis=1))), 1e-7)


def analog_correction(
    base, analog_y, d_nearest, train_spacing, train_sd,
    neighbor_dispersion, *, blend, gate, max_delta
):
    if not (0 <= blend <= 1 and max_delta > 0 and
            gate in {"none", "supported"}):
        raise ValueError("Invalid analog settings.")
    base = np.asarray(base, dtype=float)
    if gate == "none":
        multiplier = np.ones_like(base)
    else:
        # Pure feature-support confidence and train-y-neighbor disagreement.
        support = np.exp(-0.5*np.square(d_nearest/max(train_spacing,1e-8)))
        agreement = 1./(1.+np.square(
            neighbor_dispersion/max(float(train_sd),1e-8)
        ))
        multiplier = support*agreement
    correction = np.clip(
        blend*multiplier*(analog_y-base), -max_delta, max_delta
    )
    return base+correction, multiplier


def tail_correction(base, train_y, *, gain, threshold_sd, max_delta):
    if gain < 0 or threshold_sd < 0 or max_delta <= 0:
        raise ValueError("Invalid tail correction settings.")
    mu = float(np.mean(train_y))
    sd = max(float(np.std(train_y)), 1e-8)
    dev = np.asarray(base, float)-mu
    shift = gain*np.sign(dev)*np.maximum(np.abs(dev)-threshold_sd*sd, 0.)
    return np.asarray(base, float)+np.clip(shift,-max_delta,max_delta)


def get_inputs(args):
    frame, groups, distance, manifest, ids, *_ = native.load_inputs(
        args.bundle.resolve(), args.peer_repo.resolve()
    )
    full, distance197 = native.prepare_197(
        args.peer_repo.resolve(), frame, distance
    )
    if not np.array_equal(
        full.iloc[:N][ID].astype(str).to_numpy(),
        frame[ID].astype(str).to_numpy()
    ):
        raise ValueError("197/138 target/feature row order mismatch.")
    weights = native.frozen_weights(args.config.resolve())
    loo = native.read_completed(
        args.fresh_fold_dir.resolve(), ids,
        frame[TARGET].to_numpy(float), weights
    )
    if not np.array_equal(
        loo[ID].astype(str).to_numpy(), frame[ID].astype(str).to_numpy()
    ):
        raise ValueError("Saved Reference26 OOF IDs/order mismatch.")
    baseline = loo["Reference26_mean"].to_numpy(float)
    if not np.isfinite(baseline).all():
        raise ValueError("Invalid Reference26 baseline.")
    return frame, full, feature_spaces(full,manifest), baseline


def model_grid(args):
    configs = [{"method":"baseline"}]
    for name in ("robust26", "pca8"):
        for k in (3, 8, 16):
            for blend in (0.1, 0.25):
                for gate in ("none", "supported"):
                    configs.append({
                        "method":"analog", "geometry":name, "k":k,
                        "blend":blend, "gate":gate,
                    })
    for gain in (0.10, 0.20, 0.40):
        for threshold in (0.5, 1.0):
            configs.append({
                "method":"tail","gain":gain,"threshold_sd":threshold
            })
    return configs


def signature(item):
    if item["method"] == "baseline":
        return "Reference26_mean_unchanged"
    if item["method"] == "tail":
        return f"tail__g{item['gain']:g}__t{item['threshold_sd']:g}"
    return (
        f"analog__{item['geometry']}__k{item['k']}__"
        f"blend{item['blend']:g}__{item['gate']}"
    )


def evaluate_folds(y, spaces, baseline, folds, *, max_delta):
    """All X/y training operations mask the heldout's y by index."""
    configs = model_grid(None)
    predictions = {signature(cfg):[] for cfg in configs}
    for pos in folds:
        train = np.delete(np.arange(N), pos)
        query = np.array([pos])
        for cfg in configs:
            key = signature(cfg)
            if cfg["method"] == "baseline":
                value = baseline[pos]
            elif cfg["method"] == "tail":
                value = tail_correction(
                    np.array([baseline[pos]]), y[train], gain=cfg["gain"],
                    threshold_sd=cfg["threshold_sd"],max_delta=max_delta
                )[0]
            else:
                z = spaces[cfg["geometry"]]
                predicted, nearest, dispersion = neighborhood(
                    z, y, train, query, k=cfg["k"]
                )
                spacing = training_spacing(z, train)
                value = analog_correction(
                    np.array([baseline[pos]]), predicted,nearest,
                    spacing, float(np.std(y[train])), dispersion,
                    blend=cfg["blend"], gate=cfg["gate"],
                    max_delta=max_delta,
                )[0][0]
            predictions[key].append(float(value))
    return predictions


def aggregate(y, reference, predictions, folds):
    obs = y[folds]
    base = reference[folds]
    biggest = set(np.argsort(np.abs(reference-y))[-2:].tolist())
    is_hard = np.array([i in biggest for i in folds])
    rows = []
    for name, estimates in predictions.items():
        p = np.asarray(estimates, float)
        e, r = p-obs, base-obs
        changes = p-base
        rows.append({
            "model":name, "n_real_heldouts":len(folds),
            "n_real_train_per_fit":137,
            "rmse_t_ha":float(np.sqrt(np.mean(e*e))),
            "mae_t_ha":float(np.mean(np.abs(e))),
            "bias_t_ha":float(np.mean(e)),
            "delta_mse_vs_ref":float(np.mean(e*e-r*r)),
            "n_predictions_modified":int((np.abs(changes)>1e-10).sum()),
            "n_improved_vs_ref":int((e*e < r*r-1e-12).sum()),
            "n_worsened_vs_ref":int((e*e > r*r+1e-12).sum()),
            "rmse_hard2_posthoc": (
                float(np.sqrt(np.mean(e[is_hard]**2)))
                if is_hard.any() else None
            ),
            "rmse_other_136": (
                float(np.sqrt(np.mean(e[~is_hard]**2)))
                if (~is_hard).any() else None
            ),
            "max_abs_change_t_ha":float(np.max(np.abs(changes))),
        })
    return pd.DataFrame(rows).sort_values(
        ["rmse_t_ha", "n_worsened_vs_ref"],kind="stable"
    ).reset_index(drop=True)


def get_spec(args):
    return {
        "schema":1,
        "protocol":"Original Reference26 mean 138 real parcel 137/1 LOOCV + fixed feature-only analog or training-only tail adjustment",
        "screen_is_posthoc_model_selection":True,
        "native_reference26_never_refitted":True,
        "grid":model_grid(args),
        "max_delta":args.max_delta,
        "source_sha256":sha(Path(__file__).resolve()),
        "frozen_bundle_manifest_sha256":sha(args.bundle.resolve()/"manifest.json"),
        "original_native_fold_spec_sha256":sha(
            args.fresh_fold_dir.resolve()/"run_spec.json"
        ),
    }


def candidate_on_59(args, full, y, spaces, ref):
    original = args.fresh_fold_dir.resolve()/"final_artifacts"
    csv=original/"predictions59_private.csv"
    manifest=original/"final_manifest.json"
    if not csv.is_file() or not manifest.is_file():
        raise FileNotFoundError("Need native Reference26 original final fit and 59 saved predictions.")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data.get("n_train") != N or manifest_data.get("n_unknown") != 59:
        raise ValueError("Final model manifest does not identify 138/59 parcels.")
    source = pd.read_csv(csv,dtype={ID:str})
    required=full.iloc[N:][ID].astype(str).to_numpy()
    if (len(source)!=59 or source[ID].duplicated().any() or
            not np.array_equal(source[ID].astype(str).to_numpy(),required)):
        raise ValueError("Target59 IDs/order mismatch.")
    baseline = source["Reference26_mean_t_ha"].to_numpy(float)
    if not np.isfinite(baseline).all():
        raise ValueError("Nonfinite saved Reference26 target predictions.")
    if args.method == "analog":
        z=spaces[args.geometry]
        predicted, near, disp=neighborhood(
            z,y,np.arange(N),np.arange(N,197),k=args.k
        )
        p, gate=analog_correction(
            baseline, predicted,near,training_spacing(z,np.arange(N)),
            np.std(y),disp,blend=args.blend,gate=args.gate,
            max_delta=args.max_delta
        )
    elif args.method == "tail":
        p=tail_correction(
            baseline,y,gain=args.gain,threshold_sd=args.threshold_sd,
            max_delta=args.max_delta
        )
        gate=np.full(59,np.nan)
    else:
        raise ValueError("For predict59 specify --method analog or tail.")
    private = args.output.resolve()/"candidate59_private.csv"
    private.parent.mkdir(parents=True,exist_ok=True)
    if private.exists():
        raise FileExistsError(f"Existing private candidate not overwritten: {private}")
    pd.DataFrame({
        ID:required, "Reference26_mean_t_ha":baseline,
        "Corrected_candidate_t_ha":p,
        "correction_t_ha":p-baseline,
        "analog_support_multiplier":gate,
    }).to_csv(private,index=False)
    print(f"[PRIVATE] Candidate 59 predictions only: {private}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    act=parser.add_mutually_exclusive_group(required=True)
    for flag in ("check","smoke","run","summary","predict59"):
        act.add_argument("--"+flag,action="store_true")
    parser.add_argument("--bundle",type=Path,default=native.BUNDLE)
    parser.add_argument("--peer-repo",type=Path,default=native.PEER_REPO)
    parser.add_argument("--config",type=Path,default=native.CONFIG)
    parser.add_argument("--fresh-fold-dir",type=Path,default=native.OUTPUT)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--max-delta",type=float,default=0.5)
    parser.add_argument("--method",choices=("analog","tail"),default="analog")
    parser.add_argument("--geometry",choices=("robust26","pca8"),default="robust26")
    parser.add_argument("--k",type=int,default=8)
    parser.add_argument("--blend",type=float,default=0.1)
    parser.add_argument("--gate",choices=("none","supported"),default="supported")
    parser.add_argument("--gain",type=float,default=0.1)
    parser.add_argument("--threshold-sd",type=float,default=0.5)
    args=parser.parse_args()
    if args.max_delta<=0:
        parser.error("max-delta must be positive")
    if args.summary:
        path=args.output.resolve()/"comparison_aggregate.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Run --run first: {path}")
        print(pd.read_csv(path).sort_values("rmse_t_ha").to_string(index=False))
        return
    frame, full, spaces, baseline=get_inputs(args)
    y=frame[TARGET].to_numpy(float)
    if args.check:
        print("[CHECK] 197 X-only rows, 138 true historical Reference26 OOF predictions.")
        print(f"[CHECK] {len(model_grid(args))} fixed candidate recipes, 137 labeled training rows / 1 real test.")
        print(f"[CHECK] Reference26 RMSE={np.sqrt(np.mean((baseline-y)**2)):.9f} t/ha.")
        return
    if args.predict59:
        candidate_on_59(args,full,y,spaces,baseline)
        return
    if args.smoke:
        folds=sorted(set(range(12)))
    else:
        folds=list(range(N))
    preds=evaluate_folds(y,spaces,baseline,folds,max_delta=args.max_delta)
    result=aggregate(y,baseline,preds,folds)
    if args.smoke:
        print("[SMOKE] 12 real heldout parcels only. Not a 138-fold score.")
        print(result.head(10).to_string(index=False))
        return
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    spec=get_spec(args)
    path=out/"run_spec.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=spec:
        raise ValueError("Recipe/source changed. Use a different --output folder.")
    path.write_text(json.dumps(spec,indent=2)+"\n",encoding="utf-8")
    result.to_csv(out/"comparison_aggregate.csv",index=False)
    raw=pd.DataFrame({ID:frame[ID].astype(str),"observed":y,
                      "Reference26_mean":baseline,**preds})
    raw.to_csv(out/"private_heldout_predictions.csv",index=False)
    print("[FULL] 138 genuine Reference26 OOF predictions; 137 TRAINING-only analog labels per fold.")
    print(result.head(15).to_string(index=False))
    print("[CAUTION] This is exploratory parameter screening on already-studied 138 labels.")
    print(f"[PRIVATE] {out}; original 59 final predictions unchanged.")


if __name__=="__main__":
    main()
