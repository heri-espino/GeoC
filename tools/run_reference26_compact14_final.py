#!/usr/bin/env python3
"""From-scratch paired 138-fold LOOCV + two independent final 138-label refits.

Reference26: ORIGINAL architecture, training-only arithmetic-mean X imputation,
             frozen local Ridge k=24 / alpha=30.
Compact14:  ORIGINAL architecture and preprocessing (median), fresh fit.

NO prior model-cache or OOF prediction files reused. Resume only files from
this experiment whose immutable SHA-based manifest matches exactly.

  python tools/run_reference26_compact14_final.py --check
  python tools/run_reference26_compact14_final.py --smoke --jobs 1
  python tools/run_reference26_compact14_final.py --run --jobs 4
  python tools/run_reference26_compact14_final.py --summary
  python tools/run_reference26_compact14_final.py --train-final --jobs 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from run_selected_champions_loo import (  # noqa: E402
    BUNDLE,
    ID,
    PEER_REPO,
    ROOT,
    TARGET,
    digest,
    fold_record,
    load_inputs,
    write_csv,
)
from threadpoolctl import threadpool_limits  # noqa: E402

from geocebada.evaluation.loocv_comprehensive_metrics import (  # noqa: E402
    all_metrics,
)
from geocebada.evaluation.peer_mean_imputation import (  # noqa: E402
    use_training_means,
)

OUTPUT = ROOT / "reports/reference26_compact14_fresh138"
CONFIG = ROOT / "configs/reference26_compact14_ensemble.json"
N = 138
SEED = 20261018
_WORKER = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_weights(config: Path):
    data = json.loads(config.read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported weight manifest.")
    weights = data["weights"]
    names = {"Reference26_mean", "Compact14"}
    if set(weights) != names:
        raise ValueError("Exactly two named ensemble weights required.")
    a = float(weights["Reference26_mean"])
    b = float(weights["Compact14"])
    if not (np.isfinite([a, b]).all() and a >= 0 and b >= 0
            and np.isclose(a + b, 1., atol=1e-13, rtol=0)):
        raise ValueError("Frozen convex weights are invalid.")
    return a, b


def provenance(bundle: Path, peer: Path, config: Path):
    source_files = [
        "src/current_champions.py",
        "src/family_4/conditional_response.py",
        "src/geoc_signal_research.py",
        "src/physical_similarity.py",
        "src/indicator_neighborhoods.py",
        "src/temporal_indicator_weights.py",
        "src/ct3_residual_conditional/models.py",
        "src/local_ridge_sufficient_statistics.py",
    ]
    files = {
        name: digest(peer / name) for name in source_files
    }
    return {
        "schema": 1,
        "experiment": "fresh native ref26 mean / compact14 original 138 loo",
        "outer_cv": "138x 137 train 1 heldout",
        "inner_cv": f"native 4-fold shuffled seed={SEED}",
        "peer_source_sha": files,
        "original_data_manifest_sha": digest(bundle / "manifest.json"),
        "training_mean_override_sha": digest(
            ROOT / "src/geocebada/evaluation/peer_mean_imputation.py"
        ),
        "runner_sha": digest(Path(__file__).resolve()),
        "weights_sha": sha(config),
        "no_legacy_fit_or_predictions_reuse": True,
    }


def lock_output(output: Path, spec: dict):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Sources or weights changed; use a NEW output directory.")
    else:
        if list(output.glob("fold_*.csv")) or list(output.glob("*.joblib")):
            raise ValueError("Found outputs without matching provenance manifest.")
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def record_path(folder: Path, pos: int):
    return folder / f"fold_{pos + 1:03d}.csv"


def verify_record(path: Path, pos: int, ids: list, y: np.ndarray, weights):
    row = pd.read_csv(path, dtype={ID: str})
    columns = {ID, "observed", "outer_train", "outer_heldout",
               "Reference26_mean", "Compact14", "Ensemble_final"}
    if len(row) != 1 or set(row.columns) != columns:
        raise ValueError(f"{path}: incorrect fold schema.")
    r = row.iloc[0]
    if (str(r[ID]) != str(ids[pos]) or
            abs(float(r["observed"]) - float(y[pos])) > 1e-9 or
            int(r["outer_train"]) != N - 1 or
            int(r["outer_heldout"]) != 1):
        raise ValueError(f"{path}: heldout ID, target, or fold count differs.")
    value = row[["Reference26_mean", "Compact14", "Ensemble_final"]
                ].to_numpy(float).ravel()
    if not np.isfinite(value).all() or not np.isclose(
        value[2], weights[0]*value[0]+weights[1]*value[1],
        atol=1e-11, rtol=0,
    ):
        raise ValueError(f"{path}: invalid frozen weighted output.")
    return r


def init_worker(bundle: str, peer: str, config: str):
    global _WORKER
    inputs = load_inputs(Path(bundle), Path(peer))
    _WORKER = (*inputs, frozen_weights(Path(config)))


def build_fresh_fit(
    frame: pd.DataFrame, groups, distance: np.ndarray, manifest,
    train: np.ndarray, inner, ColdExperts, folder: Path,
    *, mean: bool,
):
    """No joblib/model reuse. Separate native caches for the two recipes."""
    from contextlib import nullcontext

    masked = frame.copy(deep=True)
    excluded = np.delete(np.arange(N), train)
    if excluded.size:
        masked.loc[masked.index[excluded], TARGET] = np.nan
    assert np.isfinite(masked.iloc[train][TARGET].to_numpy(float)).all()
    assert masked.iloc[excluded][TARGET].isna().all()
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), (
        use_training_means() if mean else nullcontext()
    ):
        factory = ColdExperts(
            masked, groups, distance, manifest["geology"], "random",
            train, inner, folder,
        )
        fitted = factory.full_fit(train)
    if set(np.asarray(fitted["train_positions"]).tolist()) != set(train.tolist()):
        raise RuntimeError("Model was fitted on an unexpected set of labels.")
    return {
        "outer_experts": fitted,
        "train_positions": train,
        "B1_weight": 0.,
        "imputation": "arithmetic_train_means" if mean else "original_medians",
    }, masked


def predict_model(predict_saved, pack, query: np.ndarray,
                  frame: pd.DataFrame, distance, key: str):
    with threadpool_limits(limits=1):
        pred = np.asarray(
            predict_saved(pack, query, frame, distance)[key], float
        )
    if pred.shape != (len(query),) or not np.isfinite(pred).all():
        raise ValueError("Native predictor did not return valid predictions.")
    return pred


def train_fold(pos: int, output_name: str):
    if _WORKER is None:
        raise RuntimeError("Worker not initialized.")
    (frame, groups, distance, manifest, ids, ColdExperts,
     predict_saved, weights) = _WORKER
    folder = Path(output_name)
    file = record_path(folder, pos)
    y = frame[TARGET].to_numpy(float)
    if file.is_file():
        verify_record(file, pos, ids, y, weights)
        return pos+1, "cached", 0.
    train, inner = fold_record(pos)
    started = time.perf_counter()
    a, a_frame = build_fresh_fit(
        frame, groups, distance, manifest, train, inner,
        ColdExperts, folder / f"fit_cache_mean_{pos+1:03d}",
        mean=True,
    )
    b, b_frame = build_fresh_fit(
        frame, groups, distance, manifest, train, inner,
        ColdExperts, folder / f"fit_cache_compact_{pos+1:03d}",
        mean=False,
    )
    query = np.asarray([pos])
    pred_a = float(predict_model(
        predict_saved, a, query, a_frame, distance, "Reference26"
    )[0])
    pred_b = float(predict_model(
        predict_saved, b, query, b_frame, distance, "Compact14"
    )[0])
    weighted = weights[0]*pred_a+weights[1]*pred_b
    write_csv(file, pd.DataFrame([{
        ID: ids[pos], "observed": float(y[pos]),
        "outer_train": N - 1, "outer_heldout": 1,
        "Reference26_mean": pred_a,
        "Compact14": pred_b, "Ensemble_final": weighted,
    }]))
    verify_record(file, pos, ids, y, weights)
    return pos+1, "trained", time.perf_counter()-started


def read_completed(folder: Path, ids: list, y: np.ndarray, weights):
    paths = [record_path(folder, i) for i in range(N)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            f"LOOCV incomplete: {sum(p.is_file() for p in paths)}/{N} folds."
        )
    for pos, path in enumerate(paths):
        verify_record(path, pos, ids, y, weights)
    return pd.concat(
        [pd.read_csv(p, dtype={ID: str}) for p in paths], ignore_index=True,
    )


def summarize(folder: Path, frame: pd.DataFrame, ids: list, weights):
    completed = read_completed(
        folder, ids, frame[TARGET].to_numpy(float), weights
    )
    cols = ["Reference26_mean", "Compact14", "Ensemble_final"]
    result = all_metrics(
        completed[[ID, "observed", *cols]],
        {name: "fresh_138_outer_loocv_fixed_historical_weights"
         for name in cols},
    )
    result.to_csv(
        folder / "full_metrics.csv", index=False, float_format="%.16g"
    )
    print("[FRESH 138 LOOCV] Three heldout predictors:\n" +
          result[["model", "rmse_t_ha", "mae_t_ha",
                  "linf_t_ha", "r2_cv", "bias_t_ha"]].to_string(index=False),
          flush=True)
    print("[WARNING] Ensemble weights were fitted/selected on the SAME 138 "
          "target labels in the preceding 4.6M search. This is NOT a "
          "nested unbiased meta-model LOOCV.", flush=True)


def prepare_197(peer: Path, original_train: pd.DataFrame, distance):
    """Use frozen X and geometry only; never reconstruct 197 features blindly."""
    parent = peer / "deployment/model_artifacts"
    p_features = parent / "features197.parquet"
    p_distance = parent / "distance197.npy"
    if not p_features.is_file() or not p_distance.is_file():
        raise FileNotFoundError(
            "Need original verified features197.parquet and distance197.npy "
            "from todoseboso/deployment/model_artifacts; no synthetic X."
        )
    full = pd.read_parquet(p_features)
    d = np.load(p_distance, allow_pickle=False)
    if len(full) != 197 or d.shape != (197, 197):
        raise ValueError("Invalid 197-row feature / distance geometry.")
    if not np.array_equal(
        full.iloc[:N][ID].astype(str).to_numpy(),
        original_train[ID].astype(str).to_numpy(),
    ):
        raise ValueError("The first 138 parcel IDs/order differ.")
    if full[ID].duplicated().any() or full.iloc[N:][TARGET].notna().any():
        raise ValueError("Unknown 59 labels must stay missing, IDs unique.")
    for column in original_train:
        if column not in full:
            raise ValueError(f"Missing feature {column} in 197 bundle.")
        if pd.api.types.is_numeric_dtype(original_train[column]):
            np.testing.assert_allclose(
                full.iloc[:N][column].to_numpy(float),
                original_train[column].to_numpy(float),
                atol=1e-10, rtol=0, equal_nan=True,
                err_msg=column,
            )
        elif (
            full.iloc[:N][column].fillna("__MISSING__").astype(str).tolist()
            != original_train[column].fillna("__MISSING__").astype(str).tolist()
        ):
            raise ValueError(f"Metadata column {column} differs.")
    np.testing.assert_allclose(d[:N, :N], distance, atol=1e-10, rtol=0)
    if not np.isfinite(d).all():
        raise ValueError("Non-finite distances in original 197 geometry.")
    return full, d


def final_refit(
    folder: Path, frame: pd.DataFrame, groups, distance,
    manifest, ColdExperts, predict_saved, peer: Path, weights,
):
    """Fresh 138-label mean Reference26 and independent native Compact14."""
    out = folder / "final_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    names = ("reference26_mean_train138.joblib",
             "compact14_original_train138.joblib")
    complete = out / "final_manifest.json"
    if complete.exists() or any((out / n).exists() for n in names):
        raise ValueError(
            "Final fits already exist; do not silently reuse/overwrite "
            "serialized fitted models. Use a new --output folder."
        )
    full, d197 = prepare_197(peer, frame, distance)
    train = np.arange(N)
    from sklearn.model_selection import KFold
    inner = list(KFold(
        n_splits=4, shuffle=True, random_state=2026,
    ).split(train))
    packs = []
    for name, mean in zip(names, (True, False), strict=True):
        start = time.perf_counter()
        pack, _ = build_fresh_fit(
            frame, groups, distance, manifest, train, inner,
            ColdExperts, out / ("cache_mean" if mean else "cache_compact"),
            mean=mean,
        )
        name_key = "Reference26" if mean else "Compact14"
        q = np.arange(N, 197)
        estimates = predict_model(
            predict_saved, pack, q, full, d197, name_key
        )
        pack["prediction_positions"] = q
        pack["model_name"] = (
            "Reference26_mean" if mean else "Compact14"
        )
        joblib.dump(pack, out / name, compress=3)
        trusted = joblib.load(out / name)
        repeated = predict_model(
            predict_saved, trusted, q, full, d197, name_key
        )
        np.testing.assert_allclose(estimates, repeated, atol=1e-12, rtol=0)
        packs.append(estimates)
        print(f"[FINAL REFIT] {pack['model_name']}: 138 labels, "
              f"59 finite predictions, {time.perf_counter()-start:.1f}s.",
              flush=True)
    combined = weights[0]*packs[0]+weights[1]*packs[1]
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite final ensemble predictions.")
    private = pd.DataFrame({
        ID: full.iloc[N:][ID].astype(str).to_numpy(),
        "Reference26_mean_t_ha": packs[0],
        "Compact14_t_ha": packs[1],
        "Ensemble_final_t_ha": combined,
    })
    write_csv(out / "predictions59_private.csv", private)
    summary = {
        "schema_version": 1,
        "weights": {
            "Reference26_mean": weights[0],
            "Compact14": weights[1],
        },
        "n_train": N,
        "n_unknown": len(private),
        "trained_models": {
            name: sha(out / name) for name in names
        },
        "input_artifacts": {
            "features197.parquet": sha(peer / "deployment/model_artifacts/features197.parquet"),
            "distance197.npy": sha(peer / "deployment/model_artifacts/distance197.npy"),
        },
        "replayed_predictions59": True,
        "prediction59_aggregate": {
            model: {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            } for model, array in zip(
                ("Reference26_mean", "Compact14", "Ensemble_final"),
                (*packs, combined), strict=True,
            )
        },
        "selection_warning": (
            "Mixing weights optimized using all 138 known yields. "
            "Not externally validated as a meta-model."
        ),
    }
    complete.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print("[FINAL REFIT] Two independent joblib files and private 59 "
          "predictions saved. NOT yet deployed or published.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary", "train_final"):
        action.add_argument("--"+name.replace("_", "-"),
                            action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = (
        load_inputs(args.bundle.resolve(), args.peer_repo.resolve())
    )
    weights = frozen_weights(args.config.resolve())
    print(f"[FRESH] Frozen weights={weights}, 138 train labels. "
          "Reference26: MEAN only; Compact14: ORIGINAL imputation.")
    if args.check:
        print("[CHECK] Peer bundle, feature schema, source files and weights OK.")
        print("[CHECK] Each outer fold will fit two NEW native model recipes.")
        return 0
    out = args.output.resolve()
    if args.smoke:
        out = out.parent / (out.name + "_smoke")
    spec = provenance(
        args.bundle.resolve(), args.peer_repo.resolve(),
        args.config.resolve(),
    )
    lock_output(out, spec)
    if args.summary:
        summarize(out, frame, ids, weights)
        return 0
    if args.train_final:
        # Do NOT require 138 LOOCV folds to train final on the 138 known labels.
        final_refit(
            out, frame, groups, distance, manifest, ColdExperts,
            predict_saved, args.peer_repo.resolve(), weights,
        )
        return 0
    wanted = [0] if args.smoke else list(range(N))
    y = frame[TARGET].to_numpy(float)
    missing = []
    for i in wanted:
        file = record_path(out, i)
        if file.is_file():
            verify_record(file, i, ids, y, weights)
        else:
            missing.append(i)
    print(f"[FRESH] New folds {len(missing)}/{len(wanted)}; "
          f"requested workers {args.jobs}. Existing output caches from "
          "OTHER experiments are not used.", flush=True)
    if missing:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(missing)),
            initializer=init_worker,
            initargs=(str(args.bundle.resolve()),
                      str(args.peer_repo.resolve()),
                      str(args.config.resolve())),
        ) as pool:
            futures = {
                pool.submit(train_fold, i, str(out)): i for i in missing
            }
            for ix, fut in enumerate(as_completed(futures), start=1):
                row, status, sec = fut.result()
                print(f"[FRESH {ix}/{len(missing)}] "
                      f"fold={row:03d} {status} {sec:.1f}s", flush=True)
    if args.smoke:
        file = record_path(out, 0)
        verify_record(file, 0, ids, y, weights)
        print("[SMOKE] Passed exact 137/1, mean/reference and "
              "native Compact14 fit; no promotion.", flush=True)
    else:
        summarize(out, frame, ids, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
