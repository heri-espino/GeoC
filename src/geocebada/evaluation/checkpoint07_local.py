"""Checkpoint 07C Local07: nested target-matched local Ridge using new X features.

Retains the *local-neighborhood Ridge* principle of Local04D while allowing
geographic plus representation-distance neighborhoods derived from the new
HLS/SMAP matrix. All supervised fitting and representation scaling are
inside the relevant training fold. The incumbent Local04D remains frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from geocebada.evaluation.checkpoint07c import DevelopmentData, FoldRepresentations

CHOICES = ("pca16", "pca32", "pca64", "raw256")


def _geographic_xy(data: DevelopmentData, ids: list[str]) -> np.ndarray:
    """Convert frozen WGS84 parcel centroids to approximate km in a local plane."""
    names = ("base_centroid_lat", "base_centroid_lon")
    if not set(names).issubset(data.x):
        raise ValueError(
            "Local07 requires base_centroid_lat and base_centroid_lon in 07B X."
        )
    xy = data.x.loc[ids, list(names)].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite geographic coordinates; cannot construct Local07.")
    latitude = xy[:, 0]
    longitude = xy[:, 1]
    mid = float(np.deg2rad(latitude.mean()))
    return np.column_stack([111.195 * latitude, 111.195 * np.cos(mid) * longitude])


def _predict_local(
    a: np.ndarray, b: np.ndarray,
    train_xy: np.ndarray, test_xy: np.ndarray,
    y: np.ndarray, *,
    k: int, alpha: float, geo_share: float,
) -> np.ndarray:
    """One separately fitted neighbor-weighted Ridge model per target parcel."""
    from scipy.spatial.distance import cdist

    if len(a) != len(train_xy) or len(a) != len(y) or len(b) != len(test_xy):
        raise ValueError("Misaligned local training/query representations.")
    if len(a) < 4 or k < 3 or not 0 <= geo_share <= 1:
        raise ValueError("Invalid local neighborhood specification.")
    geo = cdist(test_xy, train_xy)
    feature = cdist(b, a)
    train_geo = cdist(train_xy, train_xy)
    train_feature = cdist(a, a)
    # Scale distance channels from training-only X to avoid leaking X of tests.
    geo_scale = float(np.median(train_geo[np.triu_indices(len(a), 1)]))
    feature_scale = float(np.median(train_feature[np.triu_indices(len(a), 1)]))
    geo /= max(geo_scale, 1e-6)
    feature /= max(feature_scale, 1e-6)
    distances = geo_share * geo + (1.0 - geo_share) * feature
    result = []
    for j in range(len(b)):
        chosen = np.argsort(distances[j])[:min(int(k), len(a))]
        neighborhood = distances[j, chosen]
        weights = 1.0 / (1.0 + neighborhood)
        regressor = Ridge(alpha=float(alpha))
        regressor.fit(a[chosen], y[chosen], sample_weight=weights)
        result.append(float(regressor.predict(b[j:j+1])[0]))
    return np.asarray(result, dtype=float)


def run_local07(
    data: DevelopmentData,
    *,
    split_id: str,
    output: Path,
    trials: int,
    inner_folds: int,
    seed: int,
) -> dict[str, Any]:
    """Nested tuning and resumable Local07 evaluation with paired Local04D."""
    import optuna

    output.mkdir(parents=True, exist_ok=True)
    stem = f"{split_id}__local07"
    csv_path, json_path = output / f"{stem}.csv", output / f"{stem}.json"
    if csv_path.exists() != json_path.exists():
        orphan = csv_path if csv_path.exists() else json_path
        backup = orphan.with_suffix(orphan.suffix + ".incomplete")
        if backup.exists():
            raise RuntimeError(f"Repeated partial Local07 result: {stem}")
        orphan.replace(backup)
        print(f"[LOCAL07] Preserved partial {orphan.name}; recovering.", flush=True)
    if csv_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        rows = pd.read_csv(csv_path)
        if len(rows) != 41 * 5 or meta.get("split_id") != split_id:
            raise ValueError(f"Invalid completed Local07 result: {stem}")
        return meta
    train_ids, test_ids = data.splits[split_id]
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = []
    for ia, ib in inner.split(train_ids):
        first = [train_ids[j] for j in ia]
        second = [train_ids[j] for j in ib]
        folds = FoldRepresentations(data.x, first, second)
        coords_a = _geographic_xy(data, first)
        coords_b = _geographic_xy(data, second)
        ya = data.known_y.loc[first].to_numpy(dtype=float)
        yb = data.known_y.loc[second].to_numpy(dtype=float)
        splits.append((folds, coords_a, coords_b, ya, yb))
    study = optuna.create_study(
        study_name=f"Local07_{split_id}",
        storage=f"sqlite:///{(output / f'{stem}.sqlite').resolve().as_posix()}",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        representation = trial.suggest_categorical("representation", CHOICES)
        k = trial.suggest_categorical("k", [12, 16, 20, 24, 30, 36])
        alpha = trial.suggest_categorical("alpha", [3.0, 10.0, 30.0, 100.0])
        geo_share = trial.suggest_categorical(
            "geo_share", [0.25, 0.50, 0.75, 1.0]
        )
        errors = []
        for fold, ga, gb, ya, yb in splits:
            xa, xb = fold.get(representation)
            forecast = _predict_local(
                xa, xb, ga, gb, ya, k=k, alpha=alpha, geo_share=geo_share
            )
            errors.extend((forecast - yb) ** 2)
        return float(np.sqrt(np.mean(errors)))

    completed = sum(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
    )
    if completed < trials:
        print(f"[LOCAL07] {split_id} trials {completed}/{trials}", flush=True)
        study.optimize(objective, n_trials=trials-completed, gc_after_trial=True)
    if not study.best_trials:
        raise RuntimeError(f"Local07 produced no completed trials: {split_id}")
    pars = study.best_params
    outer = FoldRepresentations(data.x, train_ids, test_ids)
    xa, xb = outer.get(pars["representation"])
    actual = data.known_y.loc[test_ids].to_numpy(dtype=float)
    forecast = _predict_local(
        xa, xb,
        _geographic_xy(data, train_ids), _geographic_xy(data, test_ids),
        data.known_y.loc[train_ids].to_numpy(dtype=float),
        k=pars["k"], alpha=pars["alpha"], geo_share=pars["geo_share"],
    )
    baseline = np.asarray([
        data.incumbent.loc[(split_id, pid), "predicted"] for pid in test_ids
    ], dtype=float)
    variants = {"Local04D": baseline, "Local07": forecast}
    for weight in (0.10, 0.25, 0.50):
        variants[f"Local04D_plus_Local07_w{weight:.2f}"] = (
            (1-weight)*baseline + weight*forecast
        )
    rows = [
        {
            "split_id": split_id, "family": "local07", "ID_POLIGONO": pid,
            "model": name, "observed": float(y), "predicted": float(pred),
        }
        for name, series in variants.items()
        for pid, y, pred in zip(test_ids, actual, series, strict=True)
    ]
    meta = {
        "split_id": split_id, "family": "local07",
        "n_complete_trials": sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ),
        "best_params": pars, "inner_cv_rmse": float(study.best_value),
        "outer_rmse_local04d": float(np.sqrt(np.mean((actual-baseline)**2))),
        "outer_rmse_local07": float(np.sqrt(np.mean((actual-forecast)**2))),
        "outer_train_count": len(train_ids), "outer_test_count": len(test_ids),
        "promoted": False,
    }
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    tmp_json = json_path.with_suffix(".json.tmp")
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)
    tmp_json.write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    tmp_csv.replace(csv_path)
    tmp_json.replace(json_path)
    return meta
