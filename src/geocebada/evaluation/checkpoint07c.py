"""Checkpoint 07C nested high-compute search over 07B X-only features.

Every model/representation/hyperparameter is selected using inner folds of
the 97 visible pseudo-training labels. Outer 41 pseudo-target labels are used
only after the choice has been frozen. Results remain development evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint07_pca import dual_pca_fold

ID = "ID_POLIGONO"
Y = "RENDIMIENTO_T_HA"
FAMILIES = ("catboost", "xgboost", "lightgbm", "krr", "pls")
REPRESENTATIONS = ("raw256", "raw512", "pca16", "pca32", "pca64")


@dataclass
class DevelopmentData:
    """Read-only predictor matrix, visible y and frozen split/method evidence."""

    x: pd.DataFrame
    known_y: pd.Series
    splits: dict[str, tuple[list[str], list[str]]]
    incumbent: pd.DataFrame


def load_development(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    membership: pd.DataFrame,
    incumbent: pd.DataFrame,
) -> DevelopmentData:
    """Check 07C coverage and align frozen Local04D pseudo-target predictions."""
    required = {ID, Y, "CONJUNTO"}
    if not required.issubset(targets):
        raise ValueError("Official target table missing required columns.")
    if not {"split_id", "family", "role", ID}.issubset(membership):
        raise ValueError("Frozen membership missing required columns.")
    if not {"split_id", "family", "method", ID, "observed", "predicted"}.issubset(
        incumbent
    ):
        raise ValueError("Incumbent prediction table missing required columns.")
    if ID not in features or not features[ID].is_unique:
        raise ValueError("Features must have unique parcel IDs.")
    blocked = {Y, "CONJUNTO", "PRODUCCION_T", "RENDIMIENTO", "target", "yield"}
    leaks = [c for c in features if c in blocked or "siap_2025" in c.lower()]
    if leaks:
        raise ValueError(f"Prohibited label/outcome feature(s): {leaks[:5]}")
    if len(features) != 197 or len(targets) != 197:
        raise ValueError("Expected 197 parcels.")
    if not targets[ID].is_unique:
        raise ValueError("Official ID duplicated.")
    frame = features.copy()
    frame[ID] = frame[ID].astype(str)
    frame = frame.set_index(ID, verify_integrity=True)
    if set(frame.index) != set(targets[ID].astype(str)):
        raise ValueError("Feature and target parcels differ.")
    x = frame.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf], np.nan
    ).astype(np.float32)
    if len(x.columns) < 1:
        raise ValueError("No numeric features.")
    labeled = targets[targets["CONJUNTO"].eq("ENTRENAMIENTO")].copy()
    hidden = targets[targets["CONJUNTO"].eq("PREDICCION")]
    if len(labeled) != 138 or len(hidden) != 59 or hidden[Y].notna().any():
        raise ValueError("Invalid 138/59 official target contract.")
    y = labeled.assign(**{ID: labeled[ID].astype(str)}).set_index(ID)[Y]
    y = pd.to_numeric(y, errors="raise")
    if y.isna().any():
        raise ValueError("Missing training yields.")

    m = membership.loc[membership["family"].eq("target_matched")].copy()
    m[ID] = m[ID].astype(str)
    baseline = incumbent.loc[
        incumbent["family"].eq("target_matched")
        & incumbent["method"].eq("Local04D")
    ].copy()
    baseline[ID] = baseline[ID].astype(str)
    if baseline.duplicated(["split_id", ID]).any():
        raise ValueError("Duplicate Local04D pseudo-target prediction.")
    baseline = baseline.set_index(["split_id", ID]).sort_index()
    splits: dict[str, tuple[list[str], list[str]]] = {}
    for split_id, group in m.groupby("split_id", sort=True):
        if group[ID].duplicated().any():
            raise ValueError(f"Duplicate split membership: {split_id}")
        train = group.loc[group["role"].eq("pseudo_train"), ID].tolist()
        test = group.loc[group["role"].eq("pseudo_target"), ID].tolist()
        if len(train) != 97 or len(test) != 41 or set(train) & set(test):
            raise ValueError(f"Wrong 97/41 pseudo-split: {split_id}")
        if set(train + test) != set(y.index):
            raise ValueError(f"Pseudo-split covers different labels: {split_id}")
        for pid in test:
            key = (split_id, pid)
            if key not in baseline.index:
                raise ValueError(f"Local04D missing prediction for {key}")
            stored_y = float(baseline.loc[key, "observed"])
            if not np.isclose(stored_y, float(y.loc[pid]), atol=1e-8):
                raise ValueError(f"Local04D observed target mismatch: {key}")
        splits[str(split_id)] = (train, test)
    if len(splits) != 16:
        raise ValueError(f"Expected 16 target-matched splits, got {len(splits)}.")
    return DevelopmentData(x=x, known_y=y, splits=splits, incumbent=baseline)


def _source_family(column: str) -> str:
    # Fairer X-only per-family budget than always selecting interactions.
    if column.startswith("int__"):
        return "interaction"
    if column.startswith("hls_"):
        return column.split("__")[0]
    if column.startswith("smap_"):
        return column.split("__")[0]
    if column.startswith("agro_"):
        return column.split("__")[0]
    return column.split("__")[0].split("_")[0]


def _balanced_columns(train: pd.DataFrame, budget: int) -> list[str]:
    """Rank completeness and diversity from training X, not y or outer test X."""
    eligible: dict[str, list[tuple[float, str]]] = {}
    for col in train.columns:
        values = train[col]
        finite = values.notna()
        if finite.sum() < max(5, int(0.4 * len(train))):
            continue
        n_unique = int(values[finite].nunique())
        if n_unique < 3:
            continue
        diversity = min(n_unique / max(int(finite.sum()), 1), 1.0)
        score = float(finite.mean() * (0.5 + 0.5 * diversity))
        eligible.setdefault(_source_family(col), []).append((-score, col))
    for group in eligible.values():
        group.sort()
    ordered = []
    families = sorted(eligible)
    for j in range(max((len(v) for v in eligible.values()), default=0)):
        for family in families:
            if j < len(eligible[family]):
                ordered.append(eligible[family][j][1])
                if len(ordered) >= budget:
                    return ordered
    return ordered


class FoldRepresentations:
    """Cache fold-local raw-feature selection and dual PCA without leaking y."""

    def __init__(self, x: pd.DataFrame, train_ids: list[str], test_ids: list[str]):
        self.x = x
        self.train_ids = train_ids
        self.test_ids = test_ids
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, representation: str) -> tuple[np.ndarray, np.ndarray]:
        if representation in self._cache:
            return self._cache[representation]
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation {representation}")
        if representation.startswith("raw"):
            n = int(representation.removeprefix("raw"))
            chosen = _balanced_columns(self.x.loc[self.train_ids], n)
            if not chosen:
                raise ValueError("No sufficiently observed raw predictors.")
            train = self.x.loc[self.train_ids, chosen].to_numpy(dtype=float)
            test = self.x.loc[self.test_ids, chosen].to_numpy(dtype=float)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            a = imputer.fit_transform(train)
            b = imputer.transform(test)
            scaler = StandardScaler()
            a = scaler.fit_transform(a)
            b = scaler.transform(b)
        else:
            rank = int(representation.removeprefix("pca"))
            if "pca_all" not in self._cache:
                a = self.x.loc[self.train_ids].to_numpy(dtype=float)
                b = self.x.loc[self.test_ids].to_numpy(dtype=float)
                pca = dual_pca_fold(a, b)
                self._cache["pca_all"] = (pca.train, pca.test)
            full_a, full_b = self._cache["pca_all"]
            a, b = full_a[:, :rank], full_b[:, :rank]
            scaler = StandardScaler()
            a = scaler.fit_transform(a)
            b = scaler.transform(b)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Nonfinite transformed features.")
        self._cache[representation] = (a, b)
        return a, b


def _suggest(trial: Any, family: str) -> dict[str, Any]:
    """Sample conditional hyperparameters from one documented study space."""
    representation = trial.suggest_categorical(
        "representation", list(REPRESENTATIONS)
    )
    params: dict[str, Any] = {"representation": representation}
    if family in ("catboost", "xgboost", "lightgbm"):
        params["n_estimators"] = trial.suggest_int(
            "n_estimators", 400, 2000, step=200
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.007, 0.18, log=True
        )
        params["max_depth"] = trial.suggest_int("max_depth", 2, 8)
        params["reg_lambda"] = trial.suggest_float(
            "reg_lambda", 0.5, 50.0, log=True
        )
        if family != "catboost":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.55, 1.0
            )
        if family == "xgboost":
            params["min_child_weight"] = trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            )
        if family == "lightgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 7, 63)
            params["min_child_samples"] = trial.suggest_int(
                "min_child_samples", 2, 20
            )
    elif family == "krr":
        params["alpha"] = trial.suggest_float("alpha", 0.01, 100, log=True)
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 0.3, log=True)
    elif family == "pls":
        params["n_components"] = trial.suggest_int("n_components", 1, 16)
    else:
        raise ValueError(f"Unknown 07C family {family}")
    return params


def _regressor(
    family: str, params: dict[str, Any], *, device: str, threads: int,
    seed: int,
) -> Any:
    if family == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=int(params["n_estimators"]),
            depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["reg_lambda"]),
            loss_function="RMSE", random_seed=seed, verbose=False,
            thread_count=threads, allow_writing_files=False,
            task_type="GPU" if device == "gpu" else "CPU",
            devices="0" if device == "gpu" else None,
        )
    if family == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            tree_method="hist", device="cuda" if device == "gpu" else "cpu",
            n_jobs=threads, random_state=seed, objective="reg:squarederror",
        )
    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            learning_rate=float(params["learning_rate"]),
            reg_lambda=float(params["reg_lambda"]),
            subsample=float(params["subsample"]), subsample_freq=1,
            colsample_bytree=float(params["colsample_bytree"]),
            n_jobs=threads, random_state=seed, verbosity=-1,
            device_type="cpu",  # No silent assumption of OpenCL GPU build.
        )
    if family == "krr":
        from sklearn.kernel_ridge import KernelRidge

        return KernelRidge(
            alpha=float(params["alpha"]), gamma=float(params["gamma"]),
            kernel="rbf",
        )
    if family == "pls":
        from sklearn.cross_decomposition import PLSRegression

        return PLSRegression(
            n_components=int(params["n_components"]), scale=False,
            max_iter=1000,
        )
    raise ValueError(f"Unknown family {family}")


def _predict_once(
    family: str,
    params: dict[str, Any],
    xy: tuple[np.ndarray, np.ndarray],
    y_train: np.ndarray,
    *,
    device: str,
    threads: int,
    seed: int,
) -> np.ndarray:
    a, b = xy
    candidate = dict(params)
    if family == "pls":
        candidate["n_components"] = min(
            int(candidate["n_components"]), len(a) - 1, a.shape[1]
        )
    model = _regressor(family, candidate, device=device, threads=threads, seed=seed)
    model.fit(a, y_train)
    prediction = np.asarray(model.predict(b)).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{family}: nonfinite prediction.")
    return prediction


def run_nested_search(
    data: DevelopmentData,
    split_id: str,
    family: str,
    *,
    output: Path,
    trials: int,
    inner_folds: int,
    seed: int,
    device: str,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune on inner pseudo-train folds, then produce one honest outer forecast.

    SQLite Optuna study and completed result JSON enable interruption/resume.
    Splits are frozen by identifier. Never evaluate outer y in objective.
    """
    import optuna

    if family not in FAMILIES:
        raise ValueError(f"Unknown model family {family}")
    if inner_folds < 2 or trials < 1 or threads < 1:
        raise ValueError("Invalid CV / trial / threading budget.")
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be cpu or gpu.")
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / f"{split_id}__{family}.csv"
    meta_path = output / f"{split_id}__{family}.json"
    if pred_path.is_file() != meta_path.is_file():
        raise RuntimeError(
            f"Incomplete 07C split result: {split_id} {family}. "
            "Inspect files before resuming; never silently overwrite."
        )
    if pred_path.is_file() and meta_path.is_file():
        import json

        prediction = pd.read_csv(pred_path)
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            saved.get("split_id") != split_id
            or saved.get("family") != family
            or len(prediction) != 41 * 5
            or set(prediction["model"].unique()) != {
                "Local04D", family,
                *(f"Local04D_plus_{family}_w{weight:.2f}"
                  for weight in (0.10, 0.25, 0.50))
            }
        ):
            raise RuntimeError(
                f"Corrupt or incompatible 07C completed result: {split_id} {family}"
            )
        return prediction, saved

    outer_train, outer_test = data.splits[split_id]
    y_train = data.known_y.loc[outer_train].to_numpy(float)
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_cache: list[tuple[FoldRepresentations, np.ndarray, np.ndarray]] = []
    for idx_a, idx_b in inner.split(outer_train):
        a = [outer_train[k] for k in idx_a]
        b = [outer_train[k] for k in idx_b]
        inner_cache.append((
            FoldRepresentations(data.x, a, b),
            data.known_y.loc[a].to_numpy(float),
            data.known_y.loc[b].to_numpy(float),
        ))
    study_path = (output / f"{split_id}__{family}.sqlite").resolve()
    study = optuna.create_study(
        study_name=f"07c_{split_id}_{family}",
        storage=f"sqlite:///{study_path.as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Any) -> float:
        params = _suggest(trial, family)
        scores = []
        for index, (repr_fold, ya, yb) in enumerate(inner_cache):
            matrix = repr_fold.get(str(params["representation"]))
            pred = _predict_once(
                family, params, matrix, ya,
                device=device, threads=threads, seed=seed + index,
            )
            scores.append(float(np.mean((pred - yb) ** 2)))
        return float(np.sqrt(np.mean(scores)))

    completed = sum(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
    )
    to_run = max(0, trials - completed)
    if to_run:
        print(
            f"[07C] {split_id} {family} complete nested trials "
            f"{completed}/{trials}; scheduling {to_run} more",
            flush=True,
        )
        study.optimize(objective, n_trials=to_run, gc_after_trial=True)
    if not study.best_trials:
        raise RuntimeError("No completed tuning trials.")
    chosen = dict(study.best_params)
    out_fold = FoldRepresentations(data.x, outer_train, outer_test)
    forecast = _predict_once(
        family, chosen, out_fold.get(str(chosen["representation"])),
        y_train, device=device, threads=threads, seed=seed,
    )
    actual = data.known_y.loc[outer_test].to_numpy(float)
    baseline = np.asarray(
        [
            data.incumbent.loc[(split_id, pid), "predicted"]
            for pid in outer_test
        ], dtype=float,
    )
    # Fixed weights only: no using held-out y to fit a residual calibrator.
    outputs: dict[str, np.ndarray] = {
        "Local04D": baseline,
        family: forecast,
    }
    for weight in (0.10, 0.25, 0.50):
        outputs[f"Local04D_plus_{family}_w{weight:.2f}"] = (
            (1 - weight) * baseline + weight * forecast
        )
    rows = []
    for method, values in outputs.items():
        for pid, obs, value in zip(outer_test, actual, values, strict=True):
            rows.append({
                "split_id": split_id, "family": family, ID: pid,
                "model": method, "observed": float(obs), "predicted": float(value),
            })
    result = pd.DataFrame(rows)
    info = {
        "split_id": split_id,
        "family": family,
        "tuning_trials_total": len(study.trials),
        "tuning_trials_complete": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "inner_cv_rmse": float(study.best_value),
        "best_params": chosen,
        "device": device if family in ("catboost", "xgboost") else "cpu",
        "outer_train_count": len(outer_train),
        "outer_test_count": len(outer_test),
        "outer_rmse_model": float(np.sqrt(np.mean((actual - forecast) ** 2))),
        "outer_rmse_local04d": float(np.sqrt(np.mean((actual - baseline) ** 2))),
        "validation": "nested development; frozen Local04D and fixed blends",
        "promoted": False,
    }
    # Atomic final write: never mark split complete if interrupted mid-export.
    csv_tmp = pred_path.with_suffix(".csv.tmp")
    json_tmp = meta_path.with_suffix(".json.tmp")
    result.to_csv(csv_tmp, index=False)
    import json
    json_tmp.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    csv_tmp.replace(pred_path)
    json_tmp.replace(meta_path)
    return result, info


def aggregate_07c_results(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize completed split-model CSVs; repeated parcels remain correlated."""
    files = sorted(output.glob("target_matched_*__*.csv"))
    if not files:
        raise FileNotFoundError(f"No completed 07C split files in {output}")
    frames = [pd.read_csv(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["split_id", "family", ID, "model"]).any():
        raise ValueError("Duplicate nested out-of-fold 07C rows.")
    rows = []
    for (family, model), group in combined.groupby(["family", "model"]):
        rmse_splits = group.groupby("split_id").apply(
            lambda sub: float(np.sqrt(
                np.mean((sub["observed"] - sub["predicted"]) ** 2)
            )), include_groups=False,
        )
        rows.append({
            "family": family, "model": model,
            "n_splits_completed": int(group["split_id"].nunique()),
            "n_repeated_predictions": len(group),
            "rmse_pooled": float(np.sqrt(np.mean(
                (group["observed"] - group["predicted"]) ** 2
            ))),
            "rmse_mean_split": float(rmse_splits.mean()),
            "rmse_worst_split": float(rmse_splits.max()),
            "n_unique_parcels": int(group[ID].nunique()),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["n_splits_completed", "rmse_mean_split"],
        ascending=[False, True],
    )
    return combined, summary
