"""Nested leave-one-parcel-out tuning for the geographic climate model family.

Outer test labels are NEVER used for hyperparameter choice. Every outer fold
uses an identical, deterministic fivefold inner split for all feature arms.
Each inner candidate predicts each of 137 outer-training labels exactly once.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from geocebada.evaluation.area_weighted_climate_comparison import Features
from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint08_local import ID
from geocebada.evaluation.climate_leave_one_out import MODELS

SEED = 20261016
INNER_SPLITS = 5
FROZEN = (24, 30.0, 1.0)
TUNED_SUFFIX = "__NestedGridTuned"


def grid_candidates(*, wide: bool = False) -> list[tuple[int, float, float]]:
    """Return reproducible hyperparameter search including the exact incumbent."""
    ks = (8, 12, 16, 20, 24, 32, 48) if wide else (12, 20, 24, 36, 48)
    alphas = (1., 3., 10., 30., 100., 300.) if wide else (3., 10., 30., 100.)
    powers = (0., .5, 1., 2.) if wide else (.5, 1., 2.)
    combinations = list(product(ks, alphas, powers))
    return [FROZEN, *(item for item in combinations if item != FROZEN)]


def _predict(
    features: Features,
    model: str,
    observed: list[str],
    targets: list[str],
    parameters: tuple[int, float, float],
) -> np.ndarray:
    base = features.baseline
    known = [base.index[pid] for pid in observed]
    queries = [base.index[pid] for pid in targets]
    masked = np.full(len(base.ids), np.nan, dtype=float)
    masked[known] = base.y.loc[observed].to_numpy(float)
    k, alpha, power = parameters
    return predict_weighted_local_ridge(
        features.spaces[model],
        base.geographic,
        masked,
        known,
        queries,
        k=k,
        alpha=alpha,
        distance_power=power,
    )


def tune_one_outer_parcel(
    features: Features,
    held: str,
    *,
    candidates: list[tuple[int, float, float]] | None = None,
    inner_splits: int = INNER_SPLITS,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each model's hyperparameters solely on other observed yields.

    The frozen incumbent is included as a scored reference. For candidates
    with equal mean inner squared error, choose the first, i.e. the incumbent.
    """
    base = features.baseline
    known = base.y.index.astype(str).tolist()
    if len(known) != 138 or len(set(known)) != 138 or len(base.ids) != 197:
        raise ValueError("Need exactly 138 observed and 197 total parcels.")
    if held not in known or len(set(models)) != len(models):
        raise ValueError("Heldout parcel or model list is invalid.")
    if not set(models).issubset(features.spaces):
        raise ValueError("Missing requested geographic feature arm.")
    options = grid_candidates() if candidates is None else list(candidates)
    if len(options) != len(set(options)) or FROZEN not in options:
        raise ValueError("Unique candidate grid must include frozen original configuration.")
    if any(k < 3 or alpha < 0 or power < 0 for k, alpha, power in options):
        raise ValueError("Invalid local Ridge hyperparameter candidate.")
    others = [pid for pid in known if pid != held]
    if inner_splits < 2 or inner_splits > len(others):
        raise ValueError("Invalid number of inner folds.")

    splitter = KFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=SEED + base.index[held],
    )
    folds = [
        ([others[i] for i in train], [others[i] for i in valid])
        for train, valid in splitter.split(others)
    ]
    if sorted(p for _, val in folds for p in val) != sorted(others):
        raise AssertionError("Inner validation folds do not cover each label once.")
    truth = float(base.y.loc[held])
    fixed = float(_predict(features, MODELS[0], others, [held], FROZEN)[0])
    prediction_rows = [{
        ID: held, "model": MODELS[0], "observed": truth, "predicted": fixed,
        "outer_train": 137, "outer_heldout": 1,
    }]
    selection_rows = []
    for model in models:
        # CV predictions for each candidate cover all 137 (other) labels once.
        squared_errors = []
        for params in options:
            squared_sum = 0.
            n_predictions = 0
            for train, valid in folds:
                forecast = _predict(features, model, train, valid, params)
                actual = base.y.loc[valid].to_numpy(float)
                squared_sum += float(np.square(forecast - actual).sum())
                n_predictions += len(valid)
            if n_predictions != 137 or not np.isfinite(squared_sum):
                raise ValueError("Nonfinite or incomplete 137-label inner CV.")
            squared_errors.append(squared_sum / n_predictions)
        best_index = min(range(len(options)), key=lambda i: (squared_errors[i], i))
        k, alpha, power = options[best_index]
        outer_prediction = float(_predict(features, model, others, [held], options[best_index])[0])
        selection_rows.append({
            ID: held, "model": model, "k": int(k), "alpha": float(alpha),
            "power": float(power), "inner_cv_rmse": float(np.sqrt(squared_errors[best_index])),
            "frozen_inner_cv_rmse": float(np.sqrt(squared_errors[options.index(FROZEN)])),
            "inner_folds": int(inner_splits), "inner_predictions": 137,
            "candidates": len(options), "selected_frozen": options[best_index] == FROZEN,
        })
        prediction_rows.append({
            ID: held, "model": model + TUNED_SUFFIX,
            "observed": truth, "predicted": outer_prediction,
            "outer_train": 137, "outer_heldout": 1,
        })
    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def summarize_nested_grid(
    predictions: pd.DataFrame, selections: pd.DataFrame, *,
    bootstrap_count: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate paired full coverage and report unbiased-by-inner-tuning OOF scores."""
    expected = {MODELS[0], *(model + TUNED_SUFFIX for model in MODELS)}
    required_predictions = {ID, "model", "observed", "predicted", "outer_train", "outer_heldout"}
    required_selections = {
        ID, "model", "k", "alpha", "power", "inner_cv_rmse", "frozen_inner_cv_rmse",
        "inner_folds", "inner_predictions", "candidates", "selected_frozen",
    }
    if not required_predictions.issubset(predictions):
        raise ValueError("Incomplete nested prediction columns.")
    if not required_selections.issubset(selections):
        raise ValueError("Incomplete nested parameter-selection columns.")
    if set(predictions["model"]) != expected or set(selections["model"]) != set(MODELS):
        raise ValueError("Missing or unexpected models in nested comparison.")
    if predictions.duplicated([ID, "model"]).any() or selections.duplicated([ID, "model"]).any():
        raise ValueError("Duplicate heldout parcel/model in nested results.")
    groups = predictions.groupby(ID)
    selection_groups = selections.groupby(ID)
    if (len(groups) != 138 or len(selection_groups) != 138
            or not groups["model"].nunique().eq(5).all()
            or not selection_groups["model"].nunique().eq(4).all()):
        raise ValueError("Nested grid requires complete 138 x 5 predictions and 138 x 4 selections.")
    if set(groups.groups) != set(selection_groups.groups):
        raise ValueError("Selection/prediction parcel sets disagree.")
    if not groups["observed"].nunique().eq(1).all():
        raise ValueError("Inconsistent observed yields across models.")
    if not predictions["outer_train"].eq(137).all() or not predictions["outer_heldout"].eq(1).all():
        raise ValueError("Invalid nested outer training/heldout sample sizes.")
    if not selections["inner_predictions"].eq(137).all():
        raise ValueError("Inner CV must predict 137 labels.")
    if not selections["inner_folds"].eq(INNER_SPLITS).all():
        raise ValueError("Nested search requires five inner folds.")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy(float)).all():
        raise ValueError("Nonfinite outer predictions.")
    if not np.isfinite(selections[["inner_cv_rmse", "frozen_inner_cv_rmse"]].to_numpy(float)).all():
        raise ValueError("Nonfinite inner scores.")
    if bootstrap_count < 100:
        raise ValueError("At least 100 paired bootstrap draws required.")

    observed = groups["observed"].first().sort_index()
    truth = observed.to_numpy(float)
    wide = predictions.pivot(index=ID, columns="model", values="predicted").reindex(observed.index)
    if wide.isna().any().any():
        raise ValueError("Incomplete paired outer predictions.")
    rows = []
    squared = {}
    for model in [MODELS[0], *(name + TUNED_SUFFIX for name in MODELS)]:
        residual = wide[model].to_numpy(float) - truth
        squared[model] = residual**2
        rows.append({
            "model": model, "heldout_parcels": 138, "training_labels_per_fit": 137,
            "rmse": float(np.sqrt(squared[model].mean())),
            "mae": float(np.abs(residual).mean()),
            "r2": float(r2_score(truth, wide[model].to_numpy(float))),
            "mean_error": float(residual.mean()),
        })
    score = pd.DataFrame(rows).sort_values("rmse", kind="stable").reset_index(drop=True)
    idx = np.random.default_rng(SEED).integers(0, 138, size=(bootstrap_count, 138))
    ref = squared[MODELS[0]]
    paired = []
    for model in [name + TUNED_SUFFIX for name in MODELS]:
        delta = squared[model] - ref
        boot_rmse = np.sqrt(squared[model][idx].mean(axis=1)) - np.sqrt(ref[idx].mean(axis=1))
        paired.append({
            "model": model, "reference": MODELS[0],
            "delta_rmse": float(np.sqrt(squared[model].mean()) - np.sqrt(ref.mean())),
            "delta_rmse_ci95_low": float(np.quantile(boot_rmse, .025)),
            "delta_rmse_ci95_high": float(np.quantile(boot_rmse, .975)),
            "delta_mse": float(delta.mean()),
            "bootstrap_unit": "parcel (spatial correlation not adjusted)",
        })
    counts = (
        selections.groupby(["model", "k", "alpha", "power"], as_index=False)
        .agg(outer_selections=(ID, "size"))
        .sort_values(["model", "outer_selections"], ascending=[True, False])
    )
    return score, pd.DataFrame(paired), counts
