"""Leave-one-parcel-out evaluation for frozen geographic climate models.

Each of 138 labeled parcels is predicted with the other 137 labels.
The 59 hidden-label parcels contribute only transductive X.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from geocebada.evaluation.area_weighted_climate_comparison import (
    BASE, TEMPORAL, TOUCHED, WEIGHTED, Features, predict_local,
)
from geocebada.evaluation.checkpoint08_local import ID

MODELS = (BASE, TOUCHED, WEIGHTED, TEMPORAL)
SEED = 20261016


def leave_one_out_predictions(
    features: Features, *, holdout_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Predict held-out parcels with 137 other observed labels per fit.

    holdout_ids may select smoke-test IDs; the summary refuses incomplete runs.
    """
    known = features.baseline.y.index.astype(str).tolist()
    if len(known) != 138 or len(set(known)) != 138:
        raise ValueError("LOOCV requires exactly 138 distinct known-yield parcels.")
    if len(features.baseline.ids) != 197:
        raise ValueError("Expected 197 parcel feature rows.")
    if set(features.spaces) != set(MODELS):
        raise ValueError("Missing or unexpected feature arm for comparable LOOCV.")
    holdouts = known if holdout_ids is None else list(holdout_ids)
    if len(holdouts) != len(set(holdouts)) or not set(holdouts).issubset(known):
        raise ValueError("Holdout IDs must be unique and have observed yields.")

    records = []
    for held in holdouts:
        train = [parcel for parcel in known if parcel != held]
        if len(train) != 137 or held in train:
            raise AssertionError("Held-out parcel leaked into training IDs.")
        # Heldout y is not inspected until all model predictions are complete.
        forecasts = {
            model: float(predict_local(features, train, [held], model)[0])
            for model in MODELS
        }
        observed = float(features.baseline.y.loc[held])
        for model, predicted in forecasts.items():
            if not np.isfinite(predicted):
                raise ValueError(f"Nonfinite prediction for {model} at {held}.")
            records.append({
                ID: held, "model": model, "observed": observed, "predicted": predicted,
                "outer_train": 137, "outer_heldout": 1,
            })
    return pd.DataFrame.from_records(records)


def leave_one_out_summary(
    predictions: pd.DataFrame, *,
    bootstrap_count: int = 2000, rng_seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute global RMSE/MAE/R2 and paired parcel-bootstrap error deltas.

    Parcel bootstrap intervals do not adjust for spatial dependence.
    """
    if bootstrap_count < 100:
        raise ValueError("At least 100 bootstrap draws are required.")
    required = {ID, "model", "observed", "predicted", "outer_train", "outer_heldout"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing LOOCV columns: {sorted(required - set(predictions))}")
    if set(predictions["model"]) != set(MODELS):
        raise ValueError("LOOCV must compare exactly the four frozen feature arms.")
    if predictions.duplicated([ID, "model"]).any():
        raise ValueError("Duplicate parcel-model LOOCV prediction.")
    by_parcel = predictions.groupby(ID, sort=True)
    if len(by_parcel) != 138 or not by_parcel["model"].nunique().eq(4).all():
        raise ValueError("LOOCV requires 138 parcels, each predicted by four arms.")
    if not by_parcel["observed"].nunique().eq(1).all():
        raise ValueError("Ground truth differs across model arms.")
    if not predictions["outer_train"].eq(137).all() or not predictions["outer_heldout"].eq(1).all():
        raise ValueError("Every LOOCV fit must have 137 train and 1 heldout.")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy(float)).all():
        raise ValueError("Nonfinite LOOCV observations or predictions.")

    observed = by_parcel["observed"].first().sort_index()
    wide = predictions.pivot(index=ID, columns="model", values="predicted").reindex(observed.index)
    if wide.isna().any().any():
        raise ValueError("Incomplete paired LOOCV predictions.")
    truth = observed.to_numpy(float)
    rows = []
    for model in MODELS:
        residual = wide[model].to_numpy(float) - truth
        rows.append({
            "model": model, "unique_heldout_parcels": len(truth),
            "training_labels_per_fit": 137,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(r2_score(truth, wide[model].to_numpy(float))),
            "mean_error": float(np.mean(residual)),
        })
    scores = pd.DataFrame(rows).sort_values("rmse", kind="stable").reset_index(drop=True)

    baseline_squared = np.square(wide[BASE].to_numpy(float) - truth)
    indices = np.random.default_rng(rng_seed).integers(
        0, len(truth), size=(bootstrap_count, len(truth)),
    )
    deltas = []
    for model in MODELS:
        if model == BASE:
            continue
        squared = np.square(wide[model].to_numpy(float) - truth)
        diff = squared - baseline_squared
        boot_mse = diff[indices].mean(axis=1)
        boot_rmse = (
            np.sqrt(squared[indices].mean(axis=1))
            - np.sqrt(baseline_squared[indices].mean(axis=1))
        )
        deltas.append({
            "model": model, "reference": BASE,
            "delta_mse": float(diff.mean()),
            "delta_rmse": float(np.sqrt(squared.mean()) - np.sqrt(baseline_squared.mean())),
            "delta_rmse_ci95_low": float(np.quantile(boot_rmse, 0.025)),
            "delta_rmse_ci95_high": float(np.quantile(boot_rmse, 0.975)),
            "delta_mse_ci95_low": float(np.quantile(boot_mse, 0.025)),
            "delta_mse_ci95_high": float(np.quantile(boot_mse, 0.975)),
            "bootstrap_unit": "parcel (spatial dependence uncorrected)",
        })
    return scores, pd.DataFrame(deltas)
