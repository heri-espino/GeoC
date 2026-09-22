"""Checkpoint 05 global-local stacking and mixture-of-experts utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


def constrain_component_param_grid(
    param_grid: list[dict[str, list[Any]]],
    *,
    parameter: str,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    n_features: int,
) -> tuple[list[dict[str, list[Any]]], int]:
    """Restrict component-count candidates to values feasible in every inner fold.

    Component models such as PLS are bounded by both feature dimension and the
    number of training rows seen by the estimator. Nested cross-fitting can make
    an inner training fold much smaller than the outer visible-label set, so a
    grid that was valid in Checkpoint 03 can become invalid in Checkpoint 05.
    """

    if int(n_features) < 1:
        raise ValueError("Component-grid constraint requires at least one feature.")
    if not inner_splits:
        raise ValueError("Component-grid constraint requires at least one inner split.")

    min_inner_train = min(
        len(np.asarray(train_idx, dtype=int))
        for train_idx, _ in inner_splits
    )
    max_components = min(int(n_features), int(min_inner_train))
    if max_components < 1:
        raise ValueError("No feasible component count is available.")

    constrained: list[dict[str, list[Any]]] = []
    for candidate in param_grid:
        current = dict(candidate)
        if parameter not in current:
            constrained.append(current)
            continue

        values = [int(value) for value in current[parameter]]
        feasible = [
            value
            for value in values
            if 1 <= value <= max_components
        ]
        if feasible:
            current[parameter] = feasible
            constrained.append(current)

    if not constrained:
        raise ValueError(
            f"No {parameter} candidate is feasible with upper bound "
            f"{max_components}."
        )
    return constrained, max_components


def _nearest_reference_distances(
    distance_matrix: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Nearest-other distance for a reference set."""

    positions = np.asarray(positions, dtype=int)
    block = np.asarray(distance_matrix, dtype=float)[np.ix_(positions, positions)].copy()
    np.fill_diagonal(block, np.inf)
    return np.min(block, axis=1)


def _relative_support(value: float, reference: np.ndarray) -> float:
    """Return high support when a query is no farther than most references."""

    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if not np.isfinite(value) or len(reference) == 0:
        return float("nan")
    return float(np.mean(reference >= float(value)))


def support_descriptors(
    frame: pd.DataFrame,
    *,
    observed_positions: list[int] | tuple[int, ...] | np.ndarray,
    query_positions: list[int] | tuple[int, ...] | np.ndarray,
    geographic_distances: np.ndarray,
    agronomic_distances: np.ndarray,
    local_predictions: np.ndarray,
    graph_predictions: np.ndarray,
    state_column: str = "meta_estado",
    municipality_column: str = "meta_municipio",
) -> pd.DataFrame:
    """Compute leakage-safe support features relative to currently visible labels."""

    observed = np.asarray(observed_positions, dtype=int)
    query = np.asarray(query_positions, dtype=int)
    if len(observed) < 2:
        raise ValueError("Support descriptors require at least two observed parcels.")
    if len(query) != len(local_predictions) or len(query) != len(graph_predictions):
        raise ValueError("Prediction arrays do not align with query positions.")

    reference_geo = _nearest_reference_distances(geographic_distances, observed)
    reference_agro = _nearest_reference_distances(agronomic_distances, observed)

    states = frame[state_column].astype(str).to_numpy()
    municipalities = frame[municipality_column].astype(str).to_numpy()
    observed_states = states[observed]
    observed_municipalities = municipalities[observed]

    rows: list[dict[str, float]] = []
    for idx, position in enumerate(query):
        geo = float(np.min(np.asarray(geographic_distances)[position, observed]))
        agro = float(np.min(np.asarray(agronomic_distances)[position, observed]))
        same_state = float(np.mean(observed_states == states[position]))
        same_municipality = float(
            np.mean(
                (observed_states == states[position])
                & (observed_municipalities == municipalities[position])
            )
        )
        rows.append(
            {
                "log_nearest_geo_km": float(np.log1p(max(geo, 0.0))),
                "log_nearest_agro_distance": float(np.log1p(max(agro, 0.0))),
                "geo_support": _relative_support(geo, reference_geo),
                "agro_support": _relative_support(agro, reference_agro),
                "same_state_fraction": same_state,
                "same_municipality_fraction": same_municipality,
                "local_graph_abs_disagreement": float(
                    abs(float(local_predictions[idx]) - float(graph_predictions[idx]))
                ),
            }
        )
    return pd.DataFrame(rows)


class ConvexStackRegressor(BaseEstimator, RegressorMixin):
    """Squared-error convex stack with optional L2 weight shrinkage."""

    def __init__(self, l2: float = 0.0):
        self.l2 = float(l2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConvexStackRegressor:
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if matrix.ndim != 2 or len(matrix) != len(target):
            raise ValueError("Convex stack training shapes do not align.")
        n_experts = matrix.shape[1]
        if n_experts < 1:
            raise ValueError("Convex stack needs at least one expert.")

        def objective(weights: np.ndarray) -> float:
            residual = target - matrix @ weights
            return float(np.mean(residual**2) + self.l2 * np.sum(weights**2))

        result = minimize(
            objective,
            x0=np.full(n_experts, 1.0 / n_experts, dtype=float),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_experts,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            options={"maxiter": 2000, "ftol": 1.0e-12},
        )
        if not result.success:
            raise RuntimeError(f"Convex-stack optimization failed: {result.message}")
        weights = np.asarray(result.x, dtype=float)
        weights = np.clip(weights, 0.0, 1.0)
        weights /= np.sum(weights)
        self.coef_ = weights
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("ConvexStackRegressor is not fitted.")
        return np.asarray(x, dtype=float) @ np.asarray(self.coef_, dtype=float)


class SupportMixtureOfExperts(BaseEstimator, RegressorMixin):
    """Softmax-gated mixture whose weights depend on support descriptors."""

    def __init__(self, l2: float = 1.0, max_iter: int = 1500):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

    def _weight_matrix(self, support: np.ndarray) -> np.ndarray:
        scaled = self.support_scaler_.transform(np.asarray(support, dtype=float))
        design = np.column_stack([np.ones(len(scaled)), scaled])
        free_logits = design @ self.gate_coef_
        logits = np.column_stack([free_logits, np.zeros(len(design))])
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def fit(
        self,
        expert_predictions: np.ndarray,
        support: np.ndarray,
        y: np.ndarray,
    ) -> SupportMixtureOfExperts:
        predictions = np.asarray(expert_predictions, dtype=float)
        support_matrix = np.asarray(support, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if predictions.ndim != 2:
            raise ValueError("Expert predictions must be a 2D matrix.")
        if support_matrix.ndim != 2:
            raise ValueError("Support descriptors must be a 2D matrix.")
        if len(predictions) != len(target) or len(support_matrix) != len(target):
            raise ValueError("Mixture-of-experts training shapes do not align.")
        if predictions.shape[1] < 2:
            raise ValueError("Mixture-of-experts requires at least two experts.")

        self.support_scaler_ = StandardScaler().fit(support_matrix)
        scaled = self.support_scaler_.transform(support_matrix)
        design = np.column_stack([np.ones(len(scaled)), scaled])
        n_free_experts = predictions.shape[1] - 1
        shape = (design.shape[1], n_free_experts)

        def unpack(vector: np.ndarray) -> np.ndarray:
            return np.asarray(vector, dtype=float).reshape(shape)

        def objective(vector: np.ndarray) -> float:
            coef = unpack(vector)
            free_logits = design @ coef
            logits = np.column_stack([free_logits, np.zeros(len(design))])
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            predicted = np.sum(weights * predictions, axis=1)
            residual = target - predicted
            penalty = self.l2 * float(np.mean(coef**2))
            return float(np.mean(residual**2) + penalty)

        result = minimize(
            objective,
            x0=np.zeros(shape[0] * shape[1], dtype=float),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1.0e-12},
        )
        if not result.success:
            raise RuntimeError(f"Mixture-of-experts optimization failed: {result.message}")
        self.gate_coef_ = unpack(np.asarray(result.x, dtype=float))
        self.n_experts_in_ = predictions.shape[1]
        return self

    def predict(
        self,
        expert_predictions: np.ndarray,
        support: np.ndarray,
    ) -> np.ndarray:
        predictions = np.asarray(expert_predictions, dtype=float)
        if predictions.shape[1] != self.n_experts_in_:
            raise ValueError("Mixture-of-experts received a different expert count.")
        weights = self._weight_matrix(np.asarray(support, dtype=float))
        return np.sum(weights * predictions, axis=1)

    def expert_weights(self, support: np.ndarray) -> np.ndarray:
        """Return sample-specific convex expert weights."""

        return self._weight_matrix(np.asarray(support, dtype=float))


@dataclass
class FittedMetaCandidate:
    """Serializable meta-model plus selected hyperparameters."""

    name: str
    kind: str
    estimator: Any
    parameters: dict[str, Any]
    uses_support: bool = False


def candidate_prediction_frame(
    ids: list[str] | np.ndarray,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    family: str,
    split_id: str,
) -> pd.DataFrame:
    """Convert candidate prediction arrays into the canonical long format."""

    rows: list[pd.DataFrame] = []
    parcel_ids = np.asarray(ids, dtype=object)
    target = np.asarray(observed, dtype=float)
    for method, predicted in predictions.items():
        values = np.asarray(predicted, dtype=float).reshape(-1)
        if len(values) != len(parcel_ids):
            raise ValueError(f"Prediction length mismatch for {method}.")
        rows.append(
            pd.DataFrame(
                {
                    "family": str(family),
                    "split_id": str(split_id),
                    "method": str(method),
                    "ID_POLIGONO": parcel_ids,
                    "observed": target,
                    "predicted": values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
