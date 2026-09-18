"""Fold-local target-aware expression discovery.

The deterministic empirical layer is X-only and can be materialized globally. This module
contains the target-aware counterpart: candidate expressions are selected only during fit,
so callers must place the transformer inside each training fold.

This is intentionally a small, auditable expression miner rather than an unrestricted genetic
programming system. Its purpose is to test whether simple recurring formulas emerge before
introducing a heavier symbolic-regression dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class ExpressionSpec:
    """One selected two-variable expression."""

    left: str
    right: str
    operation: str
    score: float

    @property
    def formula(self) -> str:
        """Return a compact human-readable formula."""

        formulas = {
            "product": f"({self.left}) * ({self.right})",
            "sum": f"({self.left}) + ({self.right})",
            "difference": f"({self.left}) - ({self.right})",
            "reverse_difference": f"({self.right}) - ({self.left})",
            "safe_ratio": f"({self.left}) / (abs({self.right}) + eps)",
            "reverse_safe_ratio": f"({self.right}) / (abs({self.left}) + eps)",
            "symmetric_change": (
                f"2*(({self.left})-({self.right})) / "
                f"(abs({self.left})+abs({self.right})+eps)"
            ),
        }
        return formulas[self.operation]


def _as_numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Discovery primitive column(s) missing: {missing}")
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def _absolute_spearman(values: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(target)
    if finite.sum() < 3:
        return float("nan")
    x = values[finite]
    y = target[finite]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    statistic = spearmanr(x, y).statistic
    return float(abs(statistic)) if np.isfinite(statistic) else float("nan")


def _evaluate_expression(
    left: np.ndarray,
    right: np.ndarray,
    operation: str,
    epsilon: float,
) -> np.ndarray:
    if operation == "product":
        return left * right
    if operation == "sum":
        return left + right
    if operation == "difference":
        return left - right
    if operation == "reverse_difference":
        return right - left
    if operation == "safe_ratio":
        return left / (np.abs(right) + epsilon)
    if operation == "reverse_safe_ratio":
        return right / (np.abs(left) + epsilon)
    if operation == "symmetric_change":
        return 2.0 * (left - right) / (np.abs(left) + np.abs(right) + epsilon)
    raise ValueError(f"Unknown expression operation: {operation}")


def _expanded_operations(operations: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for operation in operations:
        if operation == "difference":
            expanded.extend(("difference", "reverse_difference"))
        elif operation == "safe_ratio":
            expanded.extend(("safe_ratio", "reverse_safe_ratio"))
        else:
            expanded.append(operation)
    return tuple(dict.fromkeys(expanded))


def _feature_name(rank: int, spec: ExpressionSpec) -> str:
    def slug(value: str) -> str:
        return (
            value.replace("__", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

    left = slug(spec.left)
    right = slug(spec.right)
    return f"disc_expr_{rank:02d}__{spec.operation}__{left}__{right}"


class FoldLocalExpressionMiner(BaseEstimator, TransformerMixin):
    """Select simple target-associated expressions using training data only.

    Parameters are deliberately small because GeoCebada has only 138 labeled parcels.
    The transformer first ranks primitive variables by absolute Spearman association with
    y, then evaluates simple pairwise expressions among the retained primitives.

    Selection occurs only in fit. Transform applies the fixed formulas to new rows without
    reading their target.
    """

    def __init__(
        self,
        primitive_columns: tuple[str, ...],
        *,
        top_primitives: int = 12,
        top_expressions: int = 20,
        operations: tuple[str, ...] = (
            "product",
            "difference",
            "sum",
            "safe_ratio",
            "symmetric_change",
        ),
        epsilon: float = 1.0e-6,
    ) -> None:
        self.primitive_columns = primitive_columns
        self.top_primitives = top_primitives
        self.top_expressions = top_expressions
        self.operations = operations
        self.epsilon = epsilon

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "FoldLocalExpressionMiner":
        """Learn primitive ranking and expression selection from one training fold."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldLocalExpressionMiner requires a pandas DataFrame.")
        target = np.asarray(y, dtype=float)
        if len(target) != len(X):
            raise ValueError("X and y must contain the same number of rows.")
        if not np.isfinite(target).all():
            raise ValueError("Training target contains missing or infinite values.")

        columns = list(self.primitive_columns)
        numeric = _as_numeric_frame(X, columns)
        medians = numeric.median(axis=0, skipna=True)
        usable = [
            column
            for column in columns
            if np.isfinite(float(medians[column]))
        ]
        if len(usable) < 2:
            raise ValueError("Fewer than two usable discovery primitives.")

        filled = numeric.loc[:, usable].fillna(medians.loc[usable])
        primitive_scores = {
            column: _absolute_spearman(
                filled[column].to_numpy(dtype=float),
                target,
            )
            for column in usable
        }
        ranked = sorted(
            (
                (column, score)
                for column, score in primitive_scores.items()
                if np.isfinite(score)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected_primitives = [
            column for column, _ in ranked[: int(self.top_primitives)]
        ]
        if len(selected_primitives) < 2:
            raise ValueError("Primitive ranking retained fewer than two variables.")

        candidates: list[ExpressionSpec] = []
        operations = _expanded_operations(tuple(self.operations))
        for left_name, right_name in combinations(selected_primitives, 2):
            left = filled[left_name].to_numpy(dtype=float)
            right = filled[right_name].to_numpy(dtype=float)
            for operation in operations:
                values = _evaluate_expression(
                    left,
                    right,
                    operation,
                    float(self.epsilon),
                )
                score = _absolute_spearman(values, target)
                if not np.isfinite(score):
                    continue
                candidates.append(
                    ExpressionSpec(
                        left=left_name,
                        right=right_name,
                        operation=operation,
                        score=score,
                    )
                )

        candidates.sort(
            key=lambda spec: (
                -spec.score,
                spec.operation,
                spec.left,
                spec.right,
            )
        )
        selected = candidates[: int(self.top_expressions)]
        if not selected:
            raise ValueError("No finite discovery expressions were produced.")

        self.medians_ = medians.loc[usable].astype(float).to_dict()
        self.primitive_scores_ = primitive_scores
        self.selected_primitives_ = tuple(selected_primitives)
        self.expressions_ = tuple(selected)
        self.feature_names_out_ = tuple(
            _feature_name(rank, spec)
            for rank, spec in enumerate(selected, start=1)
        )
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply expressions learned in the corresponding training fold."""

        check_is_fitted(
            self,
            attributes=[
                "medians_",
                "selected_primitives_",
                "expressions_",
                "feature_names_out_",
            ],
        )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldLocalExpressionMiner requires a pandas DataFrame.")

        needed = sorted(
            {
                column
                for spec in self.expressions_
                for column in (spec.left, spec.right)
            }
        )
        numeric = _as_numeric_frame(X, needed)
        for column in needed:
            numeric[column] = numeric[column].fillna(float(self.medians_[column]))

        output: dict[str, np.ndarray] = {}
        for name, spec in zip(
            self.feature_names_out_,
            self.expressions_,
            strict=True,
        ):
            output[name] = _evaluate_expression(
                numeric[spec.left].to_numpy(dtype=float),
                numeric[spec.right].to_numpy(dtype=float),
                spec.operation,
                float(self.epsilon),
            )
        return pd.DataFrame(output, index=X.index)

    def get_feature_names_out(
        self,
        input_features: Any | None = None,
    ) -> np.ndarray:
        """Return selected expression names after fitting."""

        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return np.asarray(self.feature_names_out_, dtype=object)

    def discovery_report(self) -> pd.DataFrame:
        """Return selected formulas and training-fold association scores."""

        check_is_fitted(
            self,
            attributes=["expressions_", "feature_names_out_"],
        )
        rows = []
        for rank, (name, spec) in enumerate(
            zip(self.feature_names_out_, self.expressions_, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "rank": rank,
                    "feature": name,
                    "left": spec.left,
                    "right": spec.right,
                    "operation": spec.operation,
                    "formula": spec.formula,
                    "abs_spearman_train": spec.score,
                }
            )
        return pd.DataFrame(rows)
