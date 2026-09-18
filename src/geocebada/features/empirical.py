"""Target-free empirical feature discovery from Feature Table v1.

This module creates deterministic X-only features that describe temporal geometry,
relative condition, sensor agreement and distribution geometry. It never inspects
RENDIMIENTO_T_HA, so the resulting table can be materialized once for all 197 parcels.

Target-aware discovery lives in geocebada.features.discovery and must be fitted
inside training folds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMN = "ID_POLIGONO"


@dataclass(frozen=True)
class EmpiricalFeatureRecord:
    """Machine-readable provenance for one deterministic empirical feature."""

    column: str
    family: str
    mode: str
    inputs: tuple[str, ...]
    formula: str
    meaning: str
    rationale: str
    evidence: str = "empirical_x_only"
    references: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "column": self.column,
            "family": self.family,
            "mode": self.mode,
            "inputs": list(self.inputs),
            "formula": self.formula,
            "meaning": self.meaning,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "references": list(self.references),
            "caveats": list(self.caveats),
        }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing column(s): {missing}")


def _series_columns(prefix: str, months: Sequence[int]) -> list[str]:
    return [f"{prefix}{int(month):02d}" for month in months]


def _to_float_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )


def _nanmean_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.mean(row[finite]))
    return result


def _nanmin_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.min(row[finite]))
    return result


def _nanmax_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.max(row[finite]))
    return result


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    numerator_b, denominator_b = np.broadcast_arrays(numerator, denominator)
    finite = np.isfinite(numerator_b) & np.isfinite(denominator_b)
    safe = finite & (np.abs(denominator_b) > epsilon)
    result[safe] = numerator_b[safe] / denominator_b[safe]
    return result


def _add_feature(
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
    *,
    column: str,
    values: np.ndarray,
    family: str,
    mode: str,
    inputs: Sequence[str],
    formula: str,
    meaning: str,
    rationale: str,
    references: Sequence[str] = (),
    caveats: Sequence[str] = (),
) -> None:
    if column in data:
        raise ValueError(f"Duplicate empirical feature name: {column}")
    data[column] = np.asarray(values, dtype=float)
    records.append(
        EmpiricalFeatureRecord(
            column=column,
            family=family,
            mode=mode,
            inputs=tuple(inputs),
            formula=formula,
            meaning=meaning,
            rationale=rationale,
            references=tuple(references),
            caveats=tuple(caveats),
        )
    )


def _row_temporal_shape(months: Sequence[int], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return scale-aware and scale-free geometry descriptors per row."""

    x_all = np.asarray(months, dtype=float)
    names = (
        "total_variation",
        "roughness",
        "curvature_energy",
        "lag1_autocorr",
        "linear_trend_r2",
        "turning_fraction",
        "peak_sharpness",
        "center_of_mass_month",
        "shape_entropy",
    )
    result = {name: np.full(matrix.shape[0], np.nan) for name in names}

    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue

        x = x_all[finite]
        y = row[finite]
        dx = np.diff(x)
        dy = np.diff(y)
        slopes = dy / dx

        result["total_variation"][row_index] = float(np.sum(np.abs(dy)))
        if len(slopes) >= 2:
            slope_change = np.diff(slopes)
            result["roughness"][row_index] = float(np.mean(np.abs(slope_change)))
            result["curvature_energy"][row_index] = float(
                np.mean(np.square(slope_change))
            )
            signs = np.sign(slopes)
            nonzero = signs != 0
            signs = signs[nonzero]
            if len(signs) >= 2:
                turns = np.sum(signs[1:] != signs[:-1])
                result["turning_fraction"][row_index] = float(
                    turns / (len(signs) - 1)
                )
            else:
                result["turning_fraction"][row_index] = 0.0

        if len(y) >= 3:
            left = y[:-1]
            right = y[1:]
            if np.std(left) > 0 and np.std(right) > 0:
                result["lag1_autocorr"][row_index] = float(
                    np.corrcoef(left, right)[0, 1]
                )

        x_centered = x - x.mean()
        denominator = float(np.sum(np.square(x_centered)))
        if denominator > 0:
            slope = float(np.sum(x_centered * (y - y.mean())) / denominator)
            fitted = y.mean() + slope * x_centered
            ss_total = float(np.sum(np.square(y - y.mean())))
            if ss_total > 0:
                ss_resid = float(np.sum(np.square(y - fitted)))
                result["linear_trend_r2"][row_index] = 1.0 - ss_resid / ss_total

        sorted_y = np.sort(y)
        if len(sorted_y) >= 2:
            result["peak_sharpness"][row_index] = float(
                sorted_y[-1] - sorted_y[-2]
            )

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        span = y_max - y_min
        if span > 0:
            weights = (y - y_min) / span
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                result["center_of_mass_month"][row_index] = float(
                    np.sum(x * weights) / weight_sum
                )
                probabilities = weights / weight_sum
                positive = probabilities > 0
                entropy = -float(
                    np.sum(
                        probabilities[positive]
                        * np.log(probabilities[positive])
                    )
                )
                normalizer = np.log(len(probabilities))
                if normalizer > 0:
                    result["shape_entropy"][row_index] = entropy / normalizer

    return result


def _build_temporal_shape_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
) -> None:
    months = [int(value) for value in config["months"]]
    requested = list(config["temporal_shape"]["metrics"])

    formulas = {
        "total_variation": "sum_t |x_t - x_(t-1)|",
        "roughness": "mean_t |slope_t - slope_(t-1)|",
        "curvature_energy": "mean_t (slope_t - slope_(t-1))^2",
        "lag1_autocorr": "corr(x_t, x_(t+1))",
        "linear_trend_r2": "R2 from OLS x_t ~ month",
        "turning_fraction": "fraction of adjacent slopes whose signs change",
        "peak_sharpness": "largest monthly value - second-largest monthly value",
        "center_of_mass_month": (
            "sum(month * range_normalized_x) / sum(range_normalized_x)"
        ),
        "shape_entropy": "normalized Shannon entropy of range-normalized monthly weights",
    }
    meanings = {
        "total_variation": "Total month-to-month movement across the seasonal curve.",
        "roughness": "Average change in local monthly slope.",
        "curvature_energy": "Squared slope-change energy of the seasonal curve.",
        "lag1_autocorr": "Persistence of adjacent monthly values.",
        "linear_trend_r2": "Fraction of seasonal variation explained by a straight trend.",
        "turning_fraction": "How often the seasonal trajectory changes direction.",
        "peak_sharpness": "How isolated the seasonal maximum is from the next-highest month.",
        "center_of_mass_month": "Timing of the range-normalized seasonal signal mass.",
        "shape_entropy": "How diffuse versus concentrated the seasonal signal is over time.",
    }

    for spec in config["temporal_shape"]["series"]:
        name = str(spec["name"])
        columns = _series_columns(str(spec["prefix"]), months)
        _require_columns(frame, columns, label=f"Temporal-shape series {name}")
        matrix = _to_float_array(frame, columns)
        metrics = _row_temporal_shape(months, matrix)

        for metric in requested:
            _add_feature(
                data,
                records,
                column=f"emp_shape__{name}__{metric}",
                values=metrics[metric],
                family="temporal_shape",
                mode=str(spec["mode"]),
                inputs=columns,
                formula=formulas[metric],
                meaning=meanings[metric],
                rationale=(
                    "These descriptors expose geometry of the observed seasonal trajectory "
                    "without using yield or assuming a crop-specific response function."
                ),
                caveats=(
                    "Monthly summaries provide only seven April-October support points.",
                    "Descriptors are empirical summaries and are not phenological stages.",
                ),
            )


def _build_historical_range_condition_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
) -> None:
    months = [int(value) for value in config["months"]]

    for spec in config["historical_range_condition"]["series"]:
        name = str(spec["name"])
        current_columns = _series_columns(str(spec["current_prefix"]), months)
        historical_min = str(spec["historical_min"])
        historical_max = str(spec["historical_max"])
        _require_columns(
            frame,
            [*current_columns, historical_min, historical_max],
            label=f"Historical-range condition {name}",
        )

        current = _to_float_array(frame, current_columns)
        lower = pd.to_numeric(frame[historical_min], errors="coerce").to_numpy(
            dtype=float
        )[:, None]
        upper = pd.to_numeric(frame[historical_max], errors="coerce").to_numpy(
            dtype=float
        )[:, None]
        scale = float(spec.get("scale", 1.0))
        position = scale * _safe_divide(
            current - lower,
            upper - lower,
            epsilon=1.0e-12,
        )

        finite = np.isfinite(position)
        below = np.full(len(frame), np.nan)
        above = np.full(len(frame), np.nan)
        for row_index in range(len(frame)):
            mask = finite[row_index]
            if not mask.any():
                continue
            threshold_high = scale
            below[row_index] = float(np.mean(position[row_index, mask] < 0.0))
            above[row_index] = float(
                np.mean(position[row_index, mask] > threshold_high)
            )

        metrics = {
            "season_mean": _nanmean_rows(position),
            "season_min": _nanmin_rows(position),
            "season_max": _nanmax_rows(position),
            "season_range": _nanmax_rows(position) - _nanmin_rows(position),
            "below_history_fraction": below,
            "above_history_fraction": above,
        }
        kind = str(spec["kind"])
        prefix = "emp_vci_like_short" if kind == "vci_like_short" else "emp_histpos"
        references = (
            ("Bokusheva2016", "Serban2025")
            if kind == "vci_like_short"
            else ()
        )
        caveats = (
            "The historical reference is parcel-specific and covers only 2022-2024.",
            "This is not a long-climatology standard Vegetation Condition Index.",
            "Values are not clipped, so excursions outside the historical range are preserved.",
        )

        for metric, values in metrics.items():
            formula = (
                f"{scale:g} * (x_2025 - historical_min) / "
                "(historical_max - historical_min)"
            )
            _add_feature(
                data,
                records,
                column=f"{prefix}__{name}__{metric}",
                values=values,
                family="short_baseline_condition",
                mode=str(spec["mode"]),
                inputs=[*current_columns, historical_min, historical_max],
                formula=f"{metric} of monthly [{formula}]",
                meaning=(
                    "Position of 2025 observations relative to the parcel's short "
                    "historical observed range."
                ),
                rationale=(
                    "A normalized within-parcel historical-range position can reveal "
                    "unusual current-season conditions while remaining target-free."
                ),
                references=references,
                caveats=caveats,
            )


def _symmetric_change(
    current: np.ndarray,
    historical: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    numerator = 2.0 * (current - historical)
    denominator = np.abs(current) + np.abs(historical)
    return _safe_divide(numerator, denominator, epsilon)


def _row_positive_fraction(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.mean(row[finite] > 0))
    return result


def _row_sign_change_fraction(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        values = row[np.isfinite(row)]
        if len(values) < 2:
            continue
        signs = np.sign(values)
        signs = signs[signs != 0]
        if len(signs) < 2:
            result[row_index] = 0.0
            continue
        result[row_index] = float(np.mean(signs[1:] != signs[:-1]))
    return result


def _build_symmetric_change_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
) -> None:
    months = [int(value) for value in config["months"]]
    late_indices = [i for i, month in enumerate(months) if month >= 8]
    epsilon = float(config["symmetric_change"]["epsilon"])

    for spec in config["symmetric_change"]["pairs"]:
        name = str(spec["name"])
        historical_columns = _series_columns(
            str(spec["historical_prefix"]),
            months,
        )
        current_columns = _series_columns(str(spec["current_prefix"]), months)
        _require_columns(
            frame,
            [*historical_columns, *current_columns],
            label=f"Symmetric-change pair {name}",
        )
        historical = _to_float_array(frame, historical_columns)
        current = _to_float_array(frame, current_columns)
        change = _symmetric_change(current, historical, epsilon)

        metrics = {
            "season_mean": _nanmean_rows(change),
            "rms": np.sqrt(_nanmean_rows(np.square(change))),
            "max_abs": _nanmax_rows(np.abs(change)),
            "late_mean": _nanmean_rows(change[:, late_indices]),
            "positive_fraction": _row_positive_fraction(change),
            "sign_change_fraction": _row_sign_change_fraction(change),
        }

        for metric, values in metrics.items():
            _add_feature(
                data,
                records,
                column=f"emp_symchange__{name}__{metric}",
                values=values,
                family="symmetric_change",
                mode=str(spec["mode"]),
                inputs=[*historical_columns, *current_columns],
                formula=(
                    f"{metric} of 2*(x_2025-x_hist)/"
                    "(|x_2025|+|x_hist|+epsilon)"
                ),
                meaning=(
                    "Relative departure of 2025 from the historical monthly profile "
                    "without dividing directly by a signed baseline."
                ),
                rationale=(
                    "Symmetric relative change is numerically safer than a simple ratio "
                    "for indices whose historical values can cross zero."
                ),
                caveats=(
                    "The statistic is a mathematical contrast, not a published crop index.",
                    "Near-zero pairs are suppressed by the configured epsilon.",
                ),
            )


def _row_pair_correlations(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pearson = np.full(left.shape[0], np.nan)
    spearman = np.full(left.shape[0], np.nan)
    cosine = np.full(left.shape[0], np.nan)

    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        finite = np.isfinite(left_row) & np.isfinite(right_row)
        if finite.sum() < 3:
            continue
        a = left_row[finite]
        b = right_row[finite]

        if np.std(a) > 0 and np.std(b) > 0:
            pearson[row_index] = float(np.corrcoef(a, b)[0, 1])
            a_rank = pd.Series(a).rank(method="average").to_numpy(dtype=float)
            b_rank = pd.Series(b).rank(method="average").to_numpy(dtype=float)
            if np.std(a_rank) > 0 and np.std(b_rank) > 0:
                spearman[row_index] = float(np.corrcoef(a_rank, b_rank)[0, 1])

        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm > 0:
            cosine[row_index] = float(np.dot(a, b) / norm)

    return pearson, spearman, cosine


def _build_sensor_shape_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
) -> None:
    months = [int(value) for value in config["months"]]
    epsilon = float(config["sensor_shape"]["epsilon"])

    for spec in config["sensor_shape"]["pairs"]:
        name = str(spec["name"])
        left_columns = _series_columns(str(spec["left_prefix"]), months)
        right_columns = _series_columns(str(spec["right_prefix"]), months)
        _require_columns(
            frame,
            [*left_columns, *right_columns],
            label=f"Sensor-shape pair {name}",
        )
        left = _to_float_array(frame, left_columns)
        right = _to_float_array(frame, right_columns)
        pearson, spearman, cosine = _row_pair_correlations(left, right)

        difference = left - right
        rmse = np.sqrt(_nanmean_rows(np.square(difference)))
        combined_range = (
            np.maximum(_nanmax_rows(left), _nanmax_rows(right))
            - np.minimum(_nanmin_rows(left), _nanmin_rows(right))
        )
        normalized_rmse = _safe_divide(rmse, combined_range, epsilon)
        symmetric = _symmetric_change(left, right, epsilon)

        left_peak = np.full(len(frame), np.nan)
        right_peak = np.full(len(frame), np.nan)
        month_array = np.asarray(months, dtype=float)
        for row_index in range(len(frame)):
            left_finite = np.isfinite(left[row_index])
            right_finite = np.isfinite(right[row_index])
            if left_finite.any():
                left_values = left[row_index, left_finite]
                left_months = month_array[left_finite]
                left_peak[row_index] = left_months[int(np.argmax(left_values))]
            if right_finite.any():
                right_values = right[row_index, right_finite]
                right_months = month_array[right_finite]
                right_peak[row_index] = right_months[int(np.argmax(right_values))]

        metrics = {
            "pearson_r": pearson,
            "spearman_r": spearman,
            "cosine_similarity": cosine,
            "normalized_rmse": normalized_rmse,
            "mean_abs_symmetric_difference": _nanmean_rows(np.abs(symmetric)),
            "peak_month_abs_difference": np.abs(left_peak - right_peak),
        }

        for metric, values in metrics.items():
            _add_feature(
                data,
                records,
                column=f"emp_sensor_shape__s2_vs_planet__{name}__{metric}",
                values=values,
                family="sensor_shape_similarity",
                mode=str(spec["mode"]),
                inputs=[*left_columns, *right_columns],
                formula=f"{metric}(Sentinel-2 monthly series, Planet monthly series)",
                meaning=(
                    "Agreement or disagreement in the April-October temporal shape "
                    "reported by Sentinel-2 and Planet."
                ),
                rationale=(
                    "Cross-sensor shape agreement can expose stable crop dynamics, while "
                    "disagreement may encode sensor-specific or observation-condition effects."
                ),
                caveats=(
                    "The sensors are not assumed to be radiometrically interchangeable.",
                    "Only seven monthly support points are available.",
                ),
            )


def _build_aggregate_geometry_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[EmpiricalFeatureRecord],
) -> None:
    epsilon = float(config["aggregate_geometry"]["epsilon"])

    for spec in config["aggregate_geometry"]["series"]:
        name = str(spec["name"])
        prefix = str(spec["prefix"])
        columns = {
            stat: f"{prefix}{stat}"
            for stat in ("mean", "std", "min", "max")
        }
        _require_columns(
            frame,
            columns.values(),
            label=f"Aggregate-geometry series {name}",
        )
        arrays = {
            stat: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            for stat, column in columns.items()
        }
        span = arrays["max"] - arrays["min"]
        metrics = {
            "cv_absmean": _safe_divide(
                arrays["std"],
                np.abs(arrays["mean"]),
                epsilon,
            ),
            "std_over_range": _safe_divide(arrays["std"], span, epsilon),
            "mean_position_in_range": _safe_divide(
                arrays["mean"] - arrays["min"],
                span,
                epsilon,
            ),
            "range_asymmetry": _safe_divide(
                arrays["max"] + arrays["min"] - 2.0 * arrays["mean"],
                span,
                epsilon,
            ),
        }

        for metric, values in metrics.items():
            _add_feature(
                data,
                records,
                column=f"emp_distribution__{name}__{metric}",
                values=values,
                family="aggregate_geometry",
                mode=str(spec["mode"]),
                inputs=list(columns.values()),
                formula=f"{metric}(mean, std, min, max)",
                meaning=(
                    "Scale-normalized geometry of the source distribution summary "
                    "already present in Feature Table v1."
                ),
                rationale=(
                    "Relative dispersion and asymmetry can expose structure hidden when "
                    "mean, standard deviation and extrema enter a model independently."
                ),
                caveats=(
                    "The source summary can mix temporal and within-observation variation.",
                    "Near-zero means or ranges are suppressed by epsilon.",
                ),
            )


def _drop_invalid_features(
    table: pd.DataFrame,
    records: list[EmpiricalFeatureRecord],
    *,
    id_column: str,
) -> tuple[pd.DataFrame, list[EmpiricalFeatureRecord], dict[str, list[str]]]:
    dropped_all_null: list[str] = []
    dropped_constant: list[str] = []
    keep: list[str] = [id_column]

    record_by_column = {record.column: record for record in records}
    retained_records: list[EmpiricalFeatureRecord] = []

    for column in table.columns:
        if column == id_column:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.empty:
            dropped_all_null.append(column)
            continue
        if finite.nunique(dropna=True) <= 1:
            dropped_constant.append(column)
            continue
        keep.append(column)
        retained_records.append(record_by_column[column])

    return (
        table.loc[:, keep].copy(),
        retained_records,
        {
            "dropped_all_null": sorted(dropped_all_null),
            "dropped_constant": sorted(dropped_constant),
        },
    )


def build_empirical_feature_layer(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic target-free empirical features for all parcels."""

    id_column = str(config["identity"]["id_column"])
    if id_column not in frame.columns:
        raise KeyError(id_column)
    if frame[id_column].duplicated().any():
        raise ValueError("Feature Table v1 contains duplicate parcel IDs.")

    data: dict[str, np.ndarray] = {}
    records: list[EmpiricalFeatureRecord] = []

    _build_temporal_shape_features(frame, config, data, records)
    _build_historical_range_condition_features(frame, config, data, records)
    _build_symmetric_change_features(frame, config, data, records)
    _build_sensor_shape_features(frame, config, data, records)
    _build_aggregate_geometry_features(frame, config, data, records)

    table = pd.DataFrame({id_column: frame[id_column].astype(str), **data})
    table, records, dropped = _drop_invalid_features(
        table,
        records,
        id_column=id_column,
    )

    feature_columns = [column for column in table.columns if column != id_column]
    values = table[feature_columns].to_numpy(dtype=float)
    missing_fraction = table[feature_columns].isna().mean()

    diagnostics = {
        "rows": int(len(table)),
        "features": int(len(feature_columns)),
        "family_counts": (
            pd.Series([record.family for record in records])
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "mode_counts": (
            pd.Series([record.mode for record in records])
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "dropped_all_null": dropped["dropped_all_null"],
        "dropped_constant": dropped["dropped_constant"],
        "missing_fraction_mean": float(missing_fraction.mean()),
        "missing_fraction_max": float(missing_fraction.max()),
        "infinite_values": int(np.isinf(values).sum()),
    }
    return table, [record.as_dict() for record in records], diagnostics


def validate_empirical_feature_layer(
    table: pd.DataFrame,
    *,
    expected_ids: Sequence[str] | pd.Series,
    id_column: str = ID_COLUMN,
) -> None:
    """Validate one-to-one coverage and numeric empirical predictors."""

    if id_column not in table.columns:
        raise KeyError(id_column)
    if table[id_column].duplicated().any():
        raise ValueError("Empirical feature table contains duplicate parcel IDs.")

    observed = set(table[id_column].astype(str))
    expected = set(pd.Series(expected_ids).astype(str))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Empirical parcel coverage mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    feature_columns = [column for column in table.columns if column != id_column]
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(table[column])
    ]
    if non_numeric:
        raise TypeError(f"Empirical features must be numeric: {non_numeric[:10]}")

    values = table[feature_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Empirical feature table contains infinite values.")


def join_empirical_features(
    base: pd.DataFrame,
    empirical: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Join the empirical layer after exact parcel-coverage validation."""

    if id_column not in base.columns or id_column not in empirical.columns:
        raise KeyError(id_column)
    if base[id_column].duplicated().any() or empirical[id_column].duplicated().any():
        raise ValueError("Both inputs must be one row per parcel before joining.")

    overlapping = sorted((set(base.columns) & set(empirical.columns)) - {id_column})
    if overlapping:
        raise ValueError(
            "Empirical columns overlap existing base columns: "
            f"{overlapping[:10]}"
        )

    validate_empirical_feature_layer(
        empirical,
        expected_ids=base[id_column],
        id_column=id_column,
    )
    joined = base.merge(empirical, on=id_column, how="left", validate="one_to_one")
    if len(joined) != len(base):
        raise ValueError("Empirical join changed the number of parcel rows.")
    return joined
