"""Agronomic and nonlinear feature layer built from Feature Table v1.

This module intentionally derives only X-based features. It never inspects
RENDIMIENTO_T_HA and therefore can be built once for all 197 parcels without
using hidden targets or target-informed feature selection.

The layer is additive: Feature Table v1 remains unchanged and callers join this
table by ID_POLIGONO when they want the agronomic representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMN = "ID_POLIGONO"


@dataclass(frozen=True)
class AgronomicFeatureRecord:
    """Machine-readable provenance for one derived agronomic feature."""

    column: str
    family: str
    mode: str
    inputs: tuple[str, ...]
    formula: str
    meaning: str
    rationale: str
    evidence: str
    references: tuple[str, ...]
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


PHENOLOGY_REFERENCES = (
    "Chanev2025",
    "Sharifi2020",
    "Mirosavljevic2018",
)
THERMAL_REFERENCES = (
    "Hajkova2019",
    "McMaster2005",
    "McMaster1997",
)
WATER_REFERENCES = (
    "Bello2022",
    "Mogensen1980",
    "Zhang2015",
    "Dhonthi2024",
)
SOIL_REFERENCES = ("Holland2021",)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing column(s): {missing}")


def _to_float_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _nanmean_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.mean(row[finite]))
    return result


def _nanstd_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if finite.any():
            result[row_index] = float(np.std(row[finite], ddof=0))
    return result


def _row_slope(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row) & np.isfinite(x)
        if finite.sum() < 2:
            continue
        xx = x[finite]
        yy = row[finite]
        x_centered = xx - xx.mean()
        denominator = float(np.sum(x_centered**2))
        if denominator <= 0:
            continue
        result[row_index] = float(np.sum(x_centered * (yy - yy.mean())) / denominator)
    return result


def _row_auc(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row) & np.isfinite(x)
        if finite.sum() < 2:
            continue
        result[row_index] = float(np.trapezoid(row[finite], x=x[finite]))
    return result


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float) -> np.ndarray:
    result = np.full(len(numerator), np.nan)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    safe = finite & (np.abs(denominator) > epsilon)
    result[safe] = numerator[safe] / denominator[safe]
    return result


def _safe_sum_complete(matrix: np.ndarray) -> np.ndarray:
    result = np.full(matrix.shape[0], np.nan)
    complete = np.isfinite(matrix).all(axis=1)
    if complete.any():
        result[complete] = np.sum(matrix[complete], axis=1)
    return result


def _series_columns(prefix: str, months: Sequence[int]) -> list[str]:
    return [f"{prefix}{int(month):02d}" for month in months]


def _add_feature(
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
    *,
    column: str,
    values: np.ndarray,
    family: str,
    mode: str,
    inputs: Sequence[str],
    formula: str,
    meaning: str,
    rationale: str,
    evidence: str,
    references: Sequence[str],
    caveats: Sequence[str] = (),
) -> None:
    if column in data:
        raise ValueError(f"Duplicate agronomic feature name: {column}")
    data[column] = np.asarray(values, dtype=float)
    records.append(
        AgronomicFeatureRecord(
            column=column,
            family=family,
            mode=mode,
            inputs=tuple(inputs),
            formula=formula,
            meaning=meaning,
            rationale=rationale,
            evidence=evidence,
            references=tuple(references),
            caveats=tuple(caveats),
        )
    )


def _phenology_metrics(months: Sequence[int], matrix: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(months, dtype=float)
    early_idx = [index for index, month in enumerate(months) if month <= 6]
    late_idx = [index for index, month in enumerate(months) if month >= 8]
    greenup_idx = [index for index, month in enumerate(months) if month <= 7]
    senescence_idx = [index for index, month in enumerate(months) if month >= 8]

    amplitude = np.full(matrix.shape[0], np.nan)
    peak_month = np.full(matrix.shape[0], np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        finite_values = row[finite]
        finite_months = x[finite]
        amplitude[row_index] = float(np.max(finite_values) - np.min(finite_values))
        peak_month[row_index] = float(finite_months[int(np.argmax(finite_values))])

    month_index = {int(month): index for index, month in enumerate(months)}
    jun_aug = np.full(matrix.shape[0], np.nan)
    if 6 in month_index and 8 in month_index:
        june = matrix[:, month_index[6]]
        august = matrix[:, month_index[8]]
        finite = np.isfinite(june) & np.isfinite(august)
        jun_aug[finite] = august[finite] - june[finite]

    return {
        "auc": _row_auc(x, matrix),
        "season_mean": _nanmean_rows(matrix),
        "amplitude": amplitude,
        "peak_month": peak_month,
        "early_mean": _nanmean_rows(matrix[:, early_idx]),
        "late_mean": _nanmean_rows(matrix[:, late_idx]),
        "jun_to_aug_delta": jun_aug,
        "greenup_slope": _row_slope(x[greenup_idx], matrix[:, greenup_idx]),
        "senescence_slope": _row_slope(x[senescence_idx], matrix[:, senescence_idx]),
    }


def _build_phenology_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    months = [int(value) for value in config["months"]]
    requested_metrics = list(config["phenology"]["metrics"])
    series_results: dict[str, dict[str, np.ndarray]] = {}
    series_inputs: dict[str, list[str]] = {}

    formulas = {
        "auc": "trapezoid(month, index_value) over available Apr-Oct months",
        "season_mean": "mean(index_value Apr-Oct)",
        "amplitude": "max(index_value) - min(index_value)",
        "peak_month": "month at max(index_value)",
        "early_mean": "mean(index_value Apr-Jun)",
        "late_mean": "mean(index_value Aug-Oct)",
        "jun_to_aug_delta": "index_Aug - index_Jun",
        "greenup_slope": "OLS slope(index_value ~ month), Apr-Jul",
        "senescence_slope": "OLS slope(index_value ~ month), Aug-Oct",
    }
    meanings = {
        "auc": "Integrated seasonal vegetation-index signal.",
        "season_mean": "Average seasonal state of the selected vegetation index.",
        "amplitude": "Seasonal dynamic range of the vegetation index.",
        "peak_month": "Timing of the highest observed monthly vegetation-index value.",
        "early_mean": "Early-season vegetation condition.",
        "late_mean": "Late-season vegetation condition.",
        "jun_to_aug_delta": "Mid-season change between June and August.",
        "greenup_slope": "Approximate early-season rate of canopy development.",
        "senescence_slope": "Approximate late-season rate of canopy decline or continued growth.",
    }

    for spec in config["phenology"]["series"]:
        series_name = str(spec["name"])
        columns = _series_columns(str(spec["prefix"]), months)
        _require_columns(frame, columns, label=f"Phenology series {series_name}")
        matrix = _to_float_array(frame, columns)
        metrics = _phenology_metrics(months, matrix)
        series_results[series_name] = metrics
        series_inputs[series_name] = columns

        for metric_name in requested_metrics:
            _add_feature(
                data,
                records,
                column=f"agro_pheno__{series_name}__{metric_name}",
                values=metrics[metric_name],
                family="phenology",
                mode=str(spec["mode"]),
                inputs=columns,
                formula=formulas[metric_name],
                meaning=meanings[metric_name],
                rationale=(
                    "Barley yield relationships with remotely sensed vegetation indices depend on "
                    "crop development stage; temporal shape can retain information lost by a single "
                    "seasonal average."
                ),
                evidence="literature_backed",
                references=PHENOLOGY_REFERENCES,
                caveats=(
                    "Monthly summaries are a coarse representation of phenology.",
                    "AUC bridges missing months linearly across remaining observed month positions.",
                ),
            )

    return series_results, series_inputs


def _build_anomaly_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    series_results: Mapping[str, Mapping[str, np.ndarray]],
    series_inputs: Mapping[str, Sequence[str]],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
    *,
    epsilon: float,
) -> None:
    months = [int(value) for value in config["months"]]
    month_index = {month: index for index, month in enumerate(months)}

    for historical_name, target_name in config["phenology"]["anomaly_pairs"]:
        historical_name = str(historical_name)
        target_name = str(target_name)
        historical_columns = list(series_inputs[historical_name])
        target_columns = list(series_inputs[target_name])
        hist = _to_float_array(frame, historical_columns)
        target = _to_float_array(frame, target_columns)
        both = np.isfinite(hist) & np.isfinite(target)
        monthly_delta = np.where(both, target - hist, np.nan)
        season_delta = _nanmean_rows(monthly_delta)
        season_ratio = _safe_divide(
            series_results[target_name]["season_mean"],
            series_results[historical_name]["season_mean"],
            epsilon,
        )
        auc_delta = series_results[target_name]["auc"] - series_results[historical_name]["auc"]
        august_delta = np.full(len(frame), np.nan)
        if 8 in month_index:
            index = month_index[8]
            finite = both[:, index]
            august_delta[finite] = target[finite, index] - hist[finite, index]
        late_delta = _nanmean_rows(monthly_delta[:, [4, 5, 6]])
        peak_shift = (
            series_results[target_name]["peak_month"]
            - series_results[historical_name]["peak_month"]
        )

        prefix = historical_name.replace("hist_", "").replace("historical_", "")
        inputs = [*historical_columns, *target_columns]
        payloads = {
            "season_mean_delta": (
                season_delta,
                "mean(2025_month - historical_month)",
                "Average 2025 departure from the historical monthly seasonal profile.",
            ),
            "season_mean_ratio": (
                season_ratio,
                "mean_2025 / mean_historical",
                "Relative 2025 seasonal vegetation condition versus historical baseline.",
            ),
            "auc_delta": (
                auc_delta,
                "AUC_2025 - AUC_historical",
                "Change in integrated seasonal vegetation signal versus historical baseline.",
            ),
            "august_delta": (
                august_delta,
                "value_2025_Aug - value_historical_Aug",
                "August departure from the historical profile.",
            ),
            "late_season_delta": (
                late_delta,
                "mean(2025 - historical, Aug-Oct)",
                "Late-season departure from historical vegetation condition.",
            ),
            "peak_month_shift": (
                peak_shift,
                "peak_month_2025 - peak_month_historical",
                "Shift in timing of the seasonal vegetation-index maximum.",
            ),
        }

        for name, (values, formula, meaning) in payloads.items():
            _add_feature(
                data,
                records,
                column=f"agro_anomaly__{prefix}__{name}",
                values=values,
                family="phenology_anomaly",
                mode="competition",
                inputs=inputs,
                formula=formula,
                meaning=meaning,
                rationale=(
                    "Within-parcel departure from a historical remote-sensing baseline can separate "
                    "persistent site productivity from conditions specific to the 2025 crop cycle."
                ),
                evidence="mechanistic_proxy",
                references=PHENOLOGY_REFERENCES,
                caveats=(
                    "Historical monthly values are climatological summaries, not a crop-growth model.",
                    "Ratio features are undefined when the historical denominator is near zero.",
                ),
            )


def _build_sensor_agreement_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    series_results: Mapping[str, Mapping[str, np.ndarray]],
    series_inputs: Mapping[str, Sequence[str]],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
    *,
    epsilon: float,
) -> None:
    for left_name, right_name in config["phenology"]["sensor_agreement"]:
        left_name = str(left_name)
        right_name = str(right_name)
        left_columns = list(series_inputs[left_name])
        right_columns = list(series_inputs[right_name])
        left = _to_float_array(frame, left_columns)
        right = _to_float_array(frame, right_columns)
        overlap = np.isfinite(left) & np.isfinite(right)

        rmse = np.full(len(frame), np.nan)
        for row_index in range(len(frame)):
            finite = overlap[row_index]
            if finite.any():
                diff = left[row_index, finite] - right[row_index, finite]
                rmse[row_index] = float(np.sqrt(np.mean(diff**2)))

        difference = (
            series_results[left_name]["season_mean"]
            - series_results[right_name]["season_mean"]
        )
        ratio = _safe_divide(
            series_results[left_name]["season_mean"],
            series_results[right_name]["season_mean"],
            epsilon,
        )
        index_name = left_name.rsplit("_", 1)[-1]
        inputs = [*left_columns, *right_columns]

        for suffix, values, formula, meaning in [
            (
                "season_mean_difference",
                difference,
                "Sentinel2 season mean - Planet season mean",
                "Cross-sensor difference in seasonal mean.",
            ),
            (
                "season_mean_ratio",
                ratio,
                "Sentinel2 season mean / Planet season mean",
                "Cross-sensor ratio in seasonal mean.",
            ),
            (
                "monthly_rmse",
                rmse,
                "sqrt(mean((Sentinel2_month - Planet_month)^2))",
                "Magnitude of month-by-month disagreement between sensors.",
            ),
        ]:
            _add_feature(
                data,
                records,
                column=f"agro_sensor__s2_planet_{index_name}__{suffix}",
                values=values,
                family="sensor_agreement",
                mode="competition",
                inputs=inputs,
                formula=formula,
                meaning=meaning,
                rationale=(
                    "Agreement or disagreement between independent sensor streams can encode "
                    "robustness of crop signal and sensor-specific departures."
                ),
                evidence="experimental",
                references=PHENOLOGY_REFERENCES,
                caveats=(
                    "Sentinel-2 and Planet products are not assumed radiometrically equivalent.",
                    "These features describe agreement rather than calibrating one sensor to another.",
                ),
            )


def _climate_columns(period: str, variable: str, months: Sequence[int]) -> list[str]:
    if period == "historical":
        return [
            f"clim_official_hist__{variable}__m{month:02d}_mean" for month in months
        ]
    if period == "2025":
        return [f"clim_official_2025__{variable}__m{month:02d}" for month in months]
    raise ValueError(f"Unknown climate period: {period}")


def _build_thermal_and_precip_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
) -> dict[str, dict[str, np.ndarray]]:
    months = [int(value) for value in config["months"]]
    day_counts = np.asarray([30, 31, 30, 31, 31, 30, 31], dtype=float)
    if months != [4, 5, 6, 7, 8, 9, 10]:
        raise ValueError("Current thermal day counts require months April through October.")

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for period, mode in [("historical", "clean"), ("2025", "competition")]:
        tmax_columns = _climate_columns(period, "tmax", months)
        tmin_columns = _climate_columns(period, "tmin", months)
        prec_columns = _climate_columns(period, "prec", months)
        _require_columns(
            frame,
            [*tmax_columns, *tmin_columns, *prec_columns],
            label=f"{period} climate",
        )
        tmax = _to_float_array(frame, tmax_columns)
        tmin = _to_float_array(frame, tmin_columns)
        precip = _to_float_array(frame, prec_columns)
        tmean = (tmax + tmin) / 2.0
        dtr = tmax - tmin

        outputs[period] = {
            "tmean_season": _nanmean_rows(tmean),
            "dtr_season": _nanmean_rows(dtr),
            "precip_season": _safe_sum_complete(precip),
        }

        for index, month in enumerate(months):
            for kind, values, formula, meaning in [
                (
                    "tmean",
                    tmean[:, index],
                    "(Tmax + Tmin) / 2",
                    "Monthly mean-temperature proxy.",
                ),
                (
                    "dtr",
                    dtr[:, index],
                    "Tmax - Tmin",
                    "Monthly diurnal temperature range.",
                ),
            ]:
                _add_feature(
                    data,
                    records,
                    column=f"agro_thermal__{period}__{kind}_m{month:02d}",
                    values=values,
                    family="thermal",
                    mode=mode,
                    inputs=[tmax_columns[index], tmin_columns[index]],
                    formula=formula,
                    meaning=meaning,
                    rationale=(
                        "Temperature controls cereal development rate and can affect crop growth "
                        "differently across the season."
                    ),
                    evidence="literature_backed",
                    references=THERMAL_REFERENCES,
                    caveats=("Monthly Tmin/Tmax summaries cannot reproduce daily thermal extremes.",),
                )

        for name, values, formula, meaning in [
            (
                "season_tmean",
                outputs[period]["tmean_season"],
                "mean(monthly (Tmax + Tmin)/2, Apr-Oct)",
                "Average thermal environment over the target season.",
            ),
            (
                "season_dtr",
                outputs[period]["dtr_season"],
                "mean(monthly Tmax-Tmin, Apr-Oct)",
                "Average diurnal temperature range over the target season.",
            ),
        ]:
            _add_feature(
                data,
                records,
                column=f"agro_thermal__{period}__{name}",
                values=values,
                family="thermal",
                mode=mode,
                inputs=[*tmax_columns, *tmin_columns],
                formula=formula,
                meaning=meaning,
                rationale="Summarizes the seasonal thermal regime relevant to cereal development.",
                evidence="literature_backed",
                references=THERMAL_REFERENCES,
                caveats=("Monthly climate values are coarse temporal summaries.",),
            )

        complete = np.isfinite(tmean).all(axis=1)
        for base in [float(value) for value in config["thermal"]["base_temperatures_c"]]:
            values = np.full(len(frame), np.nan)
            if complete.any():
                values[complete] = np.sum(
                    np.maximum(tmean[complete] - base, 0.0) * day_counts,
                    axis=1,
                )
            label = str(base).replace(".", "p")
            _add_feature(
                data,
                records,
                column=f"agro_thermal__{period}__gdd_proxy_base{label}c",
                values=values,
                family="thermal",
                mode=mode,
                inputs=[*tmax_columns, *tmin_columns],
                formula=f"sum(days_m * max(Tmean_m - {base}, 0)), Apr-Oct",
                meaning="Monthly-climate approximation to accumulated thermal time.",
                rationale=(
                    "Growing-degree concepts represent cereal phenological development; multiple "
                    "fixed bases are retained rather than selecting one from the yield target."
                ),
                evidence="literature_backed",
                references=THERMAL_REFERENCES,
                caveats=(
                    "This is a monthly proxy, not daily GDD.",
                    "The base temperature is a declared scenario, not fitted to the 138 yields.",
                ),
            )

        threshold = float(config["thermal"]["heat_threshold_c"])
        heat = np.full(len(frame), np.nan)
        if complete.any():
            heat[complete] = np.sum(
                np.maximum(tmean[complete] - threshold, 0.0) * day_counts,
                axis=1,
            )
        outputs[period]["heat_excess"] = heat
        _add_feature(
            data,
            records,
            column=f"agro_thermal__{period}__heat_excess_{threshold:g}c_proxy",
            values=heat,
            family="thermal",
            mode=mode,
            inputs=[*tmax_columns, *tmin_columns],
            formula=f"sum(days_m * max(Tmean_m - {threshold}, 0)), Apr-Oct",
            meaning="Coarse cumulative heat-excess proxy.",
            rationale=(
                "High-temperature exposure can affect barley grain filling and yield; this term "
                "allows nonlinear temperature response without choosing the threshold from target data."
            ),
            evidence="mechanistic_proxy",
            references=THERMAL_REFERENCES,
            caveats=(
                "Monthly means understate short heat-wave extremes.",
                "The threshold is a scenario parameter, not a fitted critical temperature.",
            ),
        )

        early = precip[:, 0:3]
        mid = precip[:, 2:5]
        late = precip[:, 4:7]
        precip_mean = _nanmean_rows(precip)
        precip_std = _nanstd_rows(precip)
        precip_cv = _safe_divide(precip_std, np.abs(precip_mean), 1e-9)
        season_precip = outputs[period]["precip_season"]
        late_sum = _safe_sum_complete(late)
        late_fraction = _safe_divide(late_sum, season_precip, 1e-9)

        precip_features = [
            ("early_sum", _safe_sum_complete(early), "sum(precip Apr-Jun)", "Early-season rainfall."),
            ("mid_sum", _safe_sum_complete(mid), "sum(precip Jun-Aug)", "Mid-season rainfall."),
            ("late_sum", late_sum, "sum(precip Aug-Oct)", "Late-season rainfall."),
            (
                "season_cv",
                precip_cv,
                "std(monthly precip) / abs(mean(monthly precip))",
                "Within-season rainfall concentration or variability proxy.",
            ),
            (
                "late_fraction",
                late_fraction,
                "late_precip / season_precip",
                "Fraction of seasonal precipitation occurring Aug-Oct.",
            ),
        ]
        for suffix, values, formula, meaning in precip_features:
            _add_feature(
                data,
                records,
                column=f"agro_water__{period}__precip_{suffix}",
                values=values,
                family="water_timing",
                mode=mode,
                inputs=prec_columns,
                formula=formula,
                meaning=meaning,
                rationale=(
                    "Barley drought sensitivity changes by growth stage, so seasonal rainfall timing "
                    "can be more informative than total precipitation alone."
                ),
                evidence="literature_backed",
                references=WATER_REFERENCES,
                caveats=("Monthly precipitation cannot identify short within-month drought events.",),
            )

    return outputs


def _build_water_productivity_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    climate_outputs: Mapping[str, Mapping[str, np.ndarray]],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
) -> None:
    months = [int(value) for value in config["months"]]
    epsilon = float(config["water"]["epsilon"])
    precip_columns = _climate_columns("2025", "prec", months)
    precip = _to_float_array(frame, precip_columns)
    aeti_columns = [f"wapor_2025__aeti__m{month:02d}_total" for month in months]
    npp_columns = [f"wapor_2025__npp__m{month:02d}_total" for month in months]
    ndvi_columns = _series_columns(
        "sat_basic_2025_monthly__sentinel_2__ndvi_promedio__m", months
    )
    lai_columns = _series_columns(
        "sat_basic_2025_monthly__sentinel_2__lai_promedio__m", months
    )
    _require_columns(
        frame,
        [*aeti_columns, *npp_columns, *ndvi_columns, *lai_columns],
        label="WaPOR/crop water interactions",
    )

    aeti = _to_float_array(frame, aeti_columns)
    npp = _to_float_array(frame, npp_columns)
    ndvi = _to_float_array(frame, ndvi_columns)
    lai = _to_float_array(frame, lai_columns)

    for index, month in enumerate(months):
        payloads = [
            (
                "npp_per_aeti",
                _safe_divide(npp[:, index], aeti[:, index], epsilon),
                [npp_columns[index], aeti_columns[index]],
                "NPP / (AETI + eps)",
                "Remote-sensing water-productivity proxy.",
                "mechanistic_proxy",
            ),
            (
                "aeti_per_precip",
                _safe_divide(aeti[:, index], precip[:, index], epsilon),
                [aeti_columns[index], precip_columns[index]],
                "AETI / (precip + eps)",
                "Actual evapotranspiration relative to monthly precipitation.",
                "mechanistic_proxy",
            ),
            (
                "precip_minus_aeti",
                precip[:, index] - aeti[:, index],
                [precip_columns[index], aeti_columns[index]],
                "precip - AETI",
                "Simple monthly atmospheric water-balance proxy.",
                "mechanistic_proxy",
            ),
            (
                "npp_x_ndvi",
                npp[:, index] * ndvi[:, index],
                [npp_columns[index], ndvi_columns[index]],
                "NPP * NDVI",
                "Interaction between productivity and canopy greenness.",
                "experimental",
            ),
            (
                "lai_x_aeti",
                lai[:, index] * aeti[:, index],
                [lai_columns[index], aeti_columns[index]],
                "LAI * AETI",
                "Interaction between canopy amount and actual water consumption.",
                "experimental",
            ),
        ]
        for suffix, values, inputs, formula, meaning, evidence in payloads:
            _add_feature(
                data,
                records,
                column=f"agro_water__2025__{suffix}_m{month:02d}",
                values=values,
                family="water_productivity",
                mode="competition",
                inputs=inputs,
                formula=formula,
                meaning=meaning,
                rationale=(
                    "Combines water use or productivity with crop-state information to represent "
                    "non-additive water-growth relationships."
                ),
                evidence=evidence,
                references=WATER_REFERENCES,
                caveats=(
                    "WaPOR NPP is not harvested grain yield.",
                    "AETI can include evaporation, transpiration and interception.",
                    "AETI/precipitation is not a direct drought index and can exceed one.",
                ),
            )

    aeti_total = pd.to_numeric(
        frame["wapor_2025__aeti__season_total"], errors="coerce"
    ).to_numpy(dtype=float)
    npp_total = pd.to_numeric(
        frame["wapor_2025__npp__season_total"], errors="coerce"
    ).to_numpy(dtype=float)
    precip_total = climate_outputs["2025"]["precip_season"]
    ndvi_mean = _nanmean_rows(ndvi)
    lai_mean = _nanmean_rows(lai)

    season_payloads = [
        (
            "npp_per_aeti",
            _safe_divide(npp_total, aeti_total, epsilon),
            ["wapor_2025__npp__season_total", "wapor_2025__aeti__season_total"],
            "NPP_season / (AETI_season + eps)",
            "Seasonal water-productivity proxy.",
            "mechanistic_proxy",
        ),
        (
            "aeti_per_precip",
            _safe_divide(aeti_total, precip_total, epsilon),
            ["wapor_2025__aeti__season_total", *precip_columns],
            "AETI_season / (precip_season + eps)",
            "Seasonal actual evapotranspiration relative to precipitation.",
            "mechanistic_proxy",
        ),
        (
            "precip_minus_aeti",
            precip_total - aeti_total,
            [*precip_columns, "wapor_2025__aeti__season_total"],
            "precip_season - AETI_season",
            "Seasonal atmospheric water-balance proxy.",
            "mechanistic_proxy",
        ),
        (
            "npp_x_ndvi_mean",
            npp_total * ndvi_mean,
            ["wapor_2025__npp__season_total", *ndvi_columns],
            "NPP_season * mean(NDVI_Apr-Oct)",
            "Productivity times greenness interaction.",
            "experimental",
        ),
        (
            "lai_mean_x_aeti",
            lai_mean * aeti_total,
            [*lai_columns, "wapor_2025__aeti__season_total"],
            "mean(LAI_Apr-Oct) * AETI_season",
            "Canopy amount times water-use interaction.",
            "experimental",
        ),
    ]

    for suffix, values, inputs, formula, meaning, evidence in season_payloads:
        _add_feature(
            data,
            records,
            column=f"agro_water__2025__{suffix}_season",
            values=values,
            family="water_productivity",
            mode="competition",
            inputs=inputs,
            formula=formula,
            meaning=meaning,
            rationale=(
                "Represents seasonal water-use efficiency or crop-water interactions rather than "
                "forcing independent additive effects."
            ),
            evidence=evidence,
            references=WATER_REFERENCES,
            caveats=("These are remote-sensing proxies, not direct physiological measurements.",),
        )


def _build_soil_features(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
) -> None:
    epsilon = float(config["soil"]["epsilon"])
    properties = [str(value) for value in config["soil"]["depth_properties"]]

    for prop in properties:
        upper = f"soilgrids__{prop}__depth_weighted_0_30cm"
        deep = f"soilgrids__{prop}__30_60cm__mean"
        _require_columns(frame, [upper, deep], label=f"Soil depth profile {prop}")
        upper_values = pd.to_numeric(frame[upper], errors="coerce").to_numpy(dtype=float)
        deep_values = pd.to_numeric(frame[deep], errors="coerce").to_numpy(dtype=float)
        _add_feature(
            data,
            records,
            column=f"agro_soil__{prop}__gradient_0_30_minus_30_60",
            values=upper_values - deep_values,
            family="soil_profile",
            mode="clean",
            inputs=[upper, deep],
            formula="depth_weighted_0_30cm - mean_30_60cm",
            meaning=f"Vertical contrast in SoilGrids {prop}.",
            rationale=(
                "Root-zone conditions can vary with depth; a vertical contrast can carry information "
                "not visible in either layer independently."
            ),
            evidence="mechanistic_proxy",
            references=SOIL_REFERENCES,
            caveats=("SoilGrids values are modeled estimates, not parcel soil samples.",),
        )
        _add_feature(
            data,
            records,
            column=f"agro_soil__{prop}__ratio_0_30_to_30_60",
            values=_safe_divide(upper_values, deep_values, epsilon),
            family="soil_profile",
            mode="clean",
            inputs=[upper, deep],
            formula="depth_weighted_0_30cm / (mean_30_60cm + eps)",
            meaning=f"Relative upper-to-deeper profile for SoilGrids {prop}.",
            rationale="Encodes proportional vertical change in the modeled soil property.",
            evidence="experimental",
            references=SOIL_REFERENCES,
            caveats=(
                "Ratios can be unstable near a zero denominator.",
                "SoilGrids values are modeled estimates.",
            ),
        )

    def soil(prop: str) -> tuple[str, np.ndarray]:
        column = f"soilgrids__{prop}__depth_weighted_0_30cm"
        _require_columns(frame, [column], label=f"Soil property {prop}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return column, values

    soc_col, soc = soil("soc")
    cec_col, cec = soil("cec")
    n_col, nitrogen = soil("nitrogen")
    clay_col, clay = soil("clay")
    sand_col, sand = soil("sand")
    silt_col, silt = soil("silt")
    bdod_col, bdod = soil("bdod")
    ph_col, ph = soil("phh2o")

    interactions = [
        (
            "soc_x_cec",
            soc * cec,
            [soc_col, cec_col],
            "SOC_0_30 * CEC_0_30",
            "Organic-carbon times cation-exchange interaction.",
            "mechanistic_proxy",
        ),
        (
            "soc_x_nitrogen",
            soc * nitrogen,
            [soc_col, n_col],
            "SOC_0_30 * nitrogen_0_30",
            "Soil organic-carbon times nitrogen interaction.",
            "mechanistic_proxy",
        ),
        (
            "clay_x_soc",
            clay * soc,
            [clay_col, soc_col],
            "clay_0_30 * SOC_0_30",
            "Texture times organic-carbon interaction.",
            "mechanistic_proxy",
        ),
        (
            "clay_x_cec",
            clay * cec,
            [clay_col, cec_col],
            "clay_0_30 * CEC_0_30",
            "Texture times cation-exchange interaction.",
            "mechanistic_proxy",
        ),
        (
            "bdod_x_clay",
            bdod * clay,
            [bdod_col, clay_col],
            "bulk_density_0_30 * clay_0_30",
            "Bulk-density times clay interaction.",
            "experimental",
        ),
        (
            "clay_to_sand",
            _safe_divide(clay, sand, epsilon),
            [clay_col, sand_col],
            "clay_0_30 / (sand_0_30 + eps)",
            "Fine-to-coarse texture balance proxy.",
            "mechanistic_proxy",
        ),
        (
            "fine_fraction",
            clay + silt,
            [clay_col, silt_col],
            "clay_0_30 + silt_0_30",
            "Fine soil fraction proxy.",
            "mechanistic_proxy",
        ),
        (
            "phh2o_squared",
            ph**2,
            [ph_col],
            "pH_0_30 ^ 2",
            "Quadratic pH basis term for a possible non-monotone response.",
            "experimental",
        ),
    ]

    for suffix, values, inputs, formula, meaning, evidence in interactions:
        _add_feature(
            data,
            records,
            column=f"agro_soil__{suffix}",
            values=values,
            family="soil_interaction",
            mode="clean",
            inputs=inputs,
            formula=formula,
            meaning=meaning,
            rationale=(
                "Barley response to soil chemistry and exchangeable cations is not necessarily "
                "additive or linear; explicit interactions provide compact nonlinear basis terms."
            ),
            evidence=evidence,
            references=SOIL_REFERENCES,
            caveats=(
                "Interaction terms are candidate predictors, not causal estimates.",
                "SoilGrids values are modeled spatial products.",
            ),
        )


def _build_cross_domain_features(
    frame: pd.DataFrame,
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
    *,
    epsilon: float,
) -> None:
    def base(column: str) -> np.ndarray:
        _require_columns(frame, [column], label="Cross-domain input")
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

    def derived(column: str) -> np.ndarray:
        if column not in data:
            raise KeyError(f"Missing derived dependency: {column}")
        return data[column]

    siap = base("siap_hist__yield_mean_recent")
    slope = base("topo_official__slope__mean")
    elevation = base("cem15__elevation__mean")
    clay = base("soilgrids__clay__depth_weighted_0_30cm")
    soc = base("soilgrids__soc__depth_weighted_0_30cm")
    nitrogen = base("soilgrids__nitrogen__depth_weighted_0_30cm")

    ndvi_hist = derived("agro_pheno__hist_s2_ndvi__season_mean")
    evi_hist = derived("agro_pheno__hist_s2_evi__season_mean")
    lai_hist = derived("agro_pheno__hist_s2_lai__season_mean")
    ndwi_hist = derived("agro_pheno__hist_s2_ndwi__season_mean")
    ndvi_2025 = derived("agro_pheno__y2025_s2_ndvi__season_mean")
    evi_2025 = derived("agro_pheno__y2025_s2_evi__season_mean")
    lai_2025 = derived("agro_pheno__y2025_s2_lai__season_mean")
    ndwi_2025 = derived("agro_pheno__y2025_s2_ndwi__season_mean")
    ndvi_anom = derived("agro_anomaly__s2_ndvi__season_mean_delta")
    evi_anom = derived("agro_anomaly__s2_evi__season_mean_delta")
    wue = derived("agro_water__2025__npp_per_aeti_season")
    water_gap = derived("agro_water__2025__precip_minus_aeti_season")
    heat_2025 = derived("agro_thermal__2025__heat_excess_25c_proxy")
    precip_2025 = base("clim_official_2025__prec__season_sum")
    precip_hist = base("clim_official_hist__prec__season_sum")
    tmean_2025 = derived("agro_thermal__2025__season_tmean")
    tmean_hist = derived("agro_thermal__historical__season_tmean")
    npp_total = base("wapor_2025__npp__season_total")

    payloads = [
        (
            "siap_recent_yield_x_ndvi_anomaly",
            siap * ndvi_anom,
            "competition",
            ["siap_hist__yield_mean_recent", "agro_anomaly__s2_ndvi__season_mean_delta"],
            "SIAP historical yield * NDVI 2025-vs-historical anomaly",
            "Historical municipal productivity times current parcel vegetation departure.",
            "mechanistic_proxy",
        ),
        (
            "siap_recent_yield_x_evi_anomaly",
            siap * evi_anom,
            "competition",
            ["siap_hist__yield_mean_recent", "agro_anomaly__s2_evi__season_mean_delta"],
            "SIAP historical yield * EVI 2025-vs-historical anomaly",
            "Historical municipal productivity times current parcel vegetation departure.",
            "mechanistic_proxy",
        ),
        (
            "siap_recent_yield_x_npp",
            siap * npp_total,
            "competition",
            ["siap_hist__yield_mean_recent", "wapor_2025__npp__season_total"],
            "SIAP historical yield * WaPOR NPP",
            "Historical municipal productivity times current-season productivity proxy.",
            "mechanistic_proxy",
        ),
        (
            "siap_recent_yield_x_wue",
            siap * wue,
            "competition",
            ["siap_hist__yield_mean_recent", "agro_water__2025__npp_per_aeti_season"],
            "SIAP historical yield * NPP/AETI",
            "Historical productivity times current-season water-productivity proxy.",
            "experimental",
        ),
        (
            "slope_x_precip_2025",
            slope * precip_2025,
            "competition",
            ["topo_official__slope__mean", "clim_official_2025__prec__season_sum"],
            "slope * 2025 seasonal precipitation",
            "Terrain times rainfall interaction.",
            "mechanistic_proxy",
        ),
        (
            "slope_x_precip_historical",
            slope * precip_hist,
            "clean",
            ["topo_official__slope__mean", "clim_official_hist__prec__season_sum"],
            "slope * historical seasonal precipitation",
            "Terrain times rainfall interaction.",
            "mechanistic_proxy",
        ),
        (
            "elevation_x_tmean_2025",
            elevation * tmean_2025,
            "competition",
            ["cem15__elevation__mean", "agro_thermal__2025__season_tmean"],
            "elevation * 2025 seasonal Tmean",
            "Elevation times thermal-regime interaction.",
            "experimental",
        ),
        (
            "elevation_x_tmean_historical",
            elevation * tmean_hist,
            "clean",
            ["cem15__elevation__mean", "agro_thermal__historical__season_tmean"],
            "elevation * historical seasonal Tmean",
            "Elevation times thermal-regime interaction.",
            "experimental",
        ),
        (
            "clay_x_precip_2025",
            clay * precip_2025,
            "competition",
            ["soilgrids__clay__depth_weighted_0_30cm", "clim_official_2025__prec__season_sum"],
            "clay_0_30 * 2025 seasonal precipitation",
            "Soil texture times rainfall interaction.",
            "mechanistic_proxy",
        ),
        (
            "clay_x_precip_historical",
            clay * precip_hist,
            "clean",
            ["soilgrids__clay__depth_weighted_0_30cm", "clim_official_hist__prec__season_sum"],
            "clay_0_30 * historical seasonal precipitation",
            "Soil texture times rainfall interaction.",
            "mechanistic_proxy",
        ),
        (
            "soc_x_precip_2025",
            soc * precip_2025,
            "competition",
            ["soilgrids__soc__depth_weighted_0_30cm", "clim_official_2025__prec__season_sum"],
            "SOC_0_30 * 2025 seasonal precipitation",
            "Soil organic carbon times rainfall interaction.",
            "experimental",
        ),
        (
            "soc_x_precip_historical",
            soc * precip_hist,
            "clean",
            ["soilgrids__soc__depth_weighted_0_30cm", "clim_official_hist__prec__season_sum"],
            "SOC_0_30 * historical seasonal precipitation",
            "Soil organic carbon times rainfall interaction.",
            "experimental",
        ),
        (
            "nitrogen_x_ndvi_2025",
            nitrogen * ndvi_2025,
            "competition",
            [
                "soilgrids__nitrogen__depth_weighted_0_30cm",
                "agro_pheno__y2025_s2_ndvi__season_mean",
            ],
            "soil nitrogen_0_30 * 2025 NDVI mean",
            "Soil nitrogen status times canopy greenness interaction.",
            "mechanistic_proxy",
        ),
        (
            "nitrogen_x_ndvi_historical",
            nitrogen * ndvi_hist,
            "clean",
            [
                "soilgrids__nitrogen__depth_weighted_0_30cm",
                "agro_pheno__hist_s2_ndvi__season_mean",
            ],
            "soil nitrogen_0_30 * historical NDVI mean",
            "Soil nitrogen status times canopy greenness interaction.",
            "mechanistic_proxy",
        ),
        (
            "ndvi_x_ndwi_2025",
            ndvi_2025 * ndwi_2025,
            "competition",
            [
                "agro_pheno__y2025_s2_ndvi__season_mean",
                "agro_pheno__y2025_s2_ndwi__season_mean",
            ],
            "2025 NDVI mean * 2025 NDWI mean",
            "Greenness times vegetation or water-status interaction.",
            "mechanistic_proxy",
        ),
        (
            "ndvi_x_ndwi_historical",
            ndvi_hist * ndwi_hist,
            "clean",
            [
                "agro_pheno__hist_s2_ndvi__season_mean",
                "agro_pheno__hist_s2_ndwi__season_mean",
            ],
            "historical NDVI mean * historical NDWI mean",
            "Greenness times vegetation or water-status interaction.",
            "mechanistic_proxy",
        ),
        (
            "evi_x_lai_2025",
            evi_2025 * lai_2025,
            "competition",
            [
                "agro_pheno__y2025_s2_evi__season_mean",
                "agro_pheno__y2025_s2_lai__season_mean",
            ],
            "2025 EVI mean * 2025 LAI mean",
            "Canopy vigor times canopy-amount interaction.",
            "experimental",
        ),
        (
            "evi_x_lai_historical",
            evi_hist * lai_hist,
            "clean",
            [
                "agro_pheno__hist_s2_evi__season_mean",
                "agro_pheno__hist_s2_lai__season_mean",
            ],
            "historical EVI mean * historical LAI mean",
            "Canopy vigor times canopy-amount interaction.",
            "experimental",
        ),
        (
            "heat_x_positive_water_gap_2025",
            heat_2025 * np.maximum(-water_gap, 0.0),
            "competition",
            [
                "agro_thermal__2025__heat_excess_25c_proxy",
                "agro_water__2025__precip_minus_aeti_season",
            ],
            "heat_excess * max(AETI - precipitation, 0)",
            "Combined heat times atmospheric-water-gap proxy.",
            "experimental",
        ),
        (
            "ndvi_anomaly_ratio_to_historical",
            _safe_divide(ndvi_2025 - ndvi_hist, np.abs(ndvi_hist), epsilon),
            "competition",
            [
                "agro_pheno__y2025_s2_ndvi__season_mean",
                "agro_pheno__hist_s2_ndvi__season_mean",
            ],
            "(NDVI_2025 - NDVI_hist) / abs(NDVI_hist)",
            "Relative vegetation departure normalized by historical magnitude.",
            "experimental",
        ),
    ]

    for suffix, values, mode, inputs, formula, meaning, evidence in payloads:
        _add_feature(
            data,
            records,
            column=f"agro_interaction__{suffix}",
            values=values,
            family="cross_domain",
            mode=mode,
            inputs=inputs,
            formula=formula,
            meaning=meaning,
            rationale=(
                "Agronomic outcomes can depend on combinations of baseline productivity, crop state, "
                "soil, terrain, temperature and water rather than independent additive effects."
            ),
            evidence=evidence,
            references=(*PHENOLOGY_REFERENCES, *WATER_REFERENCES, *SOIL_REFERENCES),
            caveats=("These interaction terms are hypotheses for prediction, not causal claims.",),
        )


def _build_scale_transforms(
    frame: pd.DataFrame,
    data: dict[str, np.ndarray],
    records: list[AgronomicFeatureRecord],
) -> None:
    """Add a small target-free set of monotone nonlinear basis functions."""

    candidates = [
        ("area_ha", "base_area_ha", "clean"),
        ("siap_harvested_2024", "siap_hist__harvested_ha_2024", "clean"),
        ("siap_production_2024", "siap_hist__production_t_2024", "clean"),
        ("precip_hist", "clim_official_hist__prec__season_sum", "clean"),
        ("precip_2025", "clim_official_2025__prec__season_sum", "competition"),
        ("aeti_2025", "wapor_2025__aeti__season_total", "competition"),
        ("npp_2025", "wapor_2025__npp__season_total", "competition"),
    ]
    for name, column, mode in candidates:
        _require_columns(frame, [column], label=f"Scale transform {name}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        transformed = np.full(len(values), np.nan)
        finite = np.isfinite(values) & (values >= 0)
        transformed[finite] = np.log1p(values[finite])
        _add_feature(
            data,
            records,
            column=f"agro_nonlinear__log1p_{name}",
            values=transformed,
            family="nonlinear_basis",
            mode=mode,
            inputs=[column],
            formula=f"log(1 + {column})",
            meaning="Log-compressed basis term for a non-negative, potentially skewed predictor.",
            rationale=(
                "A monotone log basis can represent diminishing marginal effects for linear or "
                "regularized models without choosing a transformation from target correlations."
            ),
            evidence="experimental",
            references=(),
            caveats=("Defined only for non-negative values.",),
        )


def build_agronomic_feature_layer(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Build the additive agronomic nonlinear feature layer."""

    id_column = str(config["identity"]["id_column"])
    _require_columns(frame, [id_column], label="Agronomic feature input")
    if frame[id_column].duplicated().any():
        raise ValueError("Agronomic feature input contains duplicate parcel IDs.")

    data: dict[str, np.ndarray] = {}
    records: list[AgronomicFeatureRecord] = []
    epsilon = float(config["water"]["epsilon"])

    series_results, series_inputs = _build_phenology_features(
        frame,
        config,
        data,
        records,
    )
    _build_anomaly_features(
        frame,
        config,
        series_results,
        series_inputs,
        data,
        records,
        epsilon=epsilon,
    )
    _build_sensor_agreement_features(
        frame,
        config,
        series_results,
        series_inputs,
        data,
        records,
        epsilon=epsilon,
    )
    climate_outputs = _build_thermal_and_precip_features(frame, config, data, records)
    _build_water_productivity_features(
        frame,
        config,
        climate_outputs,
        data,
        records,
    )
    _build_soil_features(frame, config, data, records)
    _build_cross_domain_features(frame, data, records, epsilon=epsilon)
    _build_scale_transforms(frame, data, records)

    output = pd.DataFrame({id_column: frame[id_column].astype(str).to_numpy(), **data})
    if not output[id_column].is_unique:
        raise ValueError("Agronomic output ID is not unique.")

    record_by_column = {record.column: record for record in records}
    feature_columns = [column for column in output.columns if column != id_column]

    dropped_all_null: list[str] = []
    dropped_constant: list[str] = []
    if bool(config["validation"].get("drop_all_null", True)):
        dropped_all_null = [column for column in feature_columns if output[column].isna().all()]
        output = output.drop(columns=dropped_all_null)
    feature_columns = [column for column in output.columns if column != id_column]
    if bool(config["validation"].get("drop_constant", True)):
        dropped_constant = [
            column
            for column in feature_columns
            if output[column].dropna().nunique() <= 1
        ]
        output = output.drop(columns=dropped_constant)

    retained = [column for column in output.columns if column != id_column]
    retained_records = [record_by_column[column].as_dict() for column in retained]
    family_counts = (
        pd.Series([record["family"] for record in retained_records], dtype="string")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    mode_counts = (
        pd.Series([record["mode"] for record in retained_records], dtype="string")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    diagnostics = {
        "rows": int(len(output)),
        "features": int(len(retained)),
        "family_counts": {str(key): int(value) for key, value in family_counts.items()},
        "mode_counts": {str(key): int(value) for key, value in mode_counts.items()},
        "dropped_all_null": dropped_all_null,
        "dropped_constant": dropped_constant,
        "missing_fraction_mean": (
            float(output[retained].isna().mean().mean()) if retained else 0.0
        ),
        "missing_fraction_max": (
            float(output[retained].isna().mean().max()) if retained else 0.0
        ),
    }
    return output, retained_records, diagnostics


def validate_agronomic_feature_layer(
    table: pd.DataFrame,
    *,
    expected_ids: Sequence[str] | pd.Series,
    id_column: str = ID_COLUMN,
) -> None:
    """Validate one-to-one parcel coverage and numeric derived features."""

    if id_column not in table.columns:
        raise KeyError(id_column)
    if table[id_column].duplicated().any():
        raise ValueError("Agronomic feature table contains duplicate parcel IDs.")

    observed = set(table[id_column].astype(str))
    expected = set(pd.Series(expected_ids).astype(str))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Agronomic parcel coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    feature_columns = [column for column in table.columns if column != id_column]
    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(table[column])
    ]
    if non_numeric:
        raise TypeError(f"Agronomic features must be numeric: {non_numeric[:10]}")
    values = table[feature_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Agronomic feature table contains infinite values.")


def join_agronomic_features(
    base: pd.DataFrame,
    agronomic: pd.DataFrame,
    *,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Join the additive agronomic feature layer to a parcel-level base table."""

    if base[id_column].duplicated().any() or agronomic[id_column].duplicated().any():
        raise ValueError("Both inputs must be one row per parcel before joining.")
    return base.merge(agronomic, on=id_column, how="left", validate="one_to_one")
