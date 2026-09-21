"""Checkpoint 04E external-evidence enrichment utilities.

The first 04E stage audits SIAP at the most specific rule-permitted level:
Cebada grano + production cycle + Temporal + municipality. The public 2025
municipal outcome is treated as an external competition-mode covariate, never
as a hidden parcel label.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from geocebada.evaluation.checkpoint04b import predict_graph_laplacian
from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge


def normalize_siap_label(value: object) -> str:
    """Return a stable accent/case-insensitive SIAP label."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


def aggregate_siap_scope(
    detail: pd.DataFrame,
    *,
    crop: str,
    cycle: str | None,
    modality: str | None,
) -> pd.DataFrame:
    """Aggregate one explicit SIAP crop/cycle/modality scope by municipality-year."""

    required = {
        "Anio",
        "cvegeo",
        "Nomcultivo",
        "Nomcicloproductivo",
        "Nommodalidad",
        "Sembrada",
        "Cosechada",
        "Siniestrada",
        "Volumenproduccion",
        "Rendimiento",
    }
    missing = required.difference(detail.columns)
    if missing:
        raise KeyError(f"Detailed SIAP table missing columns: {sorted(missing)}")

    work = detail.copy()
    mask = work["Nomcultivo"].map(normalize_siap_label).eq(normalize_siap_label(crop))
    if cycle is not None:
        mask &= work["Nomcicloproductivo"].map(normalize_siap_label).eq(
            normalize_siap_label(cycle)
        )
    if modality is not None:
        mask &= work["Nommodalidad"].map(normalize_siap_label).eq(
            normalize_siap_label(modality)
        )
    work = work.loc[mask].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "cvegeo",
                "Anio",
                "siap_yield_t_ha",
                "siap_reported_yield_t_ha",
                "siap_sembrada_ha",
                "siap_cosechada_ha",
                "siap_siniestrada_ha",
                "siap_production_t",
                "siap_damage_rate",
                "siap_rows",
            ]
        )

    for column in [
        "Anio",
        "Sembrada",
        "Cosechada",
        "Siniestrada",
        "Volumenproduccion",
        "Rendimiento",
    ]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["cvegeo"] = work["cvegeo"].astype("string").str.zfill(5)
    work["_reported_mass"] = work["Rendimiento"] * work["Cosechada"]

    grouped = (
        work.groupby(["cvegeo", "Anio"], as_index=False)
        .agg(
            siap_sembrada_ha=("Sembrada", "sum"),
            siap_cosechada_ha=("Cosechada", "sum"),
            siap_siniestrada_ha=("Siniestrada", "sum"),
            siap_production_t=("Volumenproduccion", "sum"),
            _reported_mass=("_reported_mass", "sum"),
            siap_rows=("Nomcultivo", "size"),
        )
        .sort_values(["cvegeo", "Anio"])
    )
    harvested = grouped["siap_cosechada_ha"].replace(0, np.nan)
    planted = grouped["siap_sembrada_ha"].replace(0, np.nan)
    grouped["siap_yield_t_ha"] = grouped["siap_production_t"] / harvested
    grouped["siap_reported_yield_t_ha"] = grouped["_reported_mass"] / harvested
    grouped["siap_damage_rate"] = grouped["siap_siniestrada_ha"] / planted
    return grouped.drop(columns="_reported_mass")


def _historical_stats(
    annual: pd.DataFrame,
    *,
    years: Sequence[int],
    prefix: str,
) -> pd.DataFrame:
    """Summarize historical municipal SIAP yield for one scope."""

    year_values = {int(value) for value in years}
    hist = annual.loc[annual["Anio"].astype("Int64").isin(year_values)].copy()
    rows: list[dict[str, float | str]] = []
    for cvegeo, group in hist.groupby("cvegeo", sort=True):
        x = pd.to_numeric(group["Anio"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group["siap_yield_t_ha"], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        trend = float(np.polyfit(x[finite], y[finite], 1)[0]) if finite.sum() >= 2 else np.nan
        rows.append(
            {
                "cvegeo": str(cvegeo).zfill(5),
                f"{prefix}_mean": float(np.nanmean(y)) if np.isfinite(y).any() else np.nan,
                f"{prefix}_std": float(np.nanstd(y)) if np.isfinite(y).any() else np.nan,
                f"{prefix}_trend": trend,
                f"{prefix}_years": int(finite.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_siap_external_panel(
    detail: pd.DataFrame,
    *,
    crop: str,
    cycle: str,
    modality: str,
    competition_year: int,
    historical_years: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build audited exact/fallback SIAP municipal priors and scope diagnostics."""

    scopes = {
        "exact": (cycle, modality),
        "cycle_allmode": (cycle, None),
        "temporal_allcycle": (None, modality),
        "allgrain": (None, None),
    }
    annual_by_scope: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    panel: pd.DataFrame | None = None

    for name, (scope_cycle, scope_modality) in scopes.items():
        annual = aggregate_siap_scope(
            detail,
            crop=crop,
            cycle=scope_cycle,
            modality=scope_modality,
        )
        annual_by_scope[name] = annual
        current = annual.loc[annual["Anio"].eq(int(competition_year))].copy()
        current = current.rename(
            columns={
                "siap_yield_t_ha": f"siap_{name}_{competition_year}_yield",
                "siap_reported_yield_t_ha": f"siap_{name}_{competition_year}_reported_yield",
                "siap_cosechada_ha": f"siap_{name}_{competition_year}_harvested_ha",
                "siap_production_t": f"siap_{name}_{competition_year}_production_t",
                "siap_damage_rate": f"siap_{name}_{competition_year}_damage_rate",
                "siap_rows": f"siap_{name}_{competition_year}_rows",
            }
        )
        keep = [
            "cvegeo",
            f"siap_{name}_{competition_year}_yield",
            f"siap_{name}_{competition_year}_reported_yield",
            f"siap_{name}_{competition_year}_harvested_ha",
            f"siap_{name}_{competition_year}_production_t",
            f"siap_{name}_{competition_year}_damage_rate",
            f"siap_{name}_{competition_year}_rows",
        ]
        current = current.loc[:, keep]
        panel = current if panel is None else panel.merge(current, on="cvegeo", how="outer")
        audit_rows.append(
            {
                "scope": name,
                "cycle": scope_cycle or "ALL",
                "modality": scope_modality or "ALL",
                "rows_all_years": int(len(annual)),
                "municipalities_2025": int(current["cvegeo"].nunique()),
                "rows_2025": int(current[f"siap_{name}_{competition_year}_rows"].sum())
                if not current.empty
                else 0,
            }
        )

    if panel is None:
        panel = pd.DataFrame(columns=["cvegeo"])

    for name in ["exact", "allgrain"]:
        stats = _historical_stats(
            annual_by_scope[name],
            years=historical_years,
            prefix=f"siap_{name}_hist",
        )
        panel = panel.merge(stats, on="cvegeo", how="outer")

    proxy_columns = [
        (f"siap_exact_{competition_year}_yield", "exact_2025"),
        (f"siap_cycle_allmode_{competition_year}_yield", "cycle_allmode_2025"),
        (f"siap_temporal_allcycle_{competition_year}_yield", "temporal_allcycle_2025"),
        (f"siap_allgrain_{competition_year}_yield", "allgrain_2025"),
        ("siap_exact_hist_mean", "exact_historical_mean"),
        ("siap_allgrain_hist_mean", "allgrain_historical_mean"),
    ]
    panel["siap_prior_yield"] = np.nan
    panel["siap_prior_source"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    for column, source in proxy_columns:
        if column not in panel:
            continue
        available = panel["siap_prior_yield"].isna() & pd.to_numeric(
            panel[column], errors="coerce"
        ).notna()
        panel.loc[available, "siap_prior_yield"] = pd.to_numeric(
            panel.loc[available, column], errors="coerce"
        )
        panel.loc[available, "siap_prior_source"] = source

    exact_col = f"siap_exact_{competition_year}_yield"
    reported_col = f"siap_exact_{competition_year}_reported_yield"
    if exact_col in panel and reported_col in panel:
        panel["siap_exact_reported_minus_recomputed"] = (
            pd.to_numeric(panel[reported_col], errors="coerce")
            - pd.to_numeric(panel[exact_col], errors="coerce")
        )
    else:
        panel["siap_exact_reported_minus_recomputed"] = np.nan

    return panel.sort_values("cvegeo").reset_index(drop=True), pd.DataFrame(audit_rows)


def attach_siap_panel_to_parcels(
    parcels: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    cvegeo_column: str = "admin_cvegeo",
) -> pd.DataFrame:
    """Attach one municipal SIAP panel row to every parcel."""

    if cvegeo_column not in parcels:
        raise KeyError(f"Parcel table missing {cvegeo_column!r}.")
    mapping = parcels[["ID_POLIGONO", cvegeo_column]].copy()
    mapping["cvegeo"] = mapping[cvegeo_column].astype("string").str.zfill(5)
    mapping = mapping.drop(columns=cvegeo_column)
    municipal = panel.copy()
    municipal["cvegeo"] = municipal["cvegeo"].astype("string").str.zfill(5)
    result = mapping.merge(municipal, on="cvegeo", how="left", validate="m:1")
    if len(result) != len(parcels):
        raise ValueError("SIAP parcel attachment changed row count.")
    return result


def complete_external_prior(
    prior: Sequence[float],
    *,
    reference_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fill residual missing external priors with the visible-reference median."""

    values = np.asarray(prior, dtype=float).copy()
    refs = np.asarray(list(map(int, reference_positions)), dtype=int)
    finite_refs = values[refs][np.isfinite(values[refs])]
    if len(finite_refs) == 0:
        raise ValueError("No finite SIAP prior exists among visible reference parcels.")
    fallback = float(np.median(finite_refs))
    filled = ~np.isfinite(values)
    values[filled] = fallback
    return values, filled


def predict_affine_external_prior(
    prior: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    alpha: float,
) -> np.ndarray:
    """Calibrate the public municipal prior using only currently visible parcel labels."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = np.asarray(list(map(int, query_positions)), dtype=int)
    model = Ridge(alpha=float(alpha))
    model.fit(prior[observed].reshape(-1, 1), np.asarray(y, dtype=float)[observed])
    return np.asarray(model.predict(prior[query].reshape(-1, 1)), dtype=float)


def predict_local_external_residual(
    prior: np.ndarray,
    x: np.ndarray,
    distance_matrix: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    k: int,
    alpha: float,
    distance_power: float,
) -> np.ndarray:
    """Add a query-specific local Ridge correction to the SIAP municipal prior."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = np.asarray(list(map(int, query_positions)), dtype=int)
    residual = np.full(len(prior), np.nan, dtype=float)
    residual[observed] = np.asarray(y, dtype=float)[observed] - prior[observed]
    correction = predict_weighted_local_ridge(
        x,
        distance_matrix,
        residual,
        observed,
        query,
        k=int(k),
        alpha=float(alpha),
        distance_power=float(distance_power),
    )
    return prior[query] + correction


def predict_graph_external_residual(
    prior: np.ndarray,
    adjacency: np.ndarray,
    y: np.ndarray,
    observed_positions: Sequence[int],
    query_positions: Sequence[int],
    *,
    regularization: float,
) -> np.ndarray:
    """Add graph-propagated parcel residuals to the SIAP municipal prior."""

    observed = np.asarray(list(map(int, observed_positions)), dtype=int)
    query = np.asarray(list(map(int, query_positions)), dtype=int)
    residual = np.full(len(prior), np.nan, dtype=float)
    residual[observed] = np.asarray(y, dtype=float)[observed] - prior[observed]
    correction = predict_graph_laplacian(
        adjacency,
        residual,
        observed,
        query,
        regularization=float(regularization),
    )
    return prior[query] + correction


def blend_predictions(
    left: np.ndarray,
    right: np.ndarray,
    *,
    right_weight: float,
) -> np.ndarray:
    """Return a fixed convex blend of two prediction vectors."""

    weight = float(right_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Blend weight must lie in [0, 1].")
    return (1.0 - weight) * np.asarray(left, dtype=float) + weight * np.asarray(
        right, dtype=float
    )
