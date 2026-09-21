"""Tests for Checkpoint 04E SIAP localization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint04e import (
    aggregate_siap_scope,
    attach_siap_panel_to_parcels,
    blend_predictions,
    build_siap_external_panel,
    complete_external_prior,
    predict_affine_external_prior,
)


def _siap_detail() -> pd.DataFrame:
    rows = [
        # Exact target scope: grain + Primavera-Verano + Temporal.
        [2024, "21001", "Cebada grano", "Primavera-Verano", "Temporal", 10, 10, 0, 30, 3.0],
        [2025, "21001", "Cebada grano", "Primavera-Verano", "Temporal", 12, 10, 2, 35, 3.5],
        [2024, "13001", "Cebada grano", "Primavera-Verano", "Temporal", 8, 8, 0, 20, 2.5],
        [2025, "13001", "Cebada grano", "Primavera-Verano", "Temporal", 8, 8, 0, 24, 3.0],
        # Same crop/cycle but irrigation: must not enter exact scope.
        [2025, "21001", "Cebada grano", "Primavera-Verano", "Riego", 5, 5, 0, 25, 5.0],
        # Forage barley: must never enter grain scope.
        [2025, "21001", "Cebada forrajera en verde", "Primavera-Verano", "Temporal", 5, 5, 0, 50, 10.0],
    ]
    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )


def test_exact_siap_scope_excludes_irrigation_and_forage() -> None:
    annual = aggregate_siap_scope(
        _siap_detail(),
        crop="Cebada grano",
        cycle="Primavera-Verano",
        modality="Temporal",
    )
    row = annual.loc[(annual["cvegeo"] == "21001") & (annual["Anio"] == 2025)].iloc[0]
    assert row["siap_rows"] == 1
    assert np.isclose(row["siap_yield_t_ha"], 3.5)
    assert np.isclose(row["siap_damage_rate"], 2 / 12)


def test_siap_external_panel_prefers_exact_2025_proxy() -> None:
    panel, audit = build_siap_external_panel(
        _siap_detail(),
        crop="Cebada grano",
        cycle="Primavera-Verano",
        modality="Temporal",
        competition_year=2025,
        historical_years=[2024],
    )
    row = panel.loc[panel["cvegeo"].eq("21001")].iloc[0]
    assert np.isclose(row["siap_exact_2025_yield"], 3.5)
    assert np.isclose(row["siap_allgrain_2025_yield"], 4.0)
    assert row["siap_prior_source"] == "exact_2025"
    assert np.isclose(row["siap_prior_yield"], 3.5)
    assert set(audit["scope"]) == {
        "exact",
        "cycle_allmode",
        "temporal_allcycle",
        "allgrain",
    }


def test_attach_panel_and_prior_completion_are_one_row_per_parcel() -> None:
    panel, _ = build_siap_external_panel(
        _siap_detail(),
        crop="Cebada grano",
        cycle="Primavera-Verano",
        modality="Temporal",
        competition_year=2025,
        historical_years=[2024],
    )
    parcels = pd.DataFrame(
        {
            "ID_POLIGONO": ["A", "B", "C"],
            "admin_cvegeo": ["21001", "13001", "29001"],
        }
    )
    attached = attach_siap_panel_to_parcels(parcels, panel)
    assert len(attached) == 3
    prior = pd.to_numeric(attached["siap_prior_yield"], errors="coerce").to_numpy(float)
    filled, mask = complete_external_prior(prior, reference_positions=[0, 1])
    assert np.isfinite(filled).all()
    assert mask.tolist() == [False, False, True]
    assert np.isclose(filled[2], np.median(filled[:2]))


def test_affine_prior_and_blend_predictions() -> None:
    prior = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    pred = predict_affine_external_prior(
        prior,
        y,
        observed_positions=[0, 1, 2],
        query_positions=[3],
        alpha=0.0,
    )
    assert np.isclose(pred[0], 8.0)
    blended = blend_predictions(np.array([2.0]), np.array([4.0]), right_weight=0.25)
    assert np.isclose(blended[0], 2.5)
