"""Regression checks for the public, static GeoCebada research presentation."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"


def read_rows(path: str) -> list[dict[str, str]]:
    with (WEBSITE / "data" / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_website_snapshot_matches_canonical_results() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sync_website_data.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_website_best_model_and_split_banks_are_distinct() -> None:
    development = read_rows("checkpoint07_summary.csv")
    confirmation = read_rows("checkpoint08_summary.csv")
    assert len(development) == 13
    assert len(confirmation) == 3
    assert {int(row["n_splits_completed"]) for row in development} == {16}
    assert {int(row["n_repeated_predictions"]) for row in development} == {656}
    assert {int(row["n_unique_parcels"]) for row in development} == {125}
    assert {int(row["n_splits"]) for row in confirmation} == {16}
    assert {int(row["repeated_predictions"]) for row in confirmation} == {656}
    assert {int(row["unique_parcels"]) for row in confirmation} == {132}
    winner07 = min(development, key=lambda row: float(row["rmse_pooled"]))
    winner08 = min(confirmation, key=lambda row: float(row["rmse_pooled"]))
    assert winner07["model"] == "Local04D"
    assert winner08["model"] == "Local04D_frozen"
    assert abs(float(winner07["rmse_pooled"]) - 0.49574142871073495) < 1e-12
    assert abs(float(winner08["rmse_pooled"]) - 0.4819355585746856) < 1e-12


def test_bootstrap_not_claimed_to_score_actual_hidden_targets() -> None:
    boot = read_rows("checkpoint08_bootstrap.csv")
    assert len(boot) == 2
    assert {row["reference"] for row in boot} == {"Local04D_frozen"}
    assert {row["bootstrap_unit"] for row in boot} == {"unique_parcel"}
    assert all(0 < float(row["ci95_low"]) < float(row["ci95_high"]) for row in boot)


def test_prediction_explorer_only_contains_fixed_59() -> None:
    predictions = read_rows("predictions.csv")
    assert len(predictions) == 59
    assert all(set(row) == {"ID_POLIGONO", "RENDIMIENTO_T_HA"} for row in predictions)
    ids = [row["ID_POLIGONO"] for row in predictions]
    assert len(set(ids)) == 59
    assert all(p.startswith("AGC_") for p in ids)
    assert all(0 < float(row["RENDIMIENTO_T_HA"]) < 30 for row in predictions)


def test_site_documents_limits_and_uses_relative_assets() -> None:
    html = (WEBSITE / "index.html").read_text(encoding="utf-8")
    javascript = (WEBSITE / "app.js").read_text(encoding="utf-8")
    for anchor in ("reto", "datos", "metodo", "validacion", "resultados", "explorar", "anexo"):
        assert f'id="{anchor}"' in html
    assert "no vuelve a entrenar" in html.lower()
    assert "particiones diferentes" in html
    assert "conceptual" in html.lower()
    for snapshot in ("checkpoint07_summary.csv", "checkpoint08_summary.csv",
                     "checkpoint08_bootstrap.csv", "predictions.csv"):
        assert snapshot in javascript
    assert "https://api." not in javascript


def test_cross_validation_explains_selection_and_nested_evaluation() -> None:
    """Visible explanation must separate the label-blind protocols and tuning stages."""
    page = (WEBSITE / "index.html").read_text(encoding="utf-8")
    assert 'id="validacion"' in page
    assert "97 / SE PUEDE ENTRENAR" in page
    assert "41 / RENDIMIENTO OCULTO" in page
    assert "16 PARTICIONES REPRODUCIBLES" in page
    assert page.count('class="cv-mini-split"') == 16
    for phrase in (
        "método de Hamilton",
        "80 propuestas por repetición",
        "160 máscaras candidatas por repetición",
        "Optuna (96 ensayos por familia y split",
        "125 IDs únicos en 07; 132 en 08",
        "Los 41 rendimientos externos nunca guían Optuna",
    ):
        assert phrase in page


def test_one_page_submission_source_has_page_contract_and_scientific_limits() -> None:
    summary = (WEBSITE / "resumen.html").read_text(encoding="utf-8")
    assert "@page{size:letter portrait;margin:0}" in summary
    assert '<html lang="es">' in summary
    assert "LocalRidge_C4_all_deterministic_Geo_k24_a30_p1" in summary
    for phrase in (
        "0.495741",
        "0.481936",
        "0.493943",
        "0.510070",
        "Los RMSE de 07 y 08",
        "desconocido",
        "sólo",
    ):
        assert phrase in summary
    assert 'href="./resumen.html"' in (WEBSITE / "index.html").read_text(encoding="utf-8")
