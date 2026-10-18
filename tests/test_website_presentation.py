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
                     "checkpoint08_bootstrap.csv", "predictions.csv",
                     "checkpoint02_baseline_summary.csv", "checkpoint03b_expression_stability.csv",
                     "checkpoint04b_protocol_summary.csv", "checkpoint05_protocol_summary.csv",
                     "final_prediction_diagnostics.csv",
                     "checkpoint06_feature_family_tail_importance.csv",
                     "checkpoint06_feature_family_error_risk.csv"):
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
        "Optuna (96 ensayos por familia y partición",
        "125 IDs únicos en 07; 132 en 08",
        "Los 41 rendimientos externos nunca guían Optuna",
    ):
        assert phrase in page


def test_one_page_submission_source_has_page_contract_and_scientific_limits() -> None:
    summary = (WEBSITE / "resumen.html").read_text(encoding="utf-8")
    assert "@page{size:letter portrait;margin:0}" in summary
    assert '<html lang="es">' in summary
    assert "GeoRidge Local — Configuración Geográfica (k=24, α=30, p=1)" in summary
    assert "GeoRidge = Local04D (repositorio)" in summary
    for phrase in (
        "0.495741",
        "0.481936",
        "0.493943",
        "0.510070",
        "Los RMSE de las etapas 7 y 8",
        "desconocido",
        "sólo",
    ):
        assert phrase in summary
    assert 'href="./resumen.html"' in (WEBSITE / "index.html").read_text(encoding="utf-8")


def test_presentation_names_keep_traceability_to_repository_names() -> None:
    """Public names are plain-language; the repository name appears only as an equivalence."""
    html = (WEBSITE / "index.html").read_text(encoding="utf-8")
    javascript = (WEBSITE / "app.js").read_text(encoding="utf-8")
    assert "Geo<span>Ridge</span>" in html
    assert "registrado en el repositorio como Local04D" in html
    assert "Prithvi07" not in html and "CatBoost07" not in html
    assert '"GeoRidge · seleccionado"' in javascript
    assert '"GeoRidge fijo · seleccionado"' in javascript
    # CSV model keys stay internal and must still match the data files.
    assert 'Local04D_frozen' in javascript


def test_equation_is_prerendered_locally_without_third_party_requests() -> None:
    html = (WEBSITE / "index.html").read_text(encoding="utf-8")
    assert "katex-mathml" in html
    assert 'href="./vendor/katex/katex.min.css"' in html
    assert (WEBSITE / "vendor" / "katex" / "katex.min.css").is_file()
    assert list((WEBSITE / "vendor" / "katex" / "fonts").glob("*.woff2"))
    assert "https://cdn" not in html and "<script src=\"http" not in html


def test_baseline_context_panel_matches_published_tables() -> None:
    state_models = {
        (r["family"], r["method"]): float(r["rmse_mean"])
        for r in read_rows("checkpoint04b_protocol_summary.csv")
    }
    later = {
        (r["family"], r["method"]): float(r["rmse_mean"])
        for r in read_rows("checkpoint05_protocol_summary.csv")
    }
    assert abs(state_models[("target_matched", "GlobalMean")] - 0.849973) < 5e-7
    assert abs(later[("target_matched", "Local04D")] - 0.487845) < 5e-7
    for method in ("StateMean", "GeoKNN_10"):
        assert state_models[("target_matched", method)] > later[("target_matched", "Local04D")]
    for method in ("PLS_C0", "CatBoost_C1", "Graph04D"):
        assert later[("target_matched", method)] > later[("target_matched", "Local04D")]


def test_support_tiers_cover_exactly_the_published_59_estimates() -> None:
    predictions = {
        r["ID_POLIGONO"]: float(r["RENDIMIENTO_T_HA"])
        for r in read_rows("predictions.csv")
    }
    diagnostics = read_rows("final_prediction_diagnostics.csv")
    assert {r["ID_POLIGONO"] for r in diagnostics} == set(predictions)
    assert len(diagnostics) == 59
    assert {r["x_support_tier"] for r in diagnostics} <= {"low", "mid", "high"}
    for row in diagnostics:
        assert abs(float(row["predicted_final"]) - predictions[row["ID_POLIGONO"]]) < 1e-6
        assert float(row["local_graph_abs_disagreement"]) >= 0
    forbidden = ("observed", "y_true", "actual_yield", "reference_yield")
    assert not [c for c in diagnostics[0] if any(word in c.lower() for word in forbidden)]


def test_variable_evidence_tables_match_the_verified_schema() -> None:
    ablation = read_rows("checkpoint02_baseline_summary.csv")
    a0 = [r for r in ablation if r["ablation"] == "A0_geometry_admin"
          and r["protocol"] == "fold_state_stratified" and r["model"] == "ExtraTrees"]
    assert len(a0) == 1 and f"{float(a0[0]['rmse_mean']):.4f}" == "0.5095"

    tail = read_rows("checkpoint06_feature_family_tail_importance.csv")
    low = [r for r in tail if r["family"] == "water_productivity" and r["task"] == "low_vs_rest"]
    assert len(low) == 1 and f"{float(low[0]['mean_split_roc_auc']):.3f}" == "0.889"

    pairs: dict[tuple[str, str], dict[str, int]] = {}
    for row in read_rows("checkpoint03b_expression_stability.csv"):
        key = tuple(sorted((row["left"], row["right"])))
        entry = pairs.setdefault(key, {"fold_state_stratified": 0, "fold_municipality_grouped": 0})
        entry[row["protocol"]] = max(entry[row["protocol"]], int(float(row["selected_folds"])))
    recurring = [k for k, v in pairs.items() if min(v.values()) >= 3]
    assert len(recurring) == 4

    risk = [
        r for r in read_rows("checkpoint06_feature_family_error_risk.csv")
        if r["method"] == "Local04D"
    ]
    assert risk and max(float(r["risk_r2_log"]) for r in risk) <= 0


def test_limitations_annex_keeps_required_scientific_caveats() -> None:
    html = (WEBSITE / "index.html").read_text(encoding="utf-8")
    assert "ANEXO E / LIMITACIONES" in html
    for phrase in (
        "El error real de las 59 parcelas es desconocido",
        "Las evaluaciones no son independientes",
        "El modelo comprime los extremos",
        "No es un predictor universal",
        "no queda descartado",
        "no por índice",
        "intervalo de confianza",
    ):
        assert phrase in html
