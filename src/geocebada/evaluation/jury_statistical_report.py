"""Create private diagnostic figures and an academic-style Markdown report.

The plots can disclose individual labels and coordinates. They are NEVER
automatically copied to the public site. Public exports use aggregates only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def render_figures(
    actual: np.ndarray, estimates: dict[str, np.ndarray],
    coords: np.ndarray, spatial: pd.DataFrame,
    semivariogram: pd.DataFrame, output: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, pred in estimates.items():
        error = pred-actual
        path = directory / (name.lower()+"_hist.png")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(error, bins=17, density=True, alpha=.6)
        x = np.linspace(error.min(), error.max(), 200)
        ax.plot(x, stats.norm.pdf(x, error.mean(), error.std(ddof=1)),
                label="Referencia normal ajustada")
        ax.axvline(0, linestyle="--", color="black")
        ax.set(xlabel="Residuo LOOCV: predicción − observado (t/ha)",
               ylabel="Densidad", title=f"{name}: distribución de residuos")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

        path = directory / (name.lower()+"_qq.png")
        (q, sorted_e), (slope, intercept, corr) = stats.probplot(
            error, dist="norm", fit=True
        )
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(q, sorted_e, s=17)
        ax.plot(q, slope*q+intercept, "--", color="black",
                label=f"Referencia, correlación {corr:.3f}")
        ax.set(xlabel="Cuantiles normales teóricos",
               ylabel="Cuantiles de residuos (t/ha)",
               title=f"{name}: gráfico Q–Q")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

        for slug, x, y, xlabel, ylabel in (
            ("residual_vs_pred", pred, error, "Predicción (t/ha)",
             "Residuo LOOCV (t/ha)"),
            ("observed_vs_pred", actual, pred, "Observado (t/ha)",
             "Predicho fuera de muestra (t/ha)"),
        ):
            path = directory / (name.lower()+"_"+slug+".png")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x, y, s=18, alpha=.7)
            if slug == "residual_vs_pred":
                ax.axhline(0, color="black", linestyle="--")
            else:
                lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
                ax.plot([lo, hi], [lo, hi], "--", color="black")
            ax.set(title=name, xlabel=xlabel, ylabel=ylabel)
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            saved.append(path)

        path = directory / (name.lower()+"_map_PRIVATE.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        scale = max(np.max(abs(error)), 1e-8)
        dots = ax.scatter(coords[:, 1], coords[:, 0], c=error,
                          vmin=-scale, vmax=scale, cmap="coolwarm", s=35)
        fig.colorbar(dots, ax=ax, label="Residuo (t/ha)")
        ax.set(title=f"{name}: residuos espaciales CONFIDENCIALES",
               xlabel="Longitud", ylabel="Latitud")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    path = directory / "absolute_error_ecdf.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pred in estimates.items():
        ae = np.sort(abs(pred-actual))
        ax.step(ae, np.arange(1, len(ae)+1)/len(ae),
                where="post", label=name)
    ax.set(title="Función de distribución del error absoluto",
           xlabel="Error absoluto (t/ha)", ylabel="Fracción acumulada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    path = directory / "spatial_moran_knn.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = spatial.loc[
        spatial.statistic.eq("Moran_I")
        & spatial.spatial_variable.eq("signed_residual")
    ]
    for name, part in rows.groupby("model"):
        ax.plot(part.knn, part.estimate, marker="o", label=name)
    ax.axhline(-1/(len(actual)-1), color="black", linestyle="--")
    ax.set(title="Moran I de residuos LOOCV",
           xlabel="Número de vecinos geográficos k", ylabel="Moran I")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    path = directory / "residual_semivariogram.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, part in semivariogram.groupby("model"):
        ax.plot(part.median_distance_km, part.semivariance_t2_ha2,
                marker="o", label=name)
    ax.set(title="Semivariograma empírico de errores",
           xlabel="Distancia entre parcelas (km)",
           ylabel="Semivarianza (t/ha)²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    return saved


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    names = [col for col in columns if col in frame]
    head = "| "+" | ".join(names)+" |"
    separator = "| "+" | ".join(["---"]*len(names))+" |"
    rows = []
    for record in frame[names].to_dict("records"):
        values = []
        for name in names:
            v = record[name]
            if pd.isna(v):
                v = "N/A"
            elif isinstance(v, float):
                v = f"{v:.6g}"
            values.append(str(v).replace("|", "/").replace("\n", " "))
        rows.append("| "+" | ".join(values)+" |")
    return "\n".join([head, separator, *rows])+"\n"


def render_markdown(
    output: Path, tables: dict[str, pd.DataFrame],
    plots: list[Path], weights: tuple[float, float],
):
    sections = [
        ("Métricas predictivas", "primary_metrics.csv", [
            "model", "rmse_t_ha", "mae_t_ha", "median_ae_t_ha",
            "p95_ae_t_ha", "l1_t_ha", "l2_t_ha", "linf_t_ha",
            "r2_cv", "adjusted_r2", "bias_t_ha", "wape_pct",
        ]),
        ("Métricas robustas, relativas y de calibración",
         "specialized_metrics.csv", [
            "model", "smape_pct", "mape_pct", "rmsle",
            "huber_delta_0p5_t2_ha2", "concordance_ccc",
            "fraction_ae_le_0p25", "fraction_ae_le_0p50",
            "calibration_intercept_t_ha", "calibration_slope",
        ]),
        ("Distribución de residuos y pruebas de normalidad",
         "normality_and_distribution.csv", [
            "model", "skewness_bias_corrected",
            "excess_kurtosis_bias_corrected", "shapiro_w",
            "shapiro_p_exploratory",
            "dagostino_pearson_p_exploratory",
            "anderson_reject_at_5pct",
        ]),
        ("Heterocedasticidad y dependencia del error",
         "heteroscedasticity_and_calibration.csv", [
            "model", "absolute_error_vs_predicted_spearman_rho",
            "koenker_breusch_pagan_lm_diagnostic",
            "koenker_breusch_pagan_asymptotic_p_exploratory",
            "brown_forsythe_3_predicted_terciles_p_exploratory",
        ]),
        ("Contrastes de pérdidas emparejadas",
         "paired_loss_tests.csv", [
            "model", "loss", "mean_loss_difference",
            "fraction_improved", "wilcoxon_p_exploratory",
            "wilcoxon_p_exploratory_holm_all",
        ]),
        ("Autocorrelación espacial global",
         "global_spatial_autocorrelation.csv", [
            "model", "knn", "spatial_variable", "statistic", "estimate",
            "p_two_sided", "p_two_sided_holm_all", "p_two_sided_fdr_bh_all",
        ]),
        ("Semivariograma de errores",
         "residual_semivariogram.csv", [
            "model", "distance_bin", "median_distance_km",
            "semivariance_t2_ha2",
        ]),
        ("Desempeño por región (uso privado)",
         "private_regional_errors.csv", [
            "group", "label", "model", "n", "rmse_t_ha", "mae_t_ha",
            "bias_t_ha",
        ]),
    ]
    lines = [
        "# Informe final de validación LOOCV y diagnóstico estadístico",
        "",
        "Experimento transductivo con 138 parcelas etiquetadas y "
        "59 rendimientos desconocidos. Las 138 predicciones base "
        "son obtenidas excluyendo una etiqueta externa en cada ajuste.",
        "",
        "La fórmula fija es Reference26_mean por "
        f"{weights[0]:.12f} más Compact14 por {weights[1]:.12f}.",
        "",
        "**Advertencia:** estos pesos se seleccionaron a partir de los "
        "mismos 138 rendimientos. La evaluación del ensamble NO es "
        "una validación independiente de selección de pesos. "
        "No se conocen los errores reales en las 59 parcelas objetivo.",
        "",
    ]
    for heading, filename, fields in sections:
        lines += ["## "+heading, "", md_table(tables[filename], fields), ""]
    lines += [
        "## Interpretación metodológica",
        "",
        "- Error firmado = predicción menos rendimiento real. "
        "Sesgo positivo significa sobreestimación.",
        "- L1 y L2 son normas vectoriales totales dependientes de n; "
        "L infinito es el máximo error absoluto. No confundir "
        "con MAE o RMSE, que son promedios normalizados.",
        "- R2 predictivo compara la SSE de predicciones externas con "
        "la varianza observada total; no es porcentaje de aciertos.",
        "- R2 ajustado no es defendible sin grados de libertad "
        "efectivos bien definidos para esta arquitectura.",
        "- Shapiro, D'Agostino y Anderson evalúan compatibilidad "
        "con normalidad en errores LOOCV agrupados; no prueban "
        "independencia ni hacen que RMSE sea válido o inválido.",
        "- Jarque–Bera estadístico puede mostrarse, pero su "
        "p-valor asintótico no es fiable para n=138.",
        "- Pruebas de heterocedasticidad y Wilcoxon tienen "
        "hipótesis de independencia que no se cumplen "
        "necesariamente para ajustes externos superpuestos.",
        "- Moran I y Geary C usan distancias geográficas reales "
        "y permutaciones. Holm controla familias de "
        "comparaciones bajo condiciones habituales; FDR-BH "
        "es una sensibilidad y puede requerir suposiciones "
        "sobre la dependencia de los p-valores.",
        "- Un residuo espacialmente autocorrelacionado revela "
        "estructura residual, pero no identifica causalmente "
        "qué variable geográfica la produjo.",
        "- Los análisis estatales/municipales y las pruebas "
        "basadas en los mismos datos siguen siendo exploratorios. "
        "Para extrapolación espacial: validar dejando bloques "
        "geográficos completos fuera.",
        "",
        "## Figuras privadas (no publicar automáticamente)",
        "",
    ]
    lines.extend(f"![{p.name}](figures/{p.name})" for p in plots)
    lines += [
        "",
        "Estas figuras contienen puntos individuales y, en los mapas, "
        "coordenadas. Revisar confidencialidad antes de compartirlas.",
        "",
        "## Referencias",
        "",
        "- NIST, diagnóstico de residuos: "
        "https://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm",
        "- SciPy, Shapiro: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html",
        "- SciPy, Jarque–Bera: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html",
        "- PySAL, autocorrelación espacial: "
        "https://pysal.org/esda/stable/user-guide/global_morans_i.html",
        "- Breiman (1996), Stacked Regressions: "
        "https://doi.org/10.1023/A:1018046112532",
        "",
    ]
    path = output / "INFORME_ESTADISTICO_FINAL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (output / "academic_figures_manifest.json").write_text(
        json.dumps({
            "private_figures": [p.name for p in plots],
            "n_holdout_parcels": 138,
            "not_for_automatic_publication": True,
        }, indent=2)+"\n", encoding="utf-8",
    )
    return path
