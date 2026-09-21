# Reporte técnico LaTeX — GeoCebada

Este directorio contiene el reporte técnico integral del proyecto GeoCebada. Está pensado como
documento interno de ciencia de datos: prioriza formulación matemática, trazabilidad,
protocolos de validación, prevención de leakage, hipótesis descartadas y decisiones de cierre.

## Estructura

    reporte/tecnico/
    ├── main.tex
    ├── preamble.tex
    ├── references.bib
    ├── frontmatter/
    │   ├── resumen.tex
    │   └── contrato_metodologico.tex
    ├── sections/
    │   ├── checkpoint_01a.tex
    │   ├── ...
    │   ├── checkpoint_04f.tex
    │   └── sintesis_final.tex
    └── appendices/
        ├── notacion_metricas.tex
        ├── reproducibilidad.tex
        └── predicciones_finales.tex

main.tex define cuatro capítulos principales, uno por Checkpoint 01–04. Cada subfase se
materializa mediante \input{...} para que el documento se pueda revisar y mantener sin
convertir el archivo principal en un bloque monolítico.

## Compilación recomendada

Desde la carpeta del reporte:

    cd reporte\tecnico
    latexmk -pdf -interaction=nonstopmode main.tex

Alternativa sin latexmk:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

Las figuras se referencian mediante rutas relativas hacia ../../reports/...; por ello se
recomienda compilar con el directorio de trabajo en reporte/tecnico/.

## Fuente de verdad

Este texto resume artefactos ya congelados del repositorio. Si alguna cifra del documento
contradice un CSV/JSON generado por el runner correspondiente, el artefacto generado y su
configuración son la fuente de verdad.

La tabla canónica de predicciones finales permanece en:

    reports/checkpoint_04f/final_predictions.csv

El apéndice del reporte muestra valores redondeados sólo para lectura. No reemplaza el CSV de
precisión completa.
