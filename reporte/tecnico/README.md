# Reporte técnico LaTeX — GeoCebada

Este directorio contiene el reporte técnico integral del proyecto GeoCebada. Está pensado como
documento interno de ciencia de datos: prioriza formulación matemática, trazabilidad,
protocolos de validación, prevención de leakage, hipótesis descartadas y decisiones de cierre.

## Estilo

El documento usa el estilo local:

    reporte/tecnico/velvetblue.sty

con las opciones:

    \usepackage[general,final,compact]{velvetblue}

Esto activa Libertinus, bibliografía autor-año con BibLaTeX/Biber, hyperlinks VelvetBlue y
márgenes A4 compactos de 1.8 cm. El archivo de estilo se versiona junto al reporte para que
GitHub Actions y los entornos locales compilen exactamente la misma versión visual.

## Estructura

    reporte/tecnico/
    ├── main.tex
    ├── preamble.tex
    ├── velvetblue.sty
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
        ├── catalogo_features.tex
        ├── protocolos_validacion.tex
        ├── resultados_negativos.tex
        └── predicciones_finales.tex

main.tex define cuatro capítulos principales, uno por Checkpoint 01–04. Cada subfase se
materializa mediante \input{...} para que el documento se pueda revisar y mantener sin
convertir el archivo principal en un bloque monolítico.

## Compilación oficial

Desde la carpeta del reporte:

    cd reporte\tecnico
    latexmk -pdf -interaction=nonstopmode main.tex

latexmk detecta BibLaTeX y ejecuta Biber cuando es necesario.

Alternativa manual:

    pdflatex main.tex
    biber main
    pdflatex main.tex
    pdflatex main.tex

Las figuras se referencian mediante rutas relativas hacia ../../reports/...; por ello se
recomienda compilar con el directorio de trabajo en reporte/tecnico/.

## GitHub Actions

La compilación del reporte es manual. En GitHub:

    Actions
    → Build research outputs
    → Run workflow
    → target = paper

El target all regenera primero las figuras reproducibles respaldadas por el repositorio y
después compila el reporte. El target figures ejecuta los runners oficiales de 04A, 04B y
04D.1. Las figuras SIAP de 04E.1 se empaquetan desde los outputs congelados ya versionados,
porque la fuente raw externa de SIAP no forma parte del checkout de GitHub.

Los artifacts se conservan 30 días y llevan el SHA corto del commit en el nombre. El workflow
no hace commits automáticos.

## Fuente de verdad

Este texto resume artefactos ya congelados del repositorio. Si alguna cifra del documento
contradice un CSV/JSON generado por el runner correspondiente, el artefacto generado y su
configuración son la fuente de verdad.

La tabla canónica de predicciones finales permanece en:

    reports/checkpoint_04f/final_predictions.csv

El apéndice del reporte muestra valores redondeados sólo para lectura. No reemplaza el CSV de
precisión completa.
