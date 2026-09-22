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
    │   ├── checkpoint_05.tex
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

El entry point recomendado es:

    python reporte/tecnico/build.py

El builder valida primero que existan todos los \\input, las figuras congeladas referenciadas y
las claves bibliográficas, y luego ejecuta el comando LaTeX oficial:

    latexmk -pdf -interaction=nonstopmode main.tex

Para validar sin compilar:

    python reporte/tecnico/build.py --check

El builder no recalcula modelos, tablas científicas ni figuras. Los PNG ya versionados bajo
reports/ se consideran inputs congelados del documento.

La alternativa manual sigue siendo:

    cd reporte\tecnico
    pdflatex main.tex
    biber main
    pdflatex main.tex
    pdflatex main.tex

## GitHub Actions

La compilación del reporte es manual. En GitHub:

    Actions
    → Build research outputs
    → Run workflow
    → target = paper

Las figuras ya fueron generadas durante los checkpoints y están versionadas. El workflow no
vuelve a ejecutar 04A, 04B, 04D.1, 04E.1 ni ningún otro experimento científico.

- target = figures: empaqueta las figuras y summaries congelados que ya existen;
- target = paper: valida esos inputs y compila el PDF;
- target = all: empaqueta las figuras existentes y compila el PDF.

La compilación usa python reporte/tecnico/build.py; regenerar una figura requiere ejecutar
explícitamente el runner científico correspondiente fuera de este build documental.

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


## Referencias

El reporte usa BibLaTeX/Biber y la bibliografía canónica es:

    reporte/tecnico/references.bib

Incluye referencias metodológicas (Ridge, PLS, ExtraTrees, CatBoost, aprendizaje
semi-supervisado, Moran), literatura agronómica/de teledetección de cebada y referencias de
fuentes de datos (FIRA, CHIRPS, SoilGrids, WaPOR, SIAP e INEGI). Las citas se insertan en el
texto con `\citep{...}` / `\citet{...}`, y `build.py --check` falla si una clave citada
no existe en el archivo BibTeX.
