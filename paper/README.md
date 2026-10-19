# Informe técnico AgroCebada FIRA 2026

Borrador científico en español, independiente de reporte/tecnico/. El documento principal es paper/main.tex y sus siete secciones se editan en paper/sections/. Incluye la bibliografía íntegra de 32 referencias canónicas del proyecto, ampliada con referencias metodológicas, y un anexo de máximo dos páginas sujeto a revisión visual.

Para compilar desde paper/, ejecutar:
    pdflatex main.tex
    biber main
    pdflatex main.tex
    pdflatex main.tex

Fecha de corte de resultados: 19 de octubre de 2026. El RMSE 0.462418 pertenece a una regla fija seleccionada después de ver los 138 rendimientos; el valor 0.462892 corresponde a la selección interna condicionada a candidatos ya explorados. No representan una evaluación independiente de toda la investigación ni el error observado de las 59 parcelas ocultas. El ensayo disponible sobre 46 parcelas favoreció DBSCAN sobre K-means, por lo cual la permanencia de K-means no debe atribuirse a superioridad demostrada.

Fuentes de las métricas: documentación de la rama analysis/selected-analog-spatial-inference-20261019 y registros locales aportados el 2026-10-19. Revisar las tablas y los límites de palabras contra los archivos privados y las bases oficiales antes del envío. Los datos parcelarios no se incluyen.
