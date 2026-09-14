# Datos

La carpeta `data/` separa las fuentes oficiales de los artefactos generados por el pipeline.

- `source/`: archivos originales proporcionados por FIRA, versionados y sin modificar.
  - `tabular/`: conjuntos BÁSICO/PRO y tabla `ID_area_rendimiento_70_30`.
  - `geospatial/`: parcelas georreferenciadas.
  - `climate/precipitation/`: precipitación CHIRPS 2022–2025.
  - `climate/temperature/`: temperatura 2022–2025.
  - `topography/`: topografía INEGI CEM 4.
- `raw/`: datos de trabajo obtenidos al ingerir/descomprimir/normalizar fuentes; no versionados.
- `interim/`: resultados de limpieza, uniones y transformaciones intermedias; no versionados.
- `processed/`: matrices/tablas finales listas para entrenamiento y evaluación; no versionados.

## Reglas

1. No editar manualmente ningún archivo dentro de `data/source/`.
2. Toda transformación debe ser reproducible desde código en `src/geocebada/`.
3. Conservar `ID_parcela` como llave estable durante todo el pipeline.
4. No mezclar predicciones, modelos entrenados o resultados con las fuentes oficiales.
5. Las predicciones finales deben poder vincularse inequívocamente con `ID_parcela` y reportarse en ton/ha.
