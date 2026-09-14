# Datos

La carpeta `data/` sigue una separación estricta por etapa del pipeline.

- `raw/`: archivos originales descargados de FIRA o fuentes externas. No editar.
- `interim/`: resultados de limpieza, uniones y transformaciones intermedias.
- `processed/`: matrices/tablas finales listas para entrenamiento y evaluación.

Los contenidos de estas carpetas están ignorados por Git. Esto permite trabajar localmente con el dataset sin publicarlo accidentalmente.

## Regla principal

Toda transformación debe ser reproducible desde código. Evitar editar CSV/XLSX manualmente.

## Identificador

Conservar `ID_parcela` como llave estable durante todo el pipeline. Las predicciones finales deberán poder vincularse inequívocamente con este identificador.
