# Checkpoint 09 — resultados públicos ligeros

Este directorio comparte únicamente **métricas agregadas y la decisión** de la
corrida local de Checkpoint 09 del 12 de octubre de 2026.

- `development_summary.csv`: transcripción de la salida `--run`/`--summary`
  proporcionada por el operador, **redondeada a seis decimales** como se mostró
  en la terminal. No se afirma identidad byte a byte con el CSV local.
- `decision.json`: reproducción del bloque JSON completo compartido por el
  operador; los campos `*_pooled` conservan la precisión impresa.
- Descripción, interpretación y cierre: `docs/CHECKPOINT_09_FINDINGS.md`.

**Integridad y privacidad:** no se suben a Git predicciones por parcela
(`development_predictions.csv`, `loso_selected_predictions.csv`), la lista
`human_label_review_candidates.csv`, archivos auxiliares ni máscaras locales.
El listado de parcelas con rendimientos supuestamente problemáticos **no
constituye una identificación de registros erróneos**.

No hubo ejecución de confirmación 09 porque `LOSO_STABILITY_GATE = FAIL`.
La predicción oficial de 59 parcelas sigue en
`reports/checkpoint_05/final_predictions.csv`, sin cambios.

Fuente: stdout pegado en la conversación por quien ejecutó el experimento;
para auditar a precisión completa o comprobar salidas exactas, comparar con
los ficheros locales generados por `tools/run_checkpoint_09_robust.py`.
