# Reference26 — aritmética media: búsqueda Ridge finalizada

**Resultado reportado por usuario desde su VM el 2026-10-19.**
138 parcelas excluidas, 137 etiquetas utilizadas por ajuste. Este
documento conserva únicamente métricas agregadas; no incluye IDs,
rendimientos observados o predicciones individuales.

## LOOCV final (las mismas 138 parcelas)

| Modelo | Imputación | Vecinos k | Ridge alpha | RMSE (t/ha) | MAE (t/ha) | Sesgo (t/ha) |
|---|---|---:|---:|---:|---:|---:|
| Reference26 original | mediana | 24 | 30 | 0.4681855146560597 | 0.3336730043055529 | 0.002374634259229654 |
| Reference26 (mejor finalista) | media | 24 | 30 | **0.468138** | 0.333786 | 0.001466 |
| Reference26 (finalista) | media | 20 | 30 | 0.470112 | 0.336173 | -0.000110 |
| Reference26 (finalista) | media | 12 | 3.2 | 0.474493 | 0.339729 | 0.007165 |

La media con la configuración original mejora el RMSE frente a la mediana
en ~0.000048 t/ha, pero el MAE empeora ligeramente. No se declara
mejora estadísticamente robusta. Los demás finalistas de la malla
reducida empeoraron RMSE en los 138 heldouts.

## Protocolo de cribado previo

- 25 configuraciones de (vecinos geográficos, Ridge alpha) evaluadas
  sobre las **primeras 12 parcelas**; líder k20/alpha3.2,
  RMSE=0.330720 vs k24/alpha30 media=0.342561.
- Ocho configuraciones avanzaron a **las primeras 46 parcelas**;
  líder k12/alpha3.2, RMSE=0.541737 vs
  k24/alpha30 media=0.547385.
- Tres configuraciones avanzaron al total de 138;
  finalmente lideró k24/alpha30 media=0.468138.

Las clasificaciones cambiaron sustancialmente al ampliar la muestra.
Por tanto, el cribado inicial es exploratorio, y el RMSE mínimo tras
selección adaptativa sigue siendo optimista como estimador independiente.

## Archivos de origen (solo en VM)

- `reports/reference26_ridge_budget_mean/screen_scores.csv`
- `reports/reference26_ridge_budget_mean/advance_scores.csv`
- `reports/reference26_ridge_budget_mean/final_scores.csv`
- `reports/reference26_ridge_budget_mean/fold_001..138/k24_a30.csv`
- `reports/selected_champions_loo/loo_predictions.csv` (mediana original)

Los valores fueron copiados de las salidas mostradas por el usuario.
Los scripts de comparación deben leer de esos archivos locales,
verificar sus huellas e igualar IDs y y observados; **no** sustituirlos
por estas cifras redondeadas para una prueba emparejada de ensambles.

## Trabajo siguiente

`tools/run_loo_ensemble_screen.py`: comparar Reference26 con media
contra Local04D, Compact14, DBSCAN y otros **solo si** sus pronósticos
son auténticas predicciones de las mismas 138 parcelas LOOCV.
Primero medir complementariedad de errores y mezclas convexas
predeterminadas; no entrenar todavía un meta-modelo complejo.
