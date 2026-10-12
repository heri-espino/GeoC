# Comparaciones publicadas: Checkpoints 07 y 08

El documento que explica **todos los datos, modelos, métodos de validación cruzada, resultados y sus limitaciones** es [Metodología y resultados](../../../docs/METODOLOGIA_Y_RESULTADOS.md).

Los archivos CSV de este directorio contienen únicamente los resúmenes agregados públicos:

| Archivo | Contenido | Banco de validación |
|---|---|---|
| `three_model_summary.csv` | Local04D, CatBoost07, Local07, Prithvi07 y nueve blends fijos | 16 splits históricos de 04B; 656 predicciones repetidas; 125 parcelas únicas |
| `checkpoint08_comparison_summary.csv` | Local04D fijo, Local04D anidado y Local07 anidado | 16 splits nuevos de confirmación; 656 predicciones repetidas; 132 parcelas únicas |
| `checkpoint08_paired_cluster_bootstrap.csv` | Diferencias de MSE frente al Local04D fijo e intervalos bootstrap por parcela | Mismo banco de Checkpoint 08 |

**No mezclar el RMSE 07 con el 08 en un ranking único:** las máscaras pseudo-objetivo son distintas. Los registros de cada CSV se comparan únicamente dentro de su banco correspondiente. **Menor RMSE es mejor; `delta_mse` positivo significa que el rival empeoró.**

**Conclusión:** ninguna de las alternativas promovió un modelo distinto de Local04D. Los 59 pronósticos canónicos siguen en [final_predictions.csv](../../checkpoint_05/final_predictions.csv). No se utilizaron los rendimientos ocultos de las 59 parcelas para calcular estas métricas.

No publicar CSV individuales de predicción por fold, etiquetas, estudios SQLite, rasters, pesos Prithvi o imágenes derivados sin verificar sus permisos de distribución.
