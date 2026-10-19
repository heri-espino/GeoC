# Reference26 — LOOCV de imputacion espacial y temporal

Experimento independiente de la busqueda Ridge de 250 configuraciones. Compara el mismo modelo Reference26 con los parametros originales en exactamente 138 folds leave-one-parcel-out (137 etiquetas de entrenamiento y una ocultada en cada fold).

## Cuatro alternativas

| Metodo | Imputacion de los predictores X faltantes |
|---|---|
| median_original | Reference26 original, medianas ajustadas exclusivamente con datos de entrenamiento; utiliza las 138 predicciones LOOCV guardadas |
| spatial | Vecinos por distancia geografica normalizada del bundle y preferencia por mismo municipio/estado |
| temporal | Observaciones BASIC/PRO de la misma parcela y sensor, ponderadas por diferencia de fecha y nubosidad |
| spatiotemporal | Observaciones fechadas propias y de ocho vecinos espaciales; para columnas no reconstruibles temporalmente, relleno espacial |

El experimento no modifica valores X que ya estaban observados, las coordenadas, las proporciones geologicas, las categorias, ni los rendimientos. Las imputaciones son exclusivamente X y pueden emplear las 138 X disponibles bajo protocolo transductivo; todo y excluido se enmascara antes de entrenar. No se imputa y ni se usa la etiqueta de las 59 parcelas de prediccion.

La mediana original permanece como control congelado. Si uno de los tres metodos nuevos no cambia ningun valor X, reutiliza las predicciones originales sin nuevos ajustes.

## Datos temporales necesarios

El bundle privado GeoC/data_n/processed/current_champions/v1 conserva solo variables resumidas. Para fechas reales, el codigo requiere los archivos originales:

- data/source/tabular/Conjunto_datos_BASICO_AgroCebada2026.csv
- data/source/tabular/Conjunto_datos_PRO_AgroCebada.csv

Los metodos temporales reconstruyen solo seis tipos de entrada cuando el resumen original esta ausente: NDVI Sentinel-2 de mayo 2025; media/desviacion de NDVI historico Sentinel-2 de abril-octubre 2022–2024; media/desviacion de NDWI Sentinel-2 de abril-octubre 2025; y cambio NDVI Planet junio-agosto de 2025. No se inventan formulas originales para variables emp_shape o I_*, ni se mezclan sensores.

Cada estimacion de fecha utiliza observaciones a no mas de 35 dias de la fecha objetivo y dentro del mismo ciclo abril-octubre; las ventanas temporales historicas se mantienen dentro del anio. Las medias/desviaciones calculadas sobre cuadrillas de 14 dias son **aproximaciones** a los resumenes faltantes y pueden no reproducir exactamente el procedimiento original.

Las distancias son las distancias normalizadas guardadas por el modelo de la companera, NO kilometros. El algoritmo espacial prioriza municipio y luego estado; esto no significa que se use el rendimiento del vecino como imputacion.

## Ejecutar en PowerShell, desde GeoC

~~~
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Primero medir cuantas celdas X son recuperables, SIN entrenar modelos
python tools/run_reference26_imputation_loo.py --check

# Una parcela para verificar los tres nuevos metodos
python tools/run_reference26_imputation_loo.py --smoke --jobs 1

# Los 138 folds completos; cualquier numero de workers positivo
python tools/run_reference26_imputation_loo.py --run --jobs 12

# Resultados sin reentrenar
python tools/run_reference26_imputation_loo.py --summary
~~~

Si los archivos originales estan en otra carpeta, usar --basic RUTA y --pro RUTA en TODOS los comandos. El experimento no descarga datos ni promete que esos archivos esten disponibles localmente.

Los resultados quedan bajo reports/reference26_imputation_loo138, excluido de Git:
- imputation_audit_aggregates.csv: numero de celdas faltantes y recuperadas por metodo/variable, sin IDs de parcelas
- run_spec.json: hashes de codigo, manifest y datos crudos, para evitar mezclar modelos
- fold_XXX/metodo.csv: predicciones privadas de cada fold (permiten reanudar)
- loo138_summary.csv: cuatro RMSE, MAE, sesgo, R² y diferencia en MSE frente a mediana
- loo138_predictions_private.csv: IDs, rendimientos observados y predicciones; NUNCA publicar
- smoke/: resultado independiente de un fold, nunca mezclado con los 138

**Interpretacion.** El mejor RMSE de cuatro estrategias sirve para explorar mejoras, pero elegir un ganador con esos mismos 138 y produce optimismo de seleccion. No se debe interpretar como validacion independiente. Imputar mas celdas no garantiza reducir RMSE. Un seguimiento util es enmascarar observaciones satelitales existentes y medir tambien su error de reconstruccion.
