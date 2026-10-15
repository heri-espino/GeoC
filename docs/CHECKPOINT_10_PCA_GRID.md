# Checkpoint 10 — Un único grid de PCA, vecinos, Ridge y distancia

**Estado:** implementado, pendiente de ejecución completa sobre los datos locales de la estación de trabajo.  
**Objetivo:** comprobar si Local04D mejora cuando se aumenta el número de componentes principales de su representación **C4** y se ajustan conjuntamente la vecindad, regularización y exponente espacial.

Esta pregunta **no** quedó resuelta en Checkpoints 04D ni 08: ambos utilizaron una representación C4 de **20 componentes**. Checkpoint 07 probó PCA de 16/32/64 componentes sobre una matriz HLS/SMAP **diferente**, principalmente para Local07 y CatBoost; no constituye la comparación solicitada.

## 1. Diseño

Se utiliza la representación original `C4_all_deterministic` del proyecto: **2 082 variables** numéricas procedentes de las capas base, agronómica y empírica determinista, excluyendo las expresiones aprendidas supervisadamente.

Las 2 082 variables pasan por imputación mediana, estandarización y PCA calculados con las \(X\) de las 197 parcelas —la misma regla transductiva X-only de Checkpoint 04A—. PCA retiene hasta 160 componentes; se usan sus prefijos **20, 40, 80 y 160**, y cada prefijo se estandariza otra vez como se hacía con el embedding C4 original.

| Hiperparámetro | Grid |
|---|---|
| Componentes PCA | **20, 40, 80, 160** |
| Vecinos geográficos \(k\) | **12, 20, 24, 36** |
| Regularización Ridge \(\alpha\) | **10, 30, 100, 300** |
| Potencia de distancia \(p\) | **0.5, 1, 2** |

Total: **4 × 4 × 4 × 3 = 192 combinaciones** por split externo; es una rejilla exhaustiva de tamaño acotado, no Optuna ni un conjunto ilimitado de hiperparámetros.

El control fijo es el Local04D original: **PCA20 + \(k=24,\alpha=30,p=1\)**. El número de vecinos es pequeño frente a 80/160 componentes; Ridge lo permite, pero sus penalizaciones más grandes pueden ser importantes. La elección se decide por CV, no por varianza explicada.

## 2. Validación cruzada anidada y alcance del resultado

**Un solo protocolo:** se reutilizan las **16 máscaras X-only de confirmación ya congeladas en Checkpoint 08**, que dividen las 138 parcelas etiquetadas en **97 pseudo-entrenamiento / 41 pseudo-objetivos**.

Dentro de cada grupo de 97 rendimientos se corre un **KFold de 3 partes**:

1. Para las **192 configuraciones**, el entrenamiento y los errores del fold interno usan solamente esos 97 rendimientos.
2. Para cada dimensión de PCA (20, 40, 80, 160), se selecciona la mejor triple combinación \((k,\alpha,p)\) por RMSE interno.
3. También se selecciona **una configuración conjunta ganadora** entre los 192 candidatos usando exclusivamente el RMSE interno.
4. Se ajusta la regla seleccionada con los 97 rendimientos y se pronostican, una sola vez, los 41 reservados externos.
5. Se calculan RMSE y \(R^2\) *pooled*, RMSE medio y peor split; todas las comparaciones usan los **mismos 16 × 41 = 656 pseudo-objetivos repetidos**.

Se guardan **6 métodos emparejados** en cada split: `PCA20_frozen`, `PCA20_nested`, `PCA40_nested`, `PCA80_nested`, `PCA160_nested` y `PCA_joint_nested`.

Como los rendimientos de las 138 parcelas de origen y estas máscaras **ya se utilizaron** en Checkpoint 08, esto constituye una **exploración posterior**, no una nueva confirmación estadísticamente independiente. La validación interna evita seleccionar hiperparámetros mirando los 41 rendimientos de *su propio split externo*, pero no borra toda la historia previa de selección. Si resultase un candidato claramente superior, se requeriría validación con nuevos datos o un banco genuinamente independiente antes de reemplazar oficialmente las 59 estimaciones.

Las máscaras no se regeneran; su existencia y sus 16×138 membresías se verifican antes de iniciar. No hay acceso a los rendimientos ocultos de las 59 parcelas reales.

## 3. Controles del método

- **Sin doble trabajo sobre nuevas imágenes:** utiliza las tablas C4 ya disponibles y los manifiestos del 04A; no exige reconstruir HLS/SMAP/Prithvi.
- **Paridad del control:** antes de puntuar, reproduce con tolerancia numérica estricta los 20 PCs almacenados en el 04A. Si existe el archivo emparejado del Checkpoint 08, también comprueba que el control PCA20 reproduzca **exactamente sus pronósticos** de `confirm_01`.
- **Misma geografía:** conserva las distancias de Haversine normalizadas y el algoritmo original de Ridge local ponderado.
- **Sin fuga supervisada:** PCA utiliza únicamente \(X\); cada fold interno sólo accede a sus propios \(y\) de entrenamiento y validación interior.
- **Reanudación:** guarda un CSV de rejilla interna, pronósticos emparejados y JSON de selección por split. Huellas SHA-256 de datos, código, parámetros y membresías impiden mezclar corridas incompatibles.
- **Datos protegidos:** los detalles individuales con rendimientos reportados permanecen en `reports/checkpoint_10_pca/` (gitignored). Sólo compartir posteriormente tablas agregadas cuidadosamente revisadas.
- **Sin promoción:** no reentrena ni cambia `reports/checkpoint_05/final_predictions.csv` ni predicciones oficiales.

## 4. Ejecución en PowerShell

Desde la raíz del repositorio, con `geocebada` activo y los archivos C4 del 04A disponibles:

```powershell
git switch main
git pull --ff-only origin main
conda activate geocebada

# Verificar fuentes, 16 máscaras e identidad del PCA20.
python tools/run_checkpoint_10_pca.py --check

# Prueba aislada: 1 split × 4 PCA × 2 vecinos (alpha30, p1)
python tools/run_checkpoint_10_pca.py --smoke

# El experimento completo: 16 splits × 192 configuraciones × 3 folds internos
python tools/run_checkpoint_10_pca.py --run

# Sólo consolidar cuando se hayan completado los 16 splits
python tools/run_checkpoint_10_pca.py --summary
```

Una interrupción del proceso no exige borrar resultados; ejecuta **el mismo comando `--run`** desde el mismo commit, versión de datos y opciones. Si cambian las fuentes o el código se requiere un **nuevo directorio `--output`**, no mezclar tablas o estudios anteriores. No lanzar dos ejecuciones a la vez. El script funciona en CPU; no hace falta GPU.

## 5. Qué se reportará después de la ejecución

| Archivo local | Contenido |
|---|---|
| `comparison_summary.csv` | RMSE pooled, \(R^2\) pooled, RMSE promedio, peor split, recuentos |
| `paired_cluster_bootstrap.csv` | \(\Delta\)MSE respecto de PCA20 fijo e IC 95 %; bootstrap agrupado por parcela |
| `inner_cv_grid_summary.csv` | RMSE interno medio/SD para cada **cuádrupla** (PCA, k, alfa, p) |
| `selected_hyperparameters.csv` | Configuración conjunta elegida dentro de cada uno de los 16 splits |
| `confirm_NN__inner_grid.csv` | 192 valores de RMSE interno del split; mantiene los 4 parámetros |
| `confirm_NN__paired.csv` | Predicciones externas de los 6 métodos en las mismas 41 parcelas |
| `confirm_NN__selection.json` | Selección interna específica para cada dimensión y la búsqueda conjunta |

La comparación principal para una decisión será `PCA_joint_nested` frente a `PCA20_frozen`, *no* el mejor RMSE externo escogido a posteriori entre las cuatro curvas. El bootstrap por parcela trata las predicciones repetidas del mismo `ID_POLIGONO` como un bloque; no garantiza independencia espacial entre parcelas.

**Referencias:** [modelo Local04D de Checkpoint 04D](CHECKPOINT_04D1_FINDINGS.md), [comparación simétrica 08](CHECKPOINT_08_NESTED_LOCAL.md), [resumen metodológico](METODOLOGIA_Y_RESULTADOS.md).
