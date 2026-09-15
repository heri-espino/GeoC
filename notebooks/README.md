# Notebooks

Los notebooks son para exploración, visualización y experimentos iniciales. La lógica estable y reutilizable debe vivir en `src/geocebada/` y consumirse como una librería normal de Python.

## Primer notebook recomendado

Abrir y ejecutar de arriba hacia abajo:

```text
notebooks/01_visualizacion_datos.ipynb
```

Está pensado para compañeros que trabajan principalmente desde notebooks. La primera celda localiza automáticamente la raíz del repositorio y ejecuta una instalación editable equivalente a:

```bash
pip install -e .
```

Después carga el split oficial, BASIC y PRO usando la librería `geocebada`, muestra un resumen del dataset, visualizaciones del target, missingness, un pairplot triangular ligero con puntos rasterizados, una vista descriptiva por parcela para abril–octubre de 2025 y el contenido actualizado de `docs/VARIABLES.md`.

El notebook se versiona **sin outputs** para no inflar el repositorio. Cada usuario genera las figuras localmente al ejecutarlo.

## Preparación manual alternativa

Si se prefiere instalar el proyecto una sola vez desde el root del repositorio:

```bash
pip install -e ".[dev,geo]"
```

Después, cualquier notebook puede importar utilidades sin modificar `sys.path` ni copiar funciones:

```python
from geocebada.data import load_yield_split, partition_yield_split
from geocebada.paths import source_path

split = load_yield_split()
train, prediction = partition_yield_split(split)
```

Para descubrir funciones existentes antes de crear una nueva, consultar `docs/FUNCTION_INDEX.md` y la referencia Sphinx en `docs/api/`.

## Convención sugerida

```text
01_visualizacion_datos.ipynb
02_data_audit.ipynb
03_feature_engineering.ipynb
04_baselines.ipynb
05_model_comparison.ipynb
06_explainability.ipynb
07_final_predictions.ipynb
```

## Regla de promoción a librería

Si una función se utiliza en más de un notebook, en la app, o forma parte del pipeline reproducible, moverla a `src/geocebada/`, documentarla y cubrirla con pruebas cuando su comportamiento sea estable.

No dejar como única implementación importante una celda de notebook. Evitar que el resultado final dependa de ejecutar notebooks manualmente en un orden implícito.
