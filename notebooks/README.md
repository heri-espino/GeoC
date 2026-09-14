# Notebooks

Los notebooks son para exploración, visualización y experimentos iniciales. La lógica estable y reutilizable debe vivir en `src/geocebada/` y consumirse como una librería normal de Python.

## Preparación

Desde el root del repositorio, instalar una vez en modo editable:

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
01_data_audit.ipynb
02_eda.ipynb
03_baselines.ipynb
04_feature_engineering.ipynb
05_model_comparison.ipynb
06_explainability.ipynb
07_final_predictions.ipynb
```

## Regla de promoción a librería

Si una función se utiliza en más de un notebook, en la app, o forma parte del pipeline reproducible, moverla a `src/geocebada/`, documentarla y cubrirla con pruebas cuando su comportamiento sea estable.

No dejar como única implementación importante una celda de notebook. Evitar que el resultado final dependa de ejecutar notebooks manualmente en un orden implícito.
