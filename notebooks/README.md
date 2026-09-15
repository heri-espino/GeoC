# Notebooks

Los notebooks son para exploración, visualización y experimentos iniciales. La lógica estable y reutilizable debe vivir en `src/geocebada/` y consumirse como una librería normal de Python.

## Si es tu primera vez en el proyecto

Sigue primero el tutorial completo:

```text
docs/GETTING_STARTED.md
```

Ese tutorial supone que ya tienes Conda instalado y deja a todo el equipo trabajando con:

- ambiente Conda `geocebada`;
- Python 3.11;
- kernel de Jupyter `Python (GeoCebada)`;
- paquete `geocebada` instalado en modo editable;
- dependencias de notebooks y geoespaciales.

El flujo inicial desde la raíz del repositorio es:

```bash
conda env create -f environment.yml
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python -m ipykernel install --user --name geocebada --display-name "Python (GeoCebada)"
jupyter lab
```

## Primer notebook recomendado

Abrir y ejecutar de arriba hacia abajo:

```text
notebooks/01_visualizacion_datos.ipynb
```

Está pensado para compañeros que trabajan principalmente desde notebooks. La primera celda localiza automáticamente la raíz del repositorio y ejecuta una instalación editable defensiva. Después carga el split oficial, BASIC y PRO usando la librería `geocebada`, muestra un resumen del dataset, visualizaciones del target, missingness, un pairplot triangular ligero con puntos rasterizados, una vista descriptiva por parcela para abril–octubre de 2025 y el contenido actualizado de `docs/VARIABLES.md`.

Si ya seguiste `docs/GETTING_STARTED.md`, la instalación de la primera celda es redundante pero inocua.

El notebook se versiona **sin outputs** para no inflar el repositorio. Cada usuario genera las figuras localmente al ejecutarlo.

## Uso normal

Después de configurar la computadora una vez, normalmente basta con:

```bash
cd GeoCebada
conda activate geocebada
git pull
jupyter lab
```

Dentro de Jupyter selecciona siempre el kernel **Python (GeoCebada)**.

Después, cualquier notebook puede importar utilidades sin modificar `sys.path` ni copiar funciones:

```python
from geocebada.data import load_yield_split, partition_yield_split
from geocebada.paths import source_path

split = load_yield_split()
train, prediction = partition_yield_split(split)
```

Para descubrir funciones existentes antes de crear una nueva, consulta `docs/FUNCTION_INDEX.md` y la referencia Sphinx en `docs/api/`.

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
