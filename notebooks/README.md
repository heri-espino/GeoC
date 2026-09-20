# Notebooks

> Estado actual: Checkpoint 04 está abierto. Para modelado nuevo, leer `docs/TRANSDUCTIVE_OBJECTIVE.md` y `docs/AGENT_GUIDE.md`. Las 59 X objetivo pueden usarse en análisis X-only transductivo; los 59 y ocultos no.

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
conda env create -f environment.dev.yml
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python -m ipykernel install --user --name geocebada --display-name "Python (GeoCebada)"
jupyter lab
```

`environment.dev.yml` es sólo para desarrollo local. El despliegue de Streamlit usa `requirements.txt`.

## 01 — Visualización rápida

Abrir y ejecutar de arriba hacia abajo:

```text
notebooks/01_visualizacion_datos.ipynb
```

Está pensado para compañeros que trabajan principalmente desde notebooks. La primera celda localiza automáticamente la raíz del repositorio y ejecuta una instalación editable defensiva. Después carga el split oficial, BASIC y PRO usando la librería `geocebada`, muestra un resumen del dataset, visualizaciones del target, missingness, un pairplot triangular ligero con puntos rasterizados, una vista descriptiva por parcela para abril–octubre de 2025 y el contenido actualizado de `docs/VARIABLES.md`.

## 02 — Cobertura y alineación temporal

```text
notebooks/02_cobertura_alineacion_temporal.ipynb
```

Esta notebook estudia la cobertura temporal de BASIC/PRO y compara `ENTRENAMIENTO` contra `PREDICCION` sin usar targets ocultos. Después alinea capturas irregulares mediante k vecinos temporales, ponderación exponencial por distancia y, opcionalmente, nubosidad.

Por defecto trabaja con las columnas `*_promedio` y una rejilla cada 14 días. Puede ampliarse a todas las estadísticas. Los sensores permanecen separados.

En una máquina NVIDIA/CUDA compatible puede instalarse el extra GPU:

```bash
python -m pip install -e ".[dev,geo,gpu]"
```

y usar `BACKEND="auto"` o `BACKEND="gpu"`. La aceleración usa CuPy para las operaciones matriciales; Pandas y los groupby siguen en CPU, así que conviene medir CPU vs GPU en la máquina real.

La metodología está documentada en `docs/TEMPORAL_ALIGNMENT.md`.

Si ya seguiste `docs/GETTING_STARTED.md`, las instalaciones editables incluidas al inicio de los notebooks son redundantes pero inocuas.

Los notebooks se versionan **sin outputs** para no inflar el repositorio. Cada usuario genera las figuras localmente al ejecutarlos.

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

Los notebooks 01–02 ya existen y son exploratorios. La Feature Table v1 y los baselines de Checkpoint 02 ya se construyeron por scripts reproducibles, así que no se deben recrear como única lógica en notebooks. Para Checkpoint 04 pueden añadirse notebooks de diagnóstico de similitud, autocorrelación, covariate shift y trayectorias temporales, pero toda lógica estable de validación/modelado debe promoverse a `src/geocebada/`.

## Regla de promoción a librería

Si una función se utiliza en más de un notebook, en la app, o forma parte del pipeline reproducible, moverla a `src/geocebada/`, documentarla y cubrirla con pruebas cuando su comportamiento sea estable.

No dejar como única implementación importante una celda de notebook. Evitar que el resultado final dependa de ejecutar notebooks manualmente en un orden implícito.
