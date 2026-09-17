# Catálogo y mini muestras de datos

Antes de desarrollar features o modelos, GeoCebada mantiene un catálogo machine-readable de la estructura real de los datos y mini fixtures deterministas para prototipado.

## Objetivo

Evitar ciclos de prueba sobre archivos grandes sólo para descubrir nombres de columnas, tipos, CRS, dimensiones o convenciones de archivos. El catálogo no reemplaza a las fuentes completas y las mini muestras no deben usarse para entrenamiento ni evaluación.

## Generación

Con el ambiente `geocebada` activo e instalado con los extras geoespaciales/externos:

```bash
python -m pip install -e ".[dev,geo,external]"
python tools/build_data_catalog.py
```

Esto genera:

- `data/data_catalog.json`: estructura de fuentes oficiales y externas disponibles;
- `data/samples/tabular/`: muestras CSV pequeñas;
- `data/samples/vector/`: GeoJSON pequeños;
- `data/samples/raster/`: ventanas GeoTIFF pequeñas preservando CRS/transform;
- `data/samples/netcdf/`: subconjuntos NetCDF pequeños cuando el backend local permite abrirlos;
- `data/samples/README.md`: contrato de uso.

## Qué registra

Para tablas: columnas, dtypes inferidos, missingness, rangos numéricos y ejemplos de valores en una muestra de perfil. Para vectores: CRS, geometrías, bbox, columnas y dtypes. Para rásteres: CRS, dimensiones, bandas, dtype, NoData, resolución y bbox. Para NetCDF: dimensiones, variables, shapes y dtypes.

Las colecciones grandes no se leen completas para catalogarlas. Se registran todos los nombres de archivos y se inspeccionan representantes de cada colección. El catálogo indica cuántas filas se usaron para inferir el esquema tabular.

## Fuentes conocidas

El script busca:

- tabla oficial de split/rendimiento;
- BASIC;
- PRO;
- parcelas oficiales;
- clima y topografía oficiales;
- INEGI municipios;
- CHIRPS diario;
- WaPOR;
- SoilGrids;
- SIAP local;
- INEGI CEM local.

Una fuente ausente se registra como `available: false` en vez de hacer fallar todo el proceso.

## Uso en desarrollo

Los loaders y nuevas funciones de feature engineering deben probarse primero contra `data/samples/` cuando sea posible. Una vez que la lógica funcione con el contrato pequeño, se ejecuta sobre las fuentes completas.

Si una fuente cambia, vuelve a ejecutar `tools/build_data_catalog.py`, revisa el diff de `data/data_catalog.json` y de las mini muestras, y sólo entonces adapta el pipeline.

Las mini muestras son deliberadamente pequeñas y no preservan la distribución estadística del dataset. No deben usarse para selección de modelos, métricas, inferencia o conclusiones agronómicas.
