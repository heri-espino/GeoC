# Mini muestras reales de datos

Esta carpeta contiene artefactos pequeños y deterministas derivados de los datos reales
para desarrollar, depurar y probar loaders, joins y feature engineering sin leer siempre
las fuentes completas.

**No son un dataset de entrenamiento**, no preservan las distribuciones estadísticas y no
deben usarse para estimar métricas ni seleccionar modelos.

## Fixtures estructurales

Las carpetas existentes:

```text
tabular/
vector/
raster/
netcdf/
```

se generan con:

```powershell
python tools\build_data_catalog.py
```

Sirven para probar formatos/esquemas individuales. Sus filas o recortes no están
necesariamente relacionados entre sí y **no deben usarse para probar joins end-to-end**.

El catálogo asociado está en `data/data_catalog.json`.

## Fixtures de integración

`integration/` se genera por separado con:

```powershell
python tools\build_integration_fixtures.py
```

Este conjunto usa los **mismos 12 `ID_POLIGONO`** a través de split, geometrías, BASIC,
PRO y administración. La selección toma dos parcelas por estrato
`Estado × CONJUNTO`, buscando diversidad de área, nubosidad, missingness y cobertura
temporal sin utilizar `RENDIMIENTO_T_HA`.

Se fuerzan además tres casos espaciales conocidos para que las pruebas incluyan bordes y
discrepancias administrativas:

- `AGC_020`
- `AGC_048`
- `AGC_129`

El fixture de integración contiene:

```text
integration/
  fixture_manifest.json
  selected_parcels.csv
  split.csv
  parcels.geojson
  basic.csv
  pro.csv
  admin.csv
  municipalities.geojson
  siap.csv
  climate/
  chirps_daily/
  topography/
  soilgrids/
  wapor/
  cem/
```

Los rásteres se enmascaran alrededor de las parcelas seleccionadas y se comprimen. Los
NetCDF WaPOR se copian completos porque ya son pequeños y conservan sus 21 bandas
dekadales.

## Invariantes

`tools/build_integration_fixtures.py` valida al terminar que:

- los 12 IDs sean idénticos en split, geometría, BASIC, PRO y administración;
- split, geometría y administración tengan una fila por parcela;
- las 59 etiquetas ocultas nunca se expongan accidentalmente;
- SIAP utilice los municipios de las parcelas seleccionadas;
- los fixtures sean reproducibles desde las fuentes completas locales.

La selección/configuración está congelada en `configs/data_contract_v2.yaml`.
