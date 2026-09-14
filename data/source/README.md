# Datos fuente

Esta carpeta conserva los archivos proporcionados oficialmente para el Reto AgroCebada FIRA 2026. No deben modificarse manualmente.

- `tabular/`: tablas maestras y archivo de rendimiento 70/30.
- `geospatial/`: archivo de parcelas georreferenciadas.
- `climate/precipitation/`: datos CHIRPS de precipitación 2022–2025.
- `climate/temperature/`: datos de temperatura 2022–2025.
- `topography/`: información topográfica INEGI CEM 4.

Los ZIP originales se conservan junto con sus carpetas extraídas para mantener trazabilidad respecto de los archivos entregados por FIRA. Las transformaciones derivadas deben generarse mediante código y escribirse en `data/interim/` o `data/processed/`.
