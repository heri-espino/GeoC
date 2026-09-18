# Documentación

Esta carpeta concentra la documentación del proyecto, la referencia de la librería Python y los materiales oficiales del Reto AgroCebada FIRA 2026.

- `AGENT_GUIDE.md`: punto de entrada operativo para agentes/colaboradores; estado actual, reglas, artifacts, CV congelada y siguiente fase.
- `PROJECT_HISTORY.md`: cronología de lo que se hizo, por qué y qué quedó validado.
- `official/`: bases, lineamientos y documentos oficiales de la convocatoria. Inmutables.
- `reference/`: diccionarios y documentación técnica del dataset proporcionado por FIRA. Conservar como evidencia fuente.
- `DATA_SOURCES.md`: inventario operativo de fuentes oficiales/externas, reglas de joins, caveats y estado de adquisición.
- `DATA_CONTRACT_V2.md`: contrato machine-readable/auditable que debe pasar antes de construir features.
- `INTEGRATION_FIXTURES.md`: selección y generación de las 12 parcelas compartidas para pruebas end-to-end.
- `FEATURE_TABLE_V1.md`: contrato, fuentes y transformaciones de la tabla parcel-level model-ready.
- `AGRONOMIC_FEATURES_V1.md`: fórmulas, significado, evidencia, caveats y bibliografía de la capa agronómica/no lineal.
- `VARIABLES.md`: diccionario source-backed de variables e índices.
- `AI_USAGE.md`: registro del uso de herramientas de IA generativa y prompts relevantes.
- `FUNCTION_INDEX.md`: índice rápido y buscable de funciones/clases públicas en `src/geocebada/`.
- `api/`: referencia API de Sphinx generada desde docstrings.
- `conf.py` / `index.rst`: configuración y entrada de la documentación Sphinx.

Los documentos oficiales y de referencia se conservan sin modificar; cualquier resumen, interpretación o documentación producida por el equipo debe mantenerse separada de las fuentes originales.

## Índice rápido de funciones

Después de añadir, renombrar o eliminar una función/clase pública en `src/geocebada/`, ejecutar:

```bash
python tools/generate_function_index.py
```

Esto regenera `docs/FUNCTION_INDEX.md` usando el AST de Python, sin importar módulos ni requerir dependencias opcionales.

Antes de crear una función nueva, buscar primero en ese índice para evitar duplicados.

## Documentación Sphinx

Instalar dependencias de documentación:

```bash
pip install -e ".[docs]"
```

Construir HTML:

```bash
sphinx-build -b html docs docs/_build/html
```

Abrir después `docs/_build/html/index.html`.

Sphinx genera automáticamente:

- referencia de módulos y funciones desde docstrings;
- índice general (`genindex`);
- índice de módulos (`modindex`);
- búsqueda full-text dentro de la documentación.

La documentación de una función debe vivir principalmente en su docstring; no mantener descripciones divergentes a mano en varios lugares.
