# Streamlit Community Cloud deployment

GeoCebada Lab is prepared to run from the repository with the entrypoint:

```text
app/main.py
```

The root `requirements.txt` installs `geocebada` with its optional geospatial dependencies (`-e .[geo]`) and explicitly declares the Streamlit runtime dependencies used directly by the app, including `plotly` and `streamlit`.

The local Conda environment is intentionally stored as `environment.dev.yml`, **not** `environment.yml`. Streamlit Community Cloud gives `environment.yml` higher priority than `requirements.txt`; keeping a recognized Conda file at the repository root can therefore cause Community Cloud to ignore `requirements.txt` and launch the app without packages such as Plotly. Do not rename `environment.dev.yml` back to `environment.yml` while this deployment strategy is in use.

The application is multipage. In addition to the main laboratory, `app/pages/1_Visual_Explorer.py` provides the spatial and multivariate exploration dashboard with filters, parcel mapping, pairplots, correlation heatmaps, bivariate views, outlier screening, grouped summaries, and missingness diagnostics.

> AI agents/Codex must read `app/AGENTS.md` before changing app code, deployment dependencies, page structure, caching or Cloud behavior. That file is the app-specific maintenance contract.

## Deploy

1. Open Streamlit Community Cloud and connect the GitHub account that owns the repository.
2. Grant access to private repositories if GeoCebada remains private.
3. Create a new app from an existing repository.
4. Select repository `heri-espino/GeoCebada`, branch `main`, and entrypoint `app/main.py`.
5. Select **Python 3.11** in the advanced deployment settings so Cloud matches the team's development environment.
6. Deploy.

Community Cloud runs the app from the repository root, so GeoCebada's project-relative path helpers continue to resolve `data/source/` correctly. Streamlit discovers the page under `app/pages/` automatically.

## Dependency troubleshooting

If Cloud shows `ModuleNotFoundError` for `plotly`, `geocebada`, or another third-party package:

1. Confirm the root contains `requirements.txt`.
2. Confirm there is **no** root `environment.yml`, `Pipfile`, or `uv.lock` that can take precedence over it.
3. In **Manage app**, reboot/redeploy the application after pulling the latest `main` commit.
4. Inspect the build log and verify that Community Cloud processes `requirements.txt` and installs the editable `geocebada` package.

A change to `requirements.txt` should trigger a dependency reinstall automatically on Community Cloud.

## ONNX model artifact

Checkpoint 03D introduces the canonical generic model-deployment format.
After a successful large-model run the workstation writes:

```text
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
models/final/checkpoint03d_global.joblib
```

The ONNX graph consumes a median-imputed float32 feature matrix. Any learned
post-imputation scaling required by the selected estimator is embedded in the
graph. The JSON manifest stores the exact raw feature order and fitted median
statistics required to construct that matrix.

The 03D runner refuses to PASS unless the post-imputation deployment object
reproduces the complete Python pipeline and ONNX Runtime reproduces that
deployment object within the configured tolerance.

This artifact must not be confused with the fixed-target competition rule.
`Local04D` remains the current 59-parcel competition method unless a later
Checkpoint 05B promotion gate succeeds. If 05B retains Local04D, the 03D ONNX
file is the generic deployable global estimator, while the canonical fixed-59
competition predictions remain a separate artifact.

The binary model directory remains gitignored by default. Before production
deployment, inspect the final ONNX file size and choose an explicit delivery
mechanism (for example release artifact, object storage, or Git LFS) rather
than silently committing a large binary to ordinary Git history.

## Maintenance contract for future agents

When an AI agent changes the Streamlit app, it must keep deployment, library code and documentation synchronized.

### If a third-party import is added

- Add the dependency to `pyproject.toml` when it is a package/runtime dependency.
- Ensure `requirements.txt` installs the extra that contains it.
- Keep direct Streamlit bootstrap dependencies explicit in `requirements.txt` when that improves deployment robustness.
- Do not install packages dynamically from `app/main.py`.
- Do not reintroduce root `environment.yml` merely for local Conda usage; use `environment.dev.yml`.

### If analytical/model logic is added

- Put stable reusable logic under `src/geocebada/`.
- Keep Streamlit code focused on widgets, filtering and rendering.
- Reuse the same frozen inference pipeline that produces the official prediction table.
- Add tests and regenerate `docs/FUNCTION_INDEX.md` when the public library API changes.

### If data behavior changes

- Preserve immutable `data/source/`.
- Keep `PREDICCION` target values absent.
- Do not use row-random validation on longitudinal BASIC/PRO.
- Preserve sensor provenance and CRS checks.
- Update `docs/VARIABLES.md` or data handoffs only when source evidence supports the interpretation change.

### Before saying a deployment is healthy

Local verification:

```bash
conda activate geocebada
python -m pip install -e ".[dev,geo]"
streamlit run app/main.py
```

Repository checks when relevant:

```bash
ruff check .
pytest -q
python tools/generate_function_index.py
sphinx-build -b html -W docs docs/_build/html
```

Then verify the **actual Cloud deployment**:

1. wait for build/redeploy;
2. inspect **Manage app → logs**;
3. open the app in a fresh browser session;
4. open every page under `app/pages/`;
5. verify BASIC/PRO loading and the official split;
6. verify `Prediction Set` has no hidden yields;
7. verify geospatial views either render safely or show an explicit CRS/data warning;
8. check that large operations are not unnecessarily rerun after every widget interaction.

Passing local tests is not evidence that Streamlit Community Cloud installed the intended dependencies.

## Performance/caching guidance

BASIC and PRO are large enough that careless Streamlit reruns can make the app unusable.

- Cache deterministic data loads/transforms with `@st.cache_data` when safe.
- Cache expensive reusable models/resources with `@st.cache_resource` where appropriate.
- Avoid repeatedly loading full BASIC/PRO CSVs on each widget change.
- Use inexpensive defaults and explicit buttons/forms for costly operations.
- Sample or rasterize large pairplots rather than rendering all longitudinal rows.
- Move reusable heavy transformations to `src/geocebada/` so notebooks and app use identical behavior.

Never cache secrets or any unavailable hidden-target information.

## Visual Explorer data sources

The Visual Explorer can load:

- the official 197-row yield/split table;
- the bundled BASIC CSV;
- the bundled PRO CSV;
- a user-uploaded CSV.

For BASIC/PRO, the page can attach parcel-level `AREA_HA`, `CONJUNTO`, and the observed training target through `ID_POLIGONO`. Hidden prediction yield remains missing.

## CRS behavior

The official parcel archive is read with GeoPandas. The map only reprojects geometries when the source GeoDataFrame already contains CRS metadata. If the CRS is missing, the dashboard stops with a warning instead of guessing one. This preserves the repository's CRS safety contract.

## Privacy

The repository currently contains official challenge data and is intended to remain private unless redistribution/publication is explicitly permitted. A deployment from a private GitHub repository should therefore also be treated as a private team tool unless the team deliberately changes sharing settings after checking the competition/data terms.

## Prediction-set inspection

The `Prediction Set` tab exposes only provided covariates for `CONJUNTO == "PREDICCION"`; it does not reveal or fabricate `RENDIMIENTO_T_HA`. Its train-vs-prediction diagnostics are intended for support-overlap, missingness, and covariate-shift checks. Supervised model selection must remain based on labeled-data validation rather than repeated manual adjustment to the 59 hidden-target rows.
