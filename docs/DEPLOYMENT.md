# Streamlit Community Cloud deployment

GeoCebada Lab is prepared to run from the repository with the entrypoint:

```text
app/main.py
```

The root `requirements.txt` installs `geocebada` with its optional geospatial dependencies (`-e .[geo]`) and explicitly declares the Streamlit runtime dependencies used directly by the app, including `plotly` and `streamlit`.

The local Conda environment is intentionally stored as `environment.dev.yml`, **not** `environment.yml`. Streamlit Community Cloud gives `environment.yml` higher priority than `requirements.txt`; keeping a recognized Conda file at the repository root can therefore cause Community Cloud to ignore `requirements.txt` and launch the app without packages such as Plotly. Do not rename `environment.dev.yml` back to `environment.yml` while this deployment strategy is in use.

The application is multipage. In addition to the main laboratory, `app/pages/1_Visual_Explorer.py` provides the spatial and multivariate exploration dashboard with filters, parcel mapping, pairplots, correlation heatmaps, bivariate views, outlier screening, grouped summaries, and missingness diagnostics.

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
