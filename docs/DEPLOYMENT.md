# Streamlit Community Cloud deployment

GeoCebada Lab is prepared to run from the repository with the entrypoint:

```text
app/main.py
```

The root `requirements.txt` installs the local `geocebada` package and its runtime dependencies from `pyproject.toml`.

## Deploy

1. Open Streamlit Community Cloud and connect the GitHub account that owns the repository.
2. Grant access to private repositories if GeoCebada remains private.
3. Create a new app from an existing repository.
4. Select repository `heri-espino/GeoCebada`, branch `main`, and entrypoint `app/main.py`.
5. Deploy.

Community Cloud runs the app from the repository root, so GeoCebada's project-relative path helpers continue to resolve `data/source/` correctly.

## Privacy

The repository currently contains official challenge data and is intended to remain private unless redistribution/publication is explicitly permitted. A deployment from a private GitHub repository should therefore also be treated as a private team tool unless the team deliberately changes sharing settings after checking the competition/data terms.

## Prediction-set inspection

The `Prediction Set` tab exposes only provided covariates for `CONJUNTO == "PREDICCION"`; it does not reveal or fabricate `RENDIMIENTO_T_HA`. Its train-vs-prediction diagnostics are intended for support-overlap, missingness, and covariate-shift checks. Supervised model selection must remain based on labeled-data validation rather than repeated manual adjustment to the 59 hidden-target rows.
