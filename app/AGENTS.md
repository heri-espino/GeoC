# AGENTS.md — GeoCebada Streamlit application

> **Checkpoint 04 update:** the active scientific objective is fixed-target transductive
> reconstruction. The app may expose and use all supplied covariates of the 59 target parcels
> for documented X-only transductive diagnostics/models. Hidden FIRA yields remain unavailable.
> The old prospective forecast-cutoff question is not the active competition objective.


These instructions apply to everything under `app/` and supplement the repository-wide `AGENTS.md` and `.ai_handoff`.

Future AI agents/Codex working on the Streamlit application must read this file **before modifying the app**. Also read `docs/AGENT_GUIDE.md` for the current project phase and frozen modeling artifacts.

## Purpose of the app

The Streamlit application is the interactive interface required by the Reto AgroCebada FIRA 2026. It must remain useful both as:

1. a team-facing analytical laboratory for exploring the provided data, assumptions, features and model diagnostics; and
2. the eventual jury-facing implementation of the **frozen** predictive pipeline.

The competition explicitly requires a functional application/interface/dashboard capable of demonstrating the practical use of the predictive model and allowing interactive consultation of results. Do not let the app drift into a disconnected demo that uses different transformations or models from the reproducible library pipeline.

## Current application structure

```text
app/
├── AGENTS.md                 # this maintenance contract
├── __init__.py
├── main.py                   # GeoCebada Lab main page
├── visual_explorer.py        # reusable Streamlit-facing explorer helpers
└── pages/
    └── 1_Visual_Explorer.py  # spatial/multivariate exploration page
```

Current functionality includes:

- overview of official split/target data;
- generic data exploration;
- inspection of provided covariates for the 59 `PREDICCION` parcels;
- train-vs-prediction covariate-shift diagnostics;
- statistical tests and regression diagnostics;
- interactive feature-recipe experiments;
- exploratory regression baselines;
- BASIC/PRO exploration;
- multivariate plots, pairplots, correlations, outliers and missingness;
- parcel mapping when source geometry/CRS permit safe rendering.

When adding a major reusable workflow, update this list, root `.ai_handoff`, `docs/DEPLOYMENT.md`, and `docs/AI_USAGE.md` when AI materially contributed.

## Architecture rule: Streamlit is the interface, `geocebada` is the analytical engine

Do **not** put stable scientific/data/model logic only inside Streamlit callbacks.

Use this dependency direction:

```text
data/source + model artifacts
          ↓
src/geocebada/               # loaders, validation, transforms, statistics, models
          ↓
app/                          # widgets, layout, filters, presentation
```

If code is needed by notebooks, tests, model training or more than one page, move it to `src/geocebada/`, expose it through the appropriate package `__init__.py`, add tests, and regenerate `docs/FUNCTION_INDEX.md`.

Do not duplicate feature engineering between the app and notebooks. The final inference page must call the same frozen preprocessing/model pipeline used to generate the official predictions.

## Data and target safety

Canonical identifiers/contracts:

- parcel ID: `ID_POLIGONO`;
- target: `RENDIMIENTO_T_HA`;
- split column: `CONJUNTO`;
- labeled value: `ENTRENAMIENTO`;
- hidden-target value: `PREDICCION`.

The app may expose the supplied **covariates** of `PREDICCION` parcels for support/shift/missingness diagnostics. It must never fabricate or display hidden target values.

Supervised statistics, training, feature selection and model comparison must use labeled parcels only unless a method is explicitly unsupervised/transductive and documented as such.

Do not choose/tune a model because the final 59 predictions “look plausible”.

## Longitudinal-data safety

BASIC/PRO contain repeated `parcel × date × sensor` records. Do not present 100k rows as 100k independent yield samples.

Any supervised validation shown in the UI must group by parcel at minimum. Never row-random split BASIC/PRO.

Keep sensor provenance visible. Same-named NDVI/EVI/LAI values from Sentinel/Landsat/Planet are not automatically interchangeable.

The target corresponds to April–October 2025. For the active competition-reconstruction mode, full observed 2025 competition information may be used. If the app separately presents a prospective forecasting scenario, that scenario must declare its own cutoff and exclude later observations.

## Geospatial/CRS safety

Known source CRS values differ across products. Never guess a missing CRS.

For parcel maps:

- inspect source CRS first;
- reproject only when CRS metadata is known;
- use EPSG:4326 for web-map display only after a valid transformation;
- do not calculate metric distances/areas in geographic degrees;
- fail with a useful warning rather than silently assigning a CRS.

Generic map aggregations are visualization conveniences, not automatically canonical model features.

## Streamlit Community Cloud dependency contract

Deployment entrypoint:

```text
app/main.py
```

Repository/branch:

```text
heri-espino/GeoCebada
main
```

Cloud dependencies are controlled from the repository root by:

```text
requirements.txt
```

Current root requirement strategy:

```text
-e .[geo]
plotly>=6.0
streamlit>=1.38
```

`pyproject.toml` remains the package dependency source of truth; `requirements.txt` is the Cloud bootstrap file and should install the package rather than duplicate every internal dependency manually.

**Do not create a root `environment.yml` for local Conda setup.** Streamlit Community Cloud can prioritize it over `requirements.txt`, which previously caused the deployed app to start without Plotly. Local/team Conda configuration belongs in:

```text
environment.dev.yml
```

When adding an app import from a third-party package:

1. decide whether it is a real package dependency;
2. if yes, add it to `pyproject.toml` (base or suitable extra);
3. ensure `requirements.txt` installs the needed extra;
4. if it is a Cloud bootstrap/runtime package, make it explicit in `requirements.txt` when useful for robustness;
5. update local environment docs if teammates need it;
6. reboot/redeploy Community Cloud and inspect logs.

Do not “fix” Cloud by adding one-off `pip install` shell calls inside `app/main.py`.

## Local smoke test before deployment

From the repository root:

```bash
conda activate geocebada
python -m pip install -e ".[dev,geo]"
streamlit run app/main.py
```

At minimum verify manually:

1. main page loads without import errors;
2. official split loads;
3. `Prediction Set` contains no hidden yield values;
4. Statistical Lab renders on labeled rows;
5. Visual Explorer page appears in navigation;
6. BASIC and PRO can load;
7. map either renders safely or produces an explicit CRS/data warning;
8. pairplot/correlation/missingness controls work on a small selection;
9. no page mutates files under `data/source/`;
10. large operations do not execute eagerly on every widget rerun when caching/precomputation is appropriate.

## Cloud smoke test after deployment

After changing `requirements.txt`, `pyproject.toml`, app imports, page structure, data loading or geospatial dependencies:

1. wait for Streamlit Community Cloud to rebuild;
2. inspect **Manage app → logs**;
3. confirm dependency installation succeeded;
4. open the deployed app in a fresh/private browser session;
5. exercise both `app/main.py` and all files under `app/pages/`;
6. verify that relative/project paths resolve from `/mount/src/geocebada`;
7. verify that bundled official data needed by the app is actually present in the deployed checkout;
8. record any deployment-specific fix in `docs/DEPLOYMENT.md` and this file if it changes the contract.

Do not report the Cloud app as fixed merely because local tests pass.

## Performance and caching

Streamlit reruns the script after widget interactions. Protect the user experience:

- use `@st.cache_data` for deterministic data loads/transforms that are safe to cache;
- use `@st.cache_resource` for expensive reusable model/resources where appropriate;
- avoid re-reading the large BASIC/PRO CSVs repeatedly;
- avoid constructing full pairplots over all ~100k longitudinal rows;
- sample/rasterize exploratory plots;
- push heavy reusable transforms into `src/geocebada/`;
- use explicit buttons/forms for expensive computations if automatic reruns would be costly;
- make defaults inexpensive enough for Community Cloud.

Never cache secrets or hidden target information.

## Error handling

Prefer a clear user-facing warning over an opaque stack trace for expected runtime conditions such as:

- missing optional geospatial dependency;
- unavailable geometry archive;
- CRS metadata missing;
- no rows remaining after filters;
- insufficient numeric columns for pairplot/correlation;
- model artifact not yet frozen/available.

Do not catch broad exceptions merely to hide programming bugs. Unexpected failures should remain diagnosable from Cloud logs.

## UI/model status language

Until a final model/pipeline is frozen, label dashboard model results as **exploratory**. Checkpoint 02 now provides the project-level frozen state-stratified and municipality-grouped folds; do not imply that the dashboard's generic random K-fold baseline is the official validation result.

When the final model exists, expose at minimum:

- parcel identifier;
- predicted yield in ton/ha;
- relevant model/version metadata;
- uncertainty/diagnostics if methodologically defensible;
- interpretable feature contributions when valid;
- clear distinction between observed training yield and predicted hidden-set yield.

The application must remain consistent with the technical report and official prediction table.

## Dependency/update checklist for future agents

Whenever modifying the Streamlit app, ask:

- Did I add/remove an import? Update `pyproject.toml` / `requirements.txt` if needed.
- Did I add a reusable transformation/statistical/model function? Put it in `src/geocebada/`, not only `app/`.
- Did I create a public library function? Update tests/docstrings/exports and regenerate `docs/FUNCTION_INDEX.md`.
- Did I change how data are interpreted? Update `docs/VARIABLES.md` or data handoff only if source evidence supports it.
- Did I change deployment behavior? Update `docs/DEPLOYMENT.md`.
- Did I change a major app capability/status? Update root `.ai_handoff`.
- Did AI materially contribute? Update `docs/AI_USAGE.md`.
- Did I expose/tune using the prediction targets? Stop: hidden targets are unavailable and must remain unavailable.
- Did I test locally **and** verify Cloud after deployment-sensitive changes?

## Before finishing an app task

Run the relevant repository checks available for the change, typically:

```bash
ruff check .
pytest -q
python tools/generate_function_index.py
sphinx-build -b html -W docs docs/_build/html
```

If the public function index changes, commit the regenerated `docs/FUNCTION_INDEX.md`.

For dependency/deployment changes, also perform the local and Cloud smoke tests above. Report only checks actually executed; never claim Streamlit Cloud is healthy without observing the deployed build/runtime.
