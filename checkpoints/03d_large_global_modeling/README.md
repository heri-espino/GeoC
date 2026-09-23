# Checkpoint 03D — Large-compute global modeling

**Status:** IMPLEMENTED / WORKSTATION RUN PENDING  
**Opened:** 2026-09-23

Checkpoint 03C intentionally used small, disciplined candidate grids. Checkpoint
03D is the final additive extension of the historical global-modeling line: use
the available workstation compute budget to ask how strong a properly tuned
global regressor can actually become.

03C.1–03C.3 remain frozen historical evidence. 03D does not rewrite them.

## Scope

Competition-only, deterministic representations:

- G0: base competition table;
- G1: agronomic nonlinear layer;
- G2: base + agronomic;
- G3: base + agronomic + deterministic empirical layer.

Target-aware expression discovery is deliberately excluded. This keeps the
large benchmark easier to audit and gives the selected global estimator a clean
deployment schema.

## Large models

The default run evaluates:

- CatBoostLarge — GPU;
- XGBoostLarge — GPU;
- LightGBMLarge — CPU;
- ExtraTreesLarge — CPU;
- HistGBLarge — CPU;
- RidgeControl;
- PLSControl.

The boosted candidates use thousands of boosting iterations/trees with strong
regularization and low learning rates. The objective is not to maximize raw
parameter count; it is to give nonlinear global models a serious, nested-CV
search budget.

## Validation

03D keeps the historical Checkpoint 03 validation contract:

- frozen state-stratified outer folds;
- frozen municipality-grouped outer folds;
- 3-fold protocol-aligned inner tuning;
- one prediction per labeled parcel per protocol/pair;
- hidden 59 FIRA targets never scored.

Ranking uses worst-protocol OOF RMSE first, then mean OOF RMSE. At most one
representation from each model family is retained among the three finalists.

## Resume

Every completed outer fit is checkpointed locally. If the workstation stops:

```powershell
python tools\run_checkpoint_03d.py --resume
```

Partial files are gitignored and are removed after a successful complete run.

## ONNX deployment contract

The scientific finalists are refit on all 138 labeled parcels after tuning
over the union of fresh state-stratified and municipality-grouped inner folds.
The highest-ranked finalist that also passes ONNX converter, output-schema and
numerical-equivalence checks becomes the deployment artifact. Converter support
never changes the scientific ranking.

The runner writes:

```text
models/final/checkpoint03d_global.joblib
models/final/checkpoint03d_global.onnx
models/final/checkpoint03d_global.onnx.json
```

The ONNX graph receives the median-imputed float32 numeric matrix. For model
families that require learned scaling, that scaler is embedded inside the ONNX
graph. The adjacent JSON manifest stores the exact raw feature order and fitted
median for every feature. The runner first verifies that the post-imputation deployment object reproduces
the complete fitted Python pipeline, then loads the ONNX graph with ONNX Runtime
and refuses to PASS unless both equivalence checks satisfy the configured
tolerance.

This global ONNX model is a deployment artifact. It does **not** silently replace
the fixed-target competition winner Local04D.


### Stable converter stack

The stable PyPI release `skl2onnx==1.20.0` predates an upstream July 2026
fix for ONNX 1.22 tree attributes. Therefore the deployment extra pins
`onnx<1.22` while retaining the released converter. This is preferred over
silently installing unreleased `skl2onnx` main.

PLS is kept as a scientific control even if its converter declares an
incorrect single-target output shape under the installed scikit-learn
version. Such a model may remain a finalist but is not eligible for the
deployment artifact unless schema verification passes.

### 2026-09-23 preflight incident

The first workstation preflight correctly stopped before training when
`ExtraTreesRegressor` conversion under ONNX 1.22+ passed a boolean
`nodes_missing_value_tracks_true` attribute where ONNX requires integer
0/1 values. The repository now pins the stable compatible ONNX range and
reports per-family converter availability instead of allowing one external
converter failure to invalidate the scientific benchmark.

## Run

Install once:

```powershell
git pull
conda activate geocebada
python -m pip install -e ".[dev,models,deployment]"
```

Fail-fast check:

```powershell
python tools\run_checkpoint_03d.py --preflight
```

Then the long run:

```powershell
python tools\run_checkpoint_03d.py
```

CPU is intentional-only:

```powershell
python tools\run_checkpoint_03d.py --compute CPU --confirm-cpu y
```

## After 03D

Do not immediately promote the best global model. Review
`reports/checkpoint_03d/` first. The top 03D finalists then feed a narrow
Checkpoint 05B target-matched remix against Local04D.
