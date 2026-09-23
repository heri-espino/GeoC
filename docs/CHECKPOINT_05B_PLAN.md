# Checkpoint 05B plan — large-global remix

**Status:** BLOCKED ON CHECKPOINT 03D RESULTS

Checkpoint 05B is intentionally narrow. It exists only if Checkpoint 03D
produces stronger global experts than the small 03C models used by Checkpoint 05.

## Inputs

- frozen Local04D incumbent;
- the 2–3 model-family-diverse finalists from `reports/checkpoint_03d/finalists.csv`;
- exact target-matched pseudo-competition methodology from Checkpoint 05.

## Candidate universe

Do not rerun the broad 05 stacker zoo. For each surviving 03D global expert,
test only:

```text
(1-w) Local04D + w Global03D
w in {0.05, 0.10, 0.15, 0.20, 0.25}
```

plus, if residual diversity justifies it, one fixed equal-weight ensemble of the
two strongest 03D global experts and one simple non-negative convex stack.

## Promotion

Use the same discipline as Checkpoint 05:

1. target-matched development must improve mean and pooled RMSE;
2. leave-one-development-split-out selection must improve both;
3. only then score one preselected challenger against Local04D on a fresh
   X-only target-matched confirmation bank with a new unused seed;
4. promote only if both confirmation metrics also improve.

## Deployment

Competition and deployment artifacts remain distinct.

- If 05B promotes a model that can be represented exactly in ONNX, export and
  verify that exact final predictor.
- If Local04D remains the competition winner, retain its exact 59-target
  CSV/joblib contract and use the verified Checkpoint 03D ONNX global model for
  generic deployment. Do not label the global ONNX file as though it were
  Local04D.
