# Complete held-out metrics: Reference26 and the five analog corrections

**2026-10-19.** Dedicated evaluation branch:
`analysis/reference26-nested-full-metrics-20261019`.

This is the comprehensive follow-up to the complete 138-fold
nested correction experiment. It **does not train anything**.
It reads only the private
`reports/reference26_top5_nested/private_nested_predictions.csv`
containing 138 original measured-y / honest heldout-yhat pairs
for exactly seven predictors:

1. Reference26 mean unchanged.
2. Robust26, k=3, blend=0.25, supported gate.
3. PCA8, k=3, blend=0.25, supported gate.
4. PCA8, k=8, blend=0.25, supported gate.
5. Robust26, k=3, blend=0.10, supported gate.
6. Robust26, k=3, blend=0.10, no gate.
7. **NestedSelection**, the real fold-specific four-fold
   chooser between these candidates and the unmodified baseline.

For every heldout, the native initial forecast comes from the
137 real training observations, and the local neighbors of
corrections only use those 137 real training labels.
The selected model's four inner folds also use no outer heldout y.
The five correction *configurations*, however, were previously
chosen post hoc on the same 138 measured parcels.

## Exact metrics computed

Signed residuals: `e = predicted - observed`, units t/ha.

### 1. Error norms and sums

- `L1 = sum_i |e_i|`, units t/ha.
- `L2 = sqrt(sum_i e_i^2)`, units t/ha.
- `Linf = max_i |e_i|`, units t/ha.
- `SSE = sum_i e_i^2`, squared units.
- `MSE = SSE/138`, squared units.
- `RMSE = sqrt(MSE)`, t/ha.
- `MAE = L1/138`, t/ha.
- Median absolute error, quantiles P25/P50/P75/P90/P95/P99,
  maximum absolute error and absolute-error IQR.
- Signed error average (bias), median, P05/P95, population
  and sample standard deviation, median absolute deviation of
  residuals about the median; maximum overprediction and underprediction.

### 2. Quality of explained variation

- **Predictive** `R2 = 1-SSE/sum((y-mean(y))^2)`, not in-sample fit.
- Explained variance, Pearson r, Spearman rho, descriptive
  calibration slope and intercept `observed = a+b*predicted`.
- Normalized RMSE as % of mean, standard deviation and
  observed range; average absolute skill relative to the
  global observed-mean benchmark (descriptive, not a
  fold-trained reference).

**Adjusted R² is deliberately N/A**. The composite regularized,
tuned and gated predictor has no single defensible parameter
count p. Substituting p=26 would produce a misleading statistic.

### 3. Robust loss and extremes

- Trimmed MAE at 5% and 10% (remove the largest 6 or 13
  absolute residuals of 138), trimmed RMSE at 10%.
  These are descriptive and omit the hardest observations.
- Mean Huber loss at deltas **0.25, 0.5, 1.0 t/ha**,
  plus mean log-cosh loss.
- Fraction of genuine observations predicted to within
  **0.25, 0.5 or 1.0 t/ha** absolute error.
  This is **not predictive interval coverage**.

### 4. Relative errors and near-zero safeguards

- WAPE = `100*sum(|e|)/sum(|observed|)`.
- MAPE computed **only where |observed|>=0.1 t/ha**;
  report how many small-denominator parcels were excluded.
- MAPE with explicit denominator floor 0.1 t/ha.
- sMAPE = `200*mean(|e|/max(|observed|+|predicted|,0.2))`.
- RMSPE with 0.1 floor and median absolute percentage error
  with the same floor.

No Infinity/NaN or arbitrary unreported replacement of zero
denominators. MAPE's remaining interpretive limitations persist.

### 5. Resampling, paired model tests, influence

- Common **original-parcel bootstrap** (default 3,000 draws)
  for descriptive conditional 95% RMSE, MAE, bias and R² intervals,
  separately for each of seven predictors.
- Paired candidate-minus-native differences in RMSE, MAE and
  MSE using the **same 138 heldouts** and common bootstrap
  draws, with 95% intervals of RMSE and MAE differences.
- Number of cases whose absolute error improved/worsened/tied;
  paired two-sided binomial sign test and paired Wilcoxon
  signed-rank on differences in absolute loss.
- One-sided sign-flip tests of MSE and MAE **improvement** against
  unchanged Reference26, and familywise **maxT over the six**
  alternative methods, with the six models' sign flips
  synchronized within each parcel.
- Leave-one-original-parcel-out *deletion influence* of the
  measured RMSE for every method, including number of 137-case
  deletions that preserve improvement versus Reference26,
  largest individual fraction of total SSE and top-two SSE share.
- RMSE, MAE, signed bias and P95 error stratified by
  **observed-yield tertiles, 46 original observations each**.
  Stratification uses heldout outcomes for *diagnosis*,
  not model training, and is post hoc.
- Optional sensitivity to geographical clusters via a
  precomputed distance matrix; **ONLY** specify
  `--distance197` if its **first 138 entries are verified**
  to match original heldout row ordering. Cluster bootstrap
  is descriptive and cannot be treated as an automatically
  calibrated spatial confidence interval.

### Inferential caveats

Every interval and p-value is **exploratory**, because the
five corrections were preselected after looking at the same 138
holdout outcomes in many earlier experiments. Even maxT adjusts
only for six alternatives in **this final family**; it cannot
erase earlier adaptive searches. Moreover, iid parcel bootstrap
and sign-exchangeability tests ignore spatial dependence.
A smaller p-value is not confirmation of independent generalization.

There are **no individual measured yields for the 59 targets**.
The model's true prediction error for the 59 and prediction-
interval coverage cannot be calculated with these files.
No interval-construction model is available in this audit.
No new observations are manufactured.

## Commands (Windows PowerShell)

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch analysis/reference26-nested-full-metrics-20261019
git pull --ff-only

python -m py_compile tools/report_reference26_top5_all_metrics.py
python -m pytest tests/test_reference26_top5_all_metrics.py -q

python tools/report_reference26_top5_all_metrics.py --check
python tools/report_reference26_top5_all_metrics.py --run
python tools/report_reference26_top5_all_metrics.py --summary
```

Higher bootstrap precision (read-only; uses the same
138 genuine heldouts and does not refit models):

```powershell
python tools/report_reference26_top5_all_metrics.py --run --bootstrap 10000
```

For optional geographic-block sensitivity on an **already
verified aligned** distance197:

```powershell
python tools/report_reference26_top5_all_metrics.py --run --distance197 deployment/model_artifacts/distance197.npy --geo-blocks 10
```

If you want one console dump to send back in chat, paste the
entire output of `--summary` and optionally the content of
`reports/reference26_top5_full_metrics/metrics_complete_7.csv`,
`paired_tests_vs_reference26.csv` and
`bootstrap_95ci_7.csv`. **Do not post
`reports/reference26_top5_nested/private_nested_predictions.csv`**,
as it contains real yields and parcel IDs. The report CSVs
are **aggregate-only** and safe to inspect for public sharing.

## Output files

All output is local under the gitignored folder
`reports/reference26_top5_full_metrics/`:

- `metrics_complete_7.csv` — core + 40+ extended metrics for
  all seven predictors.
- `paired_tests_vs_reference26.csv` — all six candidate/native
  paired error differences, 95% bootstrap intervals, sign and
  Wilcoxon tests, sign-flip p and maxT.
- `bootstrap_95ci_7.csv` — RMSE/MAE/bias/R² intervals for each model.
- `stratified_yield_tertiles.csv` — 7×3 descriptive error groups.
- `influence_aggregate_7.csv` — sensitivity to removing one
  original case and concentration of SSE.
- `spatial_sensitivity_7.csv` — optional block sensitivity,
  or explicit `SKIPPED` when spatial order isn't supplied.
- `metric_definitions.json` — exact methods, units,
  source SHA, privacy and inference caveats.
- `absolute_error_ecdf_7.png` — aggregate error ECDF.
- `bootstrap_rmse_7.png` — descriptive bootstrap intervals.
- `paired_rmse_delta.png` — candidate RMSE deltas vs native.
- `rmse_by_yield_tertile.png` — error by real yield range.

Plots are aggregate visualizations; no parcel IDs are printed.
**Do not automatically push/private copy the original nested
138-row predictions into public Git**. This audit does not modify
the previous final 59 candidates or the public Todoseboso page.
