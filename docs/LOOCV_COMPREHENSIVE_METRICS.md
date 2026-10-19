# Comprehensive error metrics for genuinely comparable 138-fold LOOCV models

**Status:** metric definitions and read-only extractor implemented; the private
138-parcel predictions are on the user's machine, so this repository document
**does not claim they were recalculated here**. All known baseline numbers
below were transcribed from previously completed runs and are approximate
where the historical terminal display truncated digits.

## 1. Available verified *historical* full-138 reports

Every listed run was reported with 138 distinct held-out parcels and 137
training labels per fold. **Identical held-out labels/IDs are verified only
after running the new extractor on the private prediction CSVs**.

| Model / protocol | RMSE (t/ha) | MAE (t/ha) | R² | Data file expected on workstation |
|---|---:|---:|---:|---|
| Reference26, **mean**, k24/alpha30 | 0.468138 | 0.333786 | approximately 0.6953 | \`reports/reference26_ridge_budget_mean/fold_001..138/k24_a30.csv\` |
| Reference26, **median**, k24/alpha30 | 0.4681855146560597 | 0.3336730043055529 | approximately 0.6952 | \`reports/selected_champions_loo/loo_predictions.csv\` |
| Reference26 + geological DBSCAN | 0.469275 | 0.335668 | approximately 0.6938 | \`reports/peer_dbscan_loo138/loo_predictions.csv\` |
| Compact14 | 0.4696729969475559 | 0.333044183790906 | approximately 0.6933 | same original selected champions file |
| Local04D / GeoPCA20 + geographic Ridge | 0.485133 | 0.351122 | 0.672766 | \`reports/area_weighted_climate_loo/loo_predictions.csv\` |
| Local04D + touched-pixel climate | 0.486455 | 0.352139 | approximately 0.6710 | same 4-arm local file |
| Local04D + area-weighted climate | 0.486682 | 0.352180 | approximately 0.6707 | same 4-arm local file |
| Local04D + temporal climate | 0.487624 | not yet recovered | approximately 0.6694 | same 4-arm local file |

The stored mean-imputation tuning finalists k20/alpha30 and k12/alpha3.2
are also read if they have all 138 saved folds; historical RMSE 0.470112
and 0.474493, respectively. They remain **exploratory** after
hyperparameter selection.

**Do not mix these complete LOOCV scores** with CatBoost, Prithvi,
PLS, Gaussian projection, GMM or historical 16×41 results using
97 training labels and 41 held-out parcels. Additional models can
join this table only if they truly have **138 independently fitted
137-train/1-heldout predictions** and matching parcel IDs and observed
yields. Similar names and equal row counts are insufficient proof.

### Metrics derivable from historical summaries (approximate)

Because all these historical reports use n = 138, the identities
`L1 = 138 × MAE`, `L2 = sqrt(138) × RMSE`, and
`MSE = RMSE²` allow the following **derived rounded** values,
even before recovering per-parcel forecasts:

| Model | Total L1 (t/ha) | Total L2 (t/ha) | MSE ((t/ha)²) |
|---|---:|---:|---:|
| Reference26 mean | 46.062468 | 5.499376 | 0.219153 |
| Reference26 median | 46.046875 | 5.499934 | 0.219198 |
| Reference26 DBSCAN | 46.322184 | 5.512733 | 0.220219 |
| Compact14 | 45.960097 | 5.517408 | 0.220593 |
| Local04D | 48.454836 | 5.699022 | 0.235354 |

These are **not independently recomputed** from the raw LOOCV
predictions; rounding of historical RMSE and MAE propagates.
The full reporting script will recompute their exact values
from the saved 138 predictions when run on the local workstation.
There is **no mathematical way** to recover L∞, MedAE or upper
absolute-error quantiles from RMSE and MAE alone.

## 2. Mathematical metric definitions

Let \(n=138\), \(y_i\) be observed yield, \(\widehat y_i\) be
the held-out model prediction, \(e_i=\widehat y_i-y_i\),
\(\overline y=n^{-1}\sum_i y_i\), and
\(\mathrm{SSE}=\sum_i e_i^2\). All \(y\) are in t/ha.

**Vector norms (NOT averages):**

\[
\|e\|_1=\sum_{i=1}^n |e_i|,\qquad
\|e\|_2=\sqrt{\sum_{i=1}^n e_i^2},\qquad
\|e\|_\infty=\max_i|e_i|.
\]

The norm values use t/ha; L1 and L2 increase with sample count
and **must always be reported alongside n**. For the fixed n=138,
the more interpretable normalized quantities are

\[
\mathrm{MAE}=\frac{\|e\|_1}{n},\quad
\mathrm{MSE}=\frac{\|e\|_2^2}{n},\quad
\mathrm{RMSE}=\frac{\|e\|_2}{\sqrt n}.
\]

The main table also reports:
- **Median absolute error (MedAE)** and **90th/95th percentiles**
  of \(|e_i|\), plus maximum absolute error. These identify long-tailed
  errors and complement RMSE.
- **Signed bias** \(n^{-1}\sum_i e_i\), the population standard
  deviation of errors, and maximum positive and negative signed errors.
- **Predictive (cross-validated) R²** (also denoted Q² in some literature):
  \[
  R^2_{\mathrm{CV}}
   =1-\frac{\sum_i(y_i-\widehat y_i)^2}
            {\sum_i(y_i-\overline y)^2}.
  \]
  The comparator is the global held-out sample mean, **not a
  separately cross-validated constant-mean predictor**. This
  denominator convention should be explicit in a paper.
- **Explained variance**, \(1-\mathrm{Var}(e)/\mathrm{Var}(y)\);
  this is not identical to R² when the model is biased.
- **Pearson and Spearman prediction–observation correlations**
  (association, not calibration); **WAPE** \(=100\sum|e_i|/\sum|y_i|\),
  normalized RMSE vs \(|\overline y|\) and vs the population
  standard deviation of \(y\), and MAE skill over a global-mean
  comparator. These are secondary diagnostic metrics.
- The supplemental paired table reports differences in RMSE/MAE/MSE,
  the fraction of parcels with lower absolute error, correlation of
  signed errors, and **95% naive parcel-bootstrap intervals** for
  paired \(\Delta\mathrm{RMSE}\) (2,000 fixed-seed resamples by default).

**Adjusted R² must not be numerically fabricated.** The conventional
in-sample linear regression expression is

\[
R^2_{\mathrm{adj}}=
1-(1-R^2)\frac{n-1}{n-p-1}.
\]

Here \(p\) counts the explanatory parameters of a fixed fitted linear
model; it is **not** automatically the number of raw features.
Reference26 is a nonlinear/regularized composite with local experts,
geological gates, adaptive feature construction and refits in 138 folds.
There is no unique, defensible fixed \(p\). Substituting 26 or 14 into
this expression would make a false claim. Therefore the output
\`adjusted_r2\` is **NA (not applicable)**, accompanied by a reason.
Use predictive R² and LOOCV error metrics instead.

Avoid making MAPE a central result: if true yields are near zero,
individual percentage errors can explode. WAPE is still only a
secondary metric and must be interpreted cautiously.

## 3. Run on the local workstation (no training)

From the GeoC root:

~~~powershell
conda activate geocebada
git pull --ff-only origin experiment/nuria-champions-loocv

# Show which original saved model predictions are present
python tools/report_loocv_metrics.py --audit

# Recompute every metric from individual authentic LOOCV predictions
python tools/report_loocv_metrics.py --run

# Display publication-candidate primary metrics (no recomputation)
python tools/report_loocv_metrics.py --summary
~~~

Private LOCAL outputs (kept Git-ignored):

- \`reports/loocv_comprehensive_metrics/all_models_loocv_metrics.csv\`
- \`reports/loocv_comprehensive_metrics/paired_model_comparisons.csv\`
- \`reports/loocv_comprehensive_metrics/run_spec.json\`

The two CSV files above contain **only aggregated metrics** (no
parcel IDs, original yield columns or per-parcel predictions).
The script raises on mismatched IDs, observed yield disagreement,
invalid predictions and incompatible fold counts. Optional missing
candidate files are named as \`[SKIP]\`; the original Reference26 and
Compact14 baseline must be present.

For a genuinely compatible new model, supply an audited CSV:

~~~powershell
python tools/report_loocv_metrics.py --run --extra "CatBoost=reports/validated_138_catboost.csv"
~~~

Extras require \`ID_POLIGONO,observed,outer_train,outer_heldout\`
with counts exactly 137 and 1, and a \`predicted\` or
model-named column, with all 138 IDs and matching observed yields.
These schema checks **do not replace independent review** of whether
the extra model really refitted 138 times excluding each target.

## 4. Publish for the final technical document

Once the metrics have been reviewed:

~~~powershell
# Optional explicit export of aggregate-only files to tracked results/
python tools/report_loocv_metrics.py --publish

git status --short
git add results/peer_models/LOOCV_FULL_METRICS_138.csv results/peer_models/LOOCV_PAIRED_COMPARISONS_138.csv
git commit -m "Publish comprehensive aggregate 138-fold LOOCV metrics"
git push origin experiment/nuria-champions-loocv
~~~

**Never commit** \`data_n/\`, per-fold files, \`loo_predictions.csv\`,
the raw held-out yields or private feature tables. The exporting
function checks an explicit aggregate-column allowlist and rejects
known per-parcel columns. Review the aggregate tables before pushing.

## 5. Interpretation in the final document

A suitable table in the paper should show at least: n, RMSE,
MAE, MedAE, P95 absolute error, maximum absolute error (L∞),
mean signed error, and predictive R². The appendix can add L1,
L2, SSE/MSE, other quantiles and normalized statistics. A companion
table should report paired delta-MSE/RMSE against Reference26 mean.

Paired naive bootstrap intervals **do not account for spatial
autocorrelation**; spatial-group/block bootstrap can be added as a
sensitivity analysis if suitable groups exist. Moreover, Reference26
model choice and its hyperparameter search reused some LOOCV labels,
so the eventual winning result remains **exploratory and potentially
optimistic** rather than independent confirmation.

All metrics must be recalculated from the real CSVs before a finalized
manuscript table is labeled verified. This documentation is a
metric specification, provenance checklist and execution guide,
not a claim that private reports have already been processed.
