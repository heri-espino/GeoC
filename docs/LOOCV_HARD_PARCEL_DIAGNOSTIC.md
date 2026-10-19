# Cross-model LOOCV hard-parcel consensus (diagnostic only)

**2026-10-19. Status:** A read-only analyzer and synthetic tests have been added
to the analysis branch. **The real private per-parcel LOOCV files have NOT
been read by the GitHub-side agent; the common two parcel IDs are not yet
known.** No label deletion, retraining or model promotion has occurred.

## Question

Across already completed genuine leave-one-parcel-out (LOOCV) fits, do
the largest residuals repeatedly arise from the same parcels, and is there
a pronounced increase in cumulative squared error near the hardest two?
If yes, that establishes a **cross-model hard-case pattern**, not by itself
a recording error or a beneficial training-label exclusion.

## Authentic available sources

Use \`tools/report_loocv_metrics.py\`'s existing verified loading path to
read and align all available 138 held-out predictions (137 training labels
per fold), including these historically recorded candidates:

- Reference26 arithmetic mean, k24/alpha30; reference median;
- Reference26 mean k20/alpha30 and k12/alpha3.2 where all folds exist;
- Reference26 geological DBSCAN;
- Compact14;
- frozen Local04D (GeoPCA20) and touched-pixel, area-weighted and
  temporal-climate variants;
- if all 138 fold CSVs exist, the **fresh** Reference26 mean,
  Compact14 and their fixed-weight ensemble.

All records must have the same **138 unique ID_POLIGONO values and observed
yields**. The existing loader checks strict parity and optional 137/1
fold counts. Historical 16×41 target-matched repetitions, 46-fold subsets,
97/41 fits, LOSO results and CatBoost/Prithvi/PLS results from incompatible
protocols **must not be pooled** into these curves. Additional independently
audited genuine 138-fold candidates can use the same \`--extra Name=PATH.csv\`
contract described in \`docs/LOOCV_COMPREHENSIVE_METRICS.md\`.

## Implementation

In PowerShell at the GeoC repository root, after checking out the PR branch:

~~~powershell
conda activate geocebada
python tools/analyze_loocv_hard_parcels.py --audit
python tools/analyze_loocv_hard_parcels.py --run
python tools/analyze_loocv_hard_parcels.py --summary
~~~

The analysis **does not refit** any model and is inexpensive compared with
the original LOOCV training.

All outputs stay beneath the **gitignored** path
\`reports/loocv_hard_parcels/\`. The program creates:

| File | Use | Publication |
|---|---|---|
| \`model_error_concentration.csv\` | Models ordered by full RMSE, top 1/2/10 SSE shares; RMSE after dropping a common pair **without refitting** | Aggregate; review |
| \`sorted_cumulative_rmse_curves.csv\` | For each model, RMSE after including the 1,2,...,138 lowest-error parcels | Aggregate; review |
| \`cumulative_rmse_last28.png\` | Chart magnifying the hardest 28 additions for every model | Aggregate; review |
| \`candidate_parcels_private.csv\` | Parcel IDs, true yields, absolute errors/ranks, family-level agreement | **Never commit** |
| \`model_top10_errors_private.csv\` | Ten worst signed/absolute prediction errors per model and IDs | **Never commit** |
| \`diagnostic_manifest_private.json\` | Model-family representatives, candidate two IDs, exact top-2 and top-10 intersections | **Never commit** |

If matplotlib is not installed, the numerical curve CSVs still work.

### Exact definitions

For \(m\) a model and \(i\) a held-out parcel, write
\(e_{im}=\widehat y_{im}^{(-i)}-y_i\), \(q_{im}=e_{im}^2\).
Let \(q_{(1)m}\leq\cdots\leq q_{(138)m}\) be the sorted squared errors.

The ascending **cumulative RMSE curve** is

\[
C_m(k)=\sqrt{\frac{1}{k}\sum_{j=1}^{k}q_{(j)m}},
\qquad 1\leq k\leq138.
\]

Here \(C_m(138)\) reproduces the model's authentic full LOOCV RMSE.
The incremental change \(C_m(k)-C_m(k-1)\) shows where the largest
observed changes occur; since the order was chosen using observed
errors, an elbow is only a **descriptive diagnostic**.

The squared-error share of the \(r\) worst cases is

\[
S_m(r)=\frac{\sum_{j=138-r+1}^{138}q_{(j)m}}
{\sum_{j=1}^{138}q_{(j)m}}.
\]

A parcel with a large squared error is not necessarily *influential
in training*. This experiment only establishes which held-out cases
are hard to predict across existing models.

### Cross-model consensus

For each model, descending absolute residual rank 1 is hardest.
The program counts top-2, top-5 and top-10 appearances among one
representative of each broad family and reports mean normalized
difficulty rank. Reference26 hyperparameter/imputation/DBSCAN
variants count as **one family**, and Local04D climate variants as
**one family**. Compact14 is another family; the fixed ensemble
is reported as a model, **not an independent family vote**.
The consensus top two are selected by average family-relative rank;
the program also reports the **actual intersection** of each
representative model's top-2 and top-10 sets. A selected pair
and an empty exact top-2 intersection can occur simultaneously.

## Scientific interpretation / limitations

- Big residuals can reflect noisy targets, rare genuine agronomic
  extremes, missing predictors, or model regression toward the mean.
  Never label the two parcels erroneous on the basis of LOOCV alone.
- The 136-parcel RMSE shown after deleting the two selected *evaluation*
  errors uses **unchanged 138-parcel historical fits**. It is not an
  estimate of a newly fitted 136-label model's skill and is optimistic
  when the removed IDs are chosen for bad residuals.
- The model variants share observations, features and algorithms, so
  agreement is not independent evidence. Prior Local04D Checkpoint 06
  found top-10%-parcel error concentration (about 62.2% of SSE),
  and Checkpoint 09 robust-label methods **did not beat Local04D**
  under its original 16×41 protocol. Neither establishes bad labels.
- An extreme fold error \(e_{im}\) only concerns prediction of \(i\)
  when \(i\) was **excluded** from training. To decide whether having
  \((x_i,y_i)\) in training harms predictions for other parcels, a
  separate **refitted leave-suspect-label-out experiment** is necessary.
- The suspected fields should be audited against measurement provenance,
  area/unit definitions, geographic neighbors, and independent
  agronomic/weather/satellite information before any deletion.
- The primary objective remains predictions for the **59 hidden
  AgroCebada parcels**. The next stage, only if supported by diagnostics,
  should refit all 138, drop A, drop B, drop both; compare predictions
  of the same 59 targets and refit in each common 136 held-out fold.
  Distinguish screening on reused 138 labels from independent
  confirmation. Do not retune ensemble weights on the screened folds
  and call the resulting performance unbiased.

## Reproducibility

This PR adds the analysis tool and synthetic tests only. It does not
modify training recipes, frozen model configurations, the 59 predictions,
published website metrics, or existing LOOCV outputs. Run the script
on the workstation with original private reports, inspect the candidate
IDs **locally**, and communicate sanitized aggregate summaries before
any decisions about a final model.
