# Top-five corrected Reference26: fixed LOOCV versus nested selection

**Date:** 2026-10-19. **Status:** nested full 138/138 LOOCV
COMPLETED on the user's workstation; final 59 candidate export
implemented separately and pending local execution. Do not overwrite
the original model or its 59 predictions.

## The key distinction

The first `tools/run_reference26_analog_tail_screen.py --run` ALREADY
computed 138 true outer 137-training / one-real-heldout predictions
for all five models with **FIXED** correction hyperparameters. For
each heldout, it uses the original Reference26 fit that excludes that
yield, and neighbor yields of the remaining 137. Re-running that
evaluation would repeat the same 138 fixed corrections, NOT
independently validate the hyperparameter choice.

The five candidates are listed below in descending order of observed
performance from the original post-hoc screen:

| Method | Geometry | k | blend | gate | prior 138-LOOCV RMSE |
|---|---|---:|---:|---|---:|
| 1 | robust26 | 3 | 0.25 | supported | **0.462418** |
| 2 | pca8 | 3 | 0.25 | supported | 0.464956 |
| 3 | pca8 | 8 | 0.25 | supported | 0.465506 |
| 4 | robust26 | 3 | 0.10 | supported | 0.465538 |
| 5 | robust26 | 3 | 0.10 | none | 0.466518 |
| Unchanged control | Reference26 mean | — | — | — | 0.468138 |

Those six numbers are **already observed** on the exact same 138
real parcels, and are deliberately not called fresh confirmation.

## New experiment: nested choice of correction inside each outer fold

The actual additional experiment evaluates a **six-choice policy**
(the top five plus unmodified Reference26) with new,
fully training-allowed predictions inside each outer fold.

For each outer heldout parcel i:

1. The **outer training set** comprises the 137 original parcels
   other than i. The original complete Reference26 mean OOF
   prediction for i is already saved and validated by ID/y, so
   do not waste another expensive refit recreating it.
2. Create **four inner folds on those 137 only**, shuffled with
   fixed seed 20261019. Each inner training set has 102 or
   103 real observed labels; the inner test part has 34 or 35
   original parcels.
3. For each of the four inner folds, **refit original native
   Reference26** from scratch on only the inner training labels,
   keeping the arithmetic training-only mean imputation and native
   internal training-CV intact (see
   `run_reference26_compact14_final.build_fresh_fit`).
   Its input target frame masks **both** the outer heldout
   yield and the current inner validation yields.
4. Predict inner validation cases and apply the five candidate
   feature-analog corrections. Each uses real training **y
   only for the inner fitting subset** and observable transductive
   197-parcel X. Never read the outer heldout y.
5. Assemble four inner folds into 137 out-of-fold baseline
   and corrected predictions. Score all six options with MSE.
   Select the best correction only when its inner MSE
   decreases by at least **1% relative** to the unchanged
   inner Reference26 MSE; otherwise retain unmodified Reference26.
6. Apply that choice to i's already validated **outer**
   Reference26 mean prediction, using all 137 permitted
   observed yields as analog neighbors. The five
   candidate outer predictions are verified against the previous
   fixed-method screen, to establish numerical parity.
7. Save one private outer prediction/selection and all six
   per-candidate outer predictions. Repeat for all 138 original
   parcels.

**Total:** four additional native Reference26 fits for each
of 138 outer folds = **552 extra fits**, not
138 × five = 690. All five corrections share the SAME four
inner Reference26 predictions; no redundant native fitting.
An original fit may also carry nested native internal 4-fold
fit/selection work, so actual internal cost is higher.
The script has cache/resume, process concurrency and SHA-based
specification fingerprints. Existing native outer OOF predictions
remain untouched.

A full run reports RMSE, MAE, bias, predictive R² for all five
fixed variants and for **NestedSelection**, plus how many folds
selected each option (including the unchanged fallback).

### Evaluation caveat that cannot be fixed retroactively

These *five candidate hyperparameter specifications were picked
using the earlier full 138-LOOCV errors* and were not specified
before any label in this dataset was examined. This means that
even four-fold inner selection in every outer fold is **only nested
conditionally on that previously chosen family**. It does not
turn the same 138 previously-inspected yields into new independent
confirmation. True study-wide unbiased performance would require
defining the candidate search inside every outer fold from scratch
(including the original Reference26 search), or real new yields.

This experiment can still teach us whether the policy reliably
chooses to correct when facing a new heldout parcel given the
preselected five. If the nested-selection policy is not better,
there is no reason to promote the post-hoc winning fixed recipe
based on its small 0.005720 t/ha RMSE advantage alone.

## Commands

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/reference26-top5-nested-loocv
git pull --ff-only

python -m py_compile tools/run_reference26_top5_nested.py
python -m pytest tests/test_reference26_top5_nested.py -q

# Fast: verifies original five fixed corrected 138-fold LOOCV predictions.
python tools/run_reference26_top5_nested.py --check
python tools/run_reference26_top5_nested.py --quick

# First expensive training proof: one outer fold with four inner native fits.
python tools/run_reference26_top5_nested.py --smoke --jobs 1

# Real nested evaluation: resumable, 138 outer folds.
python tools/run_reference26_top5_nested.py --run --jobs 4
python tools/run_reference26_top5_nested.py --summary
```

Optional partial batches on a reboot-prone Windows workstation:

```powershell
python tools/run_reference26_top5_nested.py --run --jobs 4 --from-fold 1 --through-fold 46

python tools/run_reference26_top5_nested.py --run --jobs 4 --from-fold 47 --through-fold 92

python tools/run_reference26_top5_nested.py --run --jobs 4 --from-fold 93 --through-fold 138

python tools/run_reference26_top5_nested.py --summary
```

All commands above are single-line PowerShell commands.

Private outputs, not committed to Git:

- `reports/reference26_top5_nested/run_spec.json`
- `reports/reference26_top5_nested/fold_001.csv` ... `fold_138.csv`
- `reports/reference26_top5_nested/native_inner_cache/`
- `reports/reference26_top5_nested/comparison_aggregate.csv`
- `reports/reference26_top5_nested/private_nested_predictions.csv`
- `reports/reference26_top5_nested/selection_counts_aggregate.json`

The original 59 hidden target predictions, previous Reference26
joblib and deployed website remain unchanged. Do **not** interpret
the existence of 552 fits as 552 additional distinct real test
parcels: only 138 real outer heldouts are used.

## Resultados de la VM, ejecutados por el usuario — 2026-10-19

**138/138 folds completados.** Se confirmó que los cinco modelos de
corrección fija mantienen exactamente sus puntuaciones de LOOCV
preexistentes. La **política con selección interna** logró:

| Predictor | RMSE | MAE | Bias | R² CV |
|---|---:|---:|---:|---:|
| Reference26 mean sin corregir | 0.468138 | 0.333786 | 0.001466 | 0.695292 |
| **NestedSelection (4fold interno)** | **0.462892** | **0.331956** | **-0.005062** | **0.702083** |
| robust26 k3 .25 supported fijo | 0.462418 | 0.332019 | -0.004103 | 0.702693 |
| PCA8 k3 .25 supported fijo | 0.464956 | 0.334091 | -0.000414 | 0.699421 |
| PCA8 k8 .25 supported fijo | 0.465506 | 0.334439 | -0.004101 | 0.698709 |
| robust26 k3 .10 supported fijo | 0.465538 | 0.332972 | -0.000762 | 0.698668 |
| robust26 k3 .10 sin filtro fijo | 0.466518 | 0.335959 | -0.004757 | 0.697397 |

La política eligió `robust26/k3/blend0.25/supported` en **128 folds**,
`PCA8/k3/blend0.25/supported` en **5** y Reference26 original en
**5**; nunca eligió los otros tres candidatos.

La reducción observada de RMSE del selector contra la línea
base es **0.005246 t/ha (≈1.12%)** y de MAE **0.001830 t/ha**.
No obstante, las cinco recetas son seleccionadas a posteriori
tras inspeccionar previamente las mismas 138 etiquetas; no es
un test independiente del proceso histórico de selección.
El resultado pertenece al protocolo condicional descrito arriba.

**Siguiente paso implementado:** `tools/run_reference26_top5_final59.py`
prepara la decisión final usando cuatro inner folds sobre las
138 etiquetas reales y predice las 59 desconocidas con el
modelo nativo Reference26 MEAN entrenado con 138. Ver
`docs/REFERENCE26_FINAL59_SELECTED.md`. Los 59 resultados
están pendientes de ejecución en la VM, sin promoción ni
cambios en los joblibs originales.
