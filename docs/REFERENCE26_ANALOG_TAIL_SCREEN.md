# Experiment: direct analog and tail corrections to Reference26 mean

**Date 2026-10-19 — full authentic 138-fold VM screen completed; performance exploratory.**
Branch: `experiment/reference26-analog-tail-correction`. No official
prediction promotion; do not automatically merge or change Todoseboso.

## Motivation

Reference26 mean with arithmetic training-only imputation remains the
strongest interpretable model found in this competition, with authentic
138 × (137 training, 1 heldout) parcel LOOCV RMSE **0.468138 t/ha**.
A long chain of efforts with two synthetic Ridge experts, a supervised
three-class router, global Ridge and ExtraTrees trained with 256/1024
synthetic samples, 2×/5× weights on 8 hard or X-isolated observations
and balanced augmentation on ALL 137 training parcels did not improve
this benchmark.

The last VM full CV (PR #32) gave best augmented ExtraTrees 0.487998,
plain Ridge 0.492732; thus neither a larger fake dataset nor a
different global regressor should be presumed better.

A low-cost next test should work **directly on the existing frozen
Reference26 predictions**, not replace the winner.

## Candidate A: real agronomic feature analogs

For a real heldout parcel i, let `I_minus_i` denote the original
137 real **training** parcels. Standardize the 26 original champion
features using X from all 197 observable parcels, permitting
X-only transductive PCA (8 components) and comparison with original
26 robust-scaled variables.

Find k nearest real training X feature vectors to the heldout parcel.
The neighbor response estimate is the inverse-distance weighted
mean of **real allowed labels**:

```text
a_i = sum_{j in kNN(X_i; I_minus_i)}
          (w_ij y_j) / sum_j w_ij,
w_ij = 1 / max(d(X_i,X_j), 1e-5).
```

The corrected forecast is

```text
yhat_i = yhat_Reference26_LOO_i
      + clip(lambda * g_i *
         (a_i - yhat_Reference26_LOO_i), -delta, delta).
```

Two gates:
- `none`: g_i=1.
- `supported`: g_i is a product of X-only nearest-distance
  support and training-only neighbor-response agreement. Its distance
  scale is the median training X-space nearest-neighbor distance.

Candidate grid: feature representations original robust26 / PCA8,
k=3/8/16, lambda=0.1/0.25, gate=none/supported and
maximum correction delta=0.5 t/ha. This yields 24 analog candidates.
No synthetic data, no fitted classifier and no extra true labels.

Crucial: all neighbor Y and train-Y variability come exclusively from
the current 137 training parcel labels. If a hard heldout parcel is
X-dissimilar to its historical analogs, the support gate can suppress
the correction. This is deliberate and guards already-good forecasts.

## Candidate B: conservative tail expansion

The Checkpoint 06 findings showed empirical tail compression: small
observed yields often overpredicted and high yields often underpredicted.
Apply a **fixed, limited expansion** of the already-saved
Reference26 prediction away from the *other 137 training-label mean*:

```text
yhat_i = yhat_ref_i + clip(
 gamma * sign(yhat_ref_i - mean(y_train)) *
 max(|yhat_ref_i - mean(y_train)| - t*sd(y_train), 0),
 -delta, delta).
```

Try gamma=0.1/0.2/0.4 and t=0.5/1.0 SD. This is six
predeclared transformations, with the exact unmodified Reference26
prediction as a mandatory 31st row (1+24+6=31).

This is a screen, NOT a claim this form must improve anything. Earlier
tail-correction attempts were unsuccessful under different protocols.

## Leakage and selection

- Never use the current heldout y in a training neighbor average,
  training y mean, threshold or gate.
- Start from original *saved genuine* Reference26 mean 138-fold
  predictions, which were obtained without that heldout y.
- Using observable X from all 197 is permitted in this competition
  transductive setting. The 59 hidden Y values are never read.
- Do NOT fit the correction using other heldout OOF residuals as targets:
  their Reference26 models were fitted with the current heldout y,
  so a correction trained on those full 138 OOF residuals could leak.
- The user has repeatedly examined the same 138 labels and observed
  model errors. Selecting the minimum of 31 observed RMSE scores is
  **post-hoc exploratory model selection** and not independent
  confirmation of generalization.
- Improvement should be meaningful relative to 0.468138 t/ha on the
  same 138, not just the two historically hard parcels. Report
  changed/worsened cases on all 138 and on other 136.
- A score no better than Reference26 means *keep Reference26*. No
  candidate59 changes without explicit follow-up.

## Commands

```powershell
cd C:\Users\175199\GeoC
conda activate geocebada
git fetch origin
git switch experiment/reference26-analog-tail-correction
git pull --ff-only
python -m py_compile tools/run_reference26_analog_tail_screen.py
python -m pytest tests/test_reference26_analog_tail_screen.py -q
python tools/run_reference26_analog_tail_screen.py --check
python tools/run_reference26_analog_tail_screen.py --smoke
python tools/run_reference26_analog_tail_screen.py --run
python tools/run_reference26_analog_tail_screen.py --summary
```

Local private outputs (gitignored):

- `reports/reference26_analog_tail_screen/comparison_aggregate.csv`
- `reports/reference26_analog_tail_screen/private_heldout_predictions.csv`
- `reports/reference26_analog_tail_screen/run_spec.json`

A potential explicitly chosen candidate can be applied to the existing
138-fit 59-target **native Reference26 mean** predictions, but only
after evaluating this screen and only via the separate
`--predict59` flag. It produces private
`candidate59_private.csv`, never changes the original file, model
artifacts or deployment.

Example — only a demonstration, not a selected model:

```powershell
python tools/run_reference26_analog_tail_screen.py --predict59 `
  --method analog --geometry robust26 --k 8 --blend 0.1 `
  --gate supported
```

## Interpretation

This is a direct correction of the Reference26 predictions, not a
refitting of its native CT3/local Ridge/residual architecture. Its
attraction is that the original model stays fixed and the additional
information comes from genuine neighbor yields in the fold rather than
artificial pseudo-labels. Its limitation is that nearest X analogs may
not imply similar agronomic yield and Reference26 may already have
captured their useful information. Record a negative result if it fails.


## Resultados de la ejecución real en VM — 2026-10-19

El usuario ejecutó `--check`, `--smoke`, `--run` y `--summary`.
Las 31 recetas utilizaron 138 predicciones auténticas de Reference26
(137 etiquetas disponibles / 1 test por fold). Estas métricas provienen
del log agregado aportado por el usuario; los archivos por parcela
permanecen únicamente en la VM y nunca deben incluirse en Git.

| Modelo | RMSE (t/ha) | MAE (t/ha) | RMSE 2 difíciles | RMSE otras 136 |
|---|---:|---:|---:|---:|
| **Reference26 mean, sin corrección** | 0.468138 | 0.333786 | 2.065000 | 0.399583 |
| **robust26, 3 vecinos, blend 0.25, supported** | **0.462418** | **0.332019** | 2.063534 | **0.392880** |
| PCA8, 3 vecinos, blend 0.25, supported | 0.464956 | 0.334091 | 2.050346 | 0.396914 |
| PCA8, 8 vecinos, blend 0.25, supported | 0.465506 | 0.334439 | 2.062307 | 0.396657 |
| robust26, 3 vecinos, blend 0.10, supported | 0.465538 | 0.332972 | 2.064380 | 0.396536 |

Ganancia observada del primer candidato:
**RMSE -0.005720 t/ha (-1.22% relativo)** y
**MAE -0.001767 t/ha**. La corrección modificó las 138
predicciones de test, mejoró el error cuadrático de 65 y empeoró
el de 73, con cambio absoluto máximo de **0.108689 t/ha**.
El RMSE de las dos parcelas históricas extremas casi no cambió;
la mejora descriptiva está en las otras 136.

**Límite clave:** se escogió la mejor de 31 recetas por los
mismos 138 heldouts usados en varias exploraciones anteriores.
La reducción es válida como desempeño calculado en estos folds
con parámetros a posteriori, pero **NO es evidencia confirmatoria
independiente** ni demuestra desempeño futuro en las 59 parcelas
sin etiqueta. No alterar la salida final de 59 antes de evaluar
estabilidad pareada y sensibilidad espacial/de una observación.
Además, ningún resultado sugiere que `tail` por sí solo haya
superado a Reference26 (p. ej. `tail__g0.1__t1` RMSE 0.468808).

Un diagnóstico agregado y protegido de archivos privados se añadirá
a `tools/audit_reference26_analog_gain.py`: bootstrap pareado,
estabilidad al eliminar cada observación, corrección por la búsqueda
entre recetas y sensibilidad por bloques de distancia geográfica.
