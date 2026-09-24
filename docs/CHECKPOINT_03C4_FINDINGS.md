# Checkpoint 03C.4 findings — historical nested stacking experiment

**Executed:** 2026-09-20  
**Status:** HISTORICAL / CLOSED / SUPERSEDED BY CHECKPOINT 05  
**Hidden FIRA target y used:** no

Checkpoint 03C.4 tested leakage-safe nested stacking before the project moved to
the target-matched transductive validation used in Checkpoints 04–05. The
experiment is preserved because it is useful negative evidence about learned
stackers under the original frozen state-stratified and municipality-grouped
protocols.

## Validation design

For every outer fold, 03C.4 generated level-1 predictions by cross-fitting base
learners inside the outer-training labels. Base-model tuning was nested inside
those level-1 folds; the complete base-plus-stacker procedure then predicted the
untouched outer validation fold.

The experiment used 10 base learners spanning PLS, Ridge, CatBoost, ExtraTrees
and PCA+Ridge, then compared:

- EqualTop3;
- MeanAll;
- ConvexAll;
- RidgeStack;
- ElasticNetStack;
- HuberStack;
- ExtraTreesStack;
- HistGBStack.

No final predictions for the fixed 59 targets were produced.

## Main result

The simple frozen ensemble remained much more robust than learned stackers:

| meta model | state OOF RMSE | grouped OOF RMSE | worst protocol |
|---|---:|---:|---:|
| EqualTop3 | 0.5109 | **0.7481** | **0.7481** |
| MeanAll | 0.5127 | 0.7609 | 0.7609 |
| ConvexAll | 0.5110 | 0.8246 | 0.8246 |
| HistGBStack | 0.5711 | 1.3582 | 1.3582 |
| RidgeStack | 0.5343 | 1.3806 | 1.3806 |
| ExtraTreesStack | **0.4920** | 1.3871 | 1.3871 |
| HuberStack | 0.5567 | 1.4855 | 1.4855 |
| ElasticNetStack | 0.5245 | 1.6024 | 1.6024 |

ExtraTreesStack looked strongest under state-stratified CV, but collapsed under
municipality-grouped validation. This was early evidence that selecting a
stacker from the easier protocol could substantially overstate robustness.

## Interpretation in the final project

03C.4 is not an active model stage and must not be confused with Checkpoint 05.
Checkpoint 05 later revisited global/local stacking under the more appropriate
target-matched pseudo-competition contract and likewise found that a small
development gain did not survive split-excluded promotion.

The historical 03C.4 implementation and generated outputs are preserved inertly
under:

`archive/checkpoint_03c4_nested_stacking/`

They are archived for provenance only and are not imported, tested or used by
the current pipeline.
