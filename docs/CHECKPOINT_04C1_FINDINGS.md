# Checkpoint 04C.1 — Focused CatBoost anchor findings

Checkpoint 04C.1 tested whether the historically relevant C1 agronomic
CatBoost signal adds useful global diversity to the much stronger transductive
Local04D and Graph04D predictors.

The experiment used the exact frozen Checkpoint 04B pseudo-competition
memberships, 350 agronomic features, three CatBoost configurations inherited
from Checkpoint 03C.2, and three deterministic seeds per candidate. The 59
hidden FIRA yields were never loaded or scored.

## Integrity and execution

- CatBoost task type: **GPU**
- agronomic features: **350**
- CatBoost candidates: **3**
- seeds per candidate: **3**
- target-matched pseudo-splits: **16**
- hidden competition targets used: **no**

## Primary target-matched result

| method | mean split RMSE | pooled RMSE | pooled MAE |
|---|---:|---:|---:|
| Local + CatBoost 10% | 0.487842 | 0.495915 | 0.342266 |
| **Baseline Local04D** | **0.487845** | **0.495741** | 0.342501 |
| Local + CatBoost 20% | 0.488501 | 0.496728 | 0.342952 |
| Local + CatBoost 30% | 0.489822 | 0.498177 | 0.344425 |
| Graph + CatBoost 30% | 0.491439 | 0.500394 | 0.339809 |
| Baseline Graph04D | 0.494881 | 0.503346 | 0.340288 |
| CatBoost C1 depth5/lr0.06/l2=3 | 0.515282 | 0.523869 | 0.370873 |
| CatBoost C1 depth4/lr0.03/l2=3 | 0.516796 | 0.525414 | 0.371216 |
| CatBoost C1 depth4/lr0.06/l2=6 | 0.520583 | 0.528819 | 0.373011 |

The nominal mean-RMSE difference between Local+CatBoost 10% and Local04D is
only about **0.0000035 t/ha**. The blend has a slightly worse pooled RMSE.
This is numerically negligible and is not evidence that CatBoost improves the
primary predictor.

## Controlled leave-one-split-out gate

The restricted LOSO gate could choose only Local04D, the stable CatBoost anchor,
or Local+CatBoost weights 10/20/30%.

It selected:

- **Baseline Local04D: 8/16** holdouts
- **Local + CatBoost 10%: 8/16** holdouts

The selected holdout predictions had mean RMSE **0.488884** and pooled RMSE
**0.496838**, both worse than simply keeping the frozen Local04D rule.

Therefore the tiny full-development-table difference at 10% does not survive a
more conservative split-excluded selection procedure.

## Residual diversity

Target-matched residual correlations were:

| pair | correlation |
|---|---:|
| Local04D vs Graph04D | 0.9707 |
| Local04D vs CatBoost | 0.9423 |
| Graph04D vs CatBoost | 0.9143 |

CatBoost is not identical to the local or graph predictors, but its residuals
remain highly correlated with them. The available diversity is not large enough
to compensate for its weaker standalone error.

## Stress protocols

CatBoost blending is more useful as a graph modifier than as a local modifier.

The 30% Graph+CatBoost blend improves target-matched mean RMSE from 0.4949 to
0.4914 and state-random RMSE from 0.4679 to 0.4627. However, it worsens
municipality-grouped mean RMSE from 0.5279 to 0.5508.

A 10% Graph+CatBoost blend almost preserves municipality-grouped performance
(0.5283) while improving target-matched RMSE to 0.4928, but it still remains
weaker than Local04D on the primary competition-matched protocol.

## Decision

**Do not include CatBoost as a required component of the final 04F predictor.**

Keep the CatBoost predictions as a sensitivity/diversity diagnostic only.
The evidence does not justify replacing Local04D or adding a nonzero CatBoost
weight to the final rule.

Checkpoint 04F should now be deliberately small:

1. freeze Local04D as the primary predictor unless a tightly controlled
   Local/Graph combination has convincing split-excluded evidence;
2. retain Graph04D as the robustness/disagreement reference;
3. avoid support-tier routing unless the candidate universe is explicitly
   constrained and validated out-of-split;
4. generate one canonical 59-row prediction table with provenance and a separate
   uncertainty/disagreement diagnostic.
