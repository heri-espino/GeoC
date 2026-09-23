# Checkpoint 03D plan — serious global models

Checkpoint 03D answers one remaining methodological question: whether the weak
global contribution seen in Checkpoint 05 was partly caused by the deliberately
small model-search budget inherited from 03C.2.

The experiment is intentionally expensive but still nested and auditable. It
does not use the 59 hidden yields, does not alter frozen 03C results, and does
not reopen feature discovery.

## Decision path

1. Run large CatBoost/XGBoost/LightGBM/ExtraTrees/HistGB models on deterministic
   competition representations.
2. Select by worst of the two historical OOF protocols.
3. Retain three model-family-diverse global finalists.
4. Refit the finalists on all 138 labels using combined protocol-aware inner CV.
5. Export and numerically verify the strongest deployable global model in ONNX.
6. Only then open Checkpoint 05B and test whether these stronger global experts
   add stable target-matched signal to Local04D.

A better 03D global score does not itself replace the competition method. The
fixed-target decision still belongs to 05B.
