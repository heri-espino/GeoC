# Checkpoint 03A — Agronomic Feature Layer

**Status:** implementation complete; generated-artifact validation pending.  
**Opened:** 2026-09-18

Checkpoint 03A adds an **additive, target-free nonlinear/agronomic representation** over the
frozen Feature Table v1. It does not replace or mutate Feature Table v1.

## Goal

Create a compact set of interpretable nonlinear transformations before model tuning, using
agronomic structure rather than an exhaustive polynomial expansion over ~1,400 predictors.

The layer is stored separately so Checkpoint 03B can compare:

```text
B0 = Feature Table v1 only
B1 = Agronomic Features v1 only
B2 = Feature Table v1 + Agronomic Features v1
B3 = reduced/selected base + agronomic families
```

under exactly the same frozen Checkpoint 02 folds.

## Inputs

```text
data/processed/features_v1/parcel_features_all.csv
data/processed/features_v1/feature_manifest.json
configs/agronomic_features_v1.yaml
```

No raw external data are required to build this layer.

## Outputs

```text
data/processed/agronomic_features_v1/
├── parcel_agronomic_features.csv
├── feature_manifest.json
├── build_report.json
└── README.md
```

The CSV contains only `ID_POLIGONO` plus derived predictors. It deliberately excludes
`RENDIMIENTO_T_HA` and `CONJUNTO`.

Join through:

```python
from geocebada.features import join_agronomic_features

model_table = join_agronomic_features(base_table, agronomic_table)
```

The join is one-to-one by parcel.

## Feature families

The implemented layer contains:

- phenology/seasonal curve shape from monthly vegetation indices;
- 2025-vs-historical vegetation anomalies;
- Sentinel-2 vs Planet agreement/disagreement;
- temperature, diurnal range, GDD proxies and heat-excess basis terms;
- rainfall timing;
- WaPOR NPP/AETI water-productivity proxies;
- crop-state × water interactions;
- SoilGrids vertical-profile contrasts and nonlinear soil terms;
- historical productivity × current crop-state interactions;
- terrain × climate and soil × climate interactions;
- a small set of log basis terms for non-negative skewed predictors.

The detailed formula, meaning, rationale, evidence class, input columns, references and caveats
for each feature are stored in the generated `feature_manifest.json` and documented in
`docs/AGRONOMIC_FEATURES_V1.md`.

## Evidence classes

Every derived feature is classified as one of:

`literature_backed`
: the family/relationship is directly motivated by crop or barley literature.

`mechanistic_proxy`
: the feature has a clear agronomic/physical interpretation, but the exact formula is a
project proxy rather than a standard published index.

`experimental`
: a target-free nonlinear basis/interacting hypothesis to test predictively; it must not be
reported as an established agronomic law.

The supporting bibliography is versioned at:

```text
docs/references/agronomic_features_v1.bib
```

## Leakage contract

Checkpoint 03A is deliberately **X-only**.

It may use all 197 parcels because no yield labels participate in the transformations.

It does not:

- rank candidate interactions against `RENDIMIENTO_T_HA`;
- choose thresholds from the 138 labeled targets;
- reconstruct hidden targets;
- use the 59 prediction parcels as supervised validation;
- select features because they improve a full-data target correlation.

Any target-guided selection must happen inside training folds in Checkpoint 03B.

## Clean vs competition provenance

Derived features inherit the most restrictive mode of their parents.

Historical/static-only transformations remain `clean`.

Anything using 2025 satellite, target-year climate, WaPOR or other competition-mode parents is
`competition`.

The manifest stores this mode per feature.

## CHIRPS decision

CHIRPS is intentionally excluded from Agronomic Features v1.

Checkpoint 02 showed that only one daily CHIRPS feature survived Feature Table v1, so it would
be misleading to build a larger nonlinear family on top of the current CHIRPS representation.
CHIRPS remains a separate QC item and can be added in a later revision if its extraction is
fixed and validated.

## Kernel decision

Polynomial/RBF kernels are not stored as processed features.

A kernel depends on model preprocessing and hyperparameters such as scaling, degree and
`gamma`; those transforms therefore belong inside Checkpoint 03B CV pipelines.

Checkpoint 03A stores explicit interpretable transformations only.

## Reproducible build

```powershell
conda activate geocebada
python tools\build_agronomic_features_v1.py
```

The builder validates parcel coverage, removes all-null/constant derived columns, rejects
infinite values, emits a machine-readable provenance manifest and records source hashes.

## Acceptance criteria

Checkpoint 03A closes when:

1. CI passes Ruff, tests, function-index consistency and Sphinx;
2. the builder returns exactly 197 unique parcel IDs;
3. the generated table contains no target or split column;
4. no retained derived feature contains infinities;
5. the manifest covers every retained feature;
6. clean features contain no 2025/WaPOR parent;
7. generated CSV, manifest and build report are committed;
8. project handoffs/history are updated with the final observed counts.

After closure, the next phase is **Checkpoint 03B — Modeling**, initially comparing the base
table, agronomic-only and joined representations with the frozen Checkpoint 02 folds.
