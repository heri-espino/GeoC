# tools/

Command-line entry points for reproducible GeoCebada workflows.

Agents should prefer these scripts for project-wide operations rather than recreating equivalent
notebook cells.

| Script | Role | Current status |
|---|---|---|
| `audit_official_tabular.py` | Fast official-tabular schema check used by CI | active |
| `audit_data_contract_v2.py` | Full Data Contract v2 audit | Checkpoint 01 closed; rerun after source/raw changes |
| `build_data_catalog.py` | Structural data catalog | utility |
| `build_external_data_manifest.py` | Manifest/hashes for local external files | utility |
| `download_external_data.py` | External-source downloader | use only when raw sources need refresh |
| `build_integration_fixtures.py` | Deterministic 12-parcel integration fixtures | Checkpoint 01 artifact builder |
| `build_parcel_feature_table.py` | Canonical 197-row Feature Table v1 | validated; do not rerun unless inputs/code change |
| `build_checkpoint_02.py` | Feature inventory, shift diagnostics, frozen folds, ablations, baselines | Checkpoint 02 closed |
| `build_agronomic_features_v1.py` | Target-free phenology/thermal/water/soil/nonlinear layer derived from Feature Table v1 | Checkpoint 03A closed |
| `build_empirical_features_v1.py` | Target-free temporal-geometry / relative-condition / sensor-similarity layer | Checkpoint 03B closed |
| `run_expression_discovery_03b.py` | Training-fold-only expression recurrence audit over frozen Checkpoint 02 folds | Checkpoint 03B closed |
| `run_checkpoint_03b.py` | One-command local runner: rebuild empirical layer, then run fold-local discovery audit | Checkpoint 03B closed |
| `run_checkpoint_03c.py` | Fixed-fold clean/competition representation benchmark with Ridge10 and ExtraTrees | Checkpoint 03C.1 closed |
| `run_checkpoint_03c2.py` | Competition-only nested model-family benchmark with fold-safe tuning/discovery | Checkpoint 03C.2 closed |
| `run_checkpoint_03c3.py` | Deterministic OOF review of three finalists and fixed equal-weight ensembles | Checkpoint 03C.3 closed/frozen |
| `run_checkpoint_04a.py` | Build 59-target geographic/feature/temporal topology, shift, autocorrelation, support tables and figures | Checkpoint 04A complete |
| `run_checkpoint_04b.py` | Repeated transductive pseudo-competition RMSE for global, local, spatial, graph and blended methods | **Checkpoint 04B implemented; run on workstation** |
| `generate_function_index.py` | Regenerate `docs/FUNCTION_INDEX.md` from public package API | run after public API changes |
| `inspect_reference_docx.py` | Inspect reference DOCX content | utility |
| `_netcdf_catalog_worker.py` | Internal NetCDF catalog worker | internal; not a direct user workflow |

Checkpoint 03 is closed as the First Modeling Delivery. Checkpoint 04 is active. New scripts should support target topology/similarity, transductive pseudo-competition validation, local/graph/domain-adaptation models and external-evidence enrichment.

See `docs/AGENT_GUIDE.md` before adding scripts.
