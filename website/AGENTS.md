# GeoCebada website maintenance contract

This directory is a static, jury-facing Spanish presentation; not the Python analysis
engine nor an arbitrary-parcel forecasting service.

Read before making changes:

1. `docs/METODOLOGIA_Y_RESULTADOS.md` — canonical Checkpoint 02–08 methods,
   exact results and limits.
2. `website/README.md` — four canonical CSV snapshots, deployment and rationale.
3. `docs/TRANSDUCTIVE_OBJECTIVE.md` — 197 known X, 138 known y and 59 hidden y.
4. `models/README.md` and `reports/checkpoint_05/model_manifest.json` — fixed Local04D.

Invariants:

- The one-page submission document lives at `website/resumen.html` (letter print CSS).
  Keep its metrics, caveats and selected model identical to the canonical method report.
- Cross-validation teaching must explain the two distinct mask generators (04B and 08),
  97/41, 16 repetitions per bank, 3-fold inner tuning for 08 challengers, and the
  difference between pseudo-target labels and the 59 actually hidden target yields.

- Selected fixed-target model = `Local04D` unless a genuinely verified documented
  model promotion changes the canonical 59-row output and comparison narrative.
- Checkpoint 07 uses the historical 16 target-matched masks, 125 unique parcels,
  Local04D pooled RMSE 0.495741. Checkpoint 08 uses **different** 16 confirmation
  masks, 132 unique parcels, fixed Local04D pooled RMSE 0.481936.
- Never plot 07 and 08 as one ranked sample or present their difference as a
  within-model improvement. Never call the repeated 656 evaluations independent.
- All source metrics are read from CSV snapshots synchronized byte-for-byte by
  `python tools/sync_website_data.py --check` / `--write`.
- The 59-value explorer is a fixed-model predictions lookup. Never claim an
  arbitrary-new-parcel API, live training, or actual hidden-y RMSE.
- Figure geometry in the hero and neighborhood diagram is **conceptual**.
  Never fabricate actual parcel positions or imply these diagrams are maps.
- Source inputs, code, and official PDFs are immutable; never include raw labels,
  secrets, massive rasters or model weights in the site.
- Check publication permissions before exposing new artifacts.
- Keep Spanish explanatory copy, keyboard accessibility, reduced-motion support,
  responsive display and no third-party analytics.
- Prefer stable internal links and document all new dependencies if introduced.
- This static site is independent of `app/` Streamlit. Do not substitute the
  Checkpoint 03D generic global ONNX for the Local04D fixed-59 output.

After modifications, run `python tools/sync_website_data.py --check` and
`pytest -q tests/test_website_presentation.py`. Record material new AI-generated
work in `docs/AI_USAGE.md`. Deployment of a purely static website is a lightweight
workflow. Heavy figures, papers, PDFs, exports and training must remain manual.
