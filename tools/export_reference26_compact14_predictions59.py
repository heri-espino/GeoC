#!/usr/bin/env python3
"""Verify, stage and explicitly publish 59 final THREE-MODEL predictions.

Never trains any model. Loads both newly refitted 138-label joblibs and
checks their independent 59-row predictions against the stored private CSV.
Verifies 59 exact IDs, frozen convex weights, joblib SHA256 and original
197-row feature/distance assets before staging any JSON containing IDs.

    python tools/export_reference26_compact14_predictions59.py --check
    python tools/export_reference26_compact14_predictions59.py --stage
    python tools/export_reference26_compact14_predictions59.py --publish

CAUTION: --publish copies 59 IDs and inferred yields to the PUBLIC static
site; inspect the staged JSON and contest disclosure conditions first.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import run_reference26_compact14_final as runner

BASE = runner.OUTPUT
STAGE = BASE / "publication_staging" / "ensemble_predictions59.json"
SITE = (runner.ROOT.parent / "todoseboso/site/data"
        / "ensemble_predictions59.json")
KEYS = {
    "Reference26_mean": ("reference26_mean_train138.joblib",
                         "Reference26", "Reference26_mean_t_ha"),
    "Compact14": ("compact14_original_train138.joblib",
                  "Compact14", "Compact14_t_ha"),
}


def audited_predictions(
    *, source: Path, bundle: Path, peer: Path, config: Path
) -> dict:
    frame, groups, distance, manifest, ids, cls, predictor = (
        runner.load_inputs(bundle, peer)
    )
    weights = runner.frozen_weights(config)
    spec = source / "run_spec.json"
    expected = runner.provenance(bundle, peer, config)
    if not spec.is_file() or json.loads(spec.read_text()) != expected:
        raise ValueError("The full-refit experiment provenance does not match.")
    output = source / "final_artifacts"
    completed = output / "final_manifest.json"
    if not completed.is_file():
        raise FileNotFoundError("The two native 138-refit joblibs are incomplete.")
    record = json.loads(completed.read_text(encoding="utf-8"))
    if (record["n_train"] != 138 or record["n_unknown"] != 59
            or record["weights"] != {
                "Reference26_mean": weights[0], "Compact14": weights[1],
            }):
        raise ValueError("The final model manifest and weights differ.")
    full, d197 = runner.prepare_197(peer, frame, distance)
    for field, path in (
        ("features197.parquet",
         peer / "deployment/model_artifacts/features197.parquet"),
        ("distance197.npy",
         peer / "deployment/model_artifacts/distance197.npy"),
    ):
        if runner.sha(path) != record["input_artifacts"][field]:
            raise ValueError(f"Changed input artifact {field}; cannot publish.")
    stored = pd.read_csv(output / "predictions59_private.csv",
                         dtype={runner.ID: str})
    needed = {runner.ID, "Reference26_mean_t_ha",
              "Compact14_t_ha", "Ensemble_final_t_ha"}
    if set(stored.columns) != needed or len(stored) != 59:
        raise ValueError("Stored final prediction CSV has incorrect structure.")
    expected_ids = full.iloc[138:][runner.ID].astype(str).to_numpy()
    if (stored[runner.ID].duplicated().any()
            or not np.array_equal(stored[runner.ID].to_numpy(), expected_ids)):
        raise ValueError("The 59 final parcel IDs or their order differ.")
    q = np.arange(138, 197)
    results = {}
    for name, (filename, key, col) in KEYS.items():
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required fitted model {path}.")
        if runner.sha(path) != record["trained_models"][filename]:
            raise ValueError(f"Joblib SHA-256 differs: {filename}.")
        pack = joblib.load(path)
        if (pack.get("model_name") != name or
                pack.get("imputation") != (
                    "arithmetic_train_means" if name == "Reference26_mean"
                    else "original_medians"
                ) or not np.array_equal(pack.get("train_positions"),
                                        np.arange(138))):
            raise ValueError(f"Unverified model recipe: {filename}.")
        predictions = runner.predict_model(
            predictor, pack, q, full, d197, key
        )
        np.testing.assert_allclose(
            predictions, stored[col].to_numpy(float),
            rtol=0, atol=1e-9,
            err_msg=f"Python replay differs for {filename}.",
        )
        results[name] = predictions
    combined = weights[0]*results["Reference26_mean"] + (
        weights[1]*results["Compact14"]
    )
    np.testing.assert_allclose(
        stored["Ensemble_final_t_ha"].to_numpy(float),
        combined, rtol=0, atol=1e-9,
        err_msg="The final 59 mix differs from saved component predictions."
    )
    if not np.isfinite(combined).all():
        raise ValueError("Predictions must be finite.")
    rows = [
        {
            "id": str(parcel),
            "reference26_mean_t_ha": float(pred_a),
            "compact14_t_ha": float(pred_b),
            "ensemble_t_ha": float(mix),
        }
        for parcel, pred_a, pred_b, mix in zip(
            expected_ids, results["Reference26_mean"],
            results["Compact14"], combined, strict=True
        )
    ]
    return {
        "schema_version": 1,
        "scope": "competition_unknown_59_no_observed_y",
        "n_known_training": 138,
        "n_predictions": 59,
        "parity": {
            "python_replayed_both_joblibs": True,
            "max_absolute_error_to_stored_csv_t_ha": float(max(
                max(abs(results[name]-stored[col].to_numpy(float)))
                for name, (_, _, col) in KEYS.items()
            )),
            "max_absolute_ensemble_replay_error_t_ha": float(max(
                abs(combined-stored["Ensemble_final_t_ha"].to_numpy(float))
            )),
        },
        "weights": {
            "Reference26_mean": weights[0],
            "Compact14": weights[1],
        },
        "model_sha256": record["trained_models"],
        "no_true_y_available_for_59": True,
        "exploratory_weight_selection_on_original_138": True,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for key in ("check", "stage", "publish"):
        mode.add_argument("--"+key, action="store_true")
    p.add_argument("--source", type=Path, default=BASE)
    p.add_argument("--stage-file", type=Path, default=STAGE)
    p.add_argument("--site-file", type=Path, default=SITE)
    p.add_argument("--bundle", type=Path, default=runner.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=runner.PEER_REPO)
    p.add_argument("--config", type=Path, default=runner.CONFIG)
    args = p.parse_args()
    verified = audited_predictions(
        source=args.source.resolve(), bundle=args.bundle.resolve(),
        peer=args.peer_repo.resolve(), config=args.config.resolve(),
    )
    if args.check:
        print("[VERIFY59] Both new 138-refit joblibs, sha256, "
              "197 geometry and all 59 Python predictions agree. "
              "No output published.", flush=True)
        return 0
    destination = args.stage_file.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        verified, indent=2, ensure_ascii=False, allow_nan=False
    )+"\n"
    if destination.is_file() and destination.read_text() != payload:
        raise ValueError(
            "A different gallery staging file exists; "
            "review it before overwriting."
        )
    destination.write_text(payload, encoding="utf-8")
    print(f"[STAGE59] Verified 59 forecast-only rows staged privately "
          f"to {destination}; no true yields included.", flush=True)
    if args.stage:
        return 0
    site = args.site_file.resolve()
    if site.is_file():
        raise FileExistsError(
            f"{site} already exists; review manually instead of replacing."
        )
    site.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination, site)
    print("[PUBLISH59] Explicit local staging of IDs and 59 inferred "
          f"yields for public website at {site}. "
          "Review privacy/competition permission and commit separately.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
