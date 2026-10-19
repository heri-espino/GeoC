#!/usr/bin/env python3
"""Produce the 59 FINAL CANDIDATE yields from the 138-label nested policy.

FIRA AgroCebada: 138 measured parcels / 59 targets. This script:
1. Requires genuine completed 138-outer-fold native Reference26 LOOCV AND
   the audited nested five-candidate experiment from PR #35.
2. Requires original independently refitted Reference26 MEAN train138 joblib,
   and its verified 59 saved baseline predictions; never retrains/replaces it.
3. Chooses between original Reference26 and the SAME top-five corrections
   by four NEW native Reference26 inner folds on ALL 138 measured yields.
4. Applies that chosen correction with all 138 real neighbor labels
   to all 59 already-fitted Reference26 forecasts.

The chosen rule is fit on 138 known outcomes; 59 unknown outcomes are never
read. Prior top-five recipe search was post-hoc, so conditional-nested CV
is exploratory; never claim independent test performance for the targets.

Only creates PRIVATE files under reports/reference26_top5_final59/.
Does not push individual yields to Git, modify original joblibs, overwrite
prior target predictions, or update Todoseboso.

  python tools/run_reference26_top5_final59.py --check
  python tools/run_reference26_top5_final59.py --smoke
  python tools/run_reference26_top5_final59.py --run
  python tools/run_reference26_top5_final59.py --summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import run_reference26_analog_tail_screen as analog
import run_reference26_compact14_final as native
import run_reference26_top5_nested as nested
from sklearn.model_selection import KFold

N, ID, TARGET = native.N, native.ID, native.TARGET
OUTPUT = native.ROOT / "reports/reference26_top5_final59"
SEED = nested.SEED
PREDICTIONS = "predictions59_final_nested_private.csv"
SUBMISSION = "submission59_candidate_private.csv"
MANIFEST = "final_selection_manifest.json"


def require_original_final(folder, peer, frame, full, d197, weights, predict_saved):
    """Verify existing original real-train138 Reference26 artifact and 59 yhat."""
    final = folder / "final_artifacts"
    manifest_path = final / "final_manifest.json"
    csv = final / "predictions59_private.csv"
    job = final / "reference26_mean_train138.joblib"
    if not all(p.is_file() for p in (manifest_path, csv, job)):
        raise FileNotFoundError(
            "Original trained-138 MEAN Reference26 final artifacts missing. "
            "Run: python tools/run_reference26_compact14_final.py --train-final"
        )
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (provenance.get("n_train") != N or
        provenance.get("n_unknown") != 59 or
        provenance.get("weights") != {
            "Reference26_mean": weights[0], "Compact14": weights[1]
        }):
        raise ValueError("Final Reference26 138/59 provenance mismatch.")
    expected_job_sha = provenance["trained_models"].get(job.name)
    if native.sha(job) != expected_job_sha:
        raise ValueError("Original Reference26 final joblib SHA differs.")
    evidence = provenance["input_artifacts"]
    for name in ("features197.parquet", "distance197.npy"):
        filename = peer / "deployment" / "model_artifacts" / name
        if native.sha(filename) != evidence[name]:
            raise ValueError(f"Original final feature/distance changed: {name}.")
    targets = full.iloc[N:][ID].astype(str).to_numpy()
    baseline_csv = pd.read_csv(csv, dtype={ID: str})
    if (len(baseline_csv) != 59 or
        baseline_csv[ID].duplicated().any() or
        not np.array_equal(baseline_csv[ID].to_numpy(), targets)):
        raise ValueError("Original final 59 ID values/order mismatch.")
    base = baseline_csv["Reference26_mean_t_ha"].to_numpy(float)
    if base.shape != (59,) or not np.isfinite(base).all():
        raise ValueError("Nonfinite or incomplete original 59 predictions.")
    fitted = joblib.load(job)
    if (fitted.get("imputation") != "arithmetic_train_means" or
        fitted.get("model_name") != "Reference26_mean" or
        not np.array_equal(np.asarray(fitted.get("train_positions")),
                           np.arange(N))):
        raise ValueError("Wrong model object for Reference26 MEAN train138.")
    repeat = native.predict_model(
        predict_saved, fitted, np.arange(N, len(full)), full, d197,
        "Reference26"
    )
    np.testing.assert_allclose(
        repeat, base, atol=1e-9, rtol=0,
        err_msg="Native train138 model fails to reproduce original target59."
    )
    return targets, base, {
        "source_final_manifest_sha256": native.sha(manifest_path),
        "source_final_predictions_csv_sha256": native.sha(csv),
        "source_reference26_mean_train138_joblib_sha256": native.sha(job),
    }


def validate_nested_provenance(recorded, expected):
    """Distinguish executable-only drift from changes to the CV experiment.

    The 138 completed nested fold records are verified separately by
    nested.summary_from_folds(), including ID, real y, native OOF
    prediction, and fixed-candidate parity. Formatting/refactoring after
    the original VM run can change a script's SHA without changing those
    verified numeric records. This is NOT permission to silently ignore
    different input datasets, model sources, CV thresholds, or candidates.
    """
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        raise ValueError("Nested run manifest must be a JSON object.")
    # nested.provenance() has a Python tuple of five config dictionaries,
    # while JSON stores that same ordered family as a list. Compare the
    # on-disk JSON representations, not the Python container types.
    # This does not permit any changed candidate, order, threshold or data.
    normalized_expected = json.loads(json.dumps(expected, allow_nan=False))
    changed = sorted(
        key for key in set(recorded) | set(normalized_expected)
        if recorded.get(key) != normalized_expected.get(key)
        or (key in recorded) != (key in normalized_expected)
    )
    if not changed:
        return ()
    code_only = {"source_sha", "screen_source_sha"}
    critical = sorted(set(changed) - code_only)
    if critical:
        raise ValueError(
            "Nested run fingerprint mismatch in: "
            + ", ".join(critical)
            + ". Preserve original nested results and original Reference26 "
              "artifacts; do not overwrite or retrain. "
              "Inspect the changed keys in the private run_spec.json."
        )
    print(
        "[NESTED PROVENANCE] Only script hashes changed: "
        + ", ".join(changed)
        + ". All data/model/fold/recipe fingerprints match; "
          "verifying each of 138 nested folds and all five numeric "
          "candidate predictions before accepting the original run.",
        flush=True,
    )
    return tuple(changed)


def validated_inputs(args):
    (frame, groups, distance, champion_manifest, ids,
     ColdExperts, predict_saved) = native.load_inputs(
        args.bundle.resolve(), args.peer_repo.resolve()
    )
    full, d197 = native.prepare_197(
        args.peer_repo.resolve(), frame, distance
    )
    spaces = analog.feature_spaces(full, champion_manifest)
    y = frame[TARGET].to_numpy(float)
    if y.shape != (138,) or not np.isfinite(y).all():
        raise ValueError("Exactly 138 finite real training labels required.")
    if full.iloc[N:][TARGET].notna().any():
        raise ValueError("59 target yields must be genuinely unknown.")
    weights = native.frozen_weights(args.config.resolve())
    loo = native.read_completed(
        args.fresh_fold_dir.resolve(), ids, y, weights
    )
    ref = loo["Reference26_mean"].to_numpy(float)
    screen = nested.read_screen(args.screen.resolve(), ids, y, ref)
    nested_manifest = args.nested_output.resolve() / "run_spec.json"
    if not nested_manifest.is_file():
        raise FileNotFoundError(
            "Nested 138-fold result missing. Complete "
            "tools/run_reference26_top5_nested.py --run first."
        )
    recorded = json.loads(nested_manifest.read_text(encoding="utf-8"))
    expected_nested = nested.provenance(args)
    validate_nested_provenance(recorded, expected_nested)
    # Strictly verify all 138 native outer-fold choices and fixed-model
    # predictions. This step never runs expensive models again.
    report, counts = nested.summary_from_folds(
        args.nested_output.resolve(), ids, y, ref, screen, save=False
    )
    if len(report) != len(nested.CHOICES)+1 or counts.sum() != 138:
        raise ValueError("Completed nested model-selection results incomplete.")
    targets, base59, evidence = require_original_final(
        args.fresh_fold_dir.resolve(), args.peer_repo.resolve(),
        frame, full, d197, weights, predict_saved
    )
    return (
        frame, groups, distance, champion_manifest, ColdExperts,
        predict_saved, spaces, y, targets, base59, report, counts, evidence
    )


def selection_spec(args, evidence):
    return {
        "schema": 1,
        "experiment": "138 real labels select 5 analog models or unchanged native Ref26",
        "final_prediction_count": 59,
        "training_labels": 138,
        "selection_cv": "4-fold internal native Reference26 mean OOF on 138 real labels",
        "candidate_family": list(nested.TOP5),
        "includes_unchanged_native_fallback": True,
        "inner_seed": SEED,
        "minimum_relative_mse_gain": args.min_inner_gain,
        "max_delta_t_ha": args.max_delta,
        "top5_historically_selected_on_same_138": True,
        "submission_schema_unverified": True,
        "source_sha256": native.sha(Path(__file__).resolve()),
        "nested_source_sha256": native.sha(Path(nested.__file__).resolve()),
        "analog_source_sha256": native.sha(Path(analog.__file__).resolve()),
        "source_nested_spec_sha256": native.sha(
            args.nested_output.resolve() / "run_spec.json"
        ),
        "source_screen_sha256": native.sha(args.screen.resolve()),
        "source_bundle_sha256": native.sha(
            args.bundle.resolve() / "manifest.json"
        ),
        "source_peer_sha256": native.sha(
            args.peer_repo.resolve() / "src/current_champions.py"
        ),
        **evidence,
    }


def inner_splits():
    return list(
        KFold(n_splits=4, shuffle=True, random_state=SEED).split(
            np.arange(N)
        )
    )


def inner_fold_predictions(
    inner_fold, frame, groups, distance, manifest, ColdExperts,
    predict_saved, spaces, y, folder, *, max_delta
):
    fit_idx, val_idx = inner_splits()[inner_fold]
    train = np.asarray(fit_idx, int)
    val = np.asarray(val_idx, int)
    native_inner = list(
        KFold(n_splits=4, shuffle=True, random_state=native.SEED).split(
            train
        )
    )
    pack, masked = native.build_fresh_fit(
        frame, groups, distance, manifest, train, native_inner,
        ColdExperts, folder / f"fit_cache_inner_{inner_fold+1:02d}",
        mean=True
    )
    if (np.intersect1d(train, val).size or
        not masked.iloc[val][TARGET].isna().all()):
        raise AssertionError("Final selection fit saw inner validation yields.")
    base = native.predict_model(
        predict_saved, pack, val, masked, distance, "Reference26"
    )
    row = {
        ID: frame.iloc[val][ID].astype(str).to_numpy(),
        "observed": y[val],
        "inner_fold": np.full(len(val), inner_fold+1),
        "train_real_n": np.full(len(val), len(train)),
        "Reference26_mean": base,
    }
    for cfg in nested.TOP5:
        key = analog.signature(cfg)
        row[key] = nested.apply_candidate(
            cfg, spaces[cfg["geometry"]], y, train,
            val, base, max_delta
        )
    return pd.DataFrame(row)


def validate_inner_fold(row, index, frame, y):
    _, val = inner_splits()[index]
    required = {
        ID, "observed", "inner_fold", "train_real_n",
        "Reference26_mean", *nested.NAMES
    }
    if len(row) != len(val) or set(row.columns) != required:
        raise ValueError(f"Final selection inner fold {index+1} schema mismatch.")
    if not np.array_equal(
        row[ID].astype(str).to_numpy(),
        frame.iloc[val][ID].astype(str).to_numpy()
    ):
        raise ValueError(f"Inner fold {index+1} private IDs/order mismatched.")
    np.testing.assert_allclose(
        row["observed"].to_numpy(float), y[val], rtol=0, atol=1e-10
    )
    if not ((row["inner_fold"] == index+1).all() and
            (row["train_real_n"] == N-len(val)).all()):
        raise ValueError("Invalid inner fold train/validation counts.")
    values = row[["Reference26_mean", *nested.NAMES]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite inner validation predictions.")


def combine_inner_selection(y, cached_rows, min_inner_gain):
    # Align inner prediction vectors with the original 138 input positions.
    inner = pd.concat(cached_rows, ignore_index=True)
    if len(inner) != N or inner[ID].duplicated().any():
        raise ValueError("Expected exactly 138 unique inner OOF cases.")
    preds = {
        nested.BASE: inner["Reference26_mean"].to_numpy(float)
    }
    for candidate in nested.NAMES:
        preds[candidate] = inner[candidate].to_numpy(float)
    selected, losses = nested.choose_on_inner(
        inner["observed"].to_numpy(float),
        preds, min_relative_gain=min_inner_gain
    )
    return selected, losses, inner


def corrected_targets(y, spaces, targets, base59, chosen, max_delta):
    if chosen not in nested.CHOICES or len(targets) != 59:
        raise ValueError("Invalid chosen model or target count.")
    if np.asarray(base59).shape != (59,):
        raise ValueError("Requires 59 authentic native baseline predictions.")
    results = {
        ID: np.asarray(targets, str),
        "Reference26_mean_t_ha": np.asarray(base59, float),
    }
    train = np.arange(N)
    query = np.arange(N, N+59)
    for cfg in nested.TOP5:
        key = analog.signature(cfg)
        results[key+"_t_ha"] = nested.apply_candidate(
            cfg, spaces[cfg["geometry"]], y, train,
            query, np.asarray(base59, float), max_delta
        )
    field = (
        "Reference26_mean_t_ha" if chosen == nested.BASE
        else chosen+"_t_ha"
    )
    selected = np.asarray(results[field], float)
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite final 59 corrected estimates.")
    results["NestedSelected_final_t_ha"] = selected
    results["delta_vs_reference26_t_ha"] = selected-base59
    results["selected_model"] = np.repeat(chosen, 59)
    out = pd.DataFrame(results)
    if len(out) != 59 or out[ID].duplicated().any():
        raise ValueError("Invalid final 59 ID uniqueness/count.")
    return out


def final_run(args, data):
    (frame, groups, d, manifest, ColdExperts, predict_saved,
     spaces, y, targets, baseline, report, counts, evidence) = data
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = selection_spec(args, evidence)
    spec_path = output / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Final selection recipe/data changed: use a NEW --output."
            )
    else:
        if any(output.glob("inner_*.csv")):
            raise ValueError("Unverified inner cache exists without run spec.")
        spec_path.write_text(
            json.dumps(spec, indent=2)+"\n", encoding="utf-8"
        )
    if any((output / f).exists() for f in (PREDICTIONS, SUBMISSION, MANIFEST)):
        raise FileExistsError(
            "Final outputs already exist. Run --summary or choose new "
            "--output; prior 59 estimates will never be overwritten."
        )
    cached = []
    for fold in range(4):
        path = output / f"inner_{fold+1:02d}.csv"
        if path.is_file():
            part = pd.read_csv(path, dtype={ID: str})
            validate_inner_fold(part, fold, frame, y)
            print(f"[FINAL INNER {fold+1}/4] cached", flush=True)
        else:
            part = inner_fold_predictions(
                fold, frame, groups, d, manifest, ColdExperts,
                predict_saved, spaces, y, output, max_delta=args.max_delta
            )
            validate_inner_fold(part, fold, frame, y)
            native.write_csv(path, part)
            print(f"[FINAL INNER {fold+1}/4] trained", flush=True)
        cached.append(part)
    selected, losses, inner_oof = combine_inner_selection(
        y, cached, args.min_inner_gain
    )
    # Insist original input alignment, not accidental inner validation order.
    if (set(inner_oof[ID].astype(str)) != set(frame[ID].astype(str)) or
        len(inner_oof) != 138):
        raise AssertionError("Internal original 138 labels not fully covered.")
    final = corrected_targets(
        y, spaces, targets, baseline, selected, args.max_delta
    )
    sel = final["NestedSelected_final_t_ha"].to_numpy(float)
    difference = sel-baseline
    scores = {
        k: {"inner_oof_mse": float(losses[k]),
            "inner_oof_rmse": float(np.sqrt(losses[k]))}
        for k in nested.CHOICES
    }
    summary = {
        **spec,
        "selected_final_model": selected,
        "min_relative_mse_gain_over_unchanged": args.min_inner_gain,
        "inner_oof_138_validation_mse_by_model": scores,
        "original_nested_138_selection_counts": {
            str(k): int(v) for k, v in counts.items()
        },
        "original_nested_138_rmse": float(report.loc[
            report["model"] == "NestedSelection", "rmse_t_ha"
        ].iloc[0]),
        "final59_aggregate": {
            "n": len(sel),
            "baseline_mean": float(np.mean(baseline)),
            "selected_mean": float(np.mean(sel)),
            "selected_median": float(np.median(sel)),
            "selected_min": float(np.min(sel)),
            "selected_max": float(np.max(sel)),
            "max_abs_delta": float(np.max(np.abs(difference))),
            "mean_abs_delta": float(np.mean(np.abs(difference))),
        },
        "selected_rule_is_global_for_all_59": True,
        "selected_by_inner_cv_on_138_known_y_only": True,
        "official_baseline_joblib_or_predictions_not_overwritten": True,
        "historical_model_family_posthoc_selection_warning": (
            "Top-five hyperparameters selected after inspecting historical "
            "138 OOF y. Nested selection RMSE is conditional/exploratory."
        ),
        "outputs_are_local_private_not_version_controlled": True,
    }
    submission = final[[ID, "NestedSelected_final_t_ha"]].rename(
        columns={"NestedSelected_final_t_ha": TARGET}
    )
    # Guard against any output accidentally appearing during fitting.
    if any((output / f).exists() for f in (PREDICTIONS, SUBMISSION, MANIFEST)):
        raise FileExistsError("An existing final output appeared during fitting.")
    native.write_csv(output / PREDICTIONS, final)
    native.write_csv(output / SUBMISSION, submission)
    (output / MANIFEST).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    print(f"[FINAL] Selected by new 138-real-label inner 4fold: {selected}")
    print(f"[FINAL] Generated exactly {len(final)} private candidate yields.")
    print(f"[FINAL] Mean={np.mean(sel):.6f}, median={np.median(sel):.6f}, "
          f"max |delta|={np.max(np.abs(difference)):.6f} t/ha.")
    print(f"[FINAL] Full comparison: {output / PREDICTIONS}")
    print(f"[FINAL] Two-column FIRA-format CANDIDATE: {output / SUBMISSION}")
    print(f"[FINAL] Aggregate traceability: {output / MANIFEST}")
    print("[CAUTION] Submission format must be verified against competition "
          "requirements; nothing was submitted, deployed or published.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        action.add_argument("--"+name, action="store_true")
    parser.add_argument("--bundle", type=Path, default=native.BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=native.PEER_REPO)
    parser.add_argument("--config", type=Path, default=native.CONFIG)
    parser.add_argument("--fresh-fold-dir", type=Path, default=native.OUTPUT)
    parser.add_argument("--nested-output", type=Path, default=nested.OUT)
    parser.add_argument("--screen", type=Path, default=nested.SCREEN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-inner-gain", type=float, default=0.01)
    parser.add_argument("--max-delta", type=float, default=0.5)
    args = parser.parse_args()
    if not (0 <= args.min_inner_gain < 1 and args.max_delta > 0):
        parser.error("Invalid inner gain threshold or bounded correction.")
    if args.summary:
        path = args.output.resolve() / MANIFEST
        if not path.is_file():
            raise FileNotFoundError("Run --run to create private 59 first.")
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"[SUMMARY] 59 selected rule: {result['selected_final_model']}")
        print(json.dumps(result["final59_aggregate"], indent=2))
        print("[SUMMARY] Full/private file and submission candidate "
              f"located under {args.output.resolve()}")
        return
    data = validated_inputs(args)
    frame = data[0]
    y = data[7]
    evidence = data[-1]
    if args.check:
        print("[CHECK] All prior 138 Reference26 original and nested "
              "LOOCV folds verified.")
        print("[CHECK] Original 138-label Reference26 mean joblib "
              "SHA-verified and reproduced all 59 predictions.")
        print(f"[CHECK] 4 new inner refits required, 5 corrected recipes "
              f"plus baseline, private target count={len(data[8])}.")
        print(f"[CHECK] Nested parent RMSE: "
              f"{data[10].loc[data[10]['model']=='NestedSelection', 'rmse_t_ha'].iloc[0]:.9f}")
        print(f"[CHECK] Final source joblib SHA: "
              f"{evidence['source_reference26_mean_train138_joblib_sha256']}")
        return
    if args.smoke:
        part = inner_fold_predictions(
            0, frame, data[1], data[2], data[3], data[4],
            data[5], data[6], y, args.output.resolve() / "smoke",
            max_delta=args.max_delta
        )
        validate_inner_fold(part, 0, frame, y)
        print(f"[SMOKE] Genuine native MEAN Reference26 1/4 inner "
              f"model trained; {len(part)} real validation labels excluded. "
              "No 59 final output written.")
        return
    final_run(args, data)


if __name__ == "__main__":
    main()
