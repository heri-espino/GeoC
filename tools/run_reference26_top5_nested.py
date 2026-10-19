#!/usr/bin/env python3
"""Top-five Reference26 analogs: exact fixed 138 LOOCV and nested selection.

The previous screen ALREADY evaluated the five fixed corrected models on
138 authentic 137/1 heldouts. This experiment asks an additional question:
can one CHOOSE between those five and no correction using only the other
137 labels? For outer heldout i, refit NATIVE Reference26 mean on each of
four training-only inner folds (approximately 102/103 training labels),
generate feature-analog corrected inner predictions using ONLY inner-train y,
and choose based on inner OOF MSE. Apply that choice to the already-frozen
genuine outer Reference26 prediction for i. 138 x 4 = 552 additional
native Reference26 fits for full --run; cache/resume by outer fold.

The five candidate specifications themselves were previously chosen by
inspecting the SAME 138 labels. Thus this is conditional/nested selection
of a PRE-SELECTED family, not fully unbiased study-wide validation.
Never train correction weights on the full 138 saved OOF errors of the
other parcels: those OOF models may include the current outer heldout y.

  python tools/run_reference26_top5_nested.py --check
  python tools/run_reference26_top5_nested.py --quick
  python tools/run_reference26_top5_nested.py --smoke --jobs 1
  python tools/run_reference26_top5_nested.py --run --jobs 4
  python tools/run_reference26_top5_nested.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import run_reference26_analog_tail_screen as analog
import run_reference26_compact14_final as native

ROOT, N, ID, TARGET = native.ROOT, native.N, native.ID, native.TARGET
OUT = ROOT / "reports/reference26_top5_nested"
SCREEN = analog.OUT / "private_heldout_predictions.csv"
SEED = 20261019
_WORKER = None

# Exactly the five highest-ranked CORRECTED configurations from the user's
# 2026-10-19 authentic fixed-parameter 138-fold LOOCV screen.
TOP5 = (
    {"method": "analog", "geometry": "robust26", "k": 3,
     "blend": 0.25, "gate": "supported"},
    {"method": "analog", "geometry": "pca8", "k": 3,
     "blend": 0.25, "gate": "supported"},
    {"method": "analog", "geometry": "pca8", "k": 8,
     "blend": 0.25, "gate": "supported"},
    {"method": "analog", "geometry": "robust26", "k": 3,
     "blend": 0.1, "gate": "supported"},
    {"method": "analog", "geometry": "robust26", "k": 3,
     "blend": 0.1, "gate": "none"},
)
NAMES = tuple(analog.signature(c) for c in TOP5)
BASE = "Reference26_mean_unchanged"
CHOICES = (BASE, *NAMES)


def apply_candidate(cfg, z, y, train, query, base, max_delta):
    """Never references y outside train. Same implementation as initial screen."""
    value, nearest, dispersion = analog.neighborhood(
        z, y, train, query, k=cfg["k"],
    )
    result, _ = analog.analog_correction(
        np.asarray(base, float), value, nearest,
        analog.training_spacing(z, np.asarray(train, int)),
        float(np.std(y[train])), dispersion,
        blend=cfg["blend"], gate=cfg["gate"],
        max_delta=max_delta,
    )
    return np.asarray(result, float)


def choose_on_inner(y_true, predictions, *, min_relative_gain):
    """Choose candidate only if its inner MSE beats unchanged Reference26."""
    if not (0 <= min_relative_gain < 1):
        raise ValueError("min_relative_gain must be in [0, 1).")
    if set(predictions) != set(CHOICES):
        raise ValueError("Missing or unexpected inner prediction columns.")
    actual = np.asarray(y_true, float)
    errors = {}
    for name in CHOICES:
        estimate = np.asarray(predictions[name], float)
        if estimate.shape != actual.shape or not np.isfinite(estimate).all():
            raise ValueError(f"Invalid inner prediction shape: {name}")
        errors[name] = float(np.mean((estimate-actual)**2))
    best = min(CHOICES, key=lambda name: (errors[name], CHOICES.index(name)))
    if best != BASE:
        gain = (errors[BASE]-errors[best]) / max(errors[BASE], 1e-12)
        if gain < min_relative_gain:
            best = BASE
    return best, errors


def read_screen(path: Path, ids, y, outer_base):
    if not path.is_file():
        raise FileNotFoundError(
            f"Original 31-recipe 138-fold screen not found: {path}. "
            "First run tools/run_reference26_analog_tail_screen.py --run"
        )
    screen = pd.read_csv(path, dtype={ID: str})
    expected = {ID, "observed", "Reference26_mean", *NAMES}
    if len(screen) != N or screen[ID].duplicated().any() or not expected.issubset(
        screen.columns
    ):
        raise ValueError("Original fixed candidate screen is incomplete.")
    if not np.array_equal(screen[ID].astype(str).to_numpy(), np.asarray(ids)):
        raise ValueError("Screen 138 IDs/order differs from native Reference26.")
    np.testing.assert_allclose(
        screen["observed"].to_numpy(float), y, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        screen["Reference26_mean"].to_numpy(float),
        outer_base, atol=1e-10, rtol=0,
    )
    if not np.isfinite(screen[list(NAMES)].to_numpy(float)).all():
        raise ValueError("Nonfinite top-five screen predictions.")
    return screen


def metrics(y, preds):
    actual = np.asarray(y, float)
    p = np.asarray(preds, float)
    if p.shape != actual.shape or not np.isfinite(p).all():
        raise ValueError("Invalid complete heldout predictions.")
    e = p-actual
    ss = np.sum((actual-np.mean(actual))**2)
    return {
        "n_real_heldouts": len(actual),
        "outer_real_train": N-1,
        "rmse_t_ha": float(np.sqrt(np.mean(e*e))),
        "mae_t_ha": float(np.mean(np.abs(e))),
        "bias_t_ha": float(np.mean(e)),
        "r2_cv": float(1.-np.sum(e*e)/ss) if ss>0 else None,
    }


def fixed_results(frame):
    data = frame["observed"].to_numpy(float)
    summary = []
    for name in CHOICES:
        column = "Reference26_mean" if name == BASE else name
        summary.append({"model": name, "selection": "fixed_posthoc_top5",
                        **metrics(data, frame[column].to_numpy(float))})
    return pd.DataFrame(summary).sort_values("rmse_t_ha", kind="stable")


def record_path(folder, held):
    return folder / f"fold_{held+1:03d}.csv"


def verify_fold(row, pos, ids, y, reference, screened):
    if len(row) != 1:
        raise ValueError(f"Expected one saved fold for {pos}.")
    r = row.iloc[0]
    required = {
        ID, "observed", "Reference26_mean", "chosen_model",
        "NestedSelection", "inner_mse_base", "inner_mse_chosen",
        "outer_train_n", "inner_oof_n", *NAMES
    }
    if set(row.columns) != required:
        raise ValueError(f"Saved nested fold {pos} has unexpected columns.")
    if (str(r[ID]) != str(ids[pos]) or
        not np.isclose(float(r["observed"]), y[pos], atol=1e-10, rtol=0) or
        not np.isclose(float(r["Reference26_mean"]), reference[pos],
                       atol=1e-10, rtol=0) or
        int(r["outer_train_n"]) != N-1 or
        int(r["inner_oof_n"]) != N-1 or
        r["chosen_model"] not in CHOICES):
        raise ValueError(f"Nested fold {pos} mismatches original real data.")
    for name in NAMES:
        if not np.isclose(float(r[name]), float(screened.iloc[pos][name]),
                          atol=1e-9, rtol=0):
            raise ValueError(f"Candidate {name} lost LOOCV parity on fold {pos}.")
    chosen = "Reference26_mean" if r["chosen_model"] == BASE else r["chosen_model"]
    if not np.isclose(float(r["NestedSelection"]), float(r[chosen]),
                      atol=1e-10, rtol=0):
        raise ValueError("Nested selection does not match candidate prediction.")
    if not (np.isfinite(float(r["inner_mse_base"])) and
            np.isfinite(float(r["inner_mse_chosen"]))):
        raise ValueError("Nonfinite inner validation loss.")


def provenance(args):
    return {
        "schema": 1,
        "experiment": "nested correction selection among previous top-5",
        "outer": "138 LOOCV 137 original training labels, 1 real heldout",
        "inner": "4-fold native Reference26 cross-fitting on 137 outer training labels",
        "inner_model": "genuine peer ColdExperts Reference26 arithmetic training-mean imputation",
        "candidate_family_historically_preselected_on_same_138": True,
        "selection_scope": "only 137 outer training labels",
        "outer_reference26_prediction_from_original_authentic_folds": True,
        "minimum_relative_inner_mse_gain": args.min_inner_gain,
        "max_delta": args.max_delta,
        "seed": SEED,
        "top5": TOP5,
        "source_sha": analog.sha(Path(__file__).resolve()),
        "screen_source_sha": analog.sha(Path(analog.__file__).resolve()),
        "screen_private_sha": analog.sha(args.screen.resolve()),
        "native_run_spec_sha": analog.sha(
            args.fresh_fold_dir.resolve()/"run_spec.json"
        ),
        "peer_source_sha": analog.sha(
            args.peer_repo.resolve()/"src/current_champions.py"
        ),
        "bundle_sha": analog.sha(args.bundle.resolve()/"manifest.json"),
        "frozen_weights_sha": analog.sha(args.config.resolve()),
    }


def initialize(bundle, peer, config, fresh, screen_path, max_delta, inner_gain):
    global _WORKER
    frame, groups, d, manifest, ids, ColdExperts, predict_saved = (
        native.load_inputs(Path(bundle), Path(peer))
    )
    full, _ = native.prepare_197(Path(peer), frame, d)
    spaces = analog.feature_spaces(full, manifest)
    weights = native.frozen_weights(Path(config))
    loo = native.read_completed(
        Path(fresh), ids, frame[TARGET].to_numpy(float), weights
    )
    ref = loo["Reference26_mean"].to_numpy(float)
    y = frame[TARGET].to_numpy(float)
    screened = read_screen(Path(screen_path), ids, y, ref)
    _WORKER = (frame, groups, d, manifest, ids, ColdExperts,
               predict_saved, spaces, y, ref, screened, max_delta, inner_gain)


def one_outer_fold(pos, output_name):
    if _WORKER is None:
        raise RuntimeError("Outer worker was not initialized.")
    (frame, groups, d, manifest, ids, ColdExperts, predict_saved,
     spaces, y, ref, screened, max_delta, min_gain) = _WORKER
    folder = Path(output_name)
    path = record_path(folder, pos)
    if path.is_file():
        cached = pd.read_csv(path, dtype={ID: str})
        verify_fold(cached, pos, ids, y, ref, screened)
        return pos+1, "cached", str(cached.iloc[0]["chosen_model"])

    train = np.delete(np.arange(N), pos)
    if pos in train or len(train) != 137:
        raise AssertionError("Outer heldout entered train labels.")
    inner_predictions = {name: np.full(len(train), np.nan) for name in CHOICES}
    # Each val original parcel is predicted with a native model trained on
    # the OTHER ~102/103 original outer-train labels only.
    for inner_fold, (fit_idx, val_idx) in enumerate(
        KFold(n_splits=4, shuffle=True, random_state=SEED).split(train), 1
    ):
        fit_positions, val_positions = train[fit_idx], train[val_idx]
        if (pos in fit_positions or pos in val_positions or
            np.intersect1d(fit_positions, val_positions).size):
            raise AssertionError("Inner target/outer heldout training leak.")
        nested_inner = list(
            KFold(n_splits=4, shuffle=True, random_state=native.SEED).split(
                fit_positions
            )
        )
        cache = folder / "native_inner_cache" / f"outer_{pos+1:03d}" / (
            f"inner_{inner_fold:02d}"
        )
        pack, masked = native.build_fresh_fit(
            frame, groups, d, manifest, fit_positions, nested_inner,
            ColdExperts, cache, mean=True
        )
        baseline = native.predict_model(
            predict_saved, pack, val_positions, masked, d, "Reference26"
        )
        inner_predictions[BASE][val_idx] = baseline
        for cfg in TOP5:
            key = analog.signature(cfg)
            inner_predictions[key][val_idx] = apply_candidate(
                cfg, spaces[cfg["geometry"]], y, fit_positions,
                val_positions, baseline, max_delta
            )
    selected, mses = choose_on_inner(
        y[train], inner_predictions, min_relative_gain=min_gain
    )
    row = {
        ID: ids[pos], "observed": float(y[pos]),
        "Reference26_mean": float(ref[pos]), "chosen_model": selected,
        "outer_train_n": len(train), "inner_oof_n": len(train),
        "inner_mse_base": mses[BASE], "inner_mse_chosen": mses[selected],
    }
    for cfg in TOP5:
        name = analog.signature(cfg)
        outer_prediction = apply_candidate(
            cfg, spaces[cfg["geometry"]], y, train,
            np.array([pos]), np.array([ref[pos]]), max_delta
        )
        row[name] = float(outer_prediction[0])
    row["NestedSelection"] = (
        float(ref[pos]) if selected == BASE else row[selected]
    )
    result = pd.DataFrame([row])
    verify_fold(result, pos, ids, y, ref, screened)
    native.write_csv(path, result)
    return pos+1, "trained", selected


def summary_from_folds(folder, ids, y, ref, screened, *, save):
    rows = []
    for i in range(N):
        path = record_path(folder, i)
        if not path.is_file():
            raise FileNotFoundError(
                "Need all 138 completed nested folds; "
                f"missing {path.name}"
            )
        f = pd.read_csv(path, dtype={ID: str})
        verify_fold(f, i, ids, y, ref, screened)
        rows.append(f)
    table = pd.concat(rows, ignore_index=True)
    output = []
    for name in CHOICES:
        col = "Reference26_mean" if name == BASE else name
        output.append({"model": name, "selection": "fixed_posthoc_top5",
                       **metrics(y, table[col].to_numpy(float))})
    output.append({
        "model": "NestedSelection",
        "selection": "fold_specific_inner_4fold_native_selection",
        **metrics(y, table["NestedSelection"].to_numpy(float))
    })
    report = pd.DataFrame(output).sort_values(
        "rmse_t_ha", kind="stable"
    )
    if save:
        report.to_csv(folder/"comparison_aggregate.csv", index=False)
        table.to_csv(folder/"private_nested_predictions.csv", index=False)
        counts = table["chosen_model"].value_counts().to_dict()
        (folder/"selection_counts_aggregate.json").write_text(
            json.dumps(counts, indent=2)+"\n", encoding="utf-8"
        )
    return report, table["chosen_model"].value_counts()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for name in ("check", "quick", "smoke", "run", "summary"):
        modes.add_argument("--"+name, action="store_true")
    p.add_argument("--bundle", type=Path, default=native.BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=native.PEER_REPO)
    p.add_argument("--config", type=Path, default=native.CONFIG)
    p.add_argument("--fresh-fold-dir", type=Path, default=native.OUTPUT)
    p.add_argument("--screen", type=Path, default=SCREEN)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--max-delta", type=float, default=0.5)
    p.add_argument("--min-inner-gain", type=float, default=0.01)
    p.add_argument("--from-fold", type=int, default=1)
    p.add_argument("--through-fold", type=int, default=138)
    args = p.parse_args()
    if (not 1 <= args.jobs <= 8 or args.max_delta <= 0 or
        not 0 <= args.min_inner_gain < 1 or
        not 1 <= args.from_fold <= args.through_fold <= 138):
        p.error("Invalid jobs, max_delta, gain threshold, or fold interval.")

    frame, full, spaces, ref = analog.get_inputs(args)
    ids = frame[ID].astype(str).tolist()
    y = frame[TARGET].to_numpy(float)
    screened = read_screen(args.screen.resolve(), ids, y, ref)
    known = fixed_results(screened)
    if args.check:
        print("[CHECK] Verified true 138 native Reference26 (137/1) OOF; "
              "exact five post-hoc corrected OOF candidates.")
        print("[CHECK] Nested selection needs four ADDITIONAL native "
              "Ref26 fits inside each of 138 outer folds (552 total).")
        print(known.to_string(index=False))
        return
    if args.quick:
        print("[QUICK] Existing full-138 LOOCV, FIXED corrected top-five; "
              "no new fits and no inner model selection.")
        print(known.to_string(index=False))
        return

    folder = args.output.resolve() / "smoke" if args.smoke else args.output.resolve()
    if args.summary:
        scores, counts = summary_from_folds(
            folder, ids, y, ref, screened, save=True
        )
        print("[SUMMARY] 138 true heldouts with nested candidate selection.")
        print(scores.to_string(index=False))
        print("[CHOICES] " + counts.to_string())
        return

    spec = provenance(args)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/"run_spec.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Source/settings/data changed: use NEW --output.")
    else:
        if list(folder.glob("fold_*.csv")):
            raise ValueError("Found fold files with no provenance fingerprint.")
        manifest_path.write_text(
            json.dumps(spec, indent=2)+"\n", encoding="utf-8"
        )
    positions = [0] if args.smoke else list(
        range(args.from_fold-1, args.through_fold)
    )
    todo = []
    for pos in positions:
        file = record_path(folder, pos)
        if file.exists():
            cached = pd.read_csv(file, dtype={ID: str})
            verify_fold(cached, pos, ids, y, ref, screened)
        else:
            todo.append(pos)
    print(f"[NESTED] {len(todo)} new outer folds, {len(positions)} requested, "
          f"4 native inner fits each; jobs={args.jobs}.", flush=True)
    if todo:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(todo)),
            initializer=initialize,
            initargs=(
                str(args.bundle.resolve()), str(args.peer_repo.resolve()),
                str(args.config.resolve()), str(args.fresh_fold_dir.resolve()),
                str(args.screen.resolve()), args.max_delta, args.min_inner_gain
            )
        ) as pool:
            pending = {
                pool.submit(one_outer_fold, pos, str(folder)): pos for pos in todo
            }
            for i, future in enumerate(as_completed(pending), 1):
                fold, status, selected = future.result()
                print(f"[NESTED {i}/{len(todo)}] "
                      f"outer={fold:03d} {status}, chosen={selected}",
                      flush=True)
    if args.smoke:
        print("[SMOKE] Completed one outer fold + four native inner fits, "
              "not an estimate of 138-fold performance.")
    elif args.from_fold == 1 and args.through_fold == 138:
        scores, counts = summary_from_folds(
            folder, ids, y, ref, screened, save=True
        )
        print("[FULL] Nested selected policy + five fixed corrected "
              "genuine outer LOOCVs:")
        print(scores.to_string(index=False))
        print("[CHOICES] " + counts.to_string())
        print("[CAUTION] These five recipes were historically selected "
              "on these same 138 labels: results remain exploratory.")
        print(f"[PRIVATE] {folder}; no 59 model changed.")
    else:
        print("[PARTIAL] Resume or finish 138 folds for full score.")


if __name__ == "__main__":
    main()
