#!/usr/bin/env python3
"""Full 138-fold LOOCV of 250 (neighbors, Ridge alpha) Reference26 settings.

Each setting runs teammate's original ColdExperts.full_fit and predict_saved
on all 138 independent 137+1 outer splits; the ONLY model change is local
Ridge k and alpha, applied consistently during nested training and prediction.

Source: ../todoseboso; private 26-feature inputs: data_n/processed/current_champions/v1.
Original 138-fold Reference26 forecast file is reused as the comparison, not
refitted just to be a baseline.

Default: 10 geographic k x 25 Ridge alpha = 250 distinct model configurations.
The same 138 heldout parcels are used for each configuration, never selected
based on their observed yield. Choosing the eventual best of 250 *using* their
outer scores is exploratory and gives an optimistic winner estimate.

  python tools/run_reference26_ridge250_loo.py --check
  python tools/run_reference26_ridge250_loo.py --smoke
  python tools/run_reference26_ridge250_loo.py --run --jobs 12
  python tools/run_reference26_ridge250_loo.py --summary

Interrupted runs resume individual fold x hyperparameter predictions. The
private model cache for each completed fit is discarded to keep disk use low.
"""
from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from run_selected_champions_loo import (  # noqa: E402
    BUNDLE,
    ID,
    PEER_REPO,
    ROOT,
    TARGET,
    digest,
    fold_record,
    load_inputs,
    write_csv,
)
from sklearn.metrics import r2_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from geocebada.evaluation.peer_reference26_ridge_grid import (  # noqa: E402
    ALPHAS,
    DEFAULT_PAIR,
    GRID,
    NEIGHBORS,
    share_climate_fits_between_grid_candidates,
    tuned_local_ridge,
)

N = 138
NAME = "Reference26"
BASELINE = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
OUTPUT = ROOT / "reports/reference26_ridge250_loo"
_WORKER = None


def candidate_name(k: int, alpha: float) -> str:
    return f"Reference26_k{k:02d}_alpha{alpha:g}"


def candidate_slug(k: int, alpha: float) -> str:
    return f"k{k:02d}_a{alpha:g}".replace(".", "p")


def parse_candidates(kvals: str | None, avals: str | None):
    """Explicit non-default grids supported, with unique positive values."""
    try:
        ks = NEIGHBORS if kvals is None else tuple(
            int(v.strip()) for v in kvals.split(",")
        )
        alphas = ALPHAS if avals is None else tuple(
            float(v.strip()) for v in avals.split(",")
        )
    except ValueError as exc:
        raise ValueError("Use --ks integer list and --alphas numeric list.") from exc
    if (not ks or not alphas or
            len(set(ks)) != len(ks) or len(set(alphas)) != len(alphas) or
            any(k < 3 for k in ks) or
            any(not np.isfinite(a) or a <= 0 for a in alphas)):
        raise ValueError("k must be >=3, alpha >0, with no duplicated values.")
    return tuple((k, a) for k in ks for a in alphas)


def original_results(path: Path, ids: list[str], y: np.ndarray):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing original full Reference26 LOOCV predictions: {path}. "
            "Run run_selected_champions_loo.py --summary first to rebuild "
            "the CSV from its 138 saved folds."
        )
    original = pd.read_csv(path, dtype={ID: str})
    if (
        len(original) != N
        or not original[ID].is_unique
        or not {ID, "observed", NAME}.issubset(original.columns)
        or set(original[ID].astype(str)) != set(ids)
    ):
        raise ValueError("Original Reference26 baseline must have 138 matching IDs.")
    original = original.set_index(ID).reindex(ids).reset_index()
    if (
        not np.isfinite(original[["observed", NAME]].to_numpy(float)).all()
        or not np.allclose(
            original["observed"].to_numpy(float), y, rtol=0, atol=1e-9
        )
    ):
        raise ValueError("Saved Reference26 LOOCV yields differ from bundle.")
    return original[[ID, "observed", NAME]].copy()


def manifest(bundle: Path, peer: Path, configs: tuple):
    return {
        "schema": 1,
        "protocol": "same 138 saved bundle rows, 137 train + one excluded label each",
        "search": "all fixed configurations scored separately on all 138 LOOCV heldouts",
        "n_configs": len(configs),
        "k_alpha_pairs": [[k, alpha] for k, alpha in configs],
        "inner_cv": "original ColdExperts.full_fit: 4 train-only folds seed 20261018",
        "shared_cache": "CT3 only, within one fold/batch; new local Ridge and R3 per config",
        "teammate_manifest_hash": digest(bundle / "manifest.json"),
        "teammate_code_hash": digest(peer / "src/current_champions.py"),
        "teammate_bridge_hash": digest(peer / "src/geoc_bridge.py"),
        "teammate_local_hash": digest(
            peer / "src/local_ridge_sufficient_statistics.py"
        ),
        "local_override_hash": digest(
            ROOT / "src/geocebada/evaluation/peer_reference26_ridge_grid.py"
        ),
        "original_runner_hash": digest(
            ROOT / "tools/run_selected_champions_loo.py"
        ),
        "runner_hash": digest(Path(__file__).resolve()),
    }


def lock(output: Path, info: dict):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != info:
            raise ValueError(
                "The model/input/grid changed. Use a new --output directory "
                "rather than mix incompatible fold results."
            )
    elif list(output.glob("fold_*/**/*.csv")):
        raise ValueError("Saved candidate folds have no provenance.")
    else:
        path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def fold_file(output: Path, pos: int, candidate):
    return (
        output / f"fold_{pos + 1:03d}" /
        f"{candidate_slug(*candidate)}.csv"
    )


def validate_row(path: Path, pos: int, ids: list[str], y: np.ndarray, candidate):
    row = pd.read_csv(path, dtype={ID: str})
    name = candidate_name(*candidate)
    if (
        len(row) != 1 or
        set(row.columns) != {ID, "observed", name} or
        str(row.iloc[0][ID]) != ids[pos] or
        abs(float(row.iloc[0]["observed"]) - float(y[pos])) > 1e-10 or
        not np.isfinite(row[[name]].to_numpy(float)).all()
    ):
        raise ValueError(f"Incorrect cached LOOCV candidate: {path}")
    return float(row.iloc[0][name])


def worker_init(bundle: str, peer: str):
    global _WORKER
    _WORKER = load_inputs(Path(bundle), Path(peer))


def fit_one(pos: int, conf: tuple, output: Path):
    """Fit *one* variant with native nested train-only Reference26 procedure."""
    if _WORKER is None:
        raise RuntimeError("The worker has no private 138-parcel bundle.")
    frame, groups, distances, data, ids, ColdExperts, predict_saved = _WORKER
    tr, inner = fold_record(pos)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[pos], TARGET] = np.nan
    if (
        not np.isnan(masked.iloc[pos][TARGET])
        or not np.isfinite(masked.iloc[tr][TARGET].to_numpy(float)).all()
    ):
        raise ValueError("The outer held-out yield was not isolated.")
    if pos in tr or len(tr) != 137:
        raise ValueError("Expected 137 train positions and one different heldout.")

    k, alpha = conf
    directory = (
        Path(output) / "_temp_models" / f"fold_{pos+1:03d}" /
        candidate_slug(k, alpha)
    )
    with threadpool_limits(limits=1), tuned_local_ridge(k, alpha):
        factory = ColdExperts(
            masked, groups, distances, data["geology"], "random",
            tr, inner, directory,
        )
        fitted = factory.full_fit(tr)
        if set(fitted["train_positions"]) != set(tr):
            raise RuntimeError("Peer model used unexpected training positions.")
        result = predict_saved({
            "outer_experts": fitted,
            "train_positions": tr,
            "validation_positions": np.asarray([pos], dtype=int),
            "B1_weight": 0.0,
        }, np.asarray([pos], dtype=int), masked, distances)
    value = float(np.asarray(result[NAME]).item())
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite prediction at heldout {pos+1}, {conf}.")
    return value, directory


def parity_baseline(output: Path, bundle: Path, peer: Path, original: pd.DataFrame):
    """Fail fast if the optimized Ridge override changes original (24,30)."""
    global _WORKER
    if _WORKER is None:
        _WORKER = load_inputs(bundle, peer)
    frame, _groups, _distance, _manifest, ids, _factory, _predict = _WORKER
    path = fold_file(output, 0, DEFAULT_PAIR)
    if path.is_file():
        prediction = validate_row(
            path, 0, ids, frame[TARGET].to_numpy(float), DEFAULT_PAIR
        )
    else:
        value, model_cache = fit_one(0, DEFAULT_PAIR, output)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, pd.DataFrame([{
            ID: ids[0], "observed": float(frame.iloc[0][TARGET]),
            candidate_name(*DEFAULT_PAIR): value,
        }]))
        shutil.rmtree(model_cache, ignore_errors=True)
        prediction = value
    saved = float(original.iloc[0][NAME])
    difference = abs(prediction - saved)
    print(
        f"[PARITY] Reference26 k=24 alpha=30 first-fold abs "
        f"difference = {difference:.4g} t/ha", flush=True,
    )
    if difference > 1e-7:
        raise RuntimeError(
            "Reference26 original (k=24, alpha=30) numerical parity FAILED. "
            f"Difference: {difference:.8g} t/ha. Refuse grid search; "
            "inner/outer predictions must use the original Ridge convention."
        )


def run_batch(pos: int, pairs: tuple, output_str: str):
    """Save each candidate separately; repeated runs skip completed fits."""
    output = Path(output_str)
    if _WORKER is None:
        raise RuntimeError("Worker was not initialized.")
    frame, _groups, _distance, _manifest, ids, _class, _predict = _WORKER
    y = frame[TARGET].to_numpy(float)
    completed = 0
    cached = 0
    start = time.perf_counter()
    with share_climate_fits_between_grid_candidates():
        for config in pairs:
            file = fold_file(output, pos, config)
            if file.is_file():
                validate_row(file, pos, ids, y, config)
                cached += 1
                continue
            value, model_cache = fit_one(pos, config, output)
            row = pd.DataFrame([{
                ID: ids[pos], "observed": float(y[pos]),
                candidate_name(*config): value,
            }])
            file.parent.mkdir(parents=True, exist_ok=True)
            write_csv(file, row)
            # C-T3 invariant model fits are reused *in memory* during this
            # batch; other per-candidate model caches are safe to discard.
            shutil.rmtree(model_cache, ignore_errors=True)
            completed += 1
    return pos + 1, completed, cached, time.perf_counter() - start


def summarize(output: Path, baseline: pd.DataFrame, configs: tuple):
    ids = baseline[ID].astype(str).tolist()
    y = baseline["observed"].to_numpy(float)
    n = len(configs)
    scores = [{
        "model": "Reference26_original",
        "k": 24,
        "alpha": 30.0,
        "heldout": N,
        "rmse": float(np.sqrt(np.mean(
            (baseline[NAME].to_numpy(float) - y) ** 2
        ))),
        "mae": float(np.mean(np.abs(baseline[NAME].to_numpy(float) - y))),
        "bias": float(np.mean(baseline[NAME].to_numpy(float) - y)),
        "r2": float(r2_score(y, baseline[NAME])),
    }]
    baseline_pred = baseline[NAME].to_numpy(float)
    baseline_mse = float(np.mean((baseline_pred - y)**2))
    scores[0]["delta_mse_vs_original"] = 0.0
    full = baseline[[ID, "observed"]].copy()
    default_values = None
    for count, config in enumerate(configs, start=1):
        preds = np.empty(N, dtype=float)
        for pos in range(N):
            path = fold_file(output, pos, config)
            if not path.is_file():
                raise ValueError(
                    f"Missing model {count}/{n}, LOOCV fold {pos+1}: {path}"
                )
            preds[pos] = validate_row(path, pos, ids, y, config)
        name = candidate_name(*config)
        if config == DEFAULT_PAIR:
            default_values = preds
        residual = preds - y
        mse = float(np.mean(residual**2))
        scores.append({
            "model": name, "k": config[0], "alpha": config[1],
            "heldout": N,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(residual))),
            "bias": float(np.mean(residual)),
            "r2": float(r2_score(y, preds)),
            "delta_mse_vs_original": mse - baseline_mse,
        })
        full[name] = preds
    if default_values is not None:
        diff = float(np.max(np.abs(default_values - baseline_pred)))
        print(
            f"[ALL-FOLD PARITY] Original k24/alpha30 max abs difference "
            f"{diff:.4g} t/ha", flush=True,
        )
        if diff > 1e-7:
            raise RuntimeError(
                "Original model parity failed on the 138 heldout parcels: "
                f"max |prediction difference|={diff:.8g}."
            )
    summary = pd.DataFrame(scores).sort_values(
        ["rmse", "k", "alpha"], kind="stable"
    ).reset_index(drop=True)
    write_csv(output / "loo138_summary_250.csv", summary)
    # Individually observed yields must remain private / gitignored.
    write_csv(output / "loo138_predictions_private.csv", full)
    print(
        f"[REFERENCE26] Complete 138 LOOCV × {len(configs)} fixed "
        f"k/alpha models.\n"
        + summary.head(20).to_string(index=False),
        flush=True,
    )
    print(
        "[CAUTION] 250 full LOOCV configurations were compared using the "
        "same observed labels. The lowest RMSE is exploratory, and the "
        "selected winner's error is optimistically biased.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        selection.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ks", default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.jobs < 1 or args.batch_size < 1:
        parser.error("--jobs and --batch-size must be positive.")
    try:
        pairs = parse_candidates(args.ks, args.alphas)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.ks and not args.alphas and pairs != GRID:
        raise RuntimeError("Default grid unexpectedly changed.")
    bundle = args.bundle.resolve()
    peer = args.peer_repo.resolve()
    out = (
        args.output / "smoke" if args.smoke else args.output
    ).resolve()
    frame, _groups, _distance, _data, ids, _c, _p = load_inputs(bundle, peer)
    orig = original_results(
        args.baseline.resolve(), ids, frame[TARGET].to_numpy(float)
    )
    print(
        f"[REFERENCE26 GRID] {len(pairs)} combinations = "
        f"{len({k for k, _ in pairs})} geographic k × "
        f"{len({a for _, a in pairs})} Ridge alpha; "
        f"138 fixed outer heldouts × 137 labeled training parcels.",
        flush=True,
    )
    if args.check:
        return 0
    lock(out, manifest(bundle, peer, pairs))
    if args.summary:
        summarize(out, orig, pairs)
        return 0

    parity_baseline(out, bundle, peer, orig)
    if args.smoke:
        example = next(
            (config for config in pairs if config != DEFAULT_PAIR),
            DEFAULT_PAIR,
        )
        _pos, fitted, cached, elapsed = run_batch(0, (example,), str(out))
        print(
            f"[SMOKE] Original (24,30) parity passed; tested {example}. "
            f"new={fitted}, cached={cached}, elapsed={elapsed:.1f}s. "
            "No 138-fold metrics have been computed.",
            flush=True,
        )
        return 0

    tasks = []
    for pos in range(N):
        for lo in range(0, len(pairs), args.batch_size):
            chunk = pairs[lo:lo+args.batch_size]
            if all(fold_file(out, pos, pair).is_file() for pair in chunk):
                for pair in chunk:
                    validate_row(
                        fold_file(out, pos, pair), pos, ids,
                        frame[TARGET].to_numpy(float), pair,
                    )
                continue
            tasks.append((pos, chunk))
    print(
        f"[GRID RUN] Remaining task batches: {len(tasks)}, "
        f"requested workers: {args.jobs}, "
        f"active workers: {min(args.jobs, len(tasks))}.",
        flush=True,
    )
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(tasks)),
            initializer=worker_init,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            futures = {
                pool.submit(run_batch, pos, chunk, str(out)): (pos, chunk)
                for pos, chunk in tasks
            }
            done = 0
            for future in as_completed(futures):
                held, new, cached, elapsed = future.result()
                done += 1
                print(
                    f"[GRID {done}/{len(tasks)}] fold={held:03d}: "
                    f"{new} new + {cached} cached candidates; "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
    summarize(out, orig, pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
