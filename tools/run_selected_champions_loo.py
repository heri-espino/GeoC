#!/usr/bin/env python3
"""Paired LOOCV: GeoC Local04D versus teammate's TWO selected champion models.

No DBSCAN, no B1, no CatBoost. Call peer's original ColdExperts.full_fit,
including its honest nested tuning, without the extra B1 ensemble-selection
refits performed by the peer's refit_fold entrypoint.

  python tools/run_selected_champions_loo.py --check
  python tools/run_selected_champions_loo.py --smoke
  python tools/run_selected_champions_loo.py --run --jobs 4
  python tools/run_selected_champions_loo.py --summary

Each held-out parcel gets 137 labeled training examples. Independent outer
folds can run concurrently on Windows (spawn). Model fits and caches stay local.
"""
from __future__ import annotations

import argparse
import os

# Restrict numerical-library threads per independent outer process.
for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from geocebada.evaluation.peer_champions_loo import (  # noqa: E402
    ID,
    Y,
    outer_record,
    resolve_peer_bundle,
)
from run_peer_champions_loo import (  # noqa: E402
    ROOT,
    _hash,
    _load,
    _lock,
    _spec,
    _write_fold,
    predict,
)
from sklearn.metrics import r2_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

OUTPUT = ROOT / "reports/selected_champions_loo"
LOCAL = "Local04D"
SELECTED = ("Reference26", "Compact14")
MODELS = (LOCAL, *SELECTED)
_WORKER_DATA = None


def validate_selected(row: pd.DataFrame, held_id: str, held_y: float) -> None:
    """Catch cached wrong-parcel results; never mix protocols or model columns."""
    if len(row) != 1 or set(row.columns) != {ID, "observed", *MODELS}:
        raise ValueError("Wrong three-model fold schema.")
    found = row.iloc[0]
    if str(found[ID]) != held_id or abs(float(found["observed"]) - held_y) > 1e-10:
        raise ValueError("Wrong held-out ID/y in saved fold.")
    if not np.isfinite(found[list(MODELS)].to_numpy(float)).all():
        raise ValueError("Nonfinite three-model predictions.")


def fit_selected(frame, groups, distance, geology, record, cache):
    """Fit only the two selected champions, retaining their original tuning.

    Calling peer.refit_fold would independently train four full champions
    repeatedly to choose B1's blend coefficient. The original ColdExperts
    full_fit is sufficient to refit Reference26 and Compact14 (plus the
    common CT3 internals) without fitting B1.
    """
    from src.current_champions import ColdExperts

    train = np.asarray(record["train_positions"], dtype=int)
    held = np.asarray(record["validation_positions"], dtype=int)
    inner = [
        (
            np.asarray(split["train_relative_positions"], dtype=int),
            np.asarray(split["validation_relative_positions"], dtype=int),
        )
        for split in record["inner"]
    ]
    factory = ColdExperts(
        frame, groups, distance, geology, "random", train, inner, cache,
    )
    fitted = factory.full_fit(train)
    if set(fitted["train_positions"]) != set(train):
        raise RuntimeError("Peer fit has unexpected training positions.")
    if set(train) & set(held):
        raise RuntimeError("Outer held-out parcel leaked into fitted models.")
    return {
        "outer_experts": fitted,
        "train_positions": train,
        "validation_positions": held,
        # Peer predict_saved always includes its B1 output; discard it.
        "B1_weight": 0.0,
    }


def _initialize_worker(peer_dir: str, bundle_dir: str) -> None:
    """Load protected private assets exactly once in each worker process."""
    global _WORKER_DATA
    with threadpool_limits(limits=1):
        _WORKER_DATA = _load(Path(peer_dir), Path(bundle_dir))


def _compute_one(position: int, output_dir: str) -> tuple[int, str]:
    """Independent, target-masked cold fit; atomic CSV output for resuming."""
    if _WORKER_DATA is None:
        raise RuntimeError("Worker has not loaded input data.")
    (
        frame, groups, distance, manifest, ids, baseline,
        _unused_refit, predict_saved,
    ) = _WORKER_DATA
    position = int(position)
    held_id = ids[position]
    observed = float(frame.iloc[position][Y])
    folder = Path(output_dir)
    path = folder / f"fold_{position + 1:03}.csv"
    if path.is_file():
        validate_selected(pd.read_csv(path, dtype={ID: str}), held_id, observed)
        return position + 1, "cached"

    record = outer_record(position)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], Y] = np.nan
    train_positions = np.asarray(record["train_positions"], dtype=int)
    if (
        not np.isnan(float(masked.iloc[position][Y]))
        or not np.isfinite(masked.iloc[train_positions][Y].to_numpy(float)).all()
    ):
        raise RuntimeError("Outer target masking did not isolate held-out label.")

    with threadpool_limits(limits=1):
        fit = fit_selected(
            masked, groups, distance, manifest["geology"], record,
            folder / f"peer_cache/fold_{position + 1:03}",
        )
        peer_pred = predict_saved(
            fit, np.asarray([position], dtype=int), masked, distance,
        )
        known = [parcel for parcel in baseline.y.index.astype(str) if parcel != held_id]
        local = float(
            predict(
                baseline, known, [held_id], components=20,
                k=24, alpha=30.0, power=1.0,
            )[0]
        )

    result = pd.DataFrame([{
        ID: held_id,
        "observed": observed,
        LOCAL: local,
        "Reference26": float(np.asarray(peer_pred["Reference26"]).reshape(-1)[0]),
        "Compact14": float(np.asarray(peer_pred["Compact14"]).reshape(-1)[0]),
    }])
    validate_selected(result, held_id, observed)
    _write_fold(path, result)
    return position + 1, "trained"


def _report(folder: Path, *, n: int, complete: bool) -> None:
    paths = [folder / f"fold_{i:03}.csv" for i in range(1, n + 1)]
    if any(not path.is_file() for path in paths):
        raise ValueError(f"Missing one or more of the requested {n} saved folds.")
    combined = pd.concat(
        [pd.read_csv(path, dtype={ID: str}) for path in paths],
        ignore_index=True,
    )
    if len(combined) != n or not combined[ID].is_unique:
        raise ValueError("Held-out parcels repeated or missing from comparison.")
    estimates = []
    for model in MODELS:
        residual = combined[model].to_numpy(float) - combined["observed"].to_numpy(float)
        estimates.append({
            "model": model,
            "n_parcels": n,
            "training_labels_per_fold": 137,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "mean_error": float(np.mean(residual)),
            "r2": (
                float(r2_score(combined["observed"], combined[model]))
                if n >= 2 else np.nan
            ),
        })
    scores = pd.DataFrame(estimates)
    label = "LOOCV" if complete else "SMOKE (exploratory only)"
    print(f"\n[SELECTED {label}] Paired three-model results:\n"
          + scores.to_string(index=False), flush=True)
    _write_fold(folder / ("loo_predictions.csv" if complete else "smoke_predictions.csv"),
                combined)
    _write_fold(folder / ("loo_summary.csv" if complete else "smoke_summary.csv"),
                scores)
    if not complete:
        print(
            "[SELECTED SMOKE] This small sample cannot rank models or "
            "estimate the 138-fold LOOCV RMSE.",
            flush=True,
        )


def _provenance(peer_dir: Path, bundle: Path) -> dict:
    spec = _spec(peer_dir, bundle)
    spec.update({
        "schema": 2,
        "models": list(MODELS),
        "selected_only": True,
        "B1_ensemble_selection_refits_skipped": True,
        "parallelism": "independent outer folds; single-thread numerical libraries",
        "selected_runner_sha256": _hash(Path(__file__).resolve()),
    })
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        action.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--peer-repo", type=Path, default=ROOT.parent / "todoseboso")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--jobs", type=int, default=2,
        help="Independent CPU processes for outer folds; start with 2 or 4.",
    )
    parser.add_argument(
        "--smoke-folds", type=int, default=1,
        help="Use 1–5 held-out parcels in the smoke (default 1).",
    )
    parser.add_argument("--from-fold", type=int, default=1)
    parser.add_argument("--through-fold", type=int, default=138)
    args = parser.parse_args()

    if not 1 <= args.jobs <= 8:
        parser.error("Use --jobs between 1 and 8 to limit memory use.")
    if not 1 <= args.smoke_folds <= 5:
        parser.error("--smoke-folds must be between 1 and 5.")

    folder = (args.output / "smoke" if args.smoke else args.output).resolve()
    if args.summary:
        if not (folder / "run_spec.json").is_file():
            raise ValueError("Missing provenance manifest; full run not completed.")
        _report(folder, n=138, complete=True)
        return 0

    peer_dir = args.peer_repo.resolve()
    bundle = resolve_peer_bundle(ROOT, args.bundle, peer_dir)
    # Load and validate the exact same IDs / known yields before spawning workers.
    (
        frame, _groups, _distance, _manifest, ids, _baseline, _refit, _predict,
    ) = _load(peer_dir, bundle)
    spec = _provenance(peer_dir, bundle)
    print(
        f"[SELECTED] Input bundle: {bundle}\n"
        f"[SELECTED] 138 aligned parcels; 137 training labels per held-out fit.\n"
        "[SELECTED] Models: Local04D, Reference26, Compact14 (no B1, no DBSCAN).",
        flush=True,
    )
    if args.check:
        return 0

    _lock(folder, spec)
    start = 1 if args.smoke else args.from_fold
    stop = args.smoke_folds if args.smoke else args.through_fold
    if not 1 <= start <= stop <= 138:
        parser.error("Choose valid held-out fold bounds from 1 through 138.")

    pending = []
    for fold in range(start, stop + 1):
        path = folder / f"fold_{fold:03}.csv"
        if path.is_file():
            validate_selected(
                pd.read_csv(path, dtype={ID: str}),
                ids[fold - 1],
                float(frame.iloc[fold - 1][Y]),
            )
            print(f"[SELECTED {fold}/{stop}] already saved", flush=True)
        else:
            pending.append(fold - 1)

    if pending:
        print(
            f"[SELECTED] Training {len(pending)} folds using "
            f"{min(args.jobs, len(pending))} independent CPU process(es).",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(pending)),
            initializer=_initialize_worker,
            initargs=(str(peer_dir), str(bundle)),
        ) as executor:
            futures = {
                executor.submit(_compute_one, position, str(folder)): position
                for position in pending
            }
            for future in as_completed(futures):
                fold, status = future.result()
                print(f"[SELECTED {fold}/{stop}] {status}", flush=True)

    if args.smoke:
        _report(folder, n=args.smoke_folds, complete=False)
    elif len(list(folder.glob("fold_*.csv"))) == 138:
        _report(folder, n=138, complete=True)
    else:
        print(
            "[SELECTED] Partial paired experiment saved; rerun --run to resume.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
