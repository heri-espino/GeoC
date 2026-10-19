#!/usr/bin/env python3
"""Reference26, same 138 LOOCV folds, four missing-X handling strategies.

1. ORIGINAL_MEDIAN: the already-saved Reference26 native train-only medians;
2. SPATIAL: region-prioritized geographic donor interpolation of missing X;
3. TEMPORAL: same-parcel dated BASIC/PRO reconstruction of supported X;
4. SPATIOTEMPORAL: dated geographical donor kernel for supported X,
   regional spatial fallback for other missing numeric features.

No changes to observed source features or the Reference26 model/parameters.
No y interpolation, held-out label leakage, raw time-series-to-y training,
or use of hidden 59 yields. Coordinates, geology shares and categories remain.

  python tools/run_reference26_imputation_loo.py --check
  python tools/run_reference26_imputation_loo.py --smoke --jobs 1
  python tools/run_reference26_imputation_loo.py --run --jobs 12
  python tools/run_reference26_imputation_loo.py --summary

Results under ignored reports/reference26_imputation_loo138/.
"""
from __future__ import annotations

import os

for variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

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

from geocebada.evaluation.peer_x_imputation import (  # noqa: E402
    BASIC,
    PRO,
    SATELLITE_FEATURES,
    prepare,
)

OUTPUT = ROOT / "reports/reference26_imputation_loo138"
ORIGINAL = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
DEFAULT_BASIC = ROOT / "data/source/tabular" / BASIC
DEFAULT_PRO = ROOT / "data/source/tabular" / PRO
N = 138
METHODS = ("spatial", "temporal", "spatiotemporal")
MODEL = "Reference26"
_BASELINE_NAME = "median_original"
_WORKER = None


def selected_methods(text: str) -> tuple[str, ...]:
    names = tuple(v.strip() for v in text.split(","))
    if not names or len(set(names)) != len(names) or any(
        name not in METHODS for name in names
    ):
        raise ValueError(f"Methods must be distinct members of {METHODS}.")
    return tuple(name for name in METHODS if name in names)


def baseline_predictions(path: Path, ids: list[str], y: np.ndarray) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Original full Reference26 LOOCV prediction CSV is missing: {path}. "
            "Recover it with run_selected_champions_loo.py --summary."
        )
    saved = pd.read_csv(path, dtype={ID: str})
    if (
        len(saved) != N or not saved[ID].is_unique
        or not {ID, "observed", MODEL}.issubset(saved.columns)
        or set(saved[ID]) != set(ids)
    ):
        raise ValueError("Original LOOCV baseline has incomplete/mismatched 138 IDs.")
    aligned = saved.set_index(ID).loc[ids].reset_index()
    if (
        not np.isfinite(aligned[["observed", MODEL]].to_numpy(float)).all()
        or not np.allclose(
            aligned["observed"].to_numpy(float), y,
            rtol=0.0, atol=1e-9,
        )
    ):
        raise ValueError("Saved baseline has different observations from peer bundle.")
    return aligned[[ID, "observed", MODEL]].rename(
        columns={MODEL: _BASELINE_NAME}
    )


def provenance(
    bundle: Path, peer: Path, basic: Path, pro: Path,
    methods: tuple[str, ...], spatial_k: int,
) -> dict:
    return {
        "version": 1,
        "protocol": "original teammate 138 parcel positions, 137 labeled train / 1 heldout",
        "methods": list(methods),
        "original_model": "ColdExperts.full_fit with untouched Reference26 configuration",
        "original_baseline": "saved Reference26 138-fold original medians",
        "spatial_k": spatial_k,
        "X_only_transductive": "all 138 X available; never observe held-out y",
        "temporal_cycle": "2025 April-October and historical Apr-Oct 2022-24 only",
        "supported_temporal_features": list(SATELLITE_FEATURES),
        "peer_manifest_sha256": digest(bundle / "manifest.json"),
        "peer_code_sha256": digest(peer / "src/current_champions.py"),
        "peer_signal_code_sha256": digest(peer / "src/geoc_signal_research.py"),
        "basic_sha256": digest(basic),
        "pro_sha256": digest(pro),
        "imputer_sha256": digest(
            ROOT / "src/geocebada/evaluation/peer_x_imputation.py"
        ),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def lock(folder: Path, spec: dict):
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "run_spec.json"
    if file.is_file():
        if json.loads(file.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Imputer/raw inputs or model changed since first fit. "
                "Use a NEW --output directory to avoid mixing results."
            )
    elif list(folder.glob("fold_*/**/*.csv")):
        raise ValueError("Private fold predictions already exist without provenance.")
    else:
        file.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def row_file(folder: Path, pos: int, method: str) -> Path:
    return folder / f"fold_{pos + 1:03d}" / f"{method}.csv"


def validated_row(
    folder: Path, pos: int, method: str, ids: list[str], y: np.ndarray,
) -> float:
    path = row_file(folder, pos, method)
    data = pd.read_csv(path, dtype={ID: str})
    if (
        len(data) != 1
        or set(data.columns) != {ID, "observed", method}
        or str(data.iloc[0][ID]) != ids[pos]
        or abs(float(data.iloc[0]["observed"]) - float(y[pos])) > 1e-10
        or not np.isfinite(data[method].to_numpy(float)).all()
    ):
        raise ValueError(f"Invalid saved held-out prediction: {path}")
    return float(data.iloc[0][method])


def initialize_worker(
    bundle: str, peer: str, frames: dict[str, pd.DataFrame],
):
    global _WORKER
    source = load_inputs(Path(bundle), Path(peer))
    _WORKER = (*source, frames)


def do_fold(
    pos: int, methods: tuple[str, ...], output_str: str,
    baseline: float, unchanged: tuple[str, ...],
):
    """Train each changed-X variant separately, resume per fold and variant."""
    if _WORKER is None:
        raise RuntimeError("Worker was not initialized.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved, frames = _WORKER
    output = Path(output_str)
    observed = float(frame.iloc[pos][TARGET])
    train, inner = fold_record(pos)
    if pos in train or len(train) != 137:
        raise RuntimeError("LOOCV must train on exactly the other 137 parcels.")
    counts = {"trained": 0, "cached": 0, "unchanged": 0}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for method in methods:
            saved = row_file(output, pos, method)
            if saved.is_file():
                validated_row(
                    output, pos, method, ids, frame[TARGET].to_numpy(float)
                )
                counts["cached"] += 1
                continue
            if method in unchanged:
                prediction = baseline
                counts["unchanged"] += 1
            else:
                modified = frames[method].copy(deep=True)
                modified.loc[modified.index[pos], TARGET] = np.nan
                if (
                    not np.isnan(float(modified.iloc[pos][TARGET]))
                    or not np.isfinite(
                        modified.iloc[train][TARGET].to_numpy(float)
                    ).all()
                ):
                    raise RuntimeError("Held-out y contamination in imputation run.")
                temp = (
                    output / "_training_cache" /
                    f"fold_{pos + 1:03d}" / method
                )
                try:
                    factory = ColdExperts(
                        modified, groups, distance, manifest["geology"],
                        "random", train, inner, temp,
                    )
                    fitted = factory.full_fit(train)
                    if set(fitted["train_positions"]) != set(train):
                        raise ValueError("Native model used wrong training positions.")
                    predicted = predict_saved(
                        {
                            "outer_experts": fitted,
                            "train_positions": train,
                            "validation_positions": np.array([pos], int),
                            "B1_weight": 0.0,
                        },
                        np.array([pos], int), modified, distance,
                    )
                    prediction = float(
                        np.asarray(predicted[MODEL], dtype=float).item()
                    )
                finally:
                    shutil.rmtree(temp, ignore_errors=True)
                counts["trained"] += 1
            if not np.isfinite(prediction):
                raise ValueError(f"Nonfinite {method} prediction at fold {pos + 1}.")
            write_csv(
                saved,
                pd.DataFrame([{
                    ID: ids[pos], "observed": observed, method: prediction,
                }]),
            )
    return pos + 1, counts, time.perf_counter() - started


def scores(
    output: Path, original: pd.DataFrame,
    methods: tuple[str, ...], *,
    n: int,
):
    if len(original) < n:
        raise ValueError("Original baseline does not cover requested held-out folds.")
    paired = original.iloc[:n].copy()
    ids = original[ID].astype(str).tolist()
    y = original["observed"].to_numpy(float)
    for method in methods:
        paired[method] = [
            validated_row(output, i, method, ids, y)
            for i in range(n)
        ]
    actual = paired["observed"].to_numpy(float)
    original_mse = float(np.mean(
        (paired[_BASELINE_NAME].to_numpy(float) - actual)**2
    ))
    rows = []
    for method in (_BASELINE_NAME, *methods):
        errors = paired[method].to_numpy(float) - actual
        mse = float(np.mean(errors**2))
        rows.append({
            "method": method,
            "n_heldout": n,
            "training_labels_per_fit": 137,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "r2": float(r2_score(actual, paired[method])) if n > 1 else np.nan,
            "delta_mse_vs_median": mse - original_mse,
        })
    table = pd.DataFrame(rows).sort_values(
        "rmse", kind="stable"
    ).reset_index(drop=True)
    prefix = "loo138" if n == N else "smoke"
    write_csv(output / f"{prefix}_summary.csv", table)
    write_csv(output / f"{prefix}_predictions_private.csv", paired)
    print(
        f"[IMPUTATION {n}/138] All methods on exactly the same held-out parcels:\n"
        + table.to_string(index=False), flush=True,
    )
    if n < N:
        print("[SMOKE] Not a complete LOOCV estimate.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for verb in ("check", "smoke", "run", "summary"):
        mode.add_argument("--" + verb, action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--basic", type=Path, default=DEFAULT_BASIC)
    parser.add_argument("--pro", type=Path, default=DEFAULT_PRO)
    parser.add_argument("--original-results", type=Path, default=ORIGINAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--spatial-k", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive; there is no artificial maximum.")
    try:
        methods = selected_methods(args.methods)
    except ValueError as exc:
        parser.error(str(exc))
    bundle, peer = args.bundle.resolve(), args.peer_repo.resolve()
    basic, pro = args.basic.resolve(), args.pro.resolve()
    folder = (args.output / "smoke" if args.smoke else args.output).resolve()

    frame, _groups, distance, manifest, ids, _class, _predict = load_inputs(
        bundle, peer
    )
    original = baseline_predictions(
        args.original_results.resolve(),
        ids, frame[TARGET].to_numpy(float),
    )
    reconstructed = prepare(
        frame, distance, manifest["features"], basic, pro,
        spatial_k=args.spatial_k,
    )
    audit = reconstructed.audit[
        reconstructed.audit["method"].isin(methods)
    ].reset_index(drop=True)
    totals = audit.groupby("method", sort=False)[
        ["missing_before", "newly_filled", "still_missing"]
    ].sum()
    print(
        "[IMPUTATION] Reference26 source intact, 138 outer folds, "
        "137 train / 1 masked test; four strategies including native median.",
        flush=True,
    )
    print(f"[IMPUTATION] Original numeric X missing cells "
          f"(eligible features): {reconstructed.original_missing}", flush=True)
    print(f"[IMPUTATION] Raw observations for labeled parcel IDs: "
          f"{reconstructed.raw_observations}", flush=True)
    print(totals.to_string(), flush=True)
    if args.check:
        if not audit["newly_filled"].sum():
            print(
                "[CHECK] None of these imputation methods reconstructs "
                "missing predictor cells. Their predictions will be identical "
                "to the original without any new fits.",
                flush=True,
            )
        return 0

    # Avoid ever publishing a private audit by accident: only aggregate
    # method-feature counts, no individual parcel identifiers.
    lock(folder, provenance(bundle, peer, basic, pro, methods, args.spatial_k))
    write_csv(folder / "imputation_audit_aggregates.csv", audit)
    unchanged = tuple(
        method for method in methods if int(
            totals.loc[method, "newly_filled"]
        ) == 0
    )
    if args.summary:
        scores(folder, original, methods, n=N)
        return 0
    n = 1 if args.smoke else N
    need = []
    for i in range(n):
        if all(row_file(folder, i, method).is_file() for method in methods):
            for method in methods:
                validated_row(
                    folder, i, method, ids, frame[TARGET].to_numpy(float)
                )
        else:
            need.append(i)

    if need:
        count = min(len(need), args.jobs)
        print(
            f"[IMPUTATION] {len(need)} pending heldouts, "
            f"{count} user-requested CPU workers. "
            f"Methods with no X changes: {unchanged}.",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=count,
            initializer=initialize_worker,
            initargs=(str(bundle), str(peer), reconstructed.frames),
        ) as executor:
            pending = {
                executor.submit(
                    do_fold, i, methods, str(folder),
                    float(original.iloc[i][_BASELINE_NAME]), unchanged,
                ): i for i in need
            }
            for task in as_completed(pending):
                index, counts, elapsed = task.result()
                print(
                    f"[IMPUTATION {index:03d}/{n}] "
                    f"{counts}, {elapsed:.1f}s",
                    flush=True,
                )

    scores(folder, original, methods, n=n)
    if n == N:
        print(
            "[CAUTION] Method chosen after inspecting all 138 heldout "
            "results has optimistic selection bias; this is competition "
            "exploration, not an independent generalization estimate.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
