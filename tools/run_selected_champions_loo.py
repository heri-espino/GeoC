#!/usr/bin/env python3
"""LOOCV for the two selected todoseboso models, on the original private bundle.

Uses *only* her ColdExperts.full_fit and predict_saved implementations.
No GeoC PCA, Local04D, B1 training, DBSCAN, or external data reconstruction.

  python tools/run_selected_champions_loo.py --check
  python tools/run_selected_champions_loo.py --smoke
  python tools/run_selected_champions_loo.py --run --jobs 2
  python tools/run_selected_champions_loo.py --summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data_n/processed/current_champions/v1"
PEER_REPO = ROOT.parent / "todoseboso"
OUTPUT = ROOT / "reports/selected_champions_loo"
ID = "ID_POLIGONO"
TARGET = "RENDIMIENTO_T_HA"
MODELS = ("Reference26", "Compact14")
SEED = 20261018

_DATA = None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(bundle: Path, peer_repo: Path):
    """Read the original validated 26-input bundle, with its saved models disabled."""
    source = peer_repo / "src/current_champions.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing teammate's source: {source}")
    if not (bundle / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing exported champion bundle: {bundle}")
    if str(peer_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(peer_repo.resolve()))
    from src.current_champions import ColdExperts, load_bundle, predict_saved

    frame, groups, distance, manifest = load_bundle(
        bundle, verify_saved_models=False
    )
    ids = frame[ID].astype(str).tolist()
    if len(ids) != 138 or len(set(ids)) != 138:
        raise ValueError("Expected 138 unique labeled parcel IDs.")
    y = frame[TARGET].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("All 138 observed training yields must be finite.")
    if set(groups.index.astype(str)) != set(ids):
        raise ValueError("Saved groups and training IDs do not match.")
    if np.shape(distance) != (138, 138) or not np.isfinite(distance).all():
        raise ValueError("Saved geographical distance must be finite 138×138.")
    if not set(MODELS).issubset(set(manifest["models"])):
        raise ValueError("The private bundle does not include both champions.")
    if len(manifest["features"]) != 26:
        raise ValueError("Expected the original 26-feature shared bundle.")
    if not set(manifest["features"]).issubset(frame.columns):
        raise ValueError("One or more original model features are missing.")
    return frame, groups, distance, manifest, ids, ColdExperts, predict_saved


def fold_record(position: int, n: int = 138) -> tuple[np.ndarray, list]:
    """137 training labels and four inner folds, never including the held-out row."""
    if not 0 <= position < n:
        raise ValueError("Invalid held-out position.")
    train = np.delete(np.arange(n, dtype=int), position)
    inner = [
        (np.asarray(a, dtype=int), np.asarray(b, dtype=int))
        for a, b in KFold(
            n_splits=4, shuffle=True, random_state=SEED
        ).split(train)
    ]
    if not all(len(set(train[a]) & {position}) == 0 for a, _ in inner):
        raise AssertionError("Outer target entered the inner training folds.")
    return train, inner


def validate_saved(row: pd.DataFrame, parcel: str, observed: float) -> None:
    if len(row) != 1 or set(row.columns) != {ID, "observed", *MODELS}:
        raise ValueError("Saved fold has an unexpected schema.")
    result = row.iloc[0]
    if str(result[ID]) != parcel or abs(float(result["observed"]) - observed) > 1e-10:
        raise ValueError("Saved fold does not match the held-out parcel and yield.")
    if not np.isfinite(row[list(MODELS)].to_numpy(float)).all():
        raise ValueError("Saved fold has nonfinite model predictions.")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(staging, index=False, float_format="%.16g")
    staging.replace(path)


def run_spec(bundle: Path, peer_repo: Path) -> dict:
    try:
        peer_commit = subprocess.check_output(
            ["git", "-C", str(peer_repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        peer_commit = "not_available"
    return {
        "protocol": "138x leave-one-parcel-out; 137 train, 1 test",
        "models": list(MODELS),
        "inner": f"4-fold shuffled KFold, seed={SEED}, 137 outer-train rows",
        "peer_commit": peer_commit,
        "peer_model_source_sha256": digest(peer_repo / "src/current_champions.py"),
        "manifest_sha256": digest(bundle / "manifest.json"),
        "runner_sha256": digest(Path(__file__).resolve()),
    }


def lock_output(folder: Path, spec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Code or bundle changed. Choose a new --output directory.")
    else:
        if list(folder.glob("fold_*.csv")):
            raise ValueError("Untracked completed folds exist without a run specification.")
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def initialize_worker(bundle: str, peer: str) -> None:
    global _DATA
    _DATA = load_inputs(Path(bundle), Path(peer))


def fit_one(position: int, output_dir: str) -> tuple[int, str, float]:
    """Fit her two models from scratch on the same 137 labels."""
    if _DATA is None:
        raise RuntimeError("Worker inputs were not initialized.")
    frame, groups, distance, manifest, ids, ColdExperts, predict_saved = _DATA
    held_id = ids[position]
    observed = float(frame.iloc[position][TARGET])
    out = Path(output_dir)
    path = out / f"fold_{position + 1:03}.csv"
    if path.is_file():
        validate_saved(pd.read_csv(path, dtype={ID: str}), held_id, observed)
        return position + 1, "cached", 0.

    train, inner = fold_record(position)
    masked = frame.copy(deep=True)
    masked.loc[masked.index[position], TARGET] = np.nan
    if not np.isnan(float(masked.iloc[position][TARGET])):
        raise AssertionError("The excluded yield was not masked.")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        factory = ColdExperts(
            masked, groups, distance, manifest["geology"], "random",
            train, inner, out / f"cache/fold_{position + 1:03}",
        )
        fit = factory.full_fit(train)
        if set(fit["train_positions"]) != set(train):
            raise RuntimeError("The model used unexpected training positions.")
        # predict_saved includes B1 as a derived value; we save ONLY the two
        # selected models. No B1 calibration or extra refit is performed.
        pack = {
            "outer_experts": fit,
            "train_positions": train,
            "validation_positions": np.array([position]),
            "B1_weight": 0.,
        }
        predictions = predict_saved(
            pack, np.array([position]), masked, distance
        )
    row = pd.DataFrame([{
        ID: held_id,
        "observed": observed,
        "Reference26": float(np.asarray(predictions["Reference26"]).item()),
        "Compact14": float(np.asarray(predictions["Compact14"]).item()),
    }])
    validate_saved(row, held_id, observed)
    write_csv(path, row)
    return position + 1, "trained", time.perf_counter() - started


def summarize(folder: Path, *, count: int, final: bool) -> None:
    paths = [folder / f"fold_{i:03}.csv" for i in range(1, count + 1)]
    if any(not p.is_file() for p in paths):
        raise ValueError(f"Need {count} completed held-out folds for this summary.")
    frame = pd.concat(
        [pd.read_csv(p, dtype={ID: str}) for p in paths], ignore_index=True
    )
    if len(frame) != count or frame[ID].duplicated().any():
        raise ValueError("Repeated held-out parcel or incomplete saved folds.")
    scores = []
    for model in MODELS:
        actual = frame["observed"].to_numpy(float)
        residual = frame[model].to_numpy(float) - actual
        scores.append({
            "model": model,
            "held_out_parcels": count,
            "train_labels_per_fold": 137,
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "mean_error": float(np.mean(residual)),
            "r2": float(r2_score(actual, frame[model])) if count >= 2 else np.nan,
        })
    name = "loo" if final else "smoke"
    write_csv(folder / f"{name}_predictions.csv", frame)
    write_csv(folder / f"{name}_summary.csv", pd.DataFrame(scores))
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    if not final:
        print("[SMOKE] This is not the 138-fold RMSE.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run", "summary"):
        action.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--smoke-folds", type=int, default=1)
    parser.add_argument("--from-fold", type=int, default=1)
    parser.add_argument("--through-fold", type=int, default=138)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8.")
    if not 1 <= args.smoke_folds <= 5:
        parser.error("--smoke-folds must be between 1 and 5.")
    if args.smoke and (args.from_fold != 1 or args.through_fold != 138):
        parser.error("Do not combine --smoke with --from-fold/--through-fold.")

    bundle, peer = args.bundle.resolve(), args.peer_repo.resolve()
    out = (args.output / "smoke" if args.smoke else args.output).resolve()
    # Only teammate's original bundle and source. No GeoC PCA or yield files.
    frame, _groups, _distance, _manifest, ids, _class, _predict = load_inputs(
        bundle, peer
    )
    print(
        f"[PEER] Bundle: {bundle}\n"
        "[PEER] 138 labeled parcels; models: Reference26 and Compact14.",
        flush=True,
    )
    if args.check:
        print("[CHECK] Both models' source, bundle and IDs validated.", flush=True)
        return 0
    lock_output(out, run_spec(bundle, peer))
    if args.summary:
        summarize(out, count=138, final=True)
        return 0
    start = 1 if args.smoke else args.from_fold
    stop = args.smoke_folds if args.smoke else args.through_fold
    if not 1 <= start <= stop <= 138:
        parser.error("Fold bounds must lie between 1 and 138.")

    todo = []
    for fold in range(start, stop + 1):
        path = out / f"fold_{fold:03}.csv"
        if path.is_file():
            validate_saved(
                pd.read_csv(path, dtype={ID: str}),
                ids[fold - 1],
                float(frame.iloc[fold - 1][TARGET]),
            )
        else:
            todo.append(fold - 1)
    if todo:
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(todo)),
            initializer=initialize_worker,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            pending = {
                pool.submit(fit_one, pos, str(out)): pos for pos in todo
            }
            for future in as_completed(pending):
                number, status, seconds = future.result()
                print(f"[LOO {number:03d}/138] {status} ({seconds:.1f}s)", flush=True)

    if args.smoke:
        summarize(out, count=args.smoke_folds, final=False)
    elif start == 1 and stop == 138:
        summarize(out, count=138, final=True)
    else:
        print("[LOO] Partial results saved. Full summary requires all 138.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
