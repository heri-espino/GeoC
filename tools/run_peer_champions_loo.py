#!/usr/bin/env python3
"""Cold-refit CT3/Reference26/Compact14/B1 beside GeoC Local04D under paired LOOCV.

Requires a trusted local checkout of heri-espino/todoseboso for source code.
The exported private bundle is resolved first from GeoC/data_n/ (including
common nested layouts), then from the sibling checkout. Never commit data_n.

  python tools/run_peer_champions_loo.py --check
  python tools/run_peer_champions_loo.py --smoke
  python tools/run_peer_champions_loo.py --run
  python tools/run_peer_champions_loo.py --summary

Resumable, label-masked, no cached model replay, no official 59 predictions changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import ID, atomic_json
from geocebada.evaluation.checkpoint10_pca import FROZEN, inputs, load_pca_data, predict
from geocebada.evaluation.peer_champions_loo import (
    BASELINE,
    MODELS,
    PEER_MODELS,
    SEED,
    Y,
    align_peer_targets,
    outer_record,
    resolve_peer_bundle,
    summarize_complete,
    validate_fold,
)

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
DEFAULT_OUT = ROOT / "reports/peer_champions_loo"


def _hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _git(peer: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(peer), *args], text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError("Peer checkout must be a readable Git repository.") from exc


def _load(peer: Path, bundle: Path):
    """Import the original peer recipes, not an approximation or saved predictor."""
    if not (peer / "src/current_champions.py").is_file():
        raise FileNotFoundError("Missing peer src/current_champions.py; clone todoseboso.")
    if not (bundle / "manifest.json").is_file():
        raise FileNotFoundError(
            "Missing private current_champions/v1 bundle. The fork alone "
            "does not contain the 26-feature inputs, groups and distances. "
            "Request the authorized bundle/export from the teammate."
        )
    if _git(
        peer, "status", "--porcelain", "--",
        "src", "configs", "scripts/521_current_champions.py",
    ):
        raise RuntimeError(
            "Peer modeling source/config has uncommitted changes; commit or clean it."
        )
    sys.path.insert(0, str(peer))
    from src.current_champions import load_bundle, predict_saved, refit_fold

    frame, groups, distance, manifest = load_bundle(bundle, verify_saved_models=False)
    # The frozen Local04D model uses a *20-component PCA fit*, not the
    # first 20 axes of a separate 160-component decomposition. SVD sign
    # conventions may differ between fits; the historical numerical parity
    # guard in load_pca_data remains strict and is never bypassed.
    baseline = load_pca_data(ROOT, MEMBERSHIPS, max_components=20)
    ids = align_peer_targets(frame, baseline.y)
    if set(groups.index.astype(str)) != set(ids):
        raise ValueError("Peer group IDs do not match the 138 training parcels.")
    if np.asarray(distance).shape != (138, 138) or not np.isfinite(distance).all():
        raise ValueError("Peer geographic distance must be finite 138 x 138.")
    if set(PEER_MODELS) - set(manifest["models"]):
        raise ValueError("Private bundle does not declare all four peer recipes.")
    return frame, groups, distance, manifest, ids, baseline, refit_fold, predict_saved


def _spec(peer: Path, bundle: Path) -> dict:
    paths = inputs(ROOT, MEMBERSHIPS)
    paths.update({
        "peer_runner": ROOT / "tools/run_peer_champions_loo.py",
        "peer_contracts": ROOT / "src/geocebada/evaluation/peer_champions_loo.py",
        "geoc_predictor": ROOT / "src/geocebada/evaluation/checkpoint04d1.py",
        "geoc_pca": ROOT / "src/geocebada/evaluation/checkpoint10_pca.py",
    })
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required GeoC input '{key}' missing: {path}")
    return {
        "schema": 1, "protocol": "same_138_parcels_cold_refit_LOOCV",
        "training_labels_per_fit": 137, "folds": 138,
        "inner_cv": "4-fold shuffled KFold, relative indices, seed 20261018",
        "outer_label_masked": True, "seed": SEED, "models": list(MODELS),
        "geoc_x": "197-parcel target-free transductive C4 PCA20",
        "peer_features": "pre-exported 26-feature bundle; 138 observed X; no prior fitted models",
        "peer_repo": "https://github.com/heri-espino/todoseboso",
        "peer_commit": _git(peer, "rev-parse", "HEAD"),
        "peer_bundle_manifest_sha256": _hash(bundle / "manifest.json"),
        "geoc_input_sha256": {
            key: _hash(path) for key, path in sorted(paths.items())
        },
        "limitations": "Spatially dependent labels; prior model selection on these 138 y.",
    }


def _lock(output: Path, spec: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_spec.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError("Inputs, code or peer commit changed; use a NEW --output directory.")
    elif list(output.glob("fold_*.csv")):
        raise RuntimeError("Orphaned fold files have no provenance; use a new output directory.")
    else:
        atomic_json(path, spec)


def _write_fold(path: Path, result: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    result.to_csv(temp, index=False, float_format="%.16g")
    temp.replace(path)


def _summarize(output: Path) -> None:
    files = sorted(output.glob("fold_*.csv"))
    if len(files) != 138 or not (output / "run_spec.json").is_file():
        raise ValueError(f"Need exactly 138 complete, provenance-locked folds; found {len(files)}.")
    expected = {f"fold_{i:03}.csv" for i in range(1, 139)}
    if {f.name for f in files} != expected:
        raise ValueError("Missing or unexpected fold index.")
    frames = [pd.read_csv(path, dtype={ID: str}) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    scores, paired = summarize_complete(combined)
    _write_fold(output / "loo_predictions.csv", combined)
    _write_fold(output / "loo_summary.csv", scores)
    _write_fold(output / "loo_paired_differences.csv", paired)
    print("[PEER LOO] 138 labeled parcels, 137 train / 1 test per model.\n"
          + scores.to_string(index=False), flush=True)
    print("[PEER LOO] Paired differences (peer minus GeoC):\n"
          + paired.to_string(index=False), flush=True)
    print("[PEER LOO] Exploratory: repeated development on the same y and "
          "spatial dependence prevent independent confirmation.", flush=True)


def _smoke_report(output: Path, n_folds: int) -> None:
    """Display and save a preliminary five-model comparison; never rank champions."""
    paths = [output / f"fold_{i:03}.csv" for i in range(1, n_folds + 1)]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Incomplete smoke folds; cannot compare the models.")
    predictions = pd.concat(
        [pd.read_csv(path, dtype={ID: str}) for path in paths], ignore_index=True
    )
    if not predictions[ID].is_unique or len(predictions) != n_folds:
        raise ValueError("Smoke fold IDs are not distinct.")
    details = []
    for row in predictions.to_dict("records"):
        for model in MODELS:
            error = float(row[model] - row["observed"])
            details.append({
                "heldout_parcel": row[ID],
                "model": model,
                "observed": float(row["observed"]),
                "predicted": float(row[model]),
                "error": error,
                "absolute_error": abs(error),
            })
    table = pd.DataFrame(details)
    by_model = (
        table.groupby("model", as_index=False)
        .agg(
            n_heldout=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            mse=("error", lambda values: float(np.mean(np.square(values)))),
        )
    )
    by_model["rmse"] = np.sqrt(by_model.pop("mse"))
    # The smoke sample was not selected for population representativeness.
    _write_fold(output / "smoke_predictions.csv", table)
    _write_fold(output / "smoke_model_comparison.csv", by_model)
    print(
        f"\n[PEER SMOKE] Finished {n_folds} fold(s); "
        "all five models produced finite held-out predictions.",
        flush=True,
    )
    print("[PEER SMOKE] Individual held-out predictions:", flush=True)
    print(table.to_string(index=False), flush=True)
    print("[PEER SMOKE] Descriptive errors on ONLY these smoke parcels:", flush=True)
    print(by_model.to_string(index=False), flush=True)
    print(
        "[PEER SMOKE] NOT a 138-fold LOOCV estimate; do not select a winner "
        "or compare this table with published full-sample RMSE.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    for command in ("check", "smoke", "run", "summary"):
        commands.add_argument(f"--{command}", action="store_true")
    parser.add_argument("--peer-repo", type=Path, default=ROOT.parent / "todoseboso")
    parser.add_argument(
        "--bundle", type=Path, default=None,
        help="Explicit complete bundle or parent folder; default: GeoC/data_n/.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--smoke-folds", type=int, default=1,
        help="Number of outer held-out parcels in smoke (1–5); all five models per fold.",
    )
    parser.add_argument("--from-fold", type=int, default=1)
    parser.add_argument("--through-fold", type=int, default=138)
    args = parser.parse_args()
    if not 1 <= args.smoke_folds <= 5:
        parser.error("--smoke-folds must be between 1 and 5.")
    if not args.smoke and args.smoke_folds != 1:
        parser.error("--smoke-folds is only supported together with --smoke.")

    if args.summary:
        _summarize(args.output.resolve())
        return 0
    peer = args.peer_repo.resolve()
    bundle = resolve_peer_bundle(ROOT, args.bundle, peer)
    (frame, groups, distance, manifest, ids, baseline,
     refit_fold, predict_saved) = _load(peer, bundle)
    spec = _spec(peer, bundle)
    print(
        f"[PEER LOO] Bundle: {bundle}\n"
        f"[PEER LOO] Input contract PASS: 138 aligned y; "
        f"peer commit {spec['peer_commit'][:12]}; "
        "GeoC 197-X PCA20; 26-feature peer bundle.",
        flush=True,
    )
    if args.check:
        return 0

    output = (args.output / "smoke" if args.smoke else args.output).resolve()
    _lock(output, spec)
    start = 1 if args.smoke else args.from_fold
    stop = args.smoke_folds if args.smoke else args.through_fold
    if not 1 <= start <= stop <= 138:
        parser.error("Fold bounds must satisfy 1 <= from-fold <= through-fold <= 138.")

    for fold in range(start, stop + 1):
        position = fold - 1
        held_id = ids[position]
        held_y = float(frame.iloc[position][Y])
        path = output / f"fold_{fold:03}.csv"
        if path.is_file():
            validate_fold(
                pd.read_csv(path, dtype={ID: str}),
                expected_id=held_id, expected_y=held_y,
            )
            print(f"[PEER LOO {fold}/138] cached {held_id}", flush=True)
            continue
        record = outer_record(position, n=len(ids))
        # Mask the actual target BEFORE calling any peer training code.
        # It is available separately only to score predictions after fitting.
        masked = frame.copy(deep=True)
        masked.loc[masked.index[position], Y] = np.nan
        if not np.isfinite(masked.iloc[record["train_positions"]][Y].to_numpy(float)).all():
            raise RuntimeError("Unexpected missing training yield.")
        if not np.isnan(float(masked.iloc[position][Y])):
            raise AssertionError("Held-out target was not masked.")
        pack = refit_fold(
            masked, groups, distance, manifest["geology"], record,
            output / f"peer_cache/fold_{fold:03}",
        )
        if set(pack["train_positions"]) != set(record["train_positions"]):
            raise RuntimeError("Peer model used unexpected outer training indices.")
        peer_pred = predict_saved(pack, np.array([position]), masked, distance)
        geoc_train = [pid for pid in baseline.y.index.astype(str) if pid != held_id]
        geoc_pred = float(predict(baseline, geoc_train, [held_id], components=20, **FROZEN)[0])
        row = {ID: held_id, "observed": held_y, BASELINE: geoc_pred}
        for name in PEER_MODELS:
            row[name] = float(np.asarray(peer_pred[name], dtype=float).reshape(-1)[0])
        result = pd.DataFrame([row])
        validate_fold(result, expected_id=held_id, expected_y=held_y)
        _write_fold(path, result)
        print(f"[PEER LOO {fold}/138] saved {held_id}", flush=True)
    if args.smoke:
        _smoke_report(output, args.smoke_folds)
    elif len(list(output.glob("fold_*.csv"))) == 138:
        _summarize(output)
    else:
        print("[PEER LOO] Partial results only; no final score.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
