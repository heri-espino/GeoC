#!/usr/bin/env python3
"""Produce publication-ready metrics from EXISTING 138-fold LOOCV predictions.

No model fitting. Never use historical 16x41 checkpoint scores as if they were
138x1 LOO predictions. Reuse strict parcel/y/137-1 checks from ensemble tool.

PowerShell:
  python tools/report_loocv_metrics.py --audit
  python tools/report_loocv_metrics.py --run
  python tools/report_loocv_metrics.py --summary
  python tools/report_loocv_metrics.py --publish  # aggregate-only, Git-trackable

To include another genuine 138-fold model:
  python tools/report_loocv_metrics.py --run --extra CatBoost=PATH.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

import run_loo_ensemble_screen as loader
from geocebada.evaluation.area_weighted_climate_comparison import (
    BASE,
    TEMPORAL,
    TOUCHED,
    WEIGHTED,
)
from geocebada.evaluation.loocv_comprehensive_metrics import (
    MAIN_FIELDS,
    OTHER_FIELDS,
    all_metrics,
    paired_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/loocv_comprehensive_metrics"
PUBLISH = ROOT / "results/peer_models"
ID = loader.ID
N = 138
SUPPORTED_CLIMATE = (BASE, TOUCHED, WEIGHTED, TEMPORAL)


def short_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_optional(parts, sources, name, callback, provenance):
    """Missing OPTIONAL model is skipped, but incorrect/misaligned inputs fail."""
    try:
        frame = callback()
    except FileNotFoundError as exc:
        print(f"[SKIP] {name}: {exc}", flush=True)
        return
    except ValueError as exc:
        if "does not have all 138 saved fold" in str(exc):
            print(f"[SKIP] {name}: {exc}", flush=True)
            return
        raise
    parts.append(frame)
    sources[name] = provenance


def mean_variant(directory: Path, k: int, alpha: float):
    """Read exactly the 138 native mean-imputed held-outs, no median reuse."""
    # load_mean validates the fingerprint and full k24/a30 anchor existence.
    loader.load_mean(directory)
    slug = f"k{k}_a{alpha:g}".replace(".", "p")
    files = [directory / f"fold_{i:03d}" / f"{slug}.csv"
             for i in range(1, N+1)]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError(
            f"Mean candidate k={k}, alpha={alpha} lacks 138 saved folds."
        )
    combined = pd.concat([loader.read_csv(p) for p in files], ignore_index=True)
    return loader.audited_model(
        combined, f"Reference26_mean_k{k}_a{alpha:g}",
        f"Reference26_k{k:02d}_alpha{alpha:g}",
    )


def climate_models(path: Path):
    """Exactly the four already-frozen 137/1 climate arms, in long form."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen climate predictions: {path}")
    frame = loader.read_csv(path)
    loader.require_columns(
        frame, {ID, "model", "observed", "predicted",
                "outer_train", "outer_heldout"},
        "Four frozen Local04D climate LOOCV arms",
    )
    labels = {
        BASE: "Local04D",
        TOUCHED: "Local04D_pixel_climate",
        WEIGHTED: "Local04D_area_weighted_climate",
        TEMPORAL: "Local04D_temporal_climate",
    }
    if set(frame["model"].unique()) != set(labels):
        raise ValueError("Four-arm LOOCV climate source has unexpected models.")
    return [
        loader.audited_model(
            frame.loc[frame["model"].eq(name)],
            label, "predicted", require_fold_proof=True,
        ) for name, label in labels.items()
    ]


def audit(args) -> None:
    loader.describe_files(args.mean_dir.resolve(),
                          args.local_file.resolve(),
                          args.dbscan_file.resolve())
    for path, description in (
        (ROOT / "reports/reference26_mean_imputation_loo138/loo_predictions.csv",
         "Independent mean-imputation 138 LOOCV"),
        (ROOT / "reports/reference26_ridge_budget_mean/final_scores.csv",
         "Mean Ridge 138-fold finalists"),
        (ROOT / "reports/area_weighted_climate_loo/loo_summary.csv",
         "Four-arm frozen Local04D summary"),
        (ROOT / "reports/peer_dbscan_loo138/loo_predictions.csv",
         "138-fold geological DBSCAN"),
    ):
        print(f"[AUDIT] {description}: "
              f"{'FOUND' if path.is_file() else 'missing'} {path}")
    print(
        "[AUDIT] Complete 138-fold predictions required for "
        "L1, L2, Linf, MedAE, P90/P95 and exact R2. "
        "Published aggregate RMSE alone is insufficient."
    )


def recover(args):
    original, compact = loader.load_baselines(args.baseline_file.resolve())
    frames = [original, compact]
    sources = {
        "Reference26_original": "native_peer_median_138_loo",
        "Compact14": "native_peer_138_loo",
    }
    add_optional(
        frames, sources, loader.ANCHOR,
        lambda: loader.load_mean(args.mean_dir.resolve()),
        "native_reference26_mean_train_fold_only",
    )
    for name, k, alpha in (
        ("Reference26_mean_k20_a30", 20, 30.0),
        ("Reference26_mean_k12_a3.2", 12, 3.2),
    ):
        add_optional(
            frames, sources, name,
            lambda k=k, alpha=alpha: mean_variant(
                args.mean_dir.resolve(), k, alpha,
            ),
            "native_reference26_mean_138_fold_finalist",
        )
    path = args.local_file.resolve()
    try:
        climates = climate_models(path)
    except FileNotFoundError:
        add_optional(
            frames, sources, "Local04D",
            lambda: loader.read_local(path, original),
            "frozen_local04d_138_loo",
        )
    else:
        for frame in climates:
            name = next(c for c in frame if c not in (ID, "observed"))
            frames.append(frame)
            sources[name] = "frozen_geographic_climate_138_loo"
    add_optional(
        frames, sources, "Reference26_DBSCAN",
        lambda: loader.read_dbscan(args.dbscan_file.resolve()),
        "native_reference26_dbscan_138_loo",
    )
    for value in args.extra:
        name, frame = loader.load_extra(value)
        if name in sources:
            raise ValueError(f"Duplicate additional model: {name}")
        frames.append(frame)
        sources[name] = "user_supplied_138_loo_with_outer_fold_counts"

    # IMPORTANT: Baseline ID set and y equality strictly verified for every arm.
    table = loader.join_models(original, [compact, *frames[2:]])
    if set(table.columns[2:]) != set(sources):
        raise RuntimeError("Unexpected model provenance mismatch.")
    return table, sources


def run(args):
    table, sources = recover(args)
    metric_table = all_metrics(table, sources)
    baseline = (
        loader.ANCHOR if loader.ANCHOR in table
        else "Reference26_original"
    )
    paired = paired_metrics(
        table, baseline=baseline, draws=args.bootstrap, seed=args.seed
    )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(
        out / "all_models_loocv_metrics.csv", index=False,
        float_format="%.16g",
    )
    paired.to_csv(
        out / "paired_model_comparisons.csv", index=False,
        float_format="%.16g",
    )
    (out / "run_spec.json").write_text(
        json.dumps({
            "analysis": "read-only 138-fold LOOCV predictions",
            "n_heldouts": N,
            "train_labels_per_fold": 137,
            "models": sources,
            "reference": baseline,
            "bootstrap_draws": args.bootstrap,
            "bootstrap_seed": args.seed,
            "limitations": [
                "Model selection reused observed LOOCV errors; exploratory scores.",
                "Naive parcel bootstrap ignores spatial dependence.",
                "Adjusted R2 not defined for composite model with no fixed p.",
                "User supplied extra models must have genuine 137/1 provenance.",
            ],
            "metric_definitions": (
                "src/geocebada/evaluation/loocv_comprehensive_metrics.py"
            ),
            "input_provenance": (
                "scripts' existing strict 138-ID and observed-y verification"
            ),
        }, indent=2) + "\n", encoding="utf-8",
    )
    print(f"[METRICS] Strictly matched {N} held-out parcels; "
          f"{len(sources)} models: {', '.join(sources)}.")
    print(metric_table[list(MAIN_FIELDS)].to_string(index=False))
    print(f"[METRICS] Aggregate-only files saved in {out}")
    print("[METRICS] No per-parcel values written by this script.")
    print("[METRICS] Adjusted R2 deliberately NA; see methodology.")


def summary(args):
    path = args.output.resolve() / "all_models_loocv_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No saved metrics: {path}; run --run first.")
    frame = pd.read_csv(path)
    if not set((*MAIN_FIELDS, *OTHER_FIELDS)).issubset(frame.columns):
        raise ValueError("Incomplete or unrecognized metric summary.")
    print(frame[list(MAIN_FIELDS)].to_string(index=False))
    print("[METRICS] Additional columns include L1/L2/Linf, SSE/MSE, "
          "error dispersion and model comparisons.")


def publish(args):
    """Export explicit aggregate-only column allowlists, never target vectors."""
    source = args.output.resolve()
    out = args.publish_dir.resolve()
    metrics = source / "all_models_loocv_metrics.csv"
    paired = source / "paired_model_comparisons.csv"
    if not metrics.is_file() or not paired.is_file():
        raise FileNotFoundError("Run --run first; no aggregate metric report.")
    m = pd.read_csv(metrics)
    p = pd.read_csv(paired)
    allowed = set((*MAIN_FIELDS, *OTHER_FIELDS))
    if set(m.columns) != allowed:
        raise ValueError("Refuse publication: metric columns deviate from allowlist.")
    if any(col in p.columns for col in (
        ID, "observed", "predicted", "RENDIMIENTO_T_HA"
    )):
        raise ValueError("Refuse publication: per-parcel fields detected.")
    if len(m) > 25 or len(p) > 25:
        raise ValueError("Unexpected metric rows; refuse automated publication.")
    if m["n_holdouts"].ne(N).any() or m["model"].duplicated().any():
        raise ValueError("Invalid number of LOOCV observations or model names.")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metrics, out / "LOOCV_FULL_METRICS_138.csv")
    shutil.copyfile(paired, out / "LOOCV_PAIRED_COMPARISONS_138.csv")
    print(f"[PUBLISH] Aggregate-only tables copied to {out}.")
    print("[PUBLISH] Review before git add; no per-parcel ID/target files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("audit", "run", "summary", "publish"):
        action.add_argument("--"+flag, action="store_true")
    parser.add_argument("--baseline-file", type=Path, default=loader.BASELINE_FILE)
    parser.add_argument("--mean-dir", type=Path, default=loader.MEAN_DIR)
    parser.add_argument("--local-file", type=Path, default=loader.LOCAL_FILE)
    parser.add_argument("--dbscan-file", type=Path,
                        default=loader.DBSCAN_DIR / "loo_predictions.csv")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--publish-dir", type=Path, default=PUBLISH)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if args.audit:
        audit(args)
    elif args.run:
        run(args)
    elif args.summary:
        summary(args)
    else:
        publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
