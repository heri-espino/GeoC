#!/usr/bin/env python3
"""Checkpoint 07B: thousands of X-only metrics -> PCA -> Ridge/MLP benchmark.

Usage (PowerShell):
  python tools/run_checkpoint_07_highdim.py --preflight
  python tools/run_checkpoint_07_highdim.py --smoke
  python tools/run_checkpoint_07_highdim.py
  python tools/run_checkpoint_07_highdim.py --stage evaluate

No download, no overwrite of the incumbent Local04D, no hidden-y inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from geocebada.data.targets import load_yield_split
from geocebada.evaluation.checkpoint07_pca import (
    evaluate_highdim_pca,
    summarize_pca_predictions,
)
from geocebada.features.checkpoint07_highdim import (
    ID,
    build_hls_smap_lag_features,
    build_hls_time_features,
    build_nonlinear_interactions,
    build_smap_time_features,
    merge_x_only_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


def _ensure_not_lfs_pointer(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"Git LFS pointer instead of the real data: {path}\n"
                "Run: git lfs install; git lfs pull"
            )


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("all", "build", "evaluate"), default="all"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "data/processed/features_v1/parcel_features_competition.csv",
    )
    parser.add_argument(
        "--agronomic", type=Path,
        default=ROOT / "data/processed/agronomic_features_v1/"
        "parcel_agronomic_features.csv",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "data/source/tabular/"
        "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
    )
    parser.add_argument(
        "--membership", type=Path,
        default=ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv",
    )
    parser.add_argument(
        "--hls", type=Path,
        default=ROOT / "data/processed/checkpoint_07/hls_parcel_observations.csv",
    )
    parser.add_argument(
        "--smap", type=Path,
        default=ROOT / "data/processed/checkpoint_07/smap_parcel_daily.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/checkpoint_07/highdim",
    )
    parser.add_argument("--require-hls", action="store_true")
    parser.add_argument("--require-smap", action="store_true")
    parser.add_argument(
        "--include-siap-2025", action="store_true",
        help="Allow contemporaneous municipal yield proxy in X (not default).",
    )
    parser.add_argument("--max-seeds", type=int, default=180)
    parser.add_argument("--max-products", type=int, default=12000)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--skip-neural", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    for path in (args.base, args.agronomic, args.membership, args.targets):
        _ensure_not_lfs_pointer(path)
    availability = {
        "hls": args.hls.is_file(),
        "smap": args.smap.is_file(),
    }
    if args.require_hls and not availability["hls"]:
        raise FileNotFoundError(
            f"Missing HLS panel: {args.hls}. "
            "Run tools/run_checkpoint_07_processing.py --stage all"
        )
    if args.require_smap and not availability["smap"]:
        raise FileNotFoundError(
            f"Missing SMAP panel: {args.smap}. Run tools/run_checkpoint_07_smap.py"
        )
    if args.max_seeds < 0 or args.max_products < 0 or args.max_splits < 0:
        raise ValueError("Seed, product and split caps cannot be negative.")
    return {
        "hls_available": availability["hls"],
        "smap_available": availability["smap"],
        "missing_modalities": [key for key, exists in availability.items() if not exists],
        "historical_siap_2025_proxy_included": args.include_siap_2025,
    }


def _feature_build(
    args: argparse.Namespace,
    output: Path,
    qc: dict[str, object],
) -> dict[str, object]:
    base = pd.read_csv(args.base, low_memory=False)
    agronomic = pd.read_csv(args.agronomic, low_memory=False)
    ids = base[ID].astype(str).tolist()
    if len(ids) != 197 or len(set(ids)) != 197:
        raise ValueError("The X feature table must have 197 unique official parcels.")
    if not args.include_siap_2025:
        base = base.drop(
            columns=[c for c in base if c.startswith("siap_2025")],
            errors="ignore",
        )
    blocks = [agronomic]
    hls = pd.read_csv(args.hls, low_memory=False) if qc["hls_available"] else None
    smap = pd.read_csv(args.smap, low_memory=False) if qc["smap_available"] else None
    if hls is not None:
        blocks.append(build_hls_time_features(hls, ids))
    if smap is not None:
        blocks.append(build_smap_time_features(smap, ids))
    if hls is not None and smap is not None:
        blocks.append(build_hls_smap_lag_features(hls, smap, ids))
    merged = merge_x_only_blocks(base, *blocks)
    # Even if an agronomic block happens to contain municipal label proxies,
    # exclude direct public 2025 municipal outcome data in default mode.
    if not args.include_siap_2025:
        proxy_columns = [
            col for col in merged
            if "siap_2025" in col
        ]
        merged = merged.drop(columns=proxy_columns)
    seeds = 25 if args.smoke else args.max_seeds
    products = 60 if args.smoke else args.max_products
    nonlinear, manifest = build_nonlinear_interactions(
        merged, max_seeds=seeds, max_products=products
    )
    result = pd.concat([merged.reset_index(drop=True), nonlinear.reset_index(drop=True)], axis=1)
    if result.columns.duplicated().any():
        raise ValueError("Duplicate column names in expanded 07B matrix.")
    if len(result) != 197 or result[ID].duplicated().any():
        raise ValueError("Invalid expanded parcel matrix.")
    forbidden = {"RENDIMIENTO_T_HA", "CONJUNTO", "PRODUCCION_T", "RENDIMIENTO"}
    if forbidden.intersection(result.columns):
        raise ValueError("Yield/split information survived X-only merge.")
    path = output / "expanded_parcel_features.csv"
    result.to_csv(path, index=False)
    manifest["n_all_x_features"] = len(result.columns) - 1
    manifest["n_original_plus_temporal"] = len(merged.columns) - 1
    manifest["has_hls"] = hls is not None
    manifest["has_smap"] = smap is not None
    manifest["siap_2025_proxy"] = args.include_siap_2025
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[07B BUILD] {len(result)} parcels; "
        f"{len(result.columns)-1:,} features; {manifest['n_products']:,} products",
        flush=True,
    )
    return manifest


def _evaluate(
    args: argparse.Namespace,
    output: Path,
) -> dict[str, object]:
    path = output / "expanded_parcel_features.csv"
    _ensure_not_lfs_pointer(path)
    x = pd.read_csv(path, low_memory=False)
    if not args.include_siap_2025 and (
        any("siap_2025" in col for col in x.columns)
    ):
        raise ValueError("Feature matrix contains 2025 municipal outcomes.")
    targets = load_yield_split(path=args.targets, validate=True)
    membership = pd.read_csv(args.membership)
    if "target_matched" not in set(membership["family"]):
        raise ValueError("Frozen target-matched pseudo-competition splits missing.")
    split_limit = 1 if args.smoke else args.max_splits
    if args.smoke:
        print("[07B SMOKE] one split, small model grid, isolated output", flush=True)
    pred, metrics, diagnostics = evaluate_highdim_pca(
        x, targets, membership,
        variances=(0.80, 0.95) if args.smoke else (0.80, 0.95, 0.99),
        fixed_ranks=() if args.smoke else (16, 32, 64),
        alphas=(100.0,) if args.smoke else (10.0, 100.0, 1000.0),
        quadratic=not args.smoke,
        neural=not args.skip_neural,
        mlp_hidden=((16,),) if args.smoke else ((16,), (32, 16)),
        mlp_alphas=(1.0,) if args.smoke else (1.0, 10.0),
        mlp_max_iter=120 if args.smoke else 600,
        max_splits=split_limit,
    )
    summary = summarize_pca_predictions(pred)
    pred.to_csv(output / "oof_predictions.csv", index=False)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    diagnostics.to_csv(output / "pca_diagnostics.csv", index=False)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    print("[07B BENCHMARK] Top 10 PCA finalists (development, not confirmation):")
    print(summary.head(10).to_string(index=False), flush=True)
    return {
        "n_models": len(summary),
        "n_splits": int(pred["split_id"].nunique()),
        "n_repeated_oof_predictions": len(pred),
        "best_development_model": str(summary.iloc[0]["model"]),
        "best_development_pooled_rmse": float(summary.iloc[0]["rmse_pooled"]),
    }


def main() -> int:
    args = parse_args()
    qc = preflight(args)
    print("Checkpoint 07B preflight: PASS")
    print(json.dumps(qc, indent=2))
    if args.preflight:
        return 0
    output = args.output.resolve()
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    output_file = output / "expanded_parcel_features.csv"
    if args.stage in {"all", "build"}:
        if output_file.is_file() and not args.force:
            raise FileExistsError(
                f"Existing X matrix: {output_file}. "
                "Use --stage evaluate or --force to rebuild."
            )
        manifest = _feature_build(args, output, qc)
    else:
        manifest_file = output / "feature_manifest.json"
        if manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if bool(manifest["siap_2025_proxy"]) != args.include_siap_2025:
                raise ValueError("SIAP 2025 proxy option differs from feature build.")
        else:
            manifest = {}
    outcome: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _commit(),
        "smoke": args.smoke,
        "stage": args.stage,
        "preflight": qc,
        "features": manifest,
        "incumbent_modified": False,
        "hidden_target_used": False,
    }
    if args.stage in {"all", "evaluate"}:
        outcome["evaluation"] = _evaluate(args, output)
    (output / "run_report.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Checkpoint 07B: DONE. Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
