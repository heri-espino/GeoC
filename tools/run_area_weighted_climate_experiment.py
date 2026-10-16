#!/usr/bin/env python3
"""Area-weighted climate feature experiment with fixed geographic Ridge.

Run:
  python tools/run_area_weighted_climate_experiment.py --check
  python tools/run_area_weighted_climate_experiment.py --build
  python tools/run_area_weighted_climate_experiment.py --smoke
  python tools/run_area_weighted_climate_experiment.py --run
  python tools/run_area_weighted_climate_experiment.py --summary

An optional external 197-row X-only CSV with exactly four covariates:
  python tools/run_area_weighted_climate_experiment.py --run \
      --peer-features data/local/peer_four_covariates.csv \
      --output reports/area_weighted_climate_with_peer

The optional model is a reproducible four-variable global Ridge PROXY;
it is NOT a reproduction of another person's proprietary modeling code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.area_weighted_climate_comparison import (
    BASE,
    ENSEMBLE,
    FOUR,
    TEMPORAL,
    TOUCHED,
    WEIGHTED,
    make_features,
    matched_summary,
    predict_local,
    split_predictions,
)
from geocebada.evaluation.checkpoint08_local import ID, atomic_json, digest
from geocebada.evaluation.checkpoint10_pca import (
    FROZEN,
    inputs,
    load_pca_data,
    predict,
)
from geocebada.features.area_weighted_climate import (
    MONTHS,
    VARIABLES,
    build_climate_representations,
    climate_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
CLIMATE = ROOT / "data/source/climate"
GEOMETRY = ROOT / "data/source/geospatial/Parcelas_Reto_AGC_CONJUNTO_70_30.zip"
OUT = ROOT / "reports/area_weighted_climate_experiment"
SOURCE_FILES = (
    "src/geocebada/features/area_weighted_climate.py",
    "src/geocebada/evaluation/area_weighted_climate_comparison.py",
    "src/geocebada/features/parcel.py",
    "src/geocebada/evaluation/checkpoint04d1.py",
    "src/geocebada/evaluation/checkpoint10_pca.py",
    "tools/run_area_weighted_climate_experiment.py",
)


def _paths() -> tuple[dict[str, Path], dict[tuple[str, int, int], Path]]:
    official = inputs(ROOT, MEMBERSHIPS)
    official["geometry"] = GEOMETRY
    for kind, path in official.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {kind}: {path}")
    raster_files = climate_inventory(CLIMATE)
    return official, raster_files


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temp, index=False)
    temp.replace(path)


def _fingerprint(
    folder: Path, official: dict[str, Path],
    rasters: dict[tuple[str, int, int], Path], peer: Path | None,
) -> None:
    meta = {
        "schema": 1,
        "name": "geographic_weighted_ridge_area_climate",
        "year": 2025,
        "months": list(MONTHS),
        "weather_fields": list(VARIABLES),
        "raster_files": {
            "_".join(map(str, key)): {
                "file": str(path.relative_to(ROOT)),
                "sha256": digest(path),
            } for key, path in rasters.items()
        },
        "official_sha256": {label: digest(path) for label, path in official.items()},
        "implementation_sha256": {
            name: digest(ROOT / name) for name in SOURCE_FILES
        },
        "peer_features": (
            None if peer is None else {"file": str(peer), "sha256": digest(peer)}
        ),
        "frozen_geographic_ridge": {
            "components": 20, "neighbors": FROZEN["k"],
            "ridge_alpha": FROZEN["alpha"], "inverse_distance_power": FROZEN["power"],
        },
        "frozen_outer_mask_count": 16,
        "target_labels": 138,
        "unlabeled_predictions": 59,
    }
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "run_spec.json"
    if spec.exists():
        if json.loads(spec.read_text(encoding="utf-8")) != meta:
            raise ValueError(
                "Experiment inputs/code changed. Keep this output intact and choose "
                "a different --output directory for a new experiment."
            )
    else:
        if list(folder.glob("confirm_*")) or any(
            (folder / name).exists() for name in (
                "weighted_monthly.csv", "touched_monthly.csv"
            )
        ):
            raise ValueError("Unidentified saved results in output directory.")
        atomic_json(spec, meta)


def _build(folder: Path, rasters: dict[tuple[str, int, int], Path]) -> None:
    paths = (
        "weighted_monthly.csv", "touched_monthly.csv",
        "weighted_temporal.csv", "pixel_coverage_diagnostics.csv",
    )
    existing = [((folder / name).exists()) for name in paths]
    if all(existing):
        print("[CLIMATE] Target-free feature matrices already built.", flush=True)
        return
    if any(existing):
        raise ValueError("Incomplete feature build: inspect local output; do not mix files.")
    import geopandas as gpd

    geo = gpd.read_file(f"zip://{GEOMETRY}")
    weighted, touched, temporal, quality = build_climate_representations(geo, rasters)
    if (len(weighted) != 197 or len(touched) != 197
            or len(temporal) != 197 or quality[ID].nunique() != 197):
        raise ValueError("Expected 197 unique parcels across all climate tables.")
    if (quality["coverage"].lt(.95).any()
            or quality["effective_cells"].le(0).any()):
        raise ValueError(
            "Insufficient climate raster coverage (<95%) or missing grid cells. "
            "Inspect source and geometries; do not silently impute."
        )
    for name, frame in zip(
        paths, (weighted, touched, temporal, quality), strict=True,
    ):
        _write_csv(folder / name, frame)
    target_columns = [
        c for c in weighted if any(
            c.endswith(f"__2025_m{month:02d}") for month in MONTHS
        )
    ]
    rounded = weighted[target_columns].round(6)
    distinct_profiles = int(rounded.drop_duplicates().shape[0])
    counts = rounded.value_counts(dropna=False)
    stats = {
        "parcels": 197,
        "climate_pixels_monthly_rasters": len(rasters),
        "target_year_monthly_features": len(target_columns),
        "distinct_full_2025_climate_profiles_rounded6": distinct_profiles,
        "largest_identical_climate_profile_parcel_count": int(counts.iloc[0]),
        "median_effective_pixels_per_parcel_month":
            float(quality["effective_cells"].median()),
        "mean_abs_weighted_minus_touched":
            float(quality["absolute_extraction_difference"].mean()),
        "warning": "Similar or identical raster-grid series do not create new independent labels.",
    }
    atomic_json(folder / "climate_extraction_summary.json", stats)
    print(
        f"[CLIMATE] 197 parcel profiles from {len(rasters)} source monthly rasters. "
        f"{distinct_profiles} distinct 2025 profiles (rounded 6 decimals). "
        f"Mean |weighted-touched|="
        f"{stats['mean_abs_weighted_minus_touched']:.6f}.",
        flush=True,
    )


def _data(folder: Path, peer: Path | None):
    data = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
    if len(data.folds) != 16:
        raise ValueError("Expected 16 frozen matched 97/41 split IDs.")
    optional = None if peer is None else pd.read_csv(peer)
    features = make_features(
        data,
        pd.read_csv(folder / "weighted_monthly.csv"),
        pd.read_csv(folder / "touched_monthly.csv"),
        pd.read_csv(folder / "weighted_temporal.csv"),
        peer_features=optional,
    )
    ref = ROOT / "reports/checkpoint_08/confirm_01__paired.csv"
    if ref.is_file():
        prev = pd.read_csv(ref)
        prev = prev.loc[prev["model"].eq("Local04D_frozen")]
        fit, held = data.folds["confirm_01"]
        expected = prev.set_index(ID).loc[held, "predicted"].to_numpy(float)
        recomputed = predict_local(features, fit, held, BASE)
        parity = float(np.max(np.abs(expected - recomputed)))
        if parity > 1e-7:
            raise ValueError(f"Historical geographic PCA20 predictions changed: {parity}")
        print(f"[CLIMATE] Frozen geographic Ridge parity PASS, max Δ={parity:.2g}.")
    return features


def _smoke(folder: Path, peer: Path | None) -> None:
    f = _data(folder, peer)
    fit, held = f.baseline.folds["confirm_01"]
    models = [BASE, TOUCHED, WEIGHTED, TEMPORAL]
    original = {name: predict_local(f, fit, held[:3], name) for name in models}
    frozen = f.baseline.y.loc[held].copy()
    f.baseline.y.loc[held] = 999999.
    try:
        changed = {name: predict_local(f, fit, held[:3], name) for name in models}
        for model in models:
            np.testing.assert_allclose(original[model], changed[model], rtol=0., atol=1e-10)
    finally:
        f.baseline.y.loc[held] = frozen
    print(
        "[CLIMATE SMOKE] Frozen control, 4 feature arms, "
        "outer pseudo-target label isolation PASS. No 59 predictions.",
        flush=True,
    )


def _run(folder: Path, peer: Path | None) -> None:
    f = _data(folder, peer)
    names = {BASE, TOUCHED, WEIGHTED, TEMPORAL}
    if peer is not None:
        names |= {FOUR, ENSEMBLE}
    for i, split in enumerate(sorted(f.baseline.folds), 1):
        out = folder / f"{split}__paired.csv"
        meta = folder / f"{split}__selection.json"
        if out.exists() and meta.exists():
            prior = pd.read_csv(out)
            if (len(prior) != 41 * len(names) or set(prior["model"]) != names
                    or prior.groupby(ID)["model"].nunique().ne(len(names)).any()):
                raise ValueError(f"Damaged paired output in {split}.")
            print(f"[CLIMATE {i}/16] {split} complete.", flush=True)
            continue
        if out.exists() != meta.exists():
            raise ValueError(f"Partially written paired results in {split}.")
        paired, info = split_predictions(f, split)
        if len(paired) != 41 * len(names) or set(paired["model"]) != names:
            raise ValueError("Incomplete output: no partial write performed.")
        _write_csv(out, paired)
        atomic_json(meta, info)
        print(f"[CLIMATE {i}/16] {split} completed on 41 held-out parcels.", flush=True)
    _summary(folder)


def _summary(folder: Path) -> None:
    paths = sorted(folder.glob("confirm_*__paired.csv"))
    if len(paths) != 16:
        raise ValueError(f"Only {len(paths)}/16 split prediction files exist.")
    scores, bootstrap = matched_summary(
        pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    )
    _write_csv(folder / "comparison_summary.csv", scores)
    _write_csv(folder / "paired_cluster_bootstrap.csv", bootstrap)
    print("[CLIMATE] Matched 16-fold pooled RMSE (lower better):\n"
          + scores.to_string(index=False), flush=True)
    print("[CLIMATE] Paired delta MSE against original geographic Ridge:\n"
          + bootstrap.to_string(index=False), flush=True)
    print(
        "[CLIMATE] Historical outer masks are reused/exploratory; "
        "no final prediction promotion and no new labels.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for item in ("check", "build", "smoke", "run", "summary"):
        mode.add_argument(f"--{item}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--peer-features", type=Path,
        help="Optional 197-row, ID_POLIGONO + exactly 4 numeric X-only columns.",
    )
    args = parser.parse_args()
    peer = args.peer_features
    if peer is not None and not peer.is_file():
        parser.error(f"Peer features CSV not found: {peer}")
    official, rasters = _paths()
    if args.check:
        import geopandas as gpd

        geometries = gpd.read_file(f"zip://{GEOMETRY}")
        if len(geometries) != 197 or geometries.crs is None:
            raise ValueError("Expected 197 parcels with declared geometry CRS.")
        if peer is not None and len(pd.read_csv(peer)) != 197:
            raise ValueError("Peer CSV must contain exactly 197 rows.")
        print(
            f"[CLIMATE CHECK] {len(geometries)} geometries, {len(rasters)} "
            "2022–2025 rasters, frozen PCA20 dependencies PASS.", flush=True,
        )
        return 0
    _fingerprint(args.output, official, rasters, peer)
    if args.summary:
        _summary(args.output)
        return 0
    _build(args.output, rasters)
    if args.build:
        return 0
    if args.smoke:
        _smoke(args.output, peer)
        return 0
    _run(args.output, peer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
