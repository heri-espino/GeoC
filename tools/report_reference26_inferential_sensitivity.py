#!/usr/bin/env python3
"""No-retraining inferential sensitivity companion to the fresh138 jury report.

Requires the original SHA-locked fresh138 LOOCV output; all individual yields,
IDs, coordinates and diagnostics remain in ignored reports/. No --publish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_reference26_compact14_final as runner

from geocebada.evaluation.inferential_sensitivity import (
    grouped_bootstrap,
    influential_parcels,
    paired_advantage_fragility,
    spatial_grid_labels,
)

ID = runner.ID
NAMES = ("Reference26_mean", "Compact14", "Ensemble_final")
OUTPUT = runner.OUTPUT / "inferential_sensitivity"


def load_verified(bundle: Path, peer: Path, config: Path, input_dir: Path):
    frame, _, _, _, ids, _, _ = runner.load_inputs(
        bundle, peer
    )
    expected = runner.provenance(bundle, peer, config)
    record = input_dir / "run_spec.json"
    if not record.is_file() or json.loads(record.read_text()) != expected:
        raise ValueError("Private LOOCV provenance absent or differs from frozen sources.")
    weights = runner.frozen_weights(config)
    completed = runner.read_completed(
        input_dir, ids, frame[runner.TARGET].to_numpy(float), weights
    )
    if len(completed) != 138 or completed[ID].duplicated().any():
        raise ValueError("138 aligned held-out records required.")
    metadata = frame[
        [ID, "Estado", "Municipio", "base_centroid_lat", "base_centroid_lon"]
    ].copy()
    metadata[ID] = metadata[ID].astype(str)
    aligned = completed[[ID]].merge(
        metadata, on=ID, how="left", validate="one_to_one", indicator=True
    )
    if (len(aligned) != 138 or not aligned["_merge"].eq("both").all()
            or aligned[["base_centroid_lat", "base_centroid_lon"]].isna().any().any()):
        raise ValueError("Cannot align all held-out IDs to geographic metadata.")
    return completed, aligned


def run(args):
    completed, meta = load_verified(
        args.bundle.resolve(), args.peer_repo.resolve(),
        args.config.resolve(), args.input.resolve()
    )
    actual = completed["observed"].to_numpy(float)
    estimates = {name: completed[name].to_numpy(float) for name in NAMES}
    private, influence = influential_parcels(
        actual, estimates, completed[ID].to_numpy(str)
    )
    fragile = paired_advantage_fragility(actual, estimates)
    meta_state = meta["Estado"].fillna("<unknown>").astype(str)
    meta_muni = meta["Municipio"].fillna("<unknown>").astype(str)
    municipality = ("state=" + meta_state + "|municipality=" + meta_muni).to_numpy()
    lat = meta["base_centroid_lat"].to_numpy(float)
    lon = meta["base_centroid_lon"].to_numpy(float)
    blocks = [("municipality", municipality)]
    for size in (15., 30., 50.):
        for offset in (0., .5):
            tag = f"grid_{int(size)}km_shift_{offset}"
            blocks.append((tag, spatial_grid_labels(
                lat, lon, width_km=size, offset_fraction=offset
            )))
    results, audit = [], []
    for ordinal, (tag, labels) in enumerate(blocks):
        n_groups = len(np.unique(labels))
        if n_groups < args.min_groups:
            audit.append({
                "grouping": tag, "n_groups": n_groups, "status": "SKIPPED",
                "reason": f"Fewer than {args.min_groups} spatial groups.",
            })
            continue
        result = grouped_bootstrap(
            actual, estimates, labels, grouping=tag,
            replicates=args.replicates, seed=args.seed + ordinal,
            min_groups=args.min_groups,
        )
        results.append(result)
        audit.append({
            "grouping": tag, "n_groups": n_groups, "status": "COMPUTED",
            "reason": "Exploratory grouped resampling; no retraining.",
        })
    if not results:
        raise ValueError(
            "No spatial grouping has enough blocks for an informative analysis."
        )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Contains private IDs, signed errors and row-level influences: never publish.
    private.to_csv(out / "PRIVATE_parcel_influence.csv", index=False)
    # Aggregate only, but still kept in ignored reports until reviewed.
    influence.to_csv(out / "leave_one_out_deletion_summary.csv", index=False)
    fragile.to_csv(out / "paired_advantage_fragility.csv", index=False)
    pd.concat(results, ignore_index=True).to_csv(
        out / "grouped_bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(audit).to_csv(out / "spatial_partition_audit.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "source": "138 fresh 137/1 outer-LOOCV predictions",
        "source_run_spec": json.loads((args.input.resolve() / "run_spec.json").read_text()),
        "seed": args.seed, "replicates": args.replicates,
        "min_groups": args.min_groups,
        "frozen_models": list(NAMES),
        "reference": "Reference26_mean",
        "selection_bias_corrected": False,
        "models_refitted_in_bootstrap": False,
        "valid_predictive_intervals_for_59": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("[INFERENCE] Reference26 is the primary model; ensemble exploratory.")
    print("[DELETION]\n" + influence.to_string(index=False))
    print("[FRAGILITY]\n" + fragile.to_string(index=False))
    print("[SPATIAL GROUPING]\n" + pd.DataFrame(audit).to_string(index=False))
    print(f"[PRIVATE] Saved under {out}; do not commit parcel-level outputs.")
    return 0


def summary(args):
    out = args.output.resolve()
    for name in (
        "leave_one_out_deletion_summary.csv",
        "paired_advantage_fragility.csv",
        "spatial_partition_audit.csv",
        "grouped_bootstrap_intervals.csv",
    ):
        path = out / name
        if not path.is_file():
            raise FileNotFoundError(f"Run --run first; missing {path}")
        data = pd.read_csv(path)
        if name == "grouped_bootstrap_intervals.csv":
            data = data.loc[data.metric.isin([
                "rmse_t_ha", "delta_mse_vs_reference_t2_ha2",
            ])][["model", "grouping", "metric", "estimate",
                 "ci_lower_2p5", "ci_upper_97p5", "n_groups"]]
        print(f"[{name}]\n{data.to_string(index=False)}")
    print("[CAUTION] These are exploratory intervals conditional on "
          "overlapping, frozen LOOCV predictions and previously chosen models.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    parser.add_argument("--bundle", type=Path, default=runner.BUNDLE)
    parser.add_argument("--peer-repo", type=Path, default=runner.PEER_REPO)
    parser.add_argument("--config", type=Path, default=runner.CONFIG)
    parser.add_argument("--input", type=Path, default=runner.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replicates", type=int, default=3000)
    parser.add_argument("--min-groups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    return run(args) if args.run else summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
