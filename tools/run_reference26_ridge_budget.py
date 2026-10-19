#!/usr/bin/env python3
"""Budgeted Reference26 search: reuse existing 250-grid fits; run 25 -> 8 -> 3.

PowerShell, from GeoC:
    python tools/run_reference26_ridge_budget.py --audit
    python tools/run_reference26_ridge_budget.py --screen --jobs 12
    python tools/run_reference26_ridge_budget.py --advance --jobs 12
    python tools/run_reference26_ridge_budget.py --final --jobs 12
    python tools/run_reference26_ridge_budget.py --summary

This never modifies the original 250-grid run or its run_spec.json.
"""
from __future__ import annotations

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_reference26_ridge250_loo as original_grid  # noqa: E402

from geocebada.evaluation.peer_reference26_ridge_grid import DEFAULT_PAIR  # noqa: E402

ROOT = original_grid.ROOT
BUNDLE = original_grid.BUNDLE
PEER_REPO = original_grid.PEER_REPO
BASELINE = original_grid.BASELINE
SOURCE = ROOT / "reports/reference26_ridge250_loo"
OUTPUT = ROOT / "reports/reference26_ridge_budget"
SCREEN = tuple((k, a) for k in (12, 20, 24, 32, 48)
               for a in (3.2, 10.0, 30.0, 100.0, 300.0))
assert len(SCREEN) == 25 and DEFAULT_PAIR in SCREEN
STAGES = {"screen": (12, 25), "advance": (46, 8), "final": (138, 3)}
ID = original_grid.ID


def pair_list(items):
    return tuple((int(x[0]), float(x[1])) for x in items)


def manifest(bundle, peer, source):
    info = original_grid.manifest(bundle, peer, original_grid.GRID)
    return {
        "version": 1,
        "strategy": "same first 12 then 46 then all 138; top 7+anchor, top 2+anchor",
        "initial_pairs": [[k, a] for k, a in SCREEN],
        "source_250": str(source.resolve()),
        "invariant_hashes": {
            k: info[k] for k in (
                "teammate_manifest_hash", "teammate_code_hash",
                "teammate_bridge_hash", "teammate_local_hash",
                "local_override_hash", "original_runner_hash",
                "runner_hash",
            )
        },
    }


def ensure_source_compatible(source, data):
    spec = source / "run_spec.json"
    if not spec.is_file():
        print("[REUSE] No old provenance file; old results will not be reused.")
        return False
    old = json.loads(spec.read_text(encoding="utf-8"))
    for key, value in data["invariant_hashes"].items():
        if old.get(key) != value:
            raise ValueError(
                f"Old grid {key} differs; do not reuse predictions from "
                "different code/data. Keep original reports untouched."
            )
    if pair_list(old["k_alpha_pairs"]) != original_grid.GRID:
        raise ValueError("Previous grid settings do not match the original 250.")
    return True


def write_manifest(out, spec):
    out.mkdir(parents=True, exist_ok=True)
    file = out / "run_spec.json"
    if file.is_file():
        if json.loads(file.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "New script/data/source changed. Preserve the existing outputs "
                "and choose a new --output directory."
            )
    elif any(out.glob("fold_*/*.csv")):
        raise ValueError("Existing adaptive output has no provenance file.")
    else:
        file.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def selection_file(out, name):
    return out / f"{name}_selection.json"


def selected_pairs(out, name):
    if name == "screen":
        return SCREEN
    predecessor = "screen" if name == "advance" else "advance"
    previous = selection_file(out, predecessor)
    if not previous.is_file():
        raise ValueError(f"Complete --{predecessor} before --{name}.")
    json.loads(previous.read_text(encoding="utf-8"))
    if not (out / f"{predecessor}_scores.csv").is_file():
        raise ValueError(f"Missing {predecessor} results.")
    ranking = pd.read_csv(out / f"{predecessor}_scores.csv")
    if len(ranking) < STAGES[predecessor][1]:
        raise ValueError("Previous stage has incomplete model ranking.")
    top_count = 7 if name == "advance" else 2
    leading = []
    for row in ranking.itertuples(index=False):
        pair = (int(row.k), float(row.alpha))
        if pair != DEFAULT_PAIR and pair not in leading:
            leading.append(pair)
        if len(leading) == top_count:
            break
    if len(leading) < top_count:
        raise ValueError("Not enough uniquely scored configurations.")
    result = tuple(sorted((*leading, DEFAULT_PAIR)))
    if len(result) != STAGES[name][1]:
        raise RuntimeError("Unexpected shortlist length.")
    return result


def validate_original(source, bundle, peer):
    checked = manifest(bundle, peer, source)
    can_reuse = ensure_source_compatible(source, checked)
    return checked, can_reuse


def cached_old(source, output, pos, pair, ids, y):
    """Copy only a validated original fold/config into the new private folder."""
    dest = original_grid.fold_file(output, pos, pair)
    if dest.is_file():
        original_grid.validate_row(dest, pos, ids, y, pair)
        return True, False
    old = original_grid.fold_file(source, pos, pair)
    if not old.is_file():
        return False, False
    value = original_grid.validate_row(old, pos, ids, y, pair)
    # Write atomically using the original project's writer, never modify old.
    original_grid.write_csv(dest, pd.DataFrame([{
        ID: ids[pos], "observed": float(y[pos]),
        original_grid.candidate_name(*pair): value,
    }]))
    return True, True


def scores_for_stage(output, baseline, name, pairs):
    count = STAGES[name][0]
    y = baseline["observed"].to_numpy(float)[:count]
    ids = baseline[ID].astype(str).tolist()
    reference = baseline["Reference26"].to_numpy(float)[:count]
    ref_mse = float(np.mean((reference - y)**2))
    rows = []
    for k, alpha in pairs:
        prediction = []
        for pos in range(count):
            file = original_grid.fold_file(output, pos, (k, alpha))
            if not file.is_file():
                raise ValueError(f"Missing {file}")
            prediction.append(original_grid.validate_row(
                file, pos, ids, baseline["observed"].to_numpy(float), (k, alpha)
            ))
        error = np.asarray(prediction) - y
        mse = float(np.mean(error**2))
        rows.append({
            "model": original_grid.candidate_name(k, alpha),
            "k": k, "alpha": alpha, "n_common_holdouts": count,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(abs(error))),
            "delta_mse_vs_original": mse - ref_mse,
            "mean_error": float(np.mean(error)),
        })
    result = pd.DataFrame(rows).sort_values(
        ["rmse", "k", "alpha"], kind="stable"
    ).reset_index(drop=True)
    original_grid.write_csv(output / f"{name}_scores.csv", result)
    original_metric = float(np.sqrt(ref_mse))
    print(f"[{name.upper()}] Fixed first {count} parcels, original "
          f"Reference26 RMSE={original_metric:.6f} t/ha", flush=True)
    print(result.head(15).to_string(index=False), flush=True)
    selection_file(output, name).write_text(
        json.dumps({
            "stage": name, "n_common_holdouts": count,
            "evaluated_pairs": [[k, a] for k, a in pairs],
        }, indent=2) + "\n", encoding="utf-8",
    )
    print("These stage rankings are exploratory and reused for selection.",
          flush=True)


def audit(source, baseline, pairs=SCREEN):
    """Show how much old computation is usable before doing any more fits."""
    ids = baseline[ID].astype(str).tolist()
    y = baseline["observed"].to_numpy(float)
    totals = []
    total_files = 0
    unique_positions = set()
    for pos in range(138):
        folder = source / f"fold_{pos+1:03d}"
        paths = list(folder.glob("*.csv"))
        if paths:
            unique_positions.add(pos)
        total_files += len(paths)
        valid = []
        for pair in pairs:
            path = original_grid.fold_file(source, pos, pair)
            if path.is_file():
                original_grid.validate_row(path, pos, ids, y, pair)
                valid.append(pair)
        totals.append(valid)
    for threshold in (1, 12, 46, 138):
        if threshold == 1:
            count = sum(bool(x) for x in totals)
        else:
            count = sum(
                all(pair in totals[pos] for pos in range(threshold))
                for pair in pairs
            )
        if threshold == 1:
            print(f"[AUDIT] Number of parcels with any of the 25: {count}")
        else:
            print(f"[AUDIT] Configurations completed on all first "
                  f"{threshold} parcels: {count}/25")
    print(f"[AUDIT] Total old saved configuration-parcel files: {total_files}")
    print(f"[AUDIT] Distinct old held-out parcel folders with files: "
          f"{len(unique_positions)}/138")
    print(f"[AUDIT] Selected 25-configuration reusable predictions: "
          f"{sum(map(len, totals))}/3450")
    print("[AUDIT] No models were fitted and no old files were changed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ("audit", "screen", "advance", "final", "summary"):
        mode.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--reuse-from", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--peer-repo", type=Path, default=PEER_REPO)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    a = p.parse_args()
    if a.jobs < 1 or a.batch_size < 1:
        p.error("--jobs and --batch-size must be positive.")
    bundle, peer = a.bundle.resolve(), a.peer_repo.resolve()
    source, output = a.reuse_from.resolve(), a.output.resolve()
    frame, _groups, _distance, _meta, ids, _class, _pred = (
        original_grid.load_inputs(bundle, peer)
    )
    baseline = original_grid.original_results(
        a.baseline.resolve(), ids, frame[original_grid.TARGET].to_numpy(float)
    )
    info, can_reuse = validate_original(source, bundle, peer)
    if a.audit:
        if can_reuse:
            audit(source, baseline)
        return 0
    write_manifest(output, info)
    if a.summary:
        for stage in STAGES:
            path = output / f"{stage}_scores.csv"
            if path.is_file():
                print(f"\n[{stage.upper()}]\n"
                      + pd.read_csv(path).head(10).to_string(index=False))
        return 0
    name = ("screen" if a.screen else
            "advance" if a.advance else "final")
    pairs = selected_pairs(output, name)
    count = STAGES[name][0]
    print(f"[{name.upper()}] {len(pairs)} fixed settings x "
          f"first {count} common held-out parcels.", flush=True)
    write_snapshot = baseline["observed"].to_numpy(float)
    recovered = 0
    if can_reuse:
        for pos in range(count):
            for pair in pairs:
                _found, copied = cached_old(
                    source, output, pos, pair, ids, write_snapshot
                )
                recovered += int(copied)
    print(f"[REUSE] Recovered {recovered} already completed predictions.",
          flush=True)

    # Check true numerical parity before any extensive model selection.
    original_grid.parity_baseline(output, bundle, peer, baseline)
    jobs = []
    for pos in range(count):
        missing = [
            pair for pair in pairs if not original_grid.fold_file(
                output, pos, pair
            ).is_file()
        ]
        for start in range(0, len(missing), a.batch_size):
            jobs.append((pos, tuple(missing[start:start+a.batch_size])))
    print(f"[{name.upper()}] {len(jobs)} task batches left; "
          f"{min(a.jobs, len(jobs))} workers.", flush=True)
    if jobs:
        with ProcessPoolExecutor(
            max_workers=min(a.jobs, len(jobs)),
            initializer=original_grid.worker_init,
            initargs=(str(bundle), str(peer)),
        ) as pool:
            futures = [
                pool.submit(original_grid.run_batch, pos, chunk, str(output))
                for pos, chunk in jobs
            ]
            for i, future in enumerate(as_completed(futures), 1):
                held, new, cached, elapsed = future.result()
                print(f"[{name.upper()} {i}/{len(jobs)}] parcel={held:03d}, "
                      f"{new} new/{cached} cached, {elapsed:.1f}s",
                      flush=True)
    scores_for_stage(output, baseline, name, pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
