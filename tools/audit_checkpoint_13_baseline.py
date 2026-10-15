#!/usr/bin/env python3
"""Audit the frozen Local04D on Checkpoint13's *same* twelve screening parcels.

This script does NOT rerun any random trials, change run_spec.json, or touch
the 59 target forecasts. It uses the saved 100,000-candidate screen files and
the exact frozen X-only masks to calculate the incumbent's 12-target RMSE,
virtual rank among candidates, and 2x4-fold inner RMSE. Only the new audit
CSV is written inside the already gitignored experimental output directory.

Usage:
    python tools/audit_checkpoint_13_baseline.py --output reports/checkpoint_13_100k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geocebada.evaluation.checkpoint08_local import digest
from geocebada.evaluation.checkpoint10_pca import (
    FROZEN,
    inputs,
    load_pca_data,
    predict,
)
from geocebada.evaluation.checkpoint13_randomized import (
    BASELINE,
    FAMILIES,
    SCREEN_TARGETS,
    SEED,
    repeated_cv_folds,
)

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "reports/checkpoint_08/confirmation_membership.csv"
OUTPUT = ROOT / "reports/checkpoint_13_100k"


def screening_folds(
    train: list[str], *, seed: int, count: int = SCREEN_TARGETS
) -> tuple[list[str], list[str]]:
    """Match the Checkpoint13 target-first seeded permutation byte for byte."""
    if count < 1 or count >= len(train):
        raise ValueError("Invalid 12-target screen size.")
    indices = np.random.default_rng(seed).permutation(len(train))
    return (
        [train[i] for i in indices[count:]],
        [train[i] for i in indices[:count]],
    )


def baseline_inner_scores(data, split: str, *, count: int = SCREEN_TARGETS):
    """Compute both baseline screen and repeated CV from outer-training y only."""
    train, external = data.folds[split]
    fit, check = screening_folds(
        train, seed=SEED + int(split[-2:]), count=count
    )
    if set(fit + check) != set(train) or set(check).intersection(external):
        raise ValueError("Inner screening or outer membership is inconsistent.")
    actual = data.y.loc[check].to_numpy(dtype=float)
    predicted = predict(data, fit, check, components=20, **FROZEN)
    screen = float(np.sqrt(np.mean(np.square(predicted - actual))))

    folds = repeated_cv_folds(
        train, seed=SEED + 1000 + int(split[-2:]) * 4,
        repeats=2, folds=4,
    )
    squared = 0.0
    samples = 0
    for fit, check in folds:
        observed = data.y.loc[check].to_numpy(dtype=float)
        predicted = predict(data, fit, check, components=20, **FROZEN)
        squared += float(np.square(predicted - observed).sum())
        samples += len(check)
    if samples != 2 * len(train):
        raise ValueError("Unexpected number of repeated inner CV observations.")
    return screen, float(np.sqrt(squared / samples)), len(external)


def virtual_ranks(
    screened: pd.DataFrame,
    finalists: pd.DataFrame,
    baseline_screen: float,
    baseline_cv: float,
    *,
    finalists_per_family: int,
) -> dict:
    """Answer whether the baseline would survive the first selection step."""
    needed = {"candidate_id", "family", "screen_rmse"}
    if not needed.issubset(screened):
        raise ValueError("Invalid saved screen columns.")
    if screened["candidate_id"].duplicated().any():
        raise ValueError("Duplicated candidate screen entry.")
    if not np.isfinite(screened["screen_rmse"]).all():
        raise ValueError("Missing or nonfinite screening errors.")
    if set(screened["family"]) != set(FAMILIES):
        raise ValueError("Screen family names do not match the frozen run.")
    pca = screened.loc[screened["family"].eq("PCA_Ridge"), "screen_rmse"]
    best_by_family = screened.groupby("family")["screen_rmse"].min()
    selection = finalists["inner_rmse"].astype(float)
    if not np.isfinite(selection).all():
        raise ValueError("Invalid saved final repeated-CV scores.")
    if len(finalists) < len(FAMILIES):
        raise ValueError("Too few final selection candidates.")

    # 1-based hypothetical rank if the incumbent were added as one candidate.
    # Lower rank is better; tie-breaking was not part of the actual experiment.
    rank_all = 1 + int(screened["screen_rmse"].lt(baseline_screen).sum())
    rank_pca = 1 + int(pca.lt(baseline_screen).sum())
    threshold = float(pca.nsmallest(finalists_per_family).iloc[-1])
    rank_cv = 1 + int(selection.lt(baseline_cv).sum())
    return {
        "incumbent_virtual_screen_rank_all": rank_all,
        "incumbent_virtual_screen_rank_pca_ridge": rank_pca,
        "pca_ridge_12th_screen_rmse": threshold,
        "would_pass_pca_ridge_top12_screen": rank_pca <= finalists_per_family,
        "best_saved_screen_rmse_overall": float(screened["screen_rmse"].min()),
        "best_saved_screen_rmse_pca_ridge": float(best_by_family["PCA_Ridge"]),
        "incumbent_virtual_inner_cv_rank_among_finalists": rank_cv,
        "best_saved_finalist_inner_cv_rmse": float(selection.min()),
        "saved_finalist_count": len(finalists),
    }


def _load_screen(folder: Path, split: str, candidates: pd.DataFrame) -> pd.DataFrame:
    frames = [
        pd.read_csv(p)
        for p in sorted(folder.glob(f"{split}__screen__*.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No saved screening files for {split}.")
    results = pd.concat(frames, ignore_index=True)
    if len(results) != len(candidates):
        raise ValueError(
            f"Saved {len(results)} screens in {split}, "
            f"but declared {len(candidates)} candidates."
        )
    if results["candidate_id"].tolist() != candidates["candidate_id"].tolist():
        raise ValueError("Cached trial IDs differ from the candidate sequence.")
    if results["family"].tolist() != candidates["family"].tolist():
        raise ValueError("Cached trial families do not match candidate families.")
    return results


def _verify_provenance(folder: Path) -> dict:
    path = folder / "run_spec.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment fingerprint: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if (spec.get("n_random_configs") != 100000
            or spec.get("smoke")
            or list(spec.get("families", [])) != list(FAMILIES)
            or spec.get("screen_targets") != SCREEN_TARGETS
            or spec.get("final_cv") != {"folds": 4, "repeats": 2}):
        raise ValueError("Not the expected original 100k Checkpoint13 protocol.")
    paths = inputs(ROOT, MEMBERSHIPS)
    for name, old_hash in spec["source_sha256"].items():
        if digest(paths[name]) != old_hash:
            raise ValueError(f"Input changed since the 100k run: {name}")
    for path, old_hash in spec["code_sha256"].items():
        if digest(ROOT / path) != old_hash:
            raise ValueError(
                f"Experiment code changed since the 100k run: {path}"
            )
    return spec


def audit(folder: Path) -> pd.DataFrame:
    """Reconstruct Local04D's original 12-target and 2x4-fold error per split."""
    spec = _verify_provenance(folder)
    candidates = pd.read_csv(folder / "candidates.csv")
    if (len(candidates) != spec["n_random_configs"]
            or not candidates["candidate_id"].is_unique):
        raise ValueError("Incomplete 100k candidate manifest.")
    data = load_pca_data(ROOT, MEMBERSHIPS, max_components=160)
    if data.parity_max_diff > 1e-6:
        raise RuntimeError("Historical original PCA20 coordinate mismatch.")
    rows = []
    for split in sorted(data.folds):
        screen, inner, n_external = baseline_inner_scores(data, split)
        saved = _load_screen(folder, split, candidates)
        reranked_path = folder / f"{split}__reranked.csv"
        if not reranked_path.exists():
            raise FileNotFoundError(f"Missing finalists for {split}.")
        ranking = pd.read_csv(reranked_path)
        if set(ranking["family"]) != set(FAMILIES):
            raise ValueError(f"Incomplete family reranking in {split}.")
        score = virtual_ranks(
            saved, ranking, screen, inner,
            finalists_per_family=spec["finalists_per_family"],
        )
        rows.append({
            "split_id": split,
            "incumbent": BASELINE,
            "n_screen_targets": spec["screen_targets"],
            "n_inner_cv_predictions": 194,
            "n_outer_held_out": n_external,
            "n_random_models_screened": len(saved),
            "local04d_screen_rmse": screen,
            "local04d_repeated_inner_cv_rmse": inner,
            **score,
        })
        print(
            f"[13 AUDIT] {split}: baseline 12-parcel RMSE={screen:.6f}, "
            f"screen rank={score['incumbent_virtual_screen_rank_all']}/100001, "
            f"within PCA+Ridge={score['incumbent_virtual_screen_rank_pca_ridge']}; "
            f"2x4-fold RMSE={inner:.6f}.",
            flush=True,
        )
    result = pd.DataFrame(rows)
    if len(result) != 16 or not result["n_random_models_screened"].eq(100000).all():
        raise ValueError("Not all 16 frozen outer splits were audited.")
    out = folder / "baseline_screening_audit.csv"
    tmp = out.with_suffix(".csv.tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(out)
    print(
        "[13 AUDIT] Average 12-parcel baseline RMSE over 16 splits: "
        f"{result['local04d_screen_rmse'].mean():.6f} t/ha.",
        flush=True,
    )
    print(
        "[13 AUDIT] Average 2x4-fold baseline inner RMSE: "
        f"{result['local04d_repeated_inner_cv_rmse'].mean():.6f} t/ha.",
        flush=True,
    )
    print(
        "[13 AUDIT] Baseline would survive the PCA+Ridge top-12 first screen in "
        f"{int(result['would_pass_pca_ridge_top12_screen'].sum())}/16 splits.",
        flush=True,
    )
    print(
        "[13 AUDIT] Comparing screening scores is diagnostic only. "
        "No model refit or change to 59 official forecasts.",
        flush=True,
    )
    print(f"[13 AUDIT] Saved local report: {out}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
