#!/usr/bin/env python3
"""Analyze existing 138-fold LOOCV errors; no model training or promotion.

python tools/analyze_loocv_hard_parcels.py --audit
python tools/analyze_loocv_hard_parcels.py --run
python tools/analyze_loocv_hard_parcels.py --summary

All parcel-level data remain in Git-ignored reports/ on the local machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import report_loocv_metrics as source
import run_loo_ensemble_screen as loader

ROOT = source.ROOT
N = 138
ID = loader.ID
TARGET = loader.TARGET
OUTPUT = ROOT / "reports/loocv_hard_parcels"


def family(name: str) -> str:
    if name.startswith("Reference26"):
        return "Reference26"
    if name.startswith("Compact14"):
        return "Compact14"
    if name.startswith("Local04D"):
        return "GeoPCA20"
    if name.startswith("Ensemble"):
        return "Ensemble"
    return name


def append_fresh(table: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Read all 138 genuine final-experiment folds, or skip if absent."""
    files = sorted(folder.glob("fold_???.csv"))
    if not files:
        print("[SKIP] No fresh final-experiment folds available.")
        return table
    if len(files) != N or [p.name for p in files] != [
        f"fold_{i:03d}.csv" for i in range(1, N + 1)
    ]:
        raise ValueError("Incomplete final experiment: refusing partial folds.")
    raw = pd.concat([loader.read_csv(p) for p in files], ignore_index=True)
    added = [
        loader.audited_model(raw, "Reference26_mean_fresh",
                             "Reference26_mean", require_fold_proof=True),
        loader.audited_model(raw, "Compact14_fresh",
                             "Compact14", require_fold_proof=True),
        loader.audited_model(raw, "Ensemble_fixed_weight",
                             "Ensemble_final", require_fold_proof=True),
    ]
    joined = loader.join_models(table, added)
    weights = ROOT / "configs/reference26_compact14_ensemble.json"
    if weights.is_file():
        w = json.loads(weights.read_text(encoding="utf-8"))["weights"]
        expected = (w["Reference26_mean"] * joined["Reference26_mean_fresh"]
                    + w["Compact14"] * joined["Compact14_fresh"])
        np.testing.assert_allclose(
            joined["Ensemble_fixed_weight"], expected, rtol=0, atol=1e-10
        )
    return joined


def analyze(table: pd.DataFrame):
    """Compute curves, model-tail shares, common hard parcels and raw-rank votes."""
    names = [c for c in table if c not in (ID, TARGET)]
    if len(table) != N or not names or table[ID].isna().any() or table[ID].duplicated().any():
        raise ValueError("Expected exactly 138 distinct aligned held-out parcels.")
    ids = table[ID].astype(str).to_numpy()
    y = table[TARGET].to_numpy(float)
    estimates = table[names].to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(estimates).all():
        raise ValueError("Nonfinite held-out values.")
    err = estimates - y[:, None]
    sq = err ** 2
    absolute = np.abs(err)
    rank = pd.DataFrame(absolute, columns=names).rank(
        ascending=False, method="min"
    )  # rank 1 = hardest; tied residuals receive tied ranks
    representatives = {}
    preference = [
        "Reference26_mean_fresh", "Reference26_mean", "Reference26_original",
        "Compact14", "Local04D",
    ]
    for name in preference + names:
        if name in names and family(name) != "Ensemble":
            representatives.setdefault(family(name), name)
    reps = list(representatives.values())
    if not reps:
        raise ValueError("No family representatives.")

    private = pd.DataFrame({ID: ids, "observed_private": y})
    for j, name in enumerate(names):
        private[f"abs_error__{name}"] = absolute[:, j]
        private[f"rank__{name}"] = rank[name].astype(int)
    private["family_top2_votes"] = (rank[reps] <= 2).sum(axis=1).astype(int)
    private["family_top5_votes"] = (rank[reps] <= 5).sum(axis=1).astype(int)
    private["family_top10_votes"] = (rank[reps] <= 10).sum(axis=1).astype(int)
    private["average_family_difficulty"] = ((N - rank[reps]) / (N - 1)).mean(axis=1)
    private["all_model_top2_votes"] = (rank <= 2).sum(axis=1).astype(int)
    private.sort_values(
        ["average_family_difficulty", "family_top10_votes", ID],
        ascending=[False, False, True], kind="stable", inplace=True
    )
    selected = private[ID].head(2).tolist()
    kept = ~np.isin(ids, selected)
    curve_rows, model_rows, top_rows = [], [], []
    for j, name in enumerate(names):
        order = np.argsort(absolute[:, j], kind="stable")  # easy --> hard
        accumulated = np.sqrt(np.cumsum(sq[order, j]) / np.arange(1, N + 1))
        diffs = np.diff(np.r_[np.nan, accumulated])
        curve_rows.extend({
            "model": name, "n_easiest_parcels_included": k + 1,
            "n_hardest_excluded": N - k - 1,
            "cumulative_rmse_t_ha": float(accumulated[k]),
            "increment_rmse_t_ha": float(diffs[k]),
        } for k in range(N))
        worst = np.argsort(-sq[:, j], kind="stable")
        total = float(sq[:, j].sum())
        model_rows.append({
            "model": name, "family": family(name), "n_holdouts": N,
            "rmse_full_t_ha": float(np.sqrt(sq[:, j].mean())),
            "rmse_136_same_two_removed_NO_REFIT_t_ha": float(
                np.sqrt(sq[kept, j].mean())
            ),
            "sse_worst1_pct": float(100 * sq[worst[:1], j].sum() / total),
            "sse_worst2_pct": float(100 * sq[worst[:2], j].sum() / total),
            "sse_worst10_pct": float(100 * sq[worst[:10], j].sum() / total),
            "same_as_consensus_top2": len(set(ids[worst[:2]]) & set(selected)),
            "max_absolute_error_t_ha": float(absolute[worst[0], j]),
        })
        top_rows.extend({
            ID: ids[idx], "model": name, "rank_worst_first": pos,
            "signed_error_t_ha": float(err[idx, j]),
            "absolute_error_t_ha": float(absolute[idx, j]),
        } for pos, idx in enumerate(worst[:10], 1))
    common2, common10 = set(ids), set(ids)
    for name in reps:
        worst = np.argsort(-absolute[:, names.index(name)], kind="stable")
        common2 &= set(ids[worst[:2]])
        common10 &= set(ids[worst[:10]])
    manifest = {
        "representatives_by_family": representatives,
        "two_highest_consensus_private": selected,
        "intersection_top2_across_families_private": sorted(common2),
        "intersection_top10_across_families_private": sorted(common10),
        "limitation": "Existing held-out predictions trimmed post-hoc; NO models refitted. "
                      "Correlated families do not independently validate mislabeling.",
    }
    return (
        pd.DataFrame(model_rows).sort_values("rmse_full_t_ha"),
        pd.DataFrame(curve_rows), private, pd.DataFrame(top_rows), manifest
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ("audit", "run", "summary"):
        group.add_argument("--" + action, action="store_true")
    parser.add_argument("--baseline-file", type=Path, default=loader.BASELINE_FILE)
    parser.add_argument("--mean-dir", type=Path, default=loader.MEAN_DIR)
    parser.add_argument("--local-file", type=Path, default=loader.LOCAL_FILE)
    parser.add_argument("--dbscan-file", type=Path,
                        default=loader.DBSCAN_DIR / "loo_predictions.csv")
    parser.add_argument("--fresh-folds", type=Path,
                        default=ROOT / "reports/reference26_compact14_fresh138")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.audit:
        source.audit(args)
        print("[AUDIT] Fresh final LOOCV folds:", args.fresh_folds)
        return 0
    out = args.output.resolve()
    if args.summary:
        for filename in ("model_error_concentration.csv", "candidate_parcels_private.csv"):
            file = out / filename
            if not file.is_file():
                raise FileNotFoundError(f"Run --run first: {file}")
            print(f"[SUMMARY] {file}")
            print(pd.read_csv(file).head(12).to_string(index=False))
        return 0
    table, _ = source.recover(args)  # strict 138 held-out IDs / yields
    table = append_fresh(table, args.fresh_folds.resolve())
    models, curves, parcels, top10, manifest = analyze(table)
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "model_error_concentration.csv", index=False)
    curves.to_csv(out / "sorted_cumulative_rmse_curves.csv", index=False)
    parcels.to_csv(out / "candidate_parcels_private.csv", index=False)
    top10.to_csv(out / "model_top10_errors_private.csv", index=False)
    (out / "diagnostic_manifest_private.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ANALYSIS] {len(models)} models; 138 heldouts; "
          f"{len(manifest['representatives_by_family'])} model families.")
    print(models.to_string(index=False))
    print("[PRIVATE] Consensus candidate IDs:", manifest["two_highest_consensus_private"])
    print("[PRIVATE] Top-2 intersection:", manifest["intersection_top2_across_families_private"])
    print("[PRIVATE] Top-10 intersection:", manifest["intersection_top10_across_families_private"])
    print("[CAUTION] The 136-parcel values above remove existing errors; "
          "they are NOT a retrained 136-label LOOCV comparison.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, part in curves.groupby("model", sort=False):
            ax.plot(part["n_easiest_parcels_included"], part["cumulative_rmse_t_ha"],
                    label=name)
        ax.set_xlim(110, 138)
        ax.set_xlabel("Number of easiest parcels included (out of 138)")
        ax.set_ylabel("Cumulative RMSE (t/ha)")
        ax.set_title("LOOCV cumulative error: hardest parcels enter last")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "cumulative_rmse_last28.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("[SKIP] Matplotlib unavailable; curve CSV still saved.")
    print(f"[PRIVATE] Results: {out}. Do NOT git-add per-parcel diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
