#!/usr/bin/env python3
"""Quick exploratory ensemble screen using SAVED 138-fold LOOCV predictions.

No base model training. Strict ID/y alignment, frozen 137/1 contracts where
provided; refuse accidental combination with 16x41 pseudo-target results.

PowerShell, from GeoC repository root:
    python tools/run_loo_ensemble_screen.py --audit
    python tools/run_loo_ensemble_screen.py --run
    python tools/run_loo_ensemble_screen.py --summary

Add a new 138-fold model with audited 137/1 rows:
    python tools/run_loo_ensemble_screen.py --run --extra CatBoost=PATH.csv

Extra files require columns ID_POLIGONO,observed,outer_train,outer_heldout and
either a model-named prediction column or 'predicted' (optionally 'model').
They must contain EXACTLY 138 unique held-outs; no matching by row order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
ID = "ID_POLIGONO"
TARGET = "observed"
N = 138
ANCHOR = "Reference26_mean"
BASELINE_FILE = ROOT / "reports/selected_champions_loo/loo_predictions.csv"
MEAN_DIR = ROOT / "reports/reference26_ridge_budget_mean"
LOCAL_FILE = ROOT / "reports/area_weighted_climate_loo/loo_predictions.csv"
PEER_FILE = ROOT / "reports/peer_champions_loo/loo_predictions.csv"
DBSCAN_DIR = ROOT / "reports/peer_dbscan_loo138"
OUTPUT = ROOT / "reports/loo_ensemble_screen"
KNOWN = {
    "Reference26_original": BASELINE_FILE,
    "Compact14": BASELINE_FILE,
    "Local04D": LOCAL_FILE,
    "Reference26_DBSCAN": DBSCAN_DIR / "loo_predictions.csv",
}


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    if not columns.issubset(frame.columns):
        raise ValueError(
            f"{label}: missing columns {sorted(columns - set(frame.columns))}"
        )


def audited_model(
    frame: pd.DataFrame, label: str, prediction_col: str,
    *, require_fold_proof: bool = False,
) -> pd.DataFrame:
    """One row for each of the 138 excluded parcels; reject non-LOOCV rows."""
    cols = {ID, TARGET, prediction_col}
    require_columns(frame, cols, label)
    if require_fold_proof:
        require_columns(frame, {"outer_train", "outer_heldout"}, label)
    if {"outer_train", "outer_heldout"}.issubset(frame.columns):
        if (
            not frame["outer_train"].eq(137).all()
            or not frame["outer_heldout"].eq(1).all()
        ):
            raise ValueError(f"{label}: not 137-train/1-heldout LOOCV.")
    if len(frame) != N or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(
            f"{label}: expected 138 different held-out parcel IDs, "
            f"found {len(frame)} rows."
        )
    values = frame[[TARGET, prediction_col]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: invalid observed yields or predictions.")
    result = frame[[ID, TARGET, prediction_col]].copy()
    result[ID] = result[ID].astype(str)
    return result.rename(columns={prediction_col: label})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={ID: str})


def load_baselines(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Original 138-fold champions predictions not found: {path}\n"
            "Restore from completed run: "
            "python tools/run_selected_champions_loo.py --summary"
        )
    frame = read_csv(path)
    original = audited_model(frame, "Reference26_original", "Reference26")
    compact = audited_model(frame, "Compact14", "Compact14")
    return original, compact


def load_mean(directory: Path) -> pd.DataFrame:
    """Read the *mean*, k=24/alpha=30, from 138 cached outer folds."""
    provenance = directory / "run_spec.json"
    if not provenance.is_file():
        raise FileNotFoundError(
            f"Missing mean-imputation source provenance: {provenance}"
        )
    spec = json.loads(provenance.read_text(encoding="utf-8"))
    if "mean" not in str(spec.get("imputation", "")).lower():
        raise ValueError("Mean baseline source is not marked as mean-imputed.")
    files = [
        directory / f"fold_{i:03d}" / "k24_a30.csv"
        for i in range(1, N + 1)
    ]
    if any(not path.is_file() for path in files):
        raise ValueError(
            "Reference26 mean k24/a30 does not have all 138 saved folds. "
            "Finish --final (or keep the saved complete directory)."
        )
    combined = pd.concat([read_csv(path) for path in files], ignore_index=True)
    return audited_model(
        combined, ANCHOR, "Reference26_k24_alpha30"
    )


def read_local(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    if path.is_file():
        raw = read_csv(path)
        require_columns(raw, {"model", "predicted"}, "Local04D history")
        from geocebada.evaluation.area_weighted_climate_comparison import BASE

        selected = raw.loc[raw["model"].eq(BASE)]
        return audited_model(
            selected, "Local04D", "predicted", require_fold_proof=True
        )
    if PEER_FILE.is_file():
        peer = read_csv(PEER_FILE)
        if "Local04D" in peer.columns:
            return audited_model(peer, "Local04D", "Local04D")
    raise FileNotFoundError(
        "Missing 138-fold Local04D predictions. Expected "
        f"{path} (long-form GeoPCA20_GeographicWeightedRidge) "
        f"or {PEER_FILE}. "
        "The 16x41 checkpoint results are NOT interchangeable."
    )


def read_dbscan(path: Path) -> pd.DataFrame:
    if path.is_file():
        raw = read_csv(path)
    else:
        files = [
            DBSCAN_DIR / f"fold_{i:03d}.csv" for i in range(1, N + 1)
        ]
        if any(not path.is_file() for path in files):
            raise FileNotFoundError(
                "DBSCAN does not have all 138 saved fold predictions."
            )
        raw = pd.concat([read_csv(file) for file in files], ignore_index=True)
    return audited_model(raw, "Reference26_DBSCAN", "Reference26_DBSCAN")


def load_extra(item: str) -> tuple[str, pd.DataFrame]:
    if "=" not in item:
        raise ValueError("--extra must be ModelName=path/to/predictions.csv.")
    name, source = item.split("=", 1)
    name = name.strip()
    if (
        not name or name in {ANCHOR, *KNOWN}
        or not all(c.isalnum() or c in "_-" for c in name)
    ):
        raise ValueError(
            f"Invalid/conflicting extra model name: {name!r}"
        )
    file = Path(source).expanduser().resolve()
    frame = read_csv(file)
    if "model" in frame:
        frame = frame.loc[frame["model"].eq(name)]
    column = name if name in frame.columns else "predicted"
    result = audited_model(
        frame, name, column, require_fold_proof=True
    )
    return name, result


def join_models(anchor: pd.DataFrame, models: list[pd.DataFrame]) -> pd.DataFrame:
    """Strict complete ID/y match, no intersection-only scoring."""
    result = anchor.copy()
    ids = set(result[ID])
    if len(ids) != N:
        raise ValueError("Anchor must have 138 unique parcels.")
    for model in models:
        other_cols = set(model.columns) - {ID, TARGET}
        if len(other_cols) != 1:
            raise ValueError("Each added model must have exactly one forecast.")
        name = next(iter(other_cols))
        if name in result:
            raise ValueError(f"Duplicated model name {name}.")
        if set(model[ID]) != ids:
            raise ValueError(
                f"{name}: IDs do not match all 138 anchor parcels. "
                "Never intersect mismatched protocols."
            )
        ordered = model.set_index(ID).reindex(result[ID])
        if not np.allclose(
            result[TARGET].to_numpy(dtype=float),
            ordered[TARGET].to_numpy(dtype=float),
            rtol=0, atol=1e-9,
        ):
            raise ValueError(
                f"{name}: observed yield mismatch; refusing to combine."
            )
        result[name] = ordered[name].to_numpy(dtype=float)
    return result


def mse(true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction-true)**2))


def metrics(
    y: np.ndarray, prediction: np.ndarray, base_mse: float,
    method: str, weights: dict[str, float],
) -> dict:
    err = prediction - y
    error_mse = mse(y, prediction)
    return {
        "method": method,
        "weights": json.dumps(weights, sort_keys=True),
        "n_holdouts": N,
        "rmse": math.sqrt(error_mse),
        "mae": float(np.mean(abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(r2_score(y, prediction)),
        "delta_mse_vs_ref26mean": error_mse - base_mse,
    }


def screen(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tiny predeclared convex grid; weights learned from y only descriptively."""
    features = [name for name in table.columns if name not in {ID, TARGET}]
    if features[0] != ANCHOR:
        raise ValueError("Reference26 mean must be the anchor.")
    y = table[TARGET].to_numpy(float)
    base = table[ANCHOR].to_numpy(float)
    base_mse = mse(y, base)
    errors = table[features].to_numpy(float) - y[:, None]
    corr = pd.DataFrame(
        np.corrcoef(errors, rowvar=False), index=features, columns=features,
    )
    pairwise = []
    for a, b in itertools.combinations(range(len(features)), 2):
        pairwise.append({
            "model_a": features[a],
            "model_b": features[b],
            "residual_correlation": float(corr.iloc[a, b]),
            "fraction_b_better_absolute_error": float(
                np.mean(abs(errors[:, b]) < abs(errors[:, a]))
            ),
        })
    rows = [metrics(y, base, base_mse, "anchor", {ANCHOR: 1.0})]
    for model in features[1:]:
        p = table[model].to_numpy(float)
        rows.append(metrics(y, p, base_mse, "single", {model: 1.0}))
        for w in (0.05, 0.10, 0.20, 0.30, 0.50):
            pred = (1-w)*base+w*p
            rows.append(metrics(
                y, pred, base_mse, "anchor_plus_one",
                {ANCHOR: round(1-w, 3), model: w},
            ))
    for left, right in itertools.combinations(features[1:], 2):
        left_pred = table[left].to_numpy(float)
        right_pred = table[right].to_numpy(float)
        for w_l, w_r in itertools.product((0.10, 0.20, 0.30), repeat=2):
            if w_l+w_r > 0.50:
                continue
            w_anchor = round(1-w_l-w_r, 3)
            pred = w_anchor*base+w_l*left_pred+w_r*right_pred
            rows.append(metrics(y, pred, base_mse, "anchor_plus_two", {
                ANCHOR: w_anchor, left: w_l, right: w_r,
            }))
    ranking = pd.DataFrame(rows).sort_values(
        ["rmse", "method", "weights"], kind="stable"
    ).reset_index(drop=True)
    return ranking, corr, pd.DataFrame(pairwise)


def describe_files(mean: Path, local: Path, dbscan: Path) -> None:
    paths = {
        "Reference26 original + Compact14 (138-row wide)": BASELINE_FILE,
        "Reference26 mean k24/a30 (138 per-fold files)": mean,
        "Local04D (138-row long)": local,
        "Local04D alternative peer wide": PEER_FILE,
        "DBSCAN (138-row wide)": dbscan,
    }
    for title, path in paths.items():
        if title.startswith("Reference26 mean"):
            found = sum(
                (mean / f"fold_{i:03d}" / "k24_a30.csv").is_file()
                for i in range(1, N + 1)
            )
            print(f"[AUDIT] {title}: {found}/{N} cached parcels at {mean}")
        else:
            print(f"[AUDIT] {title}: "
                  f"{'FOUND' if path.is_file() else 'missing'} {path}")
    print(
        "[AUDIT] CatBoost, Prithvi and Gaussian projections need TRUE "
        "138-parcel 137/1 LOOCV predictions. Historical 16x41 or other "
        "split outcomes cannot be combined with these 138 forecasts."
    )
    possible = sorted(
        path for path in (ROOT / "reports").rglob("*.csv")
        if "predictions" in path.name.lower()
        and path not in {BASELINE_FILE, local, PEER_FILE, dbscan}
        and "fold_" not in str(path.relative_to(ROOT / "reports"))
    )
    print(f"[AUDIT] Other saved prediction tables: {len(possible)} "
          "(showing up to 35 paths; not yet certified as 138-fold LOOCV)")
    for path in possible[:35]:
        print("  " + str(path.relative_to(ROOT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("audit", "run", "summary"):
        mode.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mean-dir", type=Path, default=MEAN_DIR)
    parser.add_argument("--local-file", type=Path, default=LOCAL_FILE)
    parser.add_argument("--baseline-file", type=Path, default=BASELINE_FILE)
    parser.add_argument("--dbscan-file", type=Path,
                        default=DBSCAN_DIR / "loo_predictions.csv")
    parser.add_argument("--extra", action="append", default=[],
                        help="New 138-fold 137/1 predictions, NAME=FILE.csv")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.audit:
        describe_files(
            args.mean_dir.resolve(), args.local_file.resolve(),
            args.dbscan_file.resolve(),
        )
        return 0
    if args.summary:
        path = out / "ensemble_scores.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No ensemble screen yet: {path}")
        print("[ENSEMBLE] Saved top exploratory combinations:\n" +
              pd.read_csv(path).head(20).to_string(index=False), flush=True)
        return 0

    original, compact = load_baselines(args.baseline_file.resolve())
    anchor = load_mean(args.mean_dir.resolve())
    optional = [original, compact]
    for name, action in (
        ("Local04D", lambda: read_local(args.local_file.resolve(), anchor)),
        ("Reference26_DBSCAN", lambda: read_dbscan(
            args.dbscan_file.resolve()
        )),
    ):
        try:
            optional.append(action())
        except FileNotFoundError as exc:
            print(f"[SKIP] {name}: {exc}", flush=True)
    for value in args.extra:
        _, frame = load_extra(value)
        optional.append(frame)

    combined = join_models(anchor, optional)
    ranked, residual_corr, diversity = screen(combined)
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / "ensemble_scores.csv", index=False)
    residual_corr.to_csv(out / "residual_correlation.csv")
    diversity.to_csv(out / "pairwise_diversity.csv", index=False)
    (out / "run_spec.json").write_text(json.dumps({
        "status": "exploratory OOF reuse ONLY; no model training",
        "models": list(combined.columns[2:]),
        "n_heldout": N,
        "anchor": ANCHOR,
        "model_files": {
            "reference26_mean": str(args.mean_dir.resolve()),
            "originals": str(args.baseline_file.resolve()),
            "local04d": str(args.local_file.resolve()),
            "dbscan": str(args.dbscan_file.resolve()),
            "additional_files": args.extra,
        },
        "warning": (
            "Choosing blending weights on the same 138 OOF observed yields "
            "is optimistically biased. Global OOF predictions are NOT safe "
            "meta-training data for nested heldout tests because other base "
            "predictions may have trained using the outer heldout label."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[ENSEMBLE] Matched {N} identical heldout IDs/y for: "
          + ", ".join(combined.columns[2:]), flush=True)
    print("[ENSEMBLE] Lowest RMSE exploratory blends:\n"
          + ranked.head(20).to_string(index=False), flush=True)
    print("[ENSEMBLE] Signed residual correlations:\n"
          + residual_corr.round(3).to_string(), flush=True)
    print("[CAUTION] Descriptive, NOT nested validation. "
          "Do not promote a winner or infer a reliable improvement yet.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
