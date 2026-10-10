"""Prithvi frozen-embedding Ridge heads under the same 16 target-matched splits.

All head selection happens via training-only inner folds. The pretrained HLS
encoder has never observed parcel y. Baseline Local04D is copied, not refit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint07_pca import dual_pca_fold
from geocebada.evaluation.checkpoint07c import DevelopmentData

RANKS = (8, 16, 32, 64)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _ridge_head(
    a: np.ndarray, b: np.ndarray, y: np.ndarray, *,
    rank: int, alpha: float,
) -> np.ndarray:
    """Use fold-training-only PCA, scale and Ridge on frozen Prithvi vectors."""
    pca = dual_pca_fold(a, b)
    count = min(rank, pca.train.shape[1])
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(pca.train[:, :count])
    xtest = scaler.transform(pca.test[:, :count])
    return np.asarray(
        Ridge(alpha=float(alpha)).fit(xtrain, y).predict(xtest),
        dtype=float,
    )


def _embedding_table(path: Path, expected_ids: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prithvi embedding extraction missing: {path}")
    table = pd.read_csv(path, low_memory=False)
    if (
        "ID_POLIGONO" not in table
        or table["ID_POLIGONO"].duplicated().any()
        or set(table["ID_POLIGONO"].astype(str)) != set(expected_ids)
        or len(table) != 197
    ):
        raise ValueError("Prithvi embedding table does not match the 197 parcels.")
    frame = table.set_index("ID_POLIGONO")
    if "RENDIMIENTO_T_HA" in frame.columns:
        raise ValueError("Pretrained embedding matrix may not include y.")
    matrix = frame.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Pretrained embeddings contain nonfinite values.")
    return frame


def evaluate_prithvi(
    data: DevelopmentData,
    *,
    embedding_path: Path,
    output: Path,
    inner_folds: int = 3,
    seed: int = 20261010,
    max_splits: int = 0,
) -> dict[str, Any]:
    """Evaluate honest frozen Prithvi head, checkpointing each outer split."""
    output.mkdir(parents=True, exist_ok=True)
    embedded = _embedding_table(embedding_path, list(data.x.index))
    ids = sorted(data.splits)
    if max_splits:
        ids = ids[:max_splits]
    diagnostics = []
    for ix, split_id in enumerate(ids, 1):
        csv_path = output / f"{split_id}__prithvi.csv"
        json_path = output / f"{split_id}__prithvi.json"
        if csv_path.exists() != json_path.exists():
            orphan = csv_path if csv_path.exists() else json_path
            backup = orphan.with_suffix(orphan.suffix + ".incomplete")
            if backup.exists():
                raise RuntimeError(f"Repeated partial Prithvi result: {split_id}")
            orphan.replace(backup)
            print(f"[PRITHVI] Preserved partial {orphan.name}; recovering.", flush=True)
        if csv_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if len(pd.read_csv(csv_path)) != 205 or meta.get("split_id") != split_id:
                raise RuntimeError(f"Corrupt Prithvi head result: {split_id}")
            diagnostics.append(meta)
            continue
        train, test = data.splits[split_id]
        y = data.known_y.loc[train].to_numpy(dtype=float)
        x = embedded.loc[train].to_numpy(dtype=float)
        test_x = embedded.loc[test].to_numpy(dtype=float)
        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        fold_arrays = [
            (x[a], x[b], y[a], y[b]) for a, b in inner.split(x)
        ]
        evaluated: list[tuple[float, int, float]] = []
        for rank in RANKS:
            for alpha in ALPHAS:
                sq = []
                for xa, xb, ya, yb in fold_arrays:
                    prediction = _ridge_head(
                        xa, xb, ya, rank=rank, alpha=alpha
                    )
                    sq.extend((prediction-yb)**2)
                evaluated.append((float(np.sqrt(np.mean(sq))), rank, alpha))
        score, best_rank, best_alpha = min(evaluated)
        forecast = _ridge_head(
            x, test_x, y, rank=best_rank, alpha=best_alpha
        )
        observed = data.known_y.loc[test].to_numpy(dtype=float)
        baseline = np.asarray([
            data.incumbent.loc[(split_id, pid), "predicted"] for pid in test
        ], dtype=float)
        variants = {"Local04D": baseline, "Prithvi07": forecast}
        for weight in (0.10, 0.25, 0.50):
            variants[f"Local04D_plus_Prithvi07_w{weight:.2f}"] = (
                (1-weight)*baseline + weight*forecast
            )
        rows = [
            {
                "split_id": split_id, "family": "prithvi",
                "ID_POLIGONO": pid, "model": name,
                "observed": float(obs), "predicted": float(pred),
            }
            for name, guesses in variants.items()
            for pid, obs, pred in zip(test, observed, guesses, strict=True)
        ]
        report = {
            "split_id": split_id,
            "family": "prithvi",
            "inner_cv_rmse": score,
            "rank": best_rank, "alpha": best_alpha,
            "outer_train_count": len(train),
            "outer_test_count": len(test),
            "outer_rmse_prithvi": float(np.sqrt(np.mean((observed-forecast)**2))),
            "outer_rmse_local04d": float(np.sqrt(np.mean((observed-baseline)**2))),
            "promoted": False,
        }
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        tmp_json = json_path.with_suffix(".json.tmp")
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)
        tmp_json.write_text(
            json.dumps(report, indent=2)+"\n", encoding="utf-8"
        )
        tmp_csv.replace(csv_path)
        tmp_json.replace(json_path)
        diagnostics.append(report)
        print(f"[PRITHVI HEAD {ix}/{len(ids)}] {split_id}", flush=True)
    # No sorting of incomplete family results into a headline RMSE:
    # summary is expected to be read with full 16 splits.
    return {
        "n_completed_splits": len(diagnostics),
        "n_expected_splits": len(ids),
        "embedding_dim": embedded.shape[1],
        "n_inner_head_candidates": len(RANKS)*len(ALPHAS),
    }
