"""Checkpoint 08: symmetric nested evaluation of the two local Ridge families.

The outer confirmation memberships are generated from X-only geographic
metadata. Neither their construction nor any inner Optuna objective uses
outer held-out yield labels. The 59 missing target labels are never loaded
for supervision, scored, or imputed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from geocebada.evaluation.checkpoint04a import haversine_distance_matrix
from geocebada.evaluation.checkpoint04b import normalized_distance_matrix
from geocebada.evaluation.checkpoint04d1 import predict_weighted_local_ridge
from geocebada.evaluation.checkpoint07_local import _predict_local
from geocebada.evaluation.checkpoint07c import FoldRepresentations

ID = "ID_POLIGONO"
Y = "RENDIMIENTO_T_HA"
FAMILIES = ("Local04D_nested", "Local07_nested")
K_VALUES = (8, 12, 16, 20, 24, 30, 36, 48)
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0)
GEO_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPS = ("raw256", "raw512", "pca16", "pca32", "pca64")
FROZEN = {"k": 24, "alpha": 30.0, "power": 1.0}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def input_paths(root: Path) -> dict[str, Path]:
    return {
        "base": root / "data/processed/features_v1/parcel_features_all.csv",
        "targets": root / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        "embeddings": root / "reports/checkpoint_04a/transductive_embeddings.csv",
        "highdim": root / "reports/checkpoint_07/highdim/expanded_parcel_features.csv",
        "original_splits": root / "reports/checkpoint_04b/pseudo_competition_splits.csv",
    }


def label_blind_memberships(
    metadata: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    seed: int = 20261012,
    repeats: int = 16,
    n_heldout: int = 41,
) -> pd.DataFrame:
    """New fixed outer bank. Stratification uses municipality/state X, never y.

    Target municipality shares set the desired pseudo-target proportions.
    Within each stratum selection is random, subject to feasibility. The
    closest available integer allocation is solved by greedy marginal error.
    """
    need = {ID, "CONJUNTO", "meta_estado", "meta_municipio"}
    if not need.issubset(metadata) or not metadata[ID].is_unique:
        raise ValueError("Need unique parcel IDs and X-only location metadata.")
    train = metadata.loc[metadata["CONJUNTO"].eq("ENTRENAMIENTO")].copy()
    target = metadata.loc[metadata["CONJUNTO"].eq("PREDICCION")].copy()
    if len(metadata) != 197 or len(train) != 138 or len(target) != 59:
        raise ValueError("Expected 197 parcels: 138 labeled and 59 missing-y.")
    if not {ID, "split_id", "role"}.issubset(historical):
        raise ValueError("Historical 04B split membership missing.")
    def key(frame: pd.DataFrame) -> pd.Series:
        return (frame["meta_estado"].fillna("?").astype(str) + "|"
                + frame["meta_municipio"].fillna("?").astype(str))
    train["_stratum"] = key(train)
    target["_stratum"] = key(target)
    groups = {name: sorted(block[ID].astype(str)) for name, block in train.groupby("_stratum")}
    share = target["_stratum"].value_counts(normalize=True)
    quotas = {g: n_heldout * float(share.get(g, 0.0)) for g in groups}
    counts = {g: 0 for g in groups}
    # Allocate each place where it minimizes the increase in squared distance
    # from the target-profile quota, respecting label availability.
    for _ in range(n_heldout):
        viable = [g for g in groups if counts[g] < len(groups[g])]
        if not viable:
            raise ValueError("No capacity for pseudo-target sampling.")
        g = min(viable, key=lambda s: ((counts[s] + 1 - quotas[s]) ** 2 -
                                      (counts[s] - quotas[s]) ** 2, s))
        counts[g] += 1

    known = set(train[ID].astype(str))
    prior_rows = historical.loc[
        historical["split_id"].astype(str).str.startswith("target_matched_")
    ]
    prior_sets = [
        set(block.loc[block["role"].eq("pseudo_target"), ID].astype(str))
        for _, block in prior_rows.groupby("split_id")
    ]
    if len(prior_sets) != 16:
        raise ValueError("Expected 16 historical target-matched splits.")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    selected: list[set[str]] = []
    for j in range(repeats):
        # Candidate choice is entirely X-only. Reduce overlap with existing
        # development splits and already selected confirmation memberships.
        candidates = []
        for _ in range(160):
            chosen = set()
            for group, n in counts.items():
                if n:
                    chosen.update(rng.choice(groups[group], size=n, replace=False).tolist())
            other = prior_sets + selected
            worst = max((len(chosen & prev) / len(chosen | prev)
                         for prev in other), default=0.0)
            candidates.append((worst, tuple(sorted(chosen))))
        overlap, ids = min(candidates)
        chosen = set(ids)
        if len(chosen) != n_heldout or chosen not in [set(v) for v in selected]:
            if len(chosen) != n_heldout:
                raise AssertionError("Incorrect pseudo-target count.")
        else:
            raise RuntimeError("Duplicate confirmation outer split.")
        selected.append(chosen)
        split_id = f"confirm_{j + 1:02d}"
        for pid in sorted(known):
            rows.append({
                "split_id": split_id, ID: pid,
                "role": "pseudo_target" if pid in chosen else "pseudo_train",
                "max_jaccard": overlap, "generator_seed": seed,
            })
    return pd.DataFrame(rows)


def validate_memberships(meta: pd.DataFrame, membership: pd.DataFrame,
                         *, expected: int = 16) -> dict[str, tuple[list[str], list[str]]]:
    known = set(meta.loc[meta["CONJUNTO"].eq("ENTRENAMIENTO"), ID].astype(str))
    hidden = set(meta.loc[meta["CONJUNTO"].eq("PREDICCION"), ID].astype(str))
    if len(known) != 138 or len(hidden) != 59:
        raise ValueError("Incorrect label/missing-target contract.")
    results = {}
    for split, group in membership.groupby("split_id"):
        if group[ID].duplicated().any() or set(group[ID]) != known:
            raise ValueError(f"Invalid parcel coverage in {split}")
        a = group.loc[group["role"].eq("pseudo_train"), ID].astype(str).tolist()
        b = group.loc[group["role"].eq("pseudo_target"), ID].astype(str).tolist()
        if len(a) != 97 or len(b) != 41 or set(a) & set(b):
            raise ValueError(f"Invalid 97/41 holdout {split}")
        results[str(split)] = (a, b)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} confirmation splits.")
    return results


@dataclass
class Inputs:
    x07: pd.DataFrame
    old: np.ndarray
    geographic: np.ndarray
    index: dict[str, int]
    y: pd.Series
    meta: pd.DataFrame
    folds: dict[str, tuple[list[str], list[str]]]


def load_inputs(root: Path, membership: pd.DataFrame) -> Inputs:
    paths = input_paths(root)
    meta = pd.read_csv(paths["base"], usecols=[ID, "CONJUNTO", "base_centroid_lat",
                                                "base_centroid_lon", "meta_estado",
                                                "meta_municipio"])
    meta[ID] = meta[ID].astype(str)
    folds = validate_memberships(meta, membership)
    targets = pd.read_csv(paths["targets"], usecols=[ID, "CONJUNTO", Y])
    targets[ID] = targets[ID].astype(str)
    if len(targets) != 197 or set(targets[ID]) != set(meta[ID]) or not targets[ID].is_unique:
        raise ValueError("Target identifiers or count do not match.")
    known = targets.loc[targets["CONJUNTO"].eq("ENTRENAMIENTO")].set_index(ID)[Y]
    hidden = targets.loc[targets["CONJUNTO"].eq("PREDICCION"), Y]
    if len(known) != 138 or known.isna().any() or len(hidden) != 59 or hidden.notna().any():
        raise ValueError("Hidden target leakage or missing training label.")
    y = pd.to_numeric(known, errors="raise").astype(float)
    embedded = pd.read_csv(paths["embeddings"])
    pc = [c for c in embedded.columns if c.startswith("pc")]
    pc.sort(key=lambda c: int(c.removeprefix("pc")))
    if not pc:
        raise ValueError("C4 original PCA embedding missing.")
    c4 = embedded.loc[embedded["representation"].eq("C4_all_deterministic")].copy()
    c4[ID] = c4[ID].astype(str)
    c4 = c4.set_index(ID).reindex(meta[ID])
    if c4[pc].isna().any().any():
        raise ValueError("C4 embedding not aligned with all 197 parcels.")
    old = StandardScaler().fit_transform(c4[pc].to_numpy(dtype=float))
    dist = haversine_distance_matrix(meta["base_centroid_lat"], meta["base_centroid_lon"])
    normal, _ = normalized_distance_matrix(
        dist, reference_positions=np.flatnonzero(meta["CONJUNTO"].eq("ENTRENAMIENTO").to_numpy()))
    raw = pd.read_csv(paths["highdim"], low_memory=False)
    raw[ID] = raw[ID].astype(str)
    if len(raw) != 197 or not raw[ID].is_unique or set(raw[ID]) != set(meta[ID]):
        raise ValueError("07B high-dimensional X alignment failure.")
    forbidden = [
        c for c in raw.columns
        if c.upper() in {Y, "CONJUNTO", "PRODUCCION_T", "RENDIMIENTO"}
        or "siap_2025" in c.lower()
    ]
    if forbidden:
        raise ValueError(f"Outcome leakage in 07B predictors: {forbidden[:5]}")
    x07 = (raw.set_index(ID).select_dtypes(include=[np.number])
           .replace([np.inf, -np.inf], np.nan).astype(np.float32))
    x07 = x07.reindex(meta[ID])
    return Inputs(x07=x07, old=old, geographic=normal,
                  index={pid: i for i, pid in enumerate(meta[ID])},
                  y=y, meta=meta, folds=folds)


def predict_old(data: Inputs, fit_ids: list[str], query_ids: list[str],
                pars: dict) -> np.ndarray:
    indices = [data.index[x] for x in fit_ids]
    queries = [data.index[x] for x in query_ids]
    all_y = np.full(len(data.meta), np.nan)
    all_y[indices] = data.y.loc[fit_ids].to_numpy(dtype=float)
    return predict_weighted_local_ridge(
        data.old, data.geographic, all_y, indices, queries,
        k=int(pars["k"]), alpha=float(pars["alpha"]), distance_power=float(pars["power"]),
    )


def geographic_pair(
    data: Inputs, train: list[str], test: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Single consistent coordinate projection anchored at training latitudes."""
    view = data.meta.set_index(ID)
    mid = float(np.deg2rad(view.loc[train, "base_centroid_lat"].mean()))

    def project(ids: list[str]) -> np.ndarray:
        lat = view.loc[ids, "base_centroid_lat"].to_numpy(float)
        lon = view.loc[ids, "base_centroid_lon"].to_numpy(float)
        return np.column_stack([111.195 * lat, 111.195 * np.cos(mid) * lon])

    return project(train), project(test)


def local07_folds(data: Inputs, training_ids: list[str], n_folds: int, seed: int):
    inner = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    output = []
    for ia, ib in inner.split(training_ids):
        a = [training_ids[i] for i in ia]
        b = [training_ids[i] for i in ib]
        transforms = FoldRepresentations(data.x07, a, b)
        train_coords, test_coords = geographic_pair(data, a, b)
        output.append((a, b, transforms, train_coords, test_coords))
    return output


def predict_new(data: Inputs, train: list[str], test: list[str],
                pars: dict, *, transform: FoldRepresentations | None = None,
                coordinates: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    if transform is None:
        transform = FoldRepresentations(data.x07, train, test)
    a, b = transform.get(str(pars["representation"]))
    if coordinates is None:
        coords_a, coords_b = geographic_pair(data, train, test)
    else:
        coords_a, coords_b = coordinates
    return _predict_local(a, b, coords_a, coords_b, data.y.loc[train].to_numpy(float),
                          k=int(pars["k"]), alpha=float(pars["alpha"]),
                          geo_share=float(pars["geo_share"]))


def tune_and_predict(data: Inputs, split_id: str, family: str, *,
                     trials: int, seed: int, n_folds: int,
                     folder: Path) -> pd.DataFrame:
    """Resume per-split Optuna trials; only inner training y enters objective."""
    import optuna

    train, test = data.folds[split_id]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{split_id}__{family}"
    csv_path, info_path = folder / f"{stem}.csv", folder / f"{stem}.json"
    if csv_path.exists() and info_path.exists():
        meta = json.loads(info_path.read_text(encoding="utf-8"))
        result = pd.read_csv(csv_path)
        if len(result) != 41 or meta["n_trials"] != trials or meta["split_id"] != split_id:
            raise ValueError(f"Incomplete or incompatible completed result {stem}")
        return result
    if csv_path.exists() or info_path.exists():
        raise RuntimeError(f"Partial paired result for {stem}; investigate before resuming.")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    if family == "Local04D_nested":
        folds = [([train[i] for i in ia], [train[i] for i in ib])
                 for ia, ib in kf.split(train)]
    elif family == "Local07_nested":
        folds = local07_folds(data, train, n_folds, seed)
    else:
        raise ValueError(f"Unknown family {family}")

    study = optuna.create_study(
        study_name=stem, direction="minimize",
        storage=f"sqlite:///{(folder / (stem + '.sqlite')).resolve().as_posix()}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if not study.trials and family == "Local04D_nested":
        study.enqueue_trial(FROZEN.copy())

    def objective(t):
        if family == "Local04D_nested":
            pars = {"k": t.suggest_categorical("k", K_VALUES),
                    "alpha": t.suggest_categorical("alpha", ALPHAS),
                    "power": t.suggest_categorical("power", POWERS)}
        else:
            pars = {"representation": t.suggest_categorical("representation", REPS),
                    "k": t.suggest_categorical("k", K_VALUES),
                    "alpha": t.suggest_categorical("alpha", ALPHAS),
                    "geo_share": t.suggest_categorical("geo_share", GEO_SHARES)}
        error = []
        for fold in folds:
            a, b = fold[:2]
            if family == "Local04D_nested":
                pred = predict_old(data, a, b, pars)
            else:
                pred = predict_new(data, a, b, pars, transform=fold[2],
                                   coordinates=(fold[3], fold[4]))
            error.extend((pred - data.y.loc[b].to_numpy(float)) ** 2)
        return float(np.sqrt(np.mean(error)))

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < trials:
        print(f"[08] {split_id} {family}: {done}/{trials} completed trials", flush=True)
        study.optimize(objective, n_trials=trials - done, gc_after_trial=True)
    if not study.best_trials:
        raise RuntimeError("No successful tuning trials.")
    pars = study.best_params
    yhat = (predict_old(data, train, test, pars) if family == "Local04D_nested"
            else predict_new(data, train, test, pars))
    result = pd.DataFrame({
        "split_id": split_id, ID: test, "model": family,
        "observed": data.y.loc[test].to_numpy(float),
        "predicted": yhat,
    })
    report = {
        "split_id": split_id, "family": family, "best_params": pars,
        "inner_rmse": float(study.best_value), "n_trials": trials,
        "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "outer_rmse": float(np.sqrt(np.mean((result["observed"] - result["predicted"]) ** 2))),
        "outer_training": len(train), "outer_holdout": len(test),
        "hidden_y_accessed": False,
    }
    draft = csv_path.with_suffix(".csv.tmp")
    result.to_csv(draft, index=False)
    draft.replace(csv_path)
    atomic_json(info_path, report)
    return result


def fixed_baseline(data: Inputs, split_id: str) -> pd.DataFrame:
    train, test = data.folds[split_id]
    return pd.DataFrame({
        "split_id": split_id, ID: test, "model": "Local04D_frozen",
        "observed": data.y.loc[test].to_numpy(float),
        "predicted": predict_old(data, train, test, FROZEN),
    })


def summarize(predictions: pd.DataFrame, *, bootstraps: int = 1000,
              seed: int = 20261012) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"split_id", ID, "model", "observed", "predicted"}
    if not required.issubset(predictions):
        raise ValueError("Missing prediction columns.")
    if predictions.duplicated(["split_id", ID, "model"]).any():
        raise ValueError("Duplicated held-out predictions.")
    models = {"Local04D_frozen", *FAMILIES}
    if set(predictions["model"]) != models:
        raise ValueError("Missing paired model.")
    ids = predictions.groupby(["split_id", ID])["model"].nunique()
    if len(ids) != 16 * 41 or not ids.eq(3).all():
        raise ValueError("Full 16-split paired coverage required.")
    truth_check = predictions.groupby(["split_id", ID])["observed"].nunique()
    if not truth_check.eq(1).all():
        raise ValueError("Observed yields mismatch across paired models.")
    records = []
    for name, part in predictions.groupby("model"):
        rmse_split = part.groupby("split_id").apply(
            lambda df: float(np.sqrt(np.mean((df["predicted"] - df["observed"]) ** 2))),
            include_groups=False,
        )
        records.append({
            "model": name, "n_splits": part["split_id"].nunique(),
            "repeated_predictions": len(part), "unique_parcels": part[ID].nunique(),
            "rmse_pooled": float(np.sqrt(np.mean((part["predicted"] - part["observed"]) ** 2))),
            "rmse_mean_split": float(rmse_split.mean()),
            "rmse_worst_split": float(rmse_split.max()),
        })
    summary = pd.DataFrame(records).sort_values("rmse_pooled")
    pivot = predictions.pivot(index=["split_id", ID], columns="model", values="predicted")
    ground = predictions.drop_duplicates(["split_id", ID]).set_index(["split_id", ID])["observed"]
    se = (pivot.sub(ground, axis=0)) ** 2
    se = se.sort_index()
    groups = {key: block for key, block in se.groupby(level=ID, sort=True)}
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    delta = []
    for contender in FAMILIES:
        base = float((se[contender] - se["Local04D_frozen"]).mean())
        sims = np.empty(bootstraps)
        for z in range(bootstraps):
            sampled = rng.choice(names, len(names), replace=True)
            values = np.concatenate([
                (groups[k][contender] - groups[k]["Local04D_frozen"]).to_numpy()
                for k in sampled
            ])
            sims[z] = float(values.mean())
        low, high = np.quantile(sims, [0.025, 0.975])
        delta.append({"model": contender, "reference": "Local04D_frozen",
                      "delta_mse": base, "ci95_low": float(low), "ci95_high": float(high),
                      "bootstrap_unit": "unique_parcel"})
    return summary, pd.DataFrame(delta)
