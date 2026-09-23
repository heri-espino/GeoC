#!/usr/bin/env python
"""Run Checkpoint 03D large-compute global modeling and ONNX export."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geocebada.evaluation.checkpoint03c2 import (
    build_competition_representation_specs,
    make_inner_cv_splits,
    summarize_nested_results,
)
from geocebada.evaluation.checkpoint03d import (
    _eligible_pairs,
    _fit_large_search,
    _gpu_smoke_test,
    _make_large_global_search_spec,
)
from geocebada.evaluation.parcel_modeling import validate_fixed_fold_assignments
from geocebada.features import join_agronomic_features, join_empirical_features
from geocebada.models.onnx_export import (
    export_regressor_to_onnx,
    verify_onnx_regressor,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _strip_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.split("__", 1)[-1]: value for key, value in params.items()}


def _validate_contract(joined: pd.DataFrame, config: dict[str, Any]) -> None:
    scope = config["scope"]
    inputs = config["inputs"]
    identity = config["identity"]
    validation = config["validation"]

    if str(scope["active_track"]) != "competition":
        raise ValueError("Checkpoint 03D must be competition-only.")
    if bool(scope.get("clean_track_enabled", False)):
        raise ValueError("Checkpoint 03D clean track must remain disabled.")
    if Path(str(inputs["base_table"])).name != "parcel_features_competition.csv":
        raise ValueError("Checkpoint 03D must use parcel_features_competition.csv.")

    split = joined[str(identity["split_column"])].astype(str)
    train_mask = split.eq(str(identity["train_value"]))
    target_mask = split.eq(str(identity["prediction_value"]))
    if len(joined) != int(validation["expected_total_rows"]):
        raise ValueError(f"Expected 197 rows, found {len(joined)}.")
    if int(train_mask.sum()) != int(validation["expected_training_rows"]):
        raise ValueError("Unexpected Checkpoint 03D training-row count.")
    if int(target_mask.sum()) != int(validation["expected_prediction_rows"]):
        raise ValueError("Unexpected Checkpoint 03D prediction-row count.")

    target = pd.to_numeric(joined[str(identity["target_column"])], errors="coerce")
    if target.loc[train_mask].isna().any():
        raise ValueError("Training target contains missing values.")
    if target.loc[target_mask].notna().any():
        raise ValueError("Hidden FIRA target values must remain missing.")


def _load_inputs(root: Path, config: dict[str, Any]) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    inputs = config["inputs"]
    paths = {
        key: root / str(inputs[key])
        for key in [
            "base_table",
            "base_manifest",
            "agronomic_table",
            "agronomic_manifest",
            "empirical_table",
            "empirical_manifest",
            "empirical_config",
            "frozen_folds",
        ]
    }
    base = pd.read_csv(paths["base_table"])
    agronomic = pd.read_csv(paths["agronomic_table"])
    empirical = pd.read_csv(paths["empirical_table"])
    joined = join_empirical_features(
        join_agronomic_features(base, agronomic),
        empirical,
    )
    folds = pd.read_csv(paths["frozen_folds"])
    manifests = {
        "base": _load_json(paths["base_manifest"]),
        "agronomic": _load_json(paths["agronomic_manifest"]),
        "empirical": _load_json(paths["empirical_manifest"]),
    }
    empirical_config = _load_yaml(paths["empirical_config"])
    _validate_contract(joined, config)
    return joined, folds, manifests, empirical_config, paths


def _check_deployment_stack() -> None:
    missing: list[str] = []
    for module in ["onnx", "onnxruntime", "onnxmltools", "skl2onnx"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Checkpoint 03D requires deployment extras before the long run. "
            f"Missing: {missing}. Install with python -m pip install -e "
            "'[dev,models,deployment]'."
        )


def _onnx_converter_smoke_test(config: dict[str, Any]) -> None:
    """Fail before the long run if any configured model family cannot round-trip to ONNX."""

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    rng = np.random.default_rng(20260923)
    x = rng.normal(size=(24, 5)).astype(np.float32)
    y = (
        2.5
        + 0.7 * x[:, 0]
        - 0.3 * x[:, 1]
        + 0.15 * x[:, 2] * x[:, 3]
    ).astype(np.float32)

    models: list[tuple[str, Any]] = [
        ("ridge", Ridge(alpha=1.0)),
        ("pls", PLSRegression(n_components=2, scale=True)),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=8,
                min_samples_leaf=2,
                random_state=20260923,
                n_jobs=1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=8,
                max_leaf_nodes=5,
                min_samples_leaf=4,
                random_state=20260923,
                early_stopping=False,
            ),
        ),
        (
            "catboost",
            CatBoostRegressor(
                iterations=8,
                depth=3,
                learning_rate=0.1,
                loss_function="RMSE",
                task_type="CPU",
                verbose=False,
                allow_writing_files=False,
                random_seed=20260923,
            ),
        ),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=8,
                max_depth=2,
                learning_rate=0.1,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                verbosity=0,
                random_state=20260923,
                n_jobs=1,
            ),
        ),
        (
            "lightgbm",
            LGBMRegressor(
                n_estimators=8,
                num_leaves=5,
                max_depth=3,
                learning_rate=0.1,
                min_child_samples=4,
                objective="regression",
                verbosity=-1,
                random_state=20260923,
                n_jobs=1,
            ),
        ),
    ]

    target_opset = int(config["deployment"]["target_opset"])
    atol = float(config["deployment"]["verification_atol"])
    rtol = float(config["deployment"]["verification_rtol"])

    with tempfile.TemporaryDirectory(prefix="geocebada_03d_onnx_") as tmp:
        directory = Path(tmp)
        for kind, model in models:
            model.fit(x, y)
            destination = directory / f"{kind}.onnx"
            try:
                export_regressor_to_onnx(
                    model,
                    kind=kind,
                    n_features=x.shape[1],
                    path=destination,
                    target_opset=target_opset,
                )
                verify_onnx_regressor(
                    model,
                    x,
                    path=destination,
                    atol=atol,
                    rtol=rtol,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ONNX converter preflight failed for {kind}: {exc}"
                ) from exc


def _partial_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "outer": output_dir / "_partial_outer_fold_metrics.csv",
        "oof": output_dir / "_partial_oof_predictions.csv",
        "inner": output_dir / "_partial_inner_search_results.csv",
        "state": output_dir / "_partial_state.json",
    }


def _load_partial(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _write_partial(
    *,
    paths: dict[str, Path],
    outer: pd.DataFrame,
    oof: pd.DataFrame,
    inner: pd.DataFrame,
    completed: set[str],
) -> None:
    outer.to_csv(paths["outer"], index=False)
    oof.to_csv(paths["oof"], index=False)
    inner.to_csv(paths["inner"], index=False)
    paths["state"].write_text(
        json.dumps(
            {"completed": sorted(completed)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _fit_key(protocol: str, representation: str, model: str, fold: int) -> str:
    return f"{protocol}|{representation}|{model}|{int(fold)}"


def _select_finalists(robustness: pd.DataFrame, count: int) -> pd.DataFrame:
    rows: list[pd.Series] = []
    used_models: set[str] = set()
    for _, row in robustness.iterrows():
        model = str(row["model"])
        if model in used_models:
            continue
        rows.append(row)
        used_models.add(model)
        if len(rows) >= int(count):
            break
    if not rows:
        raise RuntimeError("Checkpoint 03D produced no finalists.")
    result = pd.DataFrame(rows).reset_index(drop=True)
    result.insert(0, "finalist_rank", np.arange(1, len(result) + 1))
    return result


def _report_markdown(
    report: dict[str, Any],
    finalists: pd.DataFrame,
) -> str:
    lines = [
        "# Checkpoint 03D — Large-compute global benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Git commit: {report.get('git_commit') or 'unknown'}",
        "",
        "## Contract",
        "",
        "- active track: competition only",
        "- hidden FIRA y scored: no",
        f"- compute mode: {report['compute']}",
        f"- outer fits: {report['outer_fits']}",
        f"- ONNX verified: {report['onnx_verification']['verified']}",
        "",
        "## Global finalists",
        "",
        "| rank | representation | model | state RMSE | grouped RMSE | worst RMSE |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in finalists.itertuples(index=False):
        lines.append(
            f"| {row.finalist_rank} | {row.representation} | {row.model} | "
            f"{row.state_oof_rmse:.6f} | {row.grouped_oof_rmse:.6f} | "
            f"{row.worst_protocol_rmse:.6f} |"
        )
    lines.extend(
        [
            "",
            "The top finalist is refit on all 138 labeled parcels after hyperparameter",
            "selection across the union of state-stratified and municipality-grouped",
            "inner folds. Its model-only ONNX graph consumes the median-imputed",
            "float32 feature matrix described by the adjacent JSON manifest.",
            "",
            "03D does not replace Local04D. These finalists are inputs to the",
            "conditional Checkpoint 05B target-matched remix.",
            "",
        ]
    )
    return "\n".join(lines)


def run_checkpoint_03d(
    *,
    root: Path,
    config: dict[str, Any],
    selected_models: list[str] | None,
    selected_representations: list[str] | None,
    compute: str,
    resume: bool,
    preflight: bool,
) -> dict[str, Any]:
    joined, folds, manifests, empirical_config, source_paths = _load_inputs(
        root,
        config,
    )
    identity = config["identity"]
    validation = config["validation"]
    runtime = config["runtime"]
    outputs = config["outputs"]

    representations = build_competition_representation_specs(
        joined=joined,
        manifests=manifests,
        config=config,
        empirical_config=empirical_config,
    )
    representation_names = (
        selected_representations
        if selected_representations
        else list(representations)
    )
    model_names = (
        selected_models
        if selected_models
        else list(map(str, config["models"]))
    )
    unknown_reps = sorted(set(representation_names) - set(representations))
    unknown_models = sorted(set(model_names) - set(config["models"]))
    if unknown_reps:
        raise KeyError(f"Unknown representation(s): {unknown_reps}")
    if unknown_models:
        raise KeyError(f"Unknown model(s): {unknown_models}")

    pairs = _eligible_pairs(
        config=config,
        representation_names=representation_names,
        model_names=model_names,
    )
    if not pairs:
        raise RuntimeError("No eligible Checkpoint 03D model/representation pairs.")

    _check_deployment_stack()
    if compute == "GPU":
        _gpu_smoke_test(
            catboost_devices=str(config["compute"]["catboost_devices"]),
            xgboost_device=str(config["compute"]["xgboost_device"]),
        )
    _onnx_converter_smoke_test(config)

    if preflight:
        return {
            "status": "preflight_pass",
            "total_rows": len(joined),
            "pairs": len(pairs),
            "compute": compute,
            "onnx_stack": "available",
        }

    protocols = list(map(str, validation["outer_protocols"]))
    validate_fixed_fold_assignments(
        joined,
        folds,
        id_column=str(identity["id_column"]),
        split_column=str(identity["split_column"]),
        train_value=str(identity["train_value"]),
        fold_columns=protocols,
    )

    train_mask = joined[str(identity["split_column"])].eq(str(identity["train_value"]))
    target_mask = joined[str(identity["split_column"])].eq(
        str(identity["prediction_value"])
    )
    training = joined.loc[train_mask].copy()
    training[str(identity["id_column"])] = training[str(identity["id_column"])].astype(str)
    fold_frame = folds[
        [str(identity["id_column"]), *protocols]
    ].copy()
    fold_frame[str(identity["id_column"])] = fold_frame[
        str(identity["id_column"])
    ].astype(str)
    training = training.merge(
        fold_frame,
        on=str(identity["id_column"]),
        how="left",
        validate="one_to_one",
    )
    y = pd.to_numeric(training[str(identity["target_column"])], errors="raise").astype(float)

    output_dir = root / str(outputs["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = _partial_paths(output_dir)

    outer = _load_partial(partial["outer"]) if resume else pd.DataFrame()
    oof = _load_partial(partial["oof"]) if resume else pd.DataFrame()
    inner = _load_partial(partial["inner"]) if resume else pd.DataFrame()
    completed: set[str] = set()
    if resume and partial["state"].is_file():
        payload = _load_json(partial["state"])
        completed = set(map(str, payload.get("completed", [])))

    total_fits = sum(
        len(pd.to_numeric(training[protocol], errors="raise").astype(int).unique())
        for protocol in protocols
        for _ in pairs
    )
    fit_number = 0
    started_all = perf_counter()

    for representation_name, model_name in pairs:
        representation = representations[representation_name]
        x = training.loc[:, list(representation.features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        for protocol in protocols:
            fold_values = sorted(
                pd.to_numeric(training[protocol], errors="raise")
                .astype(int)
                .unique()
            )
            for fold in fold_values:
                fit_number += 1
                key = _fit_key(protocol, representation_name, model_name, fold)
                if key in completed:
                    if bool(runtime.get("progress", True)):
                        print(f"[{fit_number}/{total_fits}] resume skip {key}", flush=True)
                    continue

                if not outer.empty:
                    mask = (
                        outer["protocol"].astype(str).eq(protocol)
                        & outer["representation"].astype(str).eq(representation_name)
                        & outer["model"].astype(str).eq(model_name)
                        & pd.to_numeric(outer["fold"], errors="coerce").eq(int(fold))
                    )
                    outer = outer.loc[~mask].copy()
                if not oof.empty:
                    mask = (
                        oof["protocol"].astype(str).eq(protocol)
                        & oof["representation"].astype(str).eq(representation_name)
                        & oof["model"].astype(str).eq(model_name)
                        & pd.to_numeric(oof["fold"], errors="coerce").eq(int(fold))
                    )
                    oof = oof.loc[~mask].copy()
                if not inner.empty:
                    mask = (
                        inner["protocol"].astype(str).eq(protocol)
                        & inner["representation"].astype(str).eq(representation_name)
                        & inner["model"].astype(str).eq(model_name)
                        & pd.to_numeric(inner["fold"], errors="coerce").eq(int(fold))
                    )
                    inner = inner.loc[~mask].copy()

                valid_mask = training[protocol].eq(int(fold))
                current_train_mask = ~valid_mask
                outer_train = training.loc[current_train_mask].reset_index(drop=True)
                x_train = x.loc[current_train_mask].reset_index(drop=True)
                y_train = y.loc[current_train_mask].reset_index(drop=True)
                x_valid = x.loc[valid_mask]
                y_valid = y.loc[valid_mask].to_numpy(dtype=float)

                inner_splits = make_inner_cv_splits(
                    outer_train,
                    protocol=protocol,
                    n_splits=int(validation["inner_folds"]),
                    random_state=int(validation["random_state"]) + int(fold),
                    state_column=str(identity["state_column"]),
                    municipality_column=str(identity["municipality_column"]),
                )
                spec = _make_large_global_search_spec(
                    model_name=model_name,
                    model_config=config["models"][model_name],
                    representation=representation,
                    random_state=int(validation["random_state"]) + int(fold),
                    compute=compute,
                    catboost_devices=str(config["compute"]["catboost_devices"]),
                    xgboost_device=str(config["compute"]["xgboost_device"]),
                )

                if bool(runtime.get("progress", True)):
                    print(
                        f"[{fit_number}/{total_fits}] {protocol} / "
                        f"{representation_name} / {model_name} / fold {fold}",
                        flush=True,
                    )
                started = perf_counter()
                search = _fit_large_search(
                    spec=spec,
                    x_train=x_train,
                    y_train=y_train,
                    inner_splits=inner_splits,
                    search_n_jobs=int(runtime.get("search_n_jobs", 1)),
                )
                predicted = np.asarray(search.predict(x_valid), dtype=float).reshape(-1)
                elapsed = perf_counter() - started
                metrics = _metric_row(y_valid, predicted)

                outer_row = pd.DataFrame(
                    [
                        {
                            "protocol": protocol,
                            "representation": representation_name,
                            "model": model_name,
                            "kind": spec.kind,
                            "fold": int(fold),
                            "n_features": len(representation.features),
                            "n_primitives": 0,
                            "n_discovered": 0,
                            "n_train": int(current_train_mask.sum()),
                            "n_validation": int(valid_mask.sum()),
                            "inner_best_rmse": float(-search.best_score_),
                            "best_params_json": json.dumps(
                                _strip_params(search.best_params_),
                                sort_keys=True,
                            ),
                            "compute": compute if spec.kind in {"catboost", "xgboost"} else "CPU",
                            "elapsed_seconds": float(elapsed),
                            **metrics,
                        }
                    ]
                )
                ids = training.loc[valid_mask, str(identity["id_column"])].astype(str)
                oof_block = pd.DataFrame(
                    {
                        "protocol": protocol,
                        "representation": representation_name,
                        "model": model_name,
                        "fold": int(fold),
                        str(identity["id_column"]): ids.to_numpy(),
                        "observed": y_valid,
                        "predicted": predicted,
                        "residual": y_valid - predicted,
                    }
                )
                inner_rows: list[dict[str, Any]] = []
                for candidate_index, params in enumerate(search.cv_results_["params"]):
                    inner_rows.append(
                        {
                            "protocol": protocol,
                            "representation": representation_name,
                            "model": model_name,
                            "fold": int(fold),
                            "candidate": int(candidate_index),
                            "params_json": json.dumps(
                                _strip_params(dict(params)),
                                sort_keys=True,
                            ),
                            "inner_rmse_mean": float(
                                -search.cv_results_["mean_test_score"][candidate_index]
                            ),
                            "inner_rmse_std": float(
                                search.cv_results_["std_test_score"][candidate_index]
                            ),
                            "rank": int(
                                search.cv_results_["rank_test_score"][candidate_index]
                            ),
                        }
                    )
                inner_block = pd.DataFrame(inner_rows)

                outer = pd.concat([outer, outer_row], ignore_index=True)
                oof = pd.concat([oof, oof_block], ignore_index=True)
                inner = pd.concat([inner, inner_block], ignore_index=True)
                completed.add(key)
                _write_partial(
                    paths=partial,
                    outer=outer,
                    oof=oof,
                    inner=inner,
                    completed=completed,
                )

                if bool(runtime.get("progress", True)):
                    print(
                        f"    done rmse={metrics['rmse']:.5f}, "
                        f"{elapsed / 60.0:.1f} min",
                        flush=True,
                    )

    protocol_summary, robustness = summarize_nested_results(outer, oof)
    finalists = _select_finalists(
        robustness,
        int(validation["finalist_count"]),
    )

    outer.to_csv(output_dir / str(outputs["outer_fold_metrics"]), index=False)
    oof.to_csv(output_dir / str(outputs["oof_predictions"]), index=False)
    inner.to_csv(output_dir / str(outputs["inner_search_results"]), index=False)
    protocol_summary.to_csv(
        output_dir / str(outputs["protocol_summary"]),
        index=False,
    )
    robustness.to_csv(
        output_dir / str(outputs["robustness_summary"]),
        index=False,
    )
    finalists.to_csv(output_dir / str(outputs["finalists"]), index=False)

    representation_manifest = [
        representations[name].as_dict() for name in representation_names
    ]
    (output_dir / str(outputs["representation_manifest"])).write_text(
        json.dumps(
            {
                "schema_version": int(config["schema_version"]),
                "generated_at": datetime.now(UTC).isoformat(),
                "representations": representation_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    full_state = make_inner_cv_splits(
        training,
        protocol="fold_state_stratified",
        n_splits=int(validation["inner_folds"]),
        random_state=int(validation["random_state"]) + 7001,
        state_column=str(identity["state_column"]),
        municipality_column=str(identity["municipality_column"]),
    )
    full_grouped = make_inner_cv_splits(
        training,
        protocol="fold_municipality_grouped",
        n_splits=int(validation["inner_folds"]),
        random_state=int(validation["random_state"]) + 9001,
        state_column=str(identity["state_column"]),
        municipality_column=str(identity["municipality_column"]),
    )
    full_cv = [*full_state, *full_grouped]

    target_frame = joined.loc[target_mask].copy()
    target_ids = target_frame[str(identity["id_column"])].astype(str).to_numpy()
    actual_blocks: list[pd.DataFrame] = []
    fitted_finalists: dict[str, Any] = {}
    finalist_tuning: list[dict[str, Any]] = []
    onnx_verification: dict[str, Any] | None = None
    onnx_manifest_payload: dict[str, Any] | None = None

    for row in finalists.itertuples(index=False):
        representation_name = str(row.representation)
        model_name = str(row.model)
        representation = representations[representation_name]
        x_full = training.loc[:, list(representation.features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        x_target = target_frame.loc[:, list(representation.features)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        spec = _make_large_global_search_spec(
            model_name=model_name,
            model_config=config["models"][model_name],
            representation=representation,
            random_state=int(validation["random_state"]) + 11003,
            compute=compute,
            catboost_devices=str(config["compute"]["catboost_devices"]),
            xgboost_device=str(config["compute"]["xgboost_device"]),
        )
        search = _fit_large_search(
            spec=spec,
            x_train=x_full,
            y_train=y,
            inner_splits=full_cv,
            search_n_jobs=int(runtime.get("search_n_jobs", 1)),
        )
        method = f"03D_{model_name}_{representation_name}"
        predictions = np.asarray(search.predict(x_target), dtype=float).reshape(-1)
        actual_blocks.append(
            pd.DataFrame(
                {
                    str(identity["id_column"]): target_ids,
                    "method": method,
                    "predicted": predictions,
                }
            )
        )
        fitted_finalists[method] = search.best_estimator_
        finalist_tuning.append(
            {
                "finalist_rank": int(row.finalist_rank),
                "representation": representation_name,
                "model": model_name,
                "kind": spec.kind,
                "full_cv_rmse": float(-search.best_score_),
                "best_params": _strip_params(search.best_params_),
            }
        )

        if int(row.finalist_rank) == 1:
            best_pipeline = search.best_estimator_
            imputer = best_pipeline.named_steps["imputer"]
            fitted_model = best_pipeline.named_steps["model"]
            x_imputed = np.asarray(imputer.transform(x_full), dtype=np.float32)
            onnx_path = root / str(outputs["onnx_model"])
            export_regressor_to_onnx(
                fitted_model,
                kind=spec.kind,
                n_features=x_imputed.shape[1],
                path=onnx_path,
                target_opset=int(config["deployment"]["target_opset"]),
            )
            onnx_verification = verify_onnx_regressor(
                fitted_model,
                x_imputed,
                path=onnx_path,
                atol=float(config["deployment"]["verification_atol"]),
                rtol=float(config["deployment"]["verification_rtol"]),
            )
            statistics = np.asarray(imputer.statistics_, dtype=float)
            statistics = np.nan_to_num(
                statistics,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            onnx_manifest_payload = {
                "schema_version": 1,
                "checkpoint": "03D",
                "model": model_name,
                "kind": spec.kind,
                "representation": representation_name,
                "input_name": "features",
                "input_dtype": str(config["deployment"]["input_dtype"]),
                "raw_feature_order": list(representation.features),
                "preprocessing": {
                    "strategy": "median",
                    "statistics": statistics.tolist(),
                    "output_feature_count": int(x_imputed.shape[1]),
                },
                "onnx_path": str(outputs["onnx_model"]),
                "verification": onnx_verification,
            }

    actual = pd.concat(actual_blocks, ignore_index=True)
    actual.to_csv(
        output_dir / str(outputs["actual_global_predictions"]),
        index=False,
    )
    if onnx_verification is None or onnx_manifest_payload is None:
        raise RuntimeError("Checkpoint 03D did not produce a verified ONNX finalist.")

    onnx_manifest_path = root / str(outputs["onnx_manifest"])
    onnx_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_manifest_path.write_text(
        json.dumps(onnx_manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle_path = root / str(outputs["model_bundle"])
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "checkpoint": "03D",
            "generated_at": datetime.now(UTC).isoformat(),
            "finalists": finalist_tuning,
            "estimators": fitted_finalists,
            "actual_predictions": actual,
            "onnx_manifest": onnx_manifest_payload,
        },
        bundle_path,
    )

    report = {
        "schema_version": int(config["schema_version"]),
        "checkpoint": "03D",
        "stage": str(config["stage"]["name"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root),
        "compute": compute,
        "training_rows": int(train_mask.sum()),
        "prediction_rows": int(target_mask.sum()),
        "hidden_prediction_targets_used": False,
        "pairs": len(pairs),
        "outer_fits": int(len(outer)),
        "oof_prediction_rows": int(len(oof)),
        "finalists": finalist_tuning,
        "onnx_verification": onnx_verification,
        "elapsed_minutes": float((perf_counter() - started_all) / 60.0),
        "source_sha256": {
            key: _sha256(path) for key, path in source_paths.items()
        },
    }
    (output_dir / str(outputs["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / str(outputs["report_markdown"])).write_text(
        _report_markdown(report, finalists),
        encoding="utf-8",
    )

    for path in partial.values():
        if path.exists():
            path.unlink()

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/checkpoint03d.yaml"),
    )
    parser.add_argument("--models", help="Comma-separated model subset.")
    parser.add_argument("--representations", help="Comma-separated representation subset.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--compute",
        choices=("GPU", "CPU"),
        default=None,
        help="GPU is the configured default. CPU requires --confirm-cpu y.",
    )
    parser.add_argument("--confirm-cpu", choices=("y", "n"), default="n")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_yaml(config_path)

    compute = str(args.compute or config["compute"]["default"]).upper()
    if compute == "CPU" and args.confirm_cpu != "y":
        raise SystemExit(
            "CPU execution is intentional-only. Re-run with --compute CPU --confirm-cpu y."
        )

    report = run_checkpoint_03d(
        root=root,
        config=config,
        selected_models=_csv_list(args.models),
        selected_representations=_csv_list(args.representations),
        compute=compute,
        resume=bool(args.resume),
        preflight=bool(args.preflight),
    )
    if args.preflight:
        print("Checkpoint 03D preflight: PASS")
        print(f"Rows: {report['total_rows']}")
        print(f"Eligible representation/model pairs: {report['pairs']}")
        print(f"Compute: {report['compute']}")
        print("ONNX stack: available")
        return 0

    print("Checkpoint 03D large-compute global benchmark: PASS")
    print(f"Training rows: {report['training_rows']}")
    print(f"Outer fits: {report['outer_fits']}")
    print(f"OOF rows: {report['oof_prediction_rows']}")
    print(f"ONNX verified: {report['onnx_verification']['verified']}")
    print("Inspect: reports/checkpoint_03d/checkpoint_03d_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
