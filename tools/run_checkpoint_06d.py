#!/usr/bin/env python
"""Run Checkpoint 06D X-only regime discovery."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from geocebada.evaluation.checkpoint06d import (
    characterize_selected_regimes,
    decide_regime_hypothesis,
    discover_x_regimes,
    select_x_only_regime,
    selected_regime_assignments,
)
from geocebada.paths import find_project_root

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


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


def _preflight(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: root / str(value) for key, value in config["inputs"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing 06D inputs: {missing}")

    decision = json.loads(paths["checkpoint06c_report"].read_text(encoding="utf-8"))
    if bool(decision["loso_gate_passed"]):
        raise RuntimeError(
            "06D should not run because 06C LOSO passed; freeze/confirm calibration instead."
        )

    base = pd.read_csv(paths["base_table"])
    agronomic = pd.read_csv(paths["agronomic_table"])
    residual = pd.read_csv(paths["parcel_residual_summary"])
    identity = config["identity"]
    if len(base) != 197 or len(agronomic) != 197:
        raise ValueError("06D expects exactly 197 parcel X rows.")
    if (
        base[identity["id_column"]].duplicated().any()
        or agronomic[identity["id_column"]].duplicated().any()
    ):
        raise ValueError("06D found duplicate parcel IDs.")
    if set(base[identity["id_column"]].astype(str)) != set(
        agronomic[identity["id_column"]].astype(str)
    ):
        raise ValueError("06D base/agronomic ID universes differ.")
    target = base[identity["split_column"]].eq(identity["prediction_value"])
    if pd.to_numeric(base.loc[target, identity["target_column"]], errors="coerce").notna().any():
        raise ValueError("Hidden FIRA y is present; 06D refuses to run.")
    if not {"Local04D", "CatBoost_C1"}.issubset(set(residual["method"])):
        raise ValueError("06D requires Local04D and CatBoost parcel residual summaries.")

    return {
        "rows": 197,
        "labeled_rows": int(base[identity["split_column"]].eq(identity["train_value"]).sum()),
        "target_rows": int(target.sum()),
        "cluster_selection_uses_y": False,
        "all_197_x_allowed_transductively": True,
        "06c_loso_gate_passed": False,
    }


def _figures(
    directory: Path,
    pca: pd.DataFrame,
    assignments: pd.DataFrame,
    residual: pd.DataFrame,
    composition: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plot = pca.merge(assignments, on="ID_POLIGONO", validate="one_to_one")
    if {"PC1", "PC2"}.issubset(plot.columns):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for cluster, group in plot.groupby("cluster", sort=True):
            ax.scatter(group["PC1"], group["PC2"], label=f"cluster {cluster}", alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Checkpoint 06D X-only agronomic regimes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "x_regime_pca.png", dpi=180)
        plt.close(fig)

    local = residual.loc[residual["method"].eq("Local04D")].sort_values("cluster")
    if not local.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.bar(local["cluster"].astype(str), local["mean_residual"])
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("X-only cluster")
        ax.set_ylabel("Mean honest Local04D residual")
        ax.set_title("Residual bias by frozen X-only regime")
        fig.tight_layout()
        fig.savefig(directory / "local04d_residual_by_regime.png", dpi=180)
        plt.close(fig)

    if not composition.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.bar(composition["cluster"].astype(str), composition["n_labeled"], label="labeled")
        ax.bar(
            composition["cluster"].astype(str),
            composition["n_target"],
            bottom=composition["n_labeled"],
            label="target",
        )
        ax.set_xlabel("X-only cluster")
        ax.set_ylabel("Parcels")
        ax.set_title("Labeled/target support by X-only regime")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "regime_support.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/checkpoint06d.yaml"))
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        root = find_project_root(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _load_yaml(config_path)

    preflight = _preflight(root, config)
    print("Checkpoint 06D preflight: PASS")
    for key, value in preflight.items():
        print(f"{key}: {value}")
    if args.preflight:
        return 0

    inputs = config["inputs"]
    identity = config["identity"]
    agronomic = pd.read_csv(root / str(inputs["agronomic_table"]))
    manifest = json.loads(
        (root / str(inputs["agronomic_manifest"])).read_text(encoding="utf-8")
    )
    base = pd.read_csv(root / str(inputs["base_table"]))
    residual = pd.read_csv(root / str(inputs["parcel_residual_summary"]))

    xcfg = config["x_regimes"]
    candidates, _, _ = discover_x_regimes(
        agronomic,
        manifest,
        pca_variances=xcfg["pca_variances"],
        algorithms=xcfg["algorithms"],
        k_values=xcfg["k_values"],
        seeds=xcfg["seeds"],
        subsample_repeats=int(xcfg["subsample_repeats"]),
        subsample_fraction=float(xcfg["subsample_fraction"]),
    )
    selection = select_x_only_regime(
        candidates,
        min_cluster_fraction=float(xcfg["min_cluster_fraction"]),
        min_seed_ari=float(xcfg["min_seed_ari"]),
        min_subsample_ari=float(xcfg["min_subsample_ari"]),
    )
    assignments, pca = selected_regime_assignments(
        agronomic,
        manifest,
        selection,
        random_state=int(config["runtime"]["random_state"]),
    )
    composition, residual_by_cluster, effects, expert_advantage, payload = (
        characterize_selected_regimes(
            assignments,
            base,
            residual,
            split_column=str(identity["split_column"]),
            train_value=str(identity["train_value"]),
            prediction_value=str(identity["prediction_value"]),
            target_column=str(identity["target_column"]),
            state_column=str(identity["state_column"]),
            municipality_column=str(identity["municipality_column"]),
            permutation_repeats=int(config["residual_gate"]["permutation_repeats"]),
            random_state=int(config["runtime"]["random_state"]),
        )
    )
    decision = decide_regime_hypothesis(
        selection,
        payload,
        min_residual_eta2=float(config["residual_gate"]["min_residual_eta2"]),
        max_residual_permutation_p=float(
            config["residual_gate"]["max_residual_permutation_p"]
        ),
        min_residual_mean_range=float(
            config["residual_gate"]["min_residual_mean_range"]
        ),
    )

    out = root / str(config["outputs"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / str(config["outputs"]["candidates"]), index=False)
    assignments.to_csv(out / str(config["outputs"]["assignments"]), index=False)
    pca.to_csv(out / str(config["outputs"]["pca_coordinates"]), index=False)
    composition.to_csv(out / str(config["outputs"]["composition"]), index=False)
    residual_by_cluster.to_csv(
        out / str(config["outputs"]["residual_by_cluster"]), index=False
    )
    effects.to_csv(out / str(config["outputs"]["residual_effects"]), index=False)
    expert_advantage.to_csv(
        out / str(config["outputs"]["expert_advantage"]), index=False
    )
    _figures(
        out / str(config["outputs"]["figures_directory"]),
        pca,
        assignments,
        residual_by_cluster,
        composition,
    )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root)
    report = {
        "schema_version": 1,
        "checkpoint": "06D",
        "generated_at": generated_at,
        "git_commit": git_commit,
        "hidden_fira_y_used_or_scored": False,
        "preflight": preflight,
        "x_selection": selection,
        "decision": decision,
    }
    (out / str(config["outputs"]["report_json"])).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Checkpoint 06D — X-only regime discovery",
        "",
        f"Generated: {generated_at}",
        f"Git commit: {git_commit}",
        "",
        "## X-only selection",
        "",
        json.dumps(selection, indent=2, sort_keys=True),
        "",
        "## Residual-regime decision",
        "",
        f"**REGIME_HYPOTHESIS = {decision['REGIME_HYPOTHESIS']}**",
        "",
        json.dumps(decision, indent=2, sort_keys=True),
        "",
        "## Cluster composition",
        "",
        composition.to_string(index=False),
        "",
        "## Honest residuals by cluster",
        "",
        residual_by_cluster.to_string(index=False),
        "",
        "## Expert advantage by cluster",
        "",
        expert_advantage.to_string(index=False),
        "",
    ]
    if decision["REGIME_HYPOTHESIS"] == "SUPPORTED":
        lines.extend(
            [
                "## Next step",
                "",
                "Proceed to 06E with a partially pooled cluster residual correction first.",
                "Do not train independent high-capacity models inside each cluster.",
            ]
        )
    elif decision["REGIME_HYPOTHESIS"] == "AMBIGUOUS":
        lines.extend(
            [
                "## Next step",
                "",
                "Inspect regime diagnostics before implementing 06E.",
            ]
        )
    else:
        lines.extend(
            [
                "## Next step",
                "",
                "Do not implement cluster specialists; retain Local04D unless "
                "another predeclared mechanism is tested.",
            ]
        )

    (out / str(config["outputs"]["report_markdown"])).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Checkpoint 06D regime discovery: PASS")
    print(
        f"Selected X regime: {selection['algorithm']} K={selection['k']} "
        f"PCA={selection['pca_variance']}"
    )
    print(f"REGIME_HYPOTHESIS = {decision['REGIME_HYPOTHESIS']}")
    print(f"Inspect: {out.relative_to(root) / str(config['outputs']['report_markdown'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
