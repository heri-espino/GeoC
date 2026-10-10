#!/usr/bin/env python3
"""Frozen Prithvi-EO-2.0 inference + inner-validated parcel Ridge heads.

Pretrained weights and chips must exist locally; no downloads or hidden y.
Use --preflight before --stage all. --smoke extracts one chip only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from geocebada.data.targets import load_yield_split
from geocebada.evaluation.checkpoint07_prithvi import extract_frozen_embeddings
from geocebada.evaluation.checkpoint07_prithvi_head import evaluate_prithvi
from geocebada.evaluation.checkpoint07c import load_development

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "extract", "heads"), default="all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=("300", "600"), default="300")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models/checkpoint_07")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/checkpoint_07")
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "reports/checkpoint_07/highdim/expanded_parcel_features.csv",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=ROOT / "data/source/tabular/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
    )
    parser.add_argument(
        "--membership", type=Path,
        default=ROOT / "reports/checkpoint_04b/pseudo_competition_splits.csv",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "reports/checkpoint_06/parcel_oof_residuals.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/checkpoint_07_three/prithvi")
    args = parser.parse_args()

    for name in ("torch", "terratorch"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(
                f"Missing {name}. Install the Prithvi extra: "
                'python -m pip install -e ".[prithvi]"'
            )
    import torch

    if not torch.cuda.is_available():
        build_cuda = torch.version.cuda
        # A CPU-only torch wheel has no CUDA runtime even if nvidia-smi
        # reports a functioning NVIDIA GPU. Avoid a vague message.
        if build_cuda is None:
            reason = (
                "Installed PyTorch is CPU-only (torch.version.cuda=None). "
                "Restore the CUDA-enabled PyTorch wheel matching your "
                "NVIDIA driver and Python environment."
            )
        else:
            reason = (
                f"PyTorch was built with CUDA {build_cuda} but cannot use the "
                "GPU in this Windows session. Check nvidia-smi, NVIDIA "
                "driver availability and virtual-machine GPU passthrough."
            )
        raise RuntimeError(
            "Prithvi requires a working CUDA GPU. "
            f"torch={torch.__version__}; compiled_cuda={build_cuda}; "
            f"python={Path(__import__('sys').executable)}. "
            + reason
            + " Run: nvidia-smi; python -c \"import torch; "
            "print(torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available())\". "
            "See https://pytorch.org/get-started/previous-versions/ "
            "for an official CUDA wheel. Do not run the full workflow yet."
        )
    root = args.models_root / f"Prithvi-EO-2.0-{args.model}M-TL"
    config = root / "config.json"
    pts = list(root.glob("*.pt")) if root.exists() else []
    if not config.is_file() or len(pts) != 1 or pts[0].stat().st_size < 100 * 1024**2:
        raise FileNotFoundError(
            f"Missing local pretrained Prithvi {args.model}M-TL config/.pt: {root}"
        )
    chip_manifest = args.processed / "prithvi_chip_manifest.json"
    if not chip_manifest.is_file():
        raise FileNotFoundError(chip_manifest)
    entries = json.loads(chip_manifest.read_text(encoding="utf-8"))
    if len(entries) != 197:
        raise ValueError(f"Expected 197 HLS parcel chips, found {len(entries)}.")
    chips = args.processed / "prithvi_chips"
    missing = [
        x["ID_POLIGONO"] for x in entries
        if not x.get("complete") or not (chips / x.get("chip_path", "missing")).is_file()
    ]
    if missing:
        raise ValueError(f"Missing/incomplete Prithvi chips: {missing[:5]}")
    for p in (args.targets, args.features, args.membership, args.baseline):
        if not p.is_file():
            raise FileNotFoundError(f"Missing Prithvi head input: {p}")
    if args.preflight:
        print(
            f"[PRITHVI PREFLIGHT] PASS. {len(entries)} chips, "
            f"model={args.model}M-TL, CUDA={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        return 0

    destination = args.output / f"{args.model}M"
    if args.smoke:
        if args.stage == "heads":
            raise ValueError("--smoke is an encoder smoke test, not head training.")
        destination = destination / "smoke"
        extract_frozen_embeddings(
            model_root=root, chips_root=chips,
            manifest_path=chip_manifest, output=destination, limit=1,
        )
        print("[PRITHVI] Real checkpoint inference smoke passed (one chip).")
        return 0
    if args.stage in ("all", "extract"):
        meta = extract_frozen_embeddings(
            model_root=root, chips_root=chips,
            manifest_path=chip_manifest, output=destination,
        )
        print(f"[PRITHVI] all 197 embeddings: {meta}", flush=True)
    if args.stage in ("all", "heads"):
        x = pd.read_csv(args.features, low_memory=False)
        y = load_yield_split(path=args.targets, validate=True)
        members = pd.read_csv(args.membership)
        baseline = pd.read_csv(args.baseline)
        data = load_development(x, y, members, baseline)
        result = evaluate_prithvi(
            data, embedding_path=destination / "embeddings.csv",
            output=destination / "heads",
        )
        (destination / "head_report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if result["n_completed_splits"] != 16:
            raise RuntimeError("Not all 16 Prithvi head splits were completed.")
        print(f"[PRITHVI] head evaluation PASS: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
