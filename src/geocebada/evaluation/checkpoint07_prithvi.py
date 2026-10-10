"""Prithvi-EO-2.0 frozen embeddings from full local HLS 4-frame NPZ chips.

Requires PyTorch and TerraTorch, and previously downloaded licensed Prithvi
model weights. Does NOT download weights, train 300M parameters, use y, or
interpret raw satellite pixels as independent labeled examples.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def load_chip(path: Path, mean: np.ndarray, std: np.ndarray) -> tuple:
    """Convert HLS digital numbers to normalized B,C,T,H,W for frozen encoder."""
    with np.load(path, allow_pickle=False) as archive:
        raw = np.array(archive["hls_dn"], dtype=np.float32)
        valid = np.array(archive["valid_mask"], dtype=bool)
        parcel = np.array(archive["parcel_mask"], dtype=bool)
        dates = [str(x) for x in archive["acquisition_dates"].tolist()]
        latlon = np.asarray(archive["centroid_lat_lon"], dtype=np.float32)
    if raw.shape != (4, 6, 224, 224) or valid.shape != (4, 224, 224):
        raise ValueError(f"Invalid four-frame, six-band HLS chip: {path}")
    if parcel.shape != (224, 224) or len(dates) != 4 or latlon.shape != (2,):
        raise ValueError(f"Invalid Prithvi chip metadata: {path}")
    if not valid.any(axis=(1, 2)).all():
        raise ValueError(f"Prithvi chip contains an all-invalid time frame: {path}")
    if not parcel.any():
        raise ValueError(f"Empty parcel mask: {path}")
    pixels = np.transpose(raw, (1, 0, 2, 3))
    pixels = (pixels - mean[:, None, None, None]) / std[:, None, None, None]
    # Do not treat missing pixels as valid reflectances. Use neutral
    # normalized zero input and supply masks for aggregation later.
    pixels = np.where(valid[None], pixels, 0.0).astype(np.float32)
    coords = np.asarray([
        [date.fromisoformat(d).year, date.fromisoformat(d).timetuple().tm_yday]
        for d in dates
    ], dtype=np.float32)
    return pixels[None], coords[None], latlon[None], parcel, valid


def _last_tensor(output: object) -> object:
    """Extract final frozen features, refusing unrecognized backbone output."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        for element in reversed(output):
            try:
                return _last_tensor(element)
            except TypeError:
                continue
    if isinstance(output, dict):
        for key in ("features", "x", "last_hidden_state", "out"):
            if key in output:
                return _last_tensor(output[key])
    raise TypeError(
        "Unsupported TerraTorch Prithvi backbone output. "
        "Inspect exact model/version; do not generate surrogate embeddings."
    )


def pool_features(output: object, parcel_mask: np.ndarray) -> np.ndarray:
    """Return global+parcel pooled features if patch-grid geometry is available."""
    import torch
    import torch.nn.functional as F

    tensor = _last_tensor(output).detach().float()
    if tensor.ndim == 3:
        # [batch, sequence tokens, channels]; 224px / 16px = 14x14,
        # 4 frames with 1-frame temporal patch: 784 patch tokens.
        n = int(tensor.shape[1])
        if n in (785, 197):
            tensor = tensor[:, 1:]  # discard class token
            n -= 1
        if n not in (784, 196):
            raise ValueError(
                f"Unexpected Prithvi patch-token sequence length {n}; "
                "check encoder version and spatial/temporal patching."
            )
        frames = n // 196
        token_grid = tensor.reshape(1, frames, 14, 14, -1).permute(0, 4, 1, 2, 3)
    elif tensor.ndim == 5:
        token_grid = tensor  # [B, channels, time, h, w]
    elif tensor.ndim == 4:
        token_grid = tensor[:, :, None]  # [B, channels, h, w]
    else:
        raise ValueError(f"Unexpected Prithvi feature tensor shape: {tensor.shape}")
    if token_grid.shape[0] != 1:
        raise ValueError("Expected one parcel per Prithvi inference batch.")
    spatial = tuple(int(v) for v in token_grid.shape[-2:])
    mask = torch.as_tensor(parcel_mask, dtype=torch.float32, device=token_grid.device)
    mask = F.adaptive_avg_pool2d(mask[None, None], spatial)[0, 0]
    masked = (mask >= 0.25).to(token_grid.dtype)
    if masked.sum() < 1:
        # Extremely small parcels cannot map onto a coarse 16px token grid.
        masked = (mask == mask.max()).to(token_grid.dtype)
    overall = token_grid.mean(dim=(2, 3, 4))[0]
    weighted = (
        (token_grid * masked[None, None, None]).sum(dim=(2, 3, 4))[0]
        / (masked.sum() * token_grid.shape[2])
    )
    result = torch.cat([overall, weighted]).cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Prithvi embedding is not finite.")
    return result


def load_local_backbone(model_root: Path, *, device: str):
    """Build actual pretrained TerraTorch TL encoder using local model weights."""
    import torch
    from terratorch.registry import BACKBONE_REGISTRY

    cfg_path = model_root / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing predownloaded Prithvi config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    pretrained = cfg.get("pretrained_cfg", {})
    if (
        pretrained.get("num_frames") != 4
        or pretrained.get("in_chans") != 6
        or pretrained.get("img_size") != 224
        or "time" not in pretrained.get("coords_encoding", [])
        or "location" not in pretrained.get("coords_encoding", [])
    ):
        raise ValueError("Local Prithvi TL config does not match 4x6x224 chips.")
    weight_files = sorted(model_root.glob("*.pt"))
    if len(weight_files) != 1 or weight_files[0].stat().st_size < 100 * 1024**2:
        raise ValueError(
            f"Expected one nontrivial pretrained .pt checkpoint in {model_root}."
        )
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Prithvi frozen inference requires CUDA in this workflow.")
    architecture = str(cfg.get("architecture", "")).lower()
    if architecture not in {"prithvi_eo_v2_300_tl", "prithvi_eo_v2_600_tl"}:
        raise ValueError(f"Unsupported pretrained Prithvi architecture: {architecture}")
    # Use the documented TerraTorch local ckpt_path. Never silently fall back
    # to uninitialized weights or a random neural model.
    model = BACKBONE_REGISTRY.build(
        architecture,
        pretrained=True,
        ckpt_path=str(weight_files[0].resolve()),
        num_frames=4,
        coords_encoding=["time", "location"],
    )
    model.to(torch.device("cuda"))
    model.eval()
    mean = np.asarray(pretrained["mean"], dtype=np.float32)
    std = np.asarray(pretrained["std"], dtype=np.float32)
    if len(mean) != 6 or len(std) != 6 or not np.all(std > 0):
        raise ValueError("Invalid Prithvi six-band normalization stats.")
    return model, mean, std, weight_files[0]


def extract_frozen_embeddings(
    *,
    model_root: Path, chips_root: Path, manifest_path: Path,
    output: Path, limit: int = 0,
) -> dict[str, object]:
    """Resume per-parcel frozen embeddings, then atomically merge all 197."""
    import torch

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(entries) != 197:
        raise ValueError("Expected exactly 197 parcel chips.")
    if any(not item.get("complete") for item in entries):
        raise ValueError("Prithvi chip manifest contains incomplete parcels.")
    chosen = entries if not limit else entries[:limit]
    output.mkdir(parents=True, exist_ok=True)
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    model, mean, std, weights = load_local_backbone(model_root, device="cuda")
    for i, item in enumerate(chosen, 1):
        pid = str(item["ID_POLIGONO"])
        part = shards / f"{pid}.npz"
        if part.exists():
            with np.load(part, allow_pickle=False) as saved:
                if str(saved["parcel_id"]) == pid and np.isfinite(saved["embedding"]).all():
                    continue
            raise ValueError(f"Existing invalid Prithvi shard: {part}")
        chip = chips_root / item["chip_path"]
        if not chip.is_file():
            raise FileNotFoundError(chip)
        pixels, tc, lc, parcel, valid = load_chip(chip, mean, std)
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels).cuda()
            temporal = torch.from_numpy(tc).cuda()
            location = torch.from_numpy(lc).cuda()
            output_features = model(
                tensor, temporal_coords=temporal, location_coords=location
            )
            embedding = pool_features(output_features, parcel)
        tmp = part.with_suffix(".npz.tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream, parcel_id=np.asarray(pid), embedding=embedding,
                date_count=np.asarray(len(tc[0])),
                pixel_valid_fraction=np.asarray(float(valid.mean())),
            )
        tmp.replace(part)
        if i % 10 == 0 or i == len(chosen):
            print(f"[PRITHVI] embedding {i}/{len(chosen)} {pid}", flush=True)
    all_records: list[np.ndarray] = []
    shape = None
    for item in chosen:
        with np.load(shards / f"{item['ID_POLIGONO']}.npz", allow_pickle=False) as saved:
            vector = np.asarray(saved["embedding"], dtype=np.float32)
        if shape is None:
            shape = vector.shape
        if vector.shape != shape:
            raise ValueError("Inconsistent Prithvi feature dimensions.")
        all_records.append(vector)
    matrix = np.vstack(all_records)
    final = pd.DataFrame(matrix, columns=[f"prithvi__f{j:04d}" for j in range(matrix.shape[1])])
    final.insert(0, "ID_POLIGONO", [str(x["ID_POLIGONO"]) for x in chosen])
    path = output / ("embeddings_smoke.csv" if limit else "embeddings.csv")
    tmp = path.with_suffix(".csv.tmp")
    final.to_csv(tmp, index=False)
    tmp.replace(path)
    report = {
        "model_root": str(model_root), "weights": weights.name,
        "n_parcels": len(chosen), "n_dimensions": matrix.shape[1],
        "mask_pooling": "global mean + parcel-token weighted mean",
        "trained_y": False, "pretrained_weights_required": True,
    }
    (output / ("smoke_report.json" if limit else "embedding_report.json")).write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8"
    )
    return report
