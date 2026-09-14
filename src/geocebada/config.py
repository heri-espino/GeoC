"""Configuration loading for GeoCebada."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from geocebada.paths import project_path


def load_config(
    path: str | Path = "configs/base.yaml",
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML configuration file.

    Relative paths are resolved from the GeoCebada repository root.

    Parameters
    ----------
    path:
        YAML file path, absolute or relative to the repository root.
    root:
        Optional explicit repository root.

    Returns
    -------
    dict
        Parsed YAML mapping. Empty YAML files produce an empty dictionary.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = project_path(config_path, root=root)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}, got {type(config).__name__}.")
    return config
