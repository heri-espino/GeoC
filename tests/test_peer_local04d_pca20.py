"""Regression: peer bridge loads the *saved* original C4 PCA20 representation."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def test_peer_bridge_loads_frozen_pca20_reference(tmp_path, monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_peer_champions_loo", tools / "run_peer_champions_loo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    peer = tmp_path / "todoseboso"
    bundle = tmp_path / "data_n/processed/current_champions/v1"
    (peer / "src").mkdir(parents=True)
    bundle.mkdir(parents=True)
    (peer / "src/current_champions.py").touch()
    (bundle / "manifest.json").write_text("{}")
    ids = [f"P{i:03d}" for i in range(138)]
    labels = np.linspace(2, 6, 138)
    frame = pd.DataFrame({
        "ID_POLIGONO": ids,
        "RENDIMIENTO_T_HA": labels,
    })
    groups = pd.DataFrame(index=ids)
    distances = np.zeros((138, 138), dtype=float)

    fake = types.ModuleType("src.current_champions")
    fake.load_bundle = lambda *a, **kw: (
        frame, groups, distances,
        {"models": ["CT3", "Reference26", "Compact14", "B1"]},
    )
    fake.refit_fold = lambda *a: None
    fake.predict_saved = lambda *a: None
    monkeypatch.setitem(sys.modules, "src.current_champions", fake)
    monkeypatch.setattr(module, "_git", lambda *a: "")

    called = []

    def frozen_original(_root, _memberships):
        called.append(True)
        return types.SimpleNamespace(y=pd.Series(labels, index=ids))

    monkeypatch.setattr(module, "load_frozen_local04d", frozen_original)
    module._load(peer, bundle)
    assert called == [True], (
        "Local04D must read the stored frozen C4 PCA20 and validate historical "
        "41-prediction parity, not fit PCA again."
    )
