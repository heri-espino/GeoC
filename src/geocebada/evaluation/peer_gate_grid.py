"""Train-only geological gate searches for the teammate's Reference26 model.

Only SimplifiedBases.shares is replaced. Her ColdExperts, CT3, seven-feature
local Ridge, Compact14, R3 correction and source prediction code are reused.

Each gate's hyperparameters are selected on internal cross-fitted estimates
from the OUTER TRAINING parcels; held-out yields are never consulted.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from sklearn.cluster import KMeans

from geocebada.evaluation.peer_geology_dbscan import GeologicalDBSCANGate

# Original is not overridden and keeps the original native [4,8] gate.
GATE_GRIDS = {
    "kmeans_expanded": (2, 4, 6, 8, 12),
    "dbscan": tuple(
        (minimum, factor)
        for minimum in (3, 5)
        for factor in (0.75, 1.25, 2.0, 3.0)
    ),
    "support": (0.25, 0.5, 1.0, 2.0, 4.0),
    "global": (0.0,),
}
VARIANTS = ("original", *GATE_GRIDS)


def gate_for(kind, parameter):
    from src.conditional_local_phases import ExpertGate

    if kind == "kmeans_expanded":
        return ExpertGate("clusters", int(parameter))
    if kind == "dbscan":
        minimum, factor = parameter
        return GeologicalDBSCANGate(int(minimum), float(factor))
    if kind == "support":
        return ExpertGate("support", float(parameter))
    if kind == "global":
        return ExpertGate("global", 0.0)
    raise ValueError(f"Unknown gate family: {kind}")


def _parameter_key(param):
    if isinstance(param, tuple):
        return tuple(param)
    return (float(param),)


class GridShareMixin:
    """Fit her complete geology gate, replacing only its candidate family."""

    gate_family = None

    def shares(self, tr):
        tr = np.asarray(tr, dtype=int)
        key = tuple(tr)
        if key in self.share_models:
            return self.share_models[key]
        try:
            cv = self.cv(tr)
        except ValueError:
            self.logs.append({
                "kind": "geological_gate_cv_unavailable",
                "positions": tr.tolist(),
            })
            return None

        from src.signal_role_allocation import RoleSpace

        if self.gate_family not in GATE_GRIDS:
            raise ValueError(f"Unsupported geology gate: {self.gate_family}")
        climate, expert = self.route_oof(tr, cv)
        physical_frame = self.a["role_frame"]
        y = self.a["y"]
        definitions = self.a["physical_definitions"]
        scores = []
        for parameter in GATE_GRIDS[self.gate_family]:
            internal = np.full(len(tr), np.nan)
            for a, b in cv:
                aa, bb = tr[a], tr[b]
                try:
                    ac, ae = self.route_oof(aa)
                except ValueError:
                    # Exact native fallback when deeply nested CV is unavailable.
                    internal[b] = climate[b]
                    continue
                space = RoleSpace(["geologic_composition"], [], definitions)
                space.fit(physical_frame.iloc[aa])
                z = space.transform(physical_frame.iloc[aa])
                q = space.transform(physical_frame.iloc[bb])
                if self.gate_family == "kmeans_expanded":
                    distinct = len(np.unique(np.round(z, 10), axis=0))
                    if distinct < 1:
                        raise ValueError("No distinct geological training rows.")
                gate = gate_for(self.gate_family, parameter)
                gate.fit(z, y[aa], ac, ae)
                internal[b] = gate.predict(q, climate[b], expert[b])
            if not np.isfinite(internal).all():
                raise RuntimeError("Nonfinite inner predictions for geology gate.")
            scores.append({
                "param": parameter,
                "inner_mse": float(np.mean((internal - y[tr]) ** 2)),
            })

        selected = min(
            scores, key=lambda s: (
                round(s["inner_mse"], 12), _parameter_key(s["param"])
            ),
        )
        space = RoleSpace(["geologic_composition"], [], definitions)
        space.fit(physical_frame.iloc[tr])
        z = space.transform(physical_frame.iloc[tr])
        gate = gate_for(self.gate_family, selected["param"])
        gate.fit(z, y[tr], climate, expert)
        pack = {
            "space": space, "gate": gate,
            "gate_family": self.gate_family,
            "selected_param": selected["param"],
            "inner_mse": selected["inner_mse"],
            "scores": scores,
            "n_clusters": getattr(gate, "n_clusters_", None),
            "noise_fraction": getattr(gate, "noise_fraction_", None),
        }
        self.share_models[key] = pack
        return pack


@contextmanager
def configured_gate(kind: str):
    """Temporarily override ColdBases in this single isolated worker process."""
    if kind not in VARIANTS:
        raise ValueError(f"Unknown gate variant: {kind}")
    if kind == "original":
        yield
        return

    import src.current_champions as peer

    source = peer.ColdBases
    if issubclass(source, GridShareMixin):
        raise RuntimeError("Another tuned geological gate is already active.")
    modified = type(
        "GridColdBases", (GridShareMixin, source),
        {"gate_family": kind},
    )
    peer.ColdBases = modified
    try:
        yield
    finally:
        peer.ColdBases = source


def native_reference_gate(fit: dict) -> dict:
    """Serialisable training-only diagnostics for the original/tuned gate."""
    gate = fit["raw"]["shares"]
    if gate is None:
        return {"kind": "ct3_only_fallback"}
    if "gate_family" in gate:
        param = gate["selected_param"]
        if isinstance(param, tuple):
            param = list(param)
        return {
            "kind": gate["gate_family"],
            "selected_param": param,
            "inner_mse": gate["inner_mse"],
            "n_clusters": gate["n_clusters"],
            "noise_fraction": gate["noise_fraction"],
        }
    return {"kind": "original_kmeans", "selected_param": int(gate["k"])}
