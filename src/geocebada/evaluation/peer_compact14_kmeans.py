"""Add the original Reference26-style geological KMeans gate to Compact14.

Compact14 retains its original CT3 global estimator, 5-feature local Ridge,
and individually selected R3 residual strength. Only the mixing of its
global/local predictions is replaced with a geological KMeans gate, choosing
k=4 or k=8 by inner CV without reading the outer held-out yield.

The KMeans gate uses additional geology features. Consequently the modified
model is named *Compact14 + geological KMeans* rather than a 14-feature-only
model. No changes are made to teammate's source checkout.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

K_VALUES = (4, 8)


def compact_oof(factory, tr, cv):
    """Original compact global CT3 and local-five predictions from train-only CV."""
    from src.geoc_signal_research import predict_base

    tr = np.asarray(tr, dtype=int)
    ct = np.full(len(tr), np.nan)
    local = np.full(len(tr), np.nan)
    for a, b in cv:
        aa, bb = tr[a], tr[b]
        if set(aa) & set(bb):
            raise RuntimeError("Compact14 inner CV train/test parcels overlap.")
        ct[b] = factory.helper.ct3(aa).predict(factory.a["a"]["train"].iloc[bb])
        local[b] = predict_base(factory.raw_model(aa), bb, factory.a)
    if not np.isfinite(ct).all() or not np.isfinite(local).all():
        raise RuntimeError("Nonfinite Compact14 inner cross-fit predictions.")
    return ct, local


class CompactKMeansMixin:
    """Train-only k=4/8 selection for the Compact14 global/local gate."""

    def fit(self, tr):
        tr = np.asarray(tr, dtype=int)
        key = tuple(tr)
        if key in self.models:
            return self.models[key]

        # Use the peer's native compact fit and five-feature representation.
        model = super().fit(tr)
        if model["local"] is None:
            return model  # identical native CT3-only fallback for insufficient CV

        from src.conditional_local_phases import ExpertGate
        from src.signal_role_allocation import RoleSpace

        cv = self.cv(tr)
        ct, local = compact_oof(self, tr, cv)
        y = self.a["y"]
        frame = self.a["role_frame"]
        definitions = self.a["physical_definitions"]
        scores = []

        for k in K_VALUES:
            internal = np.full(len(tr), np.nan)
            for a, b in cv:
                aa, bb = tr[a], tr[b]
                try:
                    deeper = self.cv(aa)
                    ac, al = compact_oof(self, aa, deeper)
                except ValueError:
                    # Match Reference26: if deeper cross-fitting is unavailable,
                    # use the training-only climate prediction.
                    internal[b] = ct[b]
                    continue
                space = RoleSpace(["geologic_composition"], [], definitions)
                space.fit(frame.iloc[aa])
                gate = ExpertGate("clusters", k).fit(
                    space.transform(frame.iloc[aa]), y[aa], ac, al
                )
                internal[b] = gate.predict(
                    space.transform(frame.iloc[bb]), ct[b], local[b]
                )
            if not np.isfinite(internal).all():
                raise RuntimeError("Compact14 KMeans selection predicted NaNs.")
            scores.append((float(np.mean((internal - y[tr]) ** 2)), k))

        best_k = min(scores, key=lambda entry: (round(entry[0], 12), entry[1]))[1]
        space = RoleSpace(["geologic_composition"], [], definitions)
        space.fit(frame.iloc[tr])
        gate = ExpertGate("clusters", best_k).fit(
            space.transform(frame.iloc[tr]), y[tr], ct, local
        )
        model["geology_gate"] = {
            "space": space, "gate": gate, "k": int(best_k),
            "inner_scores": [
                {"k": int(k), "inner_mse": float(mse)} for mse, k in scores
            ],
        }
        return model

    def predict(self, model, query):
        if "geology_gate" not in model:
            return super().predict(model, query)

        from src.geoc_signal_research import predict_base

        query = np.asarray(query, dtype=int)
        if set(model["train_positions"]) & set(query):
            raise ValueError("Compact14 predicted an in-sample parcel.")
        ct = model["ct3"].predict(self.a["a"]["train"].iloc[query])
        local = predict_base(model["local"], query, self.a)
        saved = model["geology_gate"]
        return saved["gate"].predict(
            saved["space"].transform(self.a["role_frame"].iloc[query]),
            ct, local,
        )


@contextmanager
def compact_kmeans_factory():
    """Patch only the compact factory for the life of one outer fold."""
    import src.current_champions as peer

    original = peer.PrunedGlobalFactory
    if issubclass(original, CompactKMeansMixin):
        raise RuntimeError("Compact14 geological gate override already active.")
    peer.PrunedGlobalFactory = type(
        "Compact14GeologicalKMeansFactory",
        (CompactKMeansMixin, original),
        {},
    )
    try:
        yield
    finally:
        peer.PrunedGlobalFactory = original


def predict_compact_kmeans(pack, query, frame, distance):
    """Apply the native compact R3 correction to its gated raw prediction."""
    import src.current_champions as peer

    outer = pack["outer_experts"]
    compact = outer["raw"]["compact"]
    q = np.asarray(query, dtype=int)
    if "geology_gate" not in compact:
        return np.asarray(
            peer.predict_saved(pack, q, frame, distance)["Compact14"], float
        )

    gate = compact["geology_gate"]
    ct = compact["ct3"].predict(frame.iloc[q])
    local = peer.raw_local(compact["local"], q, frame, distance)
    predicted = gate["gate"].predict(
        gate["space"].transform(frame.iloc[q]), ct, local
    )
    correction = outer["correctors"][0]
    if correction is not None:
        complete = np.isfinite(
            frame.iloc[q][peer.REDUCTIONS["R3"]].to_numpy(float)
        ).all(axis=1)
        predicted = (
            predicted
            + outer["strengths"][0]
            * complete
            * correction.delta(frame.iloc[q])
        )
    result = np.asarray(predicted, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Compact14 geological KMeans prediction.")
    return result


def selected_k(fitted):
    compact = fitted["raw"]["compact"]
    gate = compact.get("geology_gate")
    if gate is None:
        return {"type": "ct3_only_fallback"}
    return {
        "type": "geological_kmeans",
        "k": gate["k"],
        "inner_scores": gate["inner_scores"],
    }
