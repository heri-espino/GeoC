"""Train Reference26 with a *fixed* geological K-means cluster count.

The teammate's original ColdExperts.full_fit and predict_saved remain intact.
The only change is bypassing selection between k=4 and 8 in ColdBases.shares
in favor of one explicitly fixed k. Its entire nested R3/correction machinery
is unchanged; Compact14 remains an untouched internal component.

The first-46 outer folds (or all 138 on explicit request) are evaluated for
each fixed k separately. No outer held-out target is used for choosing k.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

DEFAULT_K = (4, 8, 12, 14, 16, 20, 26)


class FixedGeologicalKMeansMixin:
    """Replicate original SimplifiedBases.shares but fix k instead of tuning it."""

    fixed_k: int | None = None

    def shares(self, tr):
        from src.conditional_local_phases import ExpertGate
        from src.signal_role_allocation import RoleSpace

        tr = np.asarray(tr, dtype=int)
        key = tuple(tr)
        if key in self.share_models:
            return self.share_models[key]
        if not isinstance(self.fixed_k, int) or self.fixed_k < 1:
            raise ValueError("Fixed geological KMeans requires positive integer k.")
        try:
            cv = self.cv(tr)
        except ValueError:
            self.logs.append({
                "kind": "fixed_kmeans_ct3_only_no_cv",
                "positions": tr.tolist(),
                "k": self.fixed_k,
            })
            return None

        # Preserve the teammate's original training-only CT3/local expert
        # cross-fit, 10-feature geological role preprocessing and smoothing.
        ct, ep = self.route_oof(tr, cv)
        space = RoleSpace(
            ["geologic_composition"], [], self.a["physical_definitions"]
        ).fit(self.a["role_frame"].iloc[tr])
        geology = space.transform(self.a["role_frame"].iloc[tr])
        gate = ExpertGate("clusters", self.fixed_k).fit(
            geology, self.a["y"][tr], ct, ep
        )
        saved = {
            "space": space,
            "gate": gate,
            "k": int(self.fixed_k),
        }
        self.share_models[key] = saved
        return saved


@contextmanager
def fixed_geological_kmeans(k: int):
    """Apply an isolated patch only while a single Reference26 fit executes."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    import src.current_champions as peer

    original = peer.ColdBases
    if issubclass(original, FixedGeologicalKMeansMixin):
        raise RuntimeError("Already running a modified geological gate.")
    peer.ColdBases = type(
        f"ColdBasesFixedK{k}",
        (FixedGeologicalKMeansMixin, original),
        {"fixed_k": k},
    )
    try:
        yield
    finally:
        peer.ColdBases = original


def gate_diagnostics(fitted: dict) -> dict:
    """Record requested k and actual fitted clusters, never private yield rows."""
    shares = fitted["raw"]["shares"]
    if shares is None:
        return {"fallback": "ct3_only"}
    return {
        "requested_k": int(shares["k"]),
        "fitted_clusters": int(shares["gate"].model_.n_clusters),
    }
