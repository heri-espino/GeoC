"""Experimental DBSCAN geological gate for Nuria's frozen champion recipes.

Only the clustering gate changes. Source feature extraction, local 24-NN Ridge,
climatic C-T3, cross-fitted experts and R3 adjustment remain the peer's own code.

Not a replacement for geographical 24-NN and not an unsupervised-only score.
DBSCAN has no predict(): held-out parcels connect only to a training core point
within the fit-specific radius, otherwise they receive the global mixing weight.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

RADIUS_FACTORS = (0.75, 1.25, 2.0, 3.0)
MIN_SAMPLES = (3, 5)
DBSCAN_GRID = tuple(
    (minimum, radius)
    for minimum in MIN_SAMPLES
    for radius in RADIUS_FACTORS
)


def adaptive_radius(z: np.ndarray, min_samples: int, radius_factor: float) -> float:
    """Scale radius by a training-only k-distance, never held-out coordinates.

    The k-distance is the distance to the (min_samples - 1)th other sample.
    Duplicate rows are supported. A tiny positive fallback handles constants.
    """
    z = np.asarray(z, dtype=float)
    if (
        z.ndim != 2 or len(z) < min_samples or not np.isfinite(z).all()
        or min_samples < 2 or radius_factor <= 0
    ):
        raise ValueError("Invalid geological training matrix or DBSCAN parameters.")
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(z)
    distances = neighbors.kneighbors(z, return_distance=True)[0][:, -1]
    positive = distances[distances > 1e-12]
    typical = float(np.median(positive)) if len(positive) else 1e-10
    return max(1e-10, radius_factor * typical)


class GeologicalDBSCANGate:
    """Shrink per-density-cluster mixing weights towards the global weight."""

    def __init__(self, min_samples: int, radius_factor: float):
        self.min_samples = min_samples
        self.radius_factor = radius_factor

    def fit(self, z: np.ndarray, y: np.ndarray, ct: np.ndarray, expert: np.ndarray):
        z = np.asarray(z, float)
        y, ct, expert = (np.asarray(a, float).reshape(-1) for a in (y, ct, expert))
        if z.ndim != 2 or any(len(a) != len(z) for a in (y, ct, expert)):
            raise ValueError("Inconsistent DBSCAN fit rows.")
        if not all(np.isfinite(a).all() for a in (z, y, ct, expert)):
            raise ValueError("DBSCAN gate requires finite training features and OOF.")
        from src.conditional_local_phases import global_weight

        self.global_ = global_weight(y, ct, expert)
        self.eps_ = adaptive_radius(z, self.min_samples, self.radius_factor)
        self.model_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit(z)
        self.labels_ = self.model_.labels_.copy()
        self.cluster_labels_ = np.unique(self.labels_[self.labels_ >= 0])
        self.core_indices_ = self.model_.core_sample_indices_.copy()
        self.core_features_ = z[self.core_indices_]
        self.core_labels_ = self.labels_[self.core_indices_]
        self.noise_fraction_ = float(np.mean(self.labels_ == -1))
        self.n_clusters_ = len(self.cluster_labels_)

        difference = expert - ct
        prior = 20.0 * float(np.mean(difference**2))
        self.weights_ = {}
        for label in self.cluster_labels_:
            mask = self.labels_ == label
            denom = float(np.sum(difference[mask] ** 2)) + prior
            numerator = float(difference[mask] @ (y[mask] - ct[mask]))
            value = (numerator + prior * self.global_) / denom if denom > 1e-12 else self.global_
            self.weights_[int(label)] = float(np.clip(value, 0, 1))
        return self

    def weights(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or not np.isfinite(q).all():
            raise ValueError("Invalid DBSCAN query features.")
        result = np.full(len(q), self.global_, dtype=float)
        if not len(self.core_indices_) or not len(q):
            return result
        distances = cdist(q, self.core_features_)
        nearest = np.argmin(distances, axis=1)
        valid = distances[np.arange(len(q)), nearest] <= self.eps_ + 1e-12
        for position in np.flatnonzero(valid):
            cluster = int(self.core_labels_[nearest[position]])
            result[position] = self.weights_[cluster]
        return result

    def predict(self, q: np.ndarray, ct: np.ndarray, expert: np.ndarray) -> np.ndarray:
        ct, expert = np.asarray(ct, float), np.asarray(expert, float)
        weight = self.weights(q)
        if weight.shape != ct.shape or ct.shape != expert.shape:
            raise ValueError("DBSCAN gate prediction dimensions differ.")
        return ct + weight * (expert - ct)


class DBSCANShareMixin:
    """Override only original SimplifiedBases.shares; retain all expert recipes."""

    def shares(self, tr):
        tr = np.asarray(tr, int)
        key = tuple(tr)
        if key in self.share_models:
            return self.share_models[key]
        try:
            cv = self.cv(tr)
        except ValueError:
            self.logs.append(dict(kind="dbscan_ct3_only_no_cv", positions=tr.tolist()))
            return None

        from src.signal_role_allocation import RoleSpace

        ct, ep = self.route_oof(tr, cv)
        frame = self.a["role_frame"]
        y = self.a["y"]
        atoms = ["geologic_composition"]
        scores = []
        for min_samples, radius_factor in DBSCAN_GRID:
            prediction = np.full(len(tr), np.nan)
            for c, d in cv:
                aa, bb = tr[c], tr[d]
                try:
                    ac, ae = self.route_oof(aa)
                except ValueError:
                    prediction[d] = ct[d]
                    continue
                space = RoleSpace(atoms, [], self.a["physical_definitions"]).fit(frame.iloc[aa])
                gate = GeologicalDBSCANGate(min_samples, radius_factor).fit(
                    space.transform(frame.iloc[aa]), y[aa], ac, ae
                )
                prediction[d] = gate.predict(
                    space.transform(frame.iloc[bb]), ct[d], ep[d]
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite cross-fitted DBSCAN geological gate.")
            scores.append({
                "inner_mse": float(np.mean((prediction - y[tr]) ** 2)),
                "min_samples": min_samples,
                "radius_factor": radius_factor,
            })

        winner = min(
            scores, key=lambda row: (
                round(row["inner_mse"], 12),
                row["min_samples"],
                row["radius_factor"],
            )
        )
        space = RoleSpace(atoms, [], self.a["physical_definitions"]).fit(frame.iloc[tr])
        gate = GeologicalDBSCANGate(
            winner["min_samples"], winner["radius_factor"]
        ).fit(space.transform(frame.iloc[tr]), y[tr], ct, ep)
        pack = {
            "space": space,
            "gate": gate,
            "kind": "dbscan_geology",
            "min_samples": winner["min_samples"],
            "radius_factor": winner["radius_factor"],
            "selected_inner_mse": winner["inner_mse"],
            "candidates": scores,
            "noise_fraction": gate.noise_fraction_,
            "n_clusters": gate.n_clusters_,
            "eps": gate.eps_,
        }
        self.share_models[key] = pack
        return pack


@contextmanager
def dbscan_refit_mode():
    """Use original refit_fold with a strictly local class override.

    Do not run concurrent folds inside one process while this override is
    active. It restores the original class even if training raises.
    """
    peer = import_module("src.current_champions")
    original = peer.ColdBases
    if issubclass(original, DBSCANShareMixin):
        raise RuntimeError("DBSCAN override is already active.")
    modified = type("DBSCANColdBases", (DBSCANShareMixin, original), {})
    peer.ColdBases = modified
    try:
        yield peer.refit_fold
    finally:
        peer.ColdBases = original
