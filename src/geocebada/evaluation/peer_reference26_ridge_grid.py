"""Consistent, temporary local Ridge hyperparameter override for Reference26.

The original todoseboso implementation uses two paths:
1. src.geoc_signal_research.predict_base -> module-level local_prediction
   imported from src.geoc_bridge (scikit-learn Ridge);
2. src.current_champions.raw_local -> module-level predict_local_ridge
   imported from src.local_ridge_sufficient_statistics (dense Ridge solve).

Every path needs the *same* k/alpha during both nested fitting and
held-out predictions. Never edit or overwrite the teammate's source files.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

import numpy as np

NEIGHBORS = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
ALPHAS = (
    0.1, 0.15, 0.22, 0.32, 0.46,
    0.68, 1.0, 1.5, 2.2, 3.2,
    4.6, 6.8, 10.0, 15.0, 22.0,
    30.0, 46.0, 68.0, 100.0, 150.0,
    220.0, 300.0, 460.0, 680.0, 1000.0,
)
DEFAULT_PAIR = (24, 30.0)
GRID = tuple((k, alpha) for k in NEIGHBORS for alpha in ALPHAS)
assert len(GRID) == 250 and DEFAULT_PAIR in GRID


def settings(k: int, alpha: float):
    if k < 3 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Local Ridge needs k>=3 and a finite alpha>0.")


@contextmanager
def tuned_local_ridge(k: int, alpha: float):
    """Patch BOTH inner fitting and outer-query Ridge paths consistently."""
    settings(k, alpha)

    import src.geoc_bridge as bridge
    import src.geoc_signal_research as signals
    import src.conditional_local_phases as phases
    import src.current_champions as champions
    from src.geoc_local04d_snapshot import predict_weighted_local_ridge
    from src.local_ridge_sufficient_statistics import predict_local_ridge

    old = {
        (bridge, "local_prediction"): bridge.local_prediction,
        (signals, "local_prediction"): signals.local_prediction,
        (phases, "local_prediction"): phases.local_prediction,
        (champions, "predict_local_ridge"): champions.predict_local_ridge,
    }

    def original_path(x, distances, observed, query, targets):
        observed = np.asarray(observed, dtype=int)
        query = np.asarray(query, dtype=int)
        targets = np.asarray(targets, dtype=float)
        if set(observed) & set(query) or len(observed) != len(targets):
            raise ValueError("Local Ridge train/test overlap or target mismatch.")
        y = np.full(len(x), np.nan)
        y[observed] = targets
        return predict_weighted_local_ridge(
            x, distances, y, observed, query,
            k=k, alpha=alpha, distance_power=1.0,
        )

    def fast_path(x, distances, observed, query, targets):
        return predict_local_ridge(
            x, distances, observed, query, targets,
            k=k, alpha=alpha,
        )

    for module, attr in list(old):
        setattr(module, attr, fast_path if attr == "predict_local_ridge" else original_path)
    try:
        yield
    finally:
        for (module, attr), previous in old.items():
            setattr(module, attr, previous)


@contextmanager
def share_climate_fits_between_grid_candidates():
    """Reuse only invariant C-T3 climate fits within one outer-fold *batch*.

    Same masked peer frame and CV splits across this batch. Model cache
    is process-local, not persisted, and gets discarded after the batch.
    Local Ridge, geology and R3 are always refitted separately for each
    (k, alpha), so reusing this independent climate branch is safe.
    """
    import src.current_champions as champions

    owner = champions.ColdBases
    original = owner.ct3
    cache = {}

    @wraps(original)
    def cached(self, tr):
        key = tuple(np.asarray(tr, dtype=int).tolist())
        if key not in cache:
            cache[key] = original(self, tr)
        return cache[key]

    owner.ct3 = cached
    try:
        yield
    finally:
        owner.ct3 = original
        cache.clear()
