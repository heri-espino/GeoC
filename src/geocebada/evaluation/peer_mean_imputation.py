"""Train-fold-only mean substitution for the original Reference26 preprocessors.

Replaces *only* the missing-X fallback statistic (median -> arithmetic mean)
inside five frozen teammate classes. Their feature sets, transforms, scaling,
nested fitting, Ridge, C-T3, geological gate and R3 selection are preserved.
No X values are prefilled in the shared 138-parcel source frame.

Preprocessors are fitted many times on different nested training subsets.
Each override computes the mean from exactly the X passed to that fit,
temporarily fills nonfinite numeric entries in a COPY of those rows, invokes
the original fit unchanged, then stores the training means in the fitted
model's usual missing-value attributes for later transform/predict calls.

Patches are process-local and restored after the fit. No patches to numpy,
pandas, other estimators, source files or y-based prediction code.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

import numpy as np
import pandas as pd

ATTRIBUTES = (
    ("src.family_4.conditional_response", "ConditionalFeatures", "cols", "median_"),
    ("src.geoc_signal_research", "SignalRepresentation", "columns", "medians_"),
    ("src.physical_similarity", "BlockRepresentation", "columns", "fields_"),
    ("src.indicator_neighborhoods", "NumericSpace", "columns", "med"),
    ("src.temporal_indicator_weights", "FixedCorrection", "columns", "median"),
)


def observed_means(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replace missing numeric X using only the supplied fitting rows."""
    x = frame
    filled = {}
    for name in columns:
        if name not in frame:
            # Let the original class raise its usual missing-column error.
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        if valid.all():
            continue
        mean = float(values[valid].mean()) if valid.any() else 0.
        if x is frame:
            x = frame.copy(deep=True)
        x[name] = np.where(valid, values, mean)
        filled[name] = mean
    return x, filled


def mean_fit(original, fields_attribute: str, imputed_attribute: str):
    """A generic sklearn-compatible train-only numerical fit override."""
    @wraps(original)
    def fit(self, frame, *args, **kwargs):
        names = list(getattr(self, fields_attribute))
        prepared, stats = observed_means(frame, names)
        result = original(self, prepared, *args, **kwargs)
        if stats:
            if imputed_attribute == "fields_":
                for name, mean in stats.items():
                    # BlockRepresentation uses a misleading 'median' key to
                    # store the numeric fallback. Preserve its schema.
                    if self.fields_[name]["numeric"]:
                        self.fields_[name]["median"] = mean
            else:
                imputed = getattr(self, imputed_attribute)
                for i, name in enumerate(names):
                    if name in stats:
                        imputed[i] = stats[name]
        return result

    return fit


@contextmanager
def use_training_means():
    """Switch all five original numeric imputers to means for one worker."""
    import importlib

    originals = []
    try:
        for module_name, class_name, fields_attr, med_attr in ATTRIBUTES:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            old = class_.fit
            # A nested override is not supported; this also prevents hidden
            # overrides colliding with the provenance assumptions.
            if getattr(old, "_reference26_mean_override", False):
                raise RuntimeError("The training-mean override is already active.")
            wrapper = mean_fit(old, fields_attr, med_attr)
            wrapper._reference26_mean_override = True
            originals.append((class_, old))
            class_.fit = wrapper
        yield
    finally:
        for class_, old in reversed(originals):
            class_.fit = old


def missing_numeric_inputs(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Descriptive X-only audit of the original 26 predictors; never y."""
    rows = []
    for name in features:
        values = frame[name]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        a = pd.to_numeric(values, errors="coerce").to_numpy(float)
        valid = np.isfinite(a)
        known = a[valid]
        if not (~valid).any():
            continue
        rows.append({
            "feature": name,
            "missing_cells": int((~valid).sum()),
            "observed_cells": int(valid.sum()),
            "mean_observed": float(np.mean(known)) if len(known) else 0.,
            "median_observed": float(np.median(known)) if len(known) else 0.,
        })
    return pd.DataFrame(
        rows,
        columns=[
            "feature", "missing_cells", "observed_cells",
            "mean_observed", "median_observed",
        ],
    )
