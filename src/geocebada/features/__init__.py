"""Feature engineering utilities."""

from geocebada.features.interactive import (
    FeatureRecipe,
    apply_feature_recipe,
    feature_recipe_to_dict,
    merge_features,
)
from geocebada.features.temporal import (
    align_temporal_knn,
    aligned_to_wide,
    make_temporal_grid,
    temporal_backend_available,
    temporal_coverage_summary,
)

__all__ = [
    "FeatureRecipe",
    "align_temporal_knn",
    "aligned_to_wide",
    "apply_feature_recipe",
    "feature_recipe_to_dict",
    "make_temporal_grid",
    "merge_features",
    "temporal_backend_available",
    "temporal_coverage_summary",
]
