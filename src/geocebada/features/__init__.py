"""Feature engineering utilities."""

from geocebada.features.agronomic import (
    build_agronomic_feature_layer,
    join_agronomic_features,
    validate_agronomic_feature_layer,
)
from geocebada.features.discovery import (
    FoldLocalExpressionAugmenter,
    FoldLocalExpressionMiner,
)
from geocebada.features.empirical import (
    build_empirical_feature_layer,
    join_empirical_features,
    validate_empirical_feature_layer,
)
from geocebada.features.interactive import (
    FeatureRecipe,
    apply_feature_recipe,
    feature_recipe_to_dict,
    merge_features,
)
from geocebada.features.parcel import (
    build_base_features,
    build_cem15_features,
    build_daily_chirps_features,
    build_official_climate_features,
    build_satellite_features,
    build_siap_features,
    build_soilgrids_features,
    build_static_raster_features,
    build_wapor_features,
    merge_feature_blocks,
    validate_parcel_feature_table,
    zonal_stats_for_raster,
)
from geocebada.features.temporal import (
    align_temporal_knn,
    aligned_to_wide,
    make_temporal_grid,
    temporal_backend_available,
    temporal_coverage_summary,
)

__all__ = [
    "FoldLocalExpressionAugmenter",
    "FoldLocalExpressionMiner",
    "build_empirical_feature_layer",
    "join_empirical_features",
    "validate_empirical_feature_layer",
    "build_agronomic_feature_layer",
    "join_agronomic_features",
    "validate_agronomic_feature_layer",
    "FeatureRecipe",
    "zonal_stats_for_raster",
    "validate_parcel_feature_table",
    "merge_feature_blocks",
    "build_wapor_features",
    "build_static_raster_features",
    "build_soilgrids_features",
    "build_siap_features",
    "build_satellite_features",
    "build_official_climate_features",
    "build_daily_chirps_features",
    "build_cem15_features",
    "build_base_features",
    "align_temporal_knn",
    "aligned_to_wide",
    "apply_feature_recipe",
    "feature_recipe_to_dict",
    "make_temporal_grid",
    "merge_features",
    "temporal_backend_available",
    "temporal_coverage_summary",
]
