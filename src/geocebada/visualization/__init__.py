"""Reusable visualizations for notebooks and the GeoCebada app."""

from geocebada.visualization.checkpoint04a import (
    plot_adversarial_roc,
    plot_moran_scatter,
    plot_nearest_distance_distribution,
    plot_pca_embedding,
    plot_primary_temporal_neighbor_panels,
    plot_similarity_vs_yield_difference,
    plot_support_ranking,
    plot_temporal_similarity_distribution,
    plot_top_feature_shift,
    plot_train_target_map,
)
from geocebada.visualization.checkpoint04b import (
    plot_actual_method_routing,
    plot_actual_vs_pseudo_support,
    plot_primary_method_ranking,
    plot_split_match_quality,
    plot_support_tier_method_rmse,
    plot_target_support_routing_scatter,
    plot_top_method_rmse_boxplot,
)
from geocebada.visualization.checkpoint04d1 import (
    plot_actual_candidate_spread,
    plot_graph_k_profiles,
    plot_local_k_profile,
    plot_loso_selection_frequency,
    plot_nested_validation_comparison,
    plot_refinement_ranking,
)
from geocebada.visualization.checkpoint04e import (
    plot_external_ranking,
    plot_siap_proxy_vs_yield,
    plot_siap_scope_coverage,
)
from geocebada.visualization.exploration import (
    correlation_heatmap,
    correlation_scatter,
    distribution_figure,
    missingness_table,
    outlier_summary,
    pairplot_figure,
    residual_diagnostic_figure,
)
from geocebada.visualization.geospatial import parcel_map_figure

__all__ = [
    "plot_external_ranking",
    "plot_siap_proxy_vs_yield",
    "plot_siap_scope_coverage",
    "plot_actual_candidate_spread",
    "plot_graph_k_profiles",
    "plot_local_k_profile",
    "plot_loso_selection_frequency",
    "plot_nested_validation_comparison",
    "plot_refinement_ranking",
    "plot_actual_method_routing",
    "plot_actual_vs_pseudo_support",
    "plot_primary_method_ranking",
    "plot_split_match_quality",
    "plot_support_tier_method_rmse",
    "plot_target_support_routing_scatter",
    "plot_top_method_rmse_boxplot",
    "plot_adversarial_roc",
    "plot_moran_scatter",
    "plot_nearest_distance_distribution",
    "plot_pca_embedding",
    "plot_primary_temporal_neighbor_panels",
    "plot_similarity_vs_yield_difference",
    "plot_support_ranking",
    "plot_temporal_similarity_distribution",
    "plot_top_feature_shift",
    "plot_train_target_map",
    "correlation_heatmap",
    "correlation_scatter",
    "distribution_figure",
    "missingness_table",
    "outlier_summary",
    "pairplot_figure",
    "parcel_map_figure",
    "residual_diagnostic_figure",
]
