"""Model evaluation, validation and diagnostics utilities."""

from geocebada.evaluation.checkpoint03c2 import (
    ModelSearchSpec,
    build_competition_representation_specs,
    evaluate_nested_model_families,
    make_inner_cv_splits,
    make_model_search_spec,
    summarize_nested_results,
    validate_nested_oof_coverage,
)
from geocebada.evaluation.checkpoint03c import (
    RepresentationSpec,
    build_layer_feature_catalog,
    build_representation_specs,
    evaluate_representation_benchmark,
    make_representation_estimator,
    summarize_representation_benchmark,
)
from geocebada.evaluation.parcel_modeling import (
    build_feature_catalog,
    build_fixed_fold_assignments,
    classify_feature_family,
    evaluate_fixed_folds,
    initial_regressors,
    resolve_ablation_features,
    summarize_feature_inventory,
    summarize_fixed_fold_scores,
    train_prediction_diagnostics,
    validate_fixed_fold_assignments,
)
from geocebada.evaluation.regression import (
    benchmark_regressors,
    default_regressors,
    summarize_benchmark,
)

__all__ = [
    "ModelSearchSpec",
    "build_competition_representation_specs",
    "evaluate_nested_model_families",
    "make_inner_cv_splits",
    "make_model_search_spec",
    "summarize_nested_results",
    "validate_nested_oof_coverage",
    "RepresentationSpec",
    "build_layer_feature_catalog",
    "build_representation_specs",
    "evaluate_representation_benchmark",
    "make_representation_estimator",
    "summarize_representation_benchmark",
    "benchmark_regressors",
    "build_feature_catalog",
    "build_fixed_fold_assignments",
    "classify_feature_family",
    "default_regressors",
    "evaluate_fixed_folds",
    "initial_regressors",
    "resolve_ablation_features",
    "summarize_benchmark",
    "summarize_feature_inventory",
    "summarize_fixed_fold_scores",
    "train_prediction_diagnostics",
    "validate_fixed_fold_assignments",
]
