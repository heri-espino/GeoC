# Checkpoint 06D — X-only regime discovery

Generated: 2026-10-06T19:33:06.598286+00:00
Git commit: a59ee24b6c8befcf2f0254ef5e60b54f9739b599

## X-only selection

{
  "algorithm": "kmeans",
  "k": 4,
  "min_cluster_fraction": 0.14720812182741116,
  "min_cluster_size": 29,
  "n_components": 11,
  "pca_variance": 0.8,
  "seed_ari_mean": 0.8808540126095037,
  "selection_status": "X_CLUSTERING_SELECTED",
  "silhouette": 0.19664443428580888,
  "subsample_ari_mean": 0.8204019163473918,
  "x_eligible": true
}

## Residual-regime decision

**REGIME_HYPOTHESIS = NOT_SUPPORTED**

{
  "REGIME_HYPOTHESIS": "NOT_SUPPORTED",
  "expert_advantage_range": 0.03375613535527802,
  "geography": {
    "cluster_municipality_nmi": 0.44842987447265725,
    "cluster_state_nmi": 0.2984425314623464
  },
  "local04d_residual_effect": {
    "method": "Local04D",
    "mse_eta2": 0.011872540839704893,
    "mse_permutation_p": 0.7621189405297352,
    "residual_eta2": 0.003997428513152935,
    "residual_mean_range": 0.08773811960491257,
    "residual_permutation_p": 0.9345327336331835
  },
  "local04d_residual_structure_supported": false,
  "selected_x_regime": {
    "algorithm": "kmeans",
    "k": 4,
    "min_cluster_fraction": 0.14720812182741116,
    "min_cluster_size": 29,
    "n_components": 11,
    "pca_variance": 0.8,
    "seed_ari_mean": 0.8808540126095037,
    "selection_status": "X_CLUSTERING_SELECTED",
    "silhouette": 0.19664443428580888,
    "subsample_ari_mean": 0.8204019163473918,
    "x_eligible": true
  },
  "x_clustering_supported": true
}

## Cluster composition

 cluster  n_all  n_labeled  n_target  target_fraction  yield_mean  yield_std
       0     57         39        18         0.315789    4.645897   0.529163
       1     29         20         9         0.310345    4.349000   0.418962
       2     41         28        13         0.317073    3.028571   0.407831
       3     70         51        19         0.271429    3.781765   0.802957

## Honest residuals by cluster

           method  cluster  n_parcels  mean_residual  median_residual  mean_squared_error  rmse_across_parcels  mean_absolute_error
      CatBoost_C1        0         37       0.099595         0.141565            0.356102             0.596742             0.416675
      CatBoost_C1        1         16       0.038389         0.046849            0.142011             0.376844             0.310325
      CatBoost_C1        2         25      -0.102473        -0.057047            0.181166             0.425636             0.316576
      CatBoost_C1        3         47      -0.045678         0.053791            0.289918             0.538440             0.370107
         Graph04D        0         37       0.078023         0.003515            0.304097             0.551450             0.354595
         Graph04D        1         16      -0.064389        -0.031114            0.136313             0.369205             0.318749
         Graph04D        2         25      -0.049961        -0.007108            0.165656             0.407009             0.311963
         Graph04D        3         47      -0.071619         0.043610            0.277312             0.526604             0.324493
         Local04D        0         37       0.007502        -0.056576            0.318392             0.564262             0.391309
         Local04D        1         16      -0.027647         0.010863            0.120028             0.346451             0.290646
         Local04D        2         25      -0.080237        -0.031562            0.177212             0.420966             0.312015
         Local04D        3         47      -0.035632         0.035594            0.252959             0.502951             0.331557
LocalCatBoostMean        0         37       0.053548         0.043675            0.328567             0.573207             0.390376
LocalCatBoostMean        1         16       0.005371         0.026662            0.121146             0.348060             0.293688
LocalCatBoostMean        2         25      -0.091355        -0.077590            0.175357             0.418757             0.311818
LocalCatBoostMean        3         47      -0.040655         0.081214            0.265086             0.514865             0.345944

## Expert advantage by cluster

 cluster  n_parcels  mean_cat_minus_local_mse  median_cat_minus_local_mse  catboost_better_fraction
       0         37                  0.037710                    0.007566                  0.459459
       1         16                  0.021983                   -0.000825                  0.500000
       2         25                  0.003954                    0.001872                  0.480000
       3         47                  0.036958                    0.017453                  0.404255

## Next step

Do not implement cluster specialists; retain Local04D unless another predeclared mechanism is tested.
