# Checkpoint 06E — supervised center/tail mixture

Generated: 2026-10-06T20:24:42.419200+00:00
Git commit: 30a6a2afbe56fa1544883bb662ee70fe5e80a2dc

## Architecture

Local04D + w_center*center_residual_expert(X) + w_low*low_offset + w_high*high_offset

low gate family: water_productivity
high gate family: thermal

## Development ranking

                        method  n_splits  n_predictions  rmse_mean  rmse_median  rmse_worst  pooled_rmse  center_pooled_rmse  tail_pooled_rmse  low_pooled_rmse  high_pooled_rmse  center_bias  low_bias  high_bias
                      Local04D        16            656   0.487845     0.468742    0.694700     0.495741            0.360770          0.664049         0.727121          0.534243     0.013806 -0.431968   0.473166
          Center_G1PCA_a1000p0        16            656   0.487933     0.468019    0.683474     0.495477            0.351396          0.671959         0.729383          0.555765     0.020759 -0.453806   0.494647
          Center_Cross_a1000p0        16            656   0.488012     0.468538    0.686606     0.495683            0.358236          0.666245         0.726085          0.544227     0.018138 -0.435277   0.483158
           Center_G1PCA_a100p0        16            656   0.488902     0.470565    0.683838     0.496411            0.350577          0.674529         0.732779          0.556479     0.023025 -0.463706   0.496504
            Center_G1PCA_a10p0        16            656   0.489097     0.471249    0.683973     0.496607            0.350750          0.674764         0.733451          0.555699     0.023456 -0.465417   0.496073
CenterTail_G1PCA_a1000p0_s0p25        16            656   0.490567     0.473283    0.705459     0.498992            0.386539          0.645862         0.716124          0.497490    -0.003892 -0.386242   0.425671
 CenterTail_G1PCA_a100p0_s0p25        16            656   0.490602     0.474302    0.705432     0.499055            0.384810          0.647733         0.718589          0.497931    -0.001626 -0.396141   0.427527
  CenterTail_G1PCA_a10p0_s0p25        16            656   0.490652     0.474740    0.705504     0.499119            0.384808          0.647867         0.719110          0.497080    -0.001196 -0.397852   0.427096
            Center_Cross_a10p0        16            656   0.491142     0.470791    0.682744     0.498484            0.350898          0.678347         0.731428          0.572273     0.027277 -0.453766   0.500724
           Center_Cross_a100p0        16            656   0.491501     0.472822    0.688076     0.499066            0.355446          0.675492         0.730826          0.564218     0.020110 -0.446134   0.499213
CenterTail_Cross_a1000p0_s0p25        16            656   0.492727     0.477049    0.709514     0.501180            0.394785          0.641983         0.714858          0.486679    -0.006514 -0.367713   0.414182
                TailGate_s0p25        16            656   0.493291     0.478743    0.719297     0.502042            0.397963          0.640479         0.716452          0.476928    -0.010846 -0.364404   0.404190
  CenterTail_Cross_a10p0_s0p25        16            656   0.493637     0.475460    0.704637     0.501781            0.385585          0.652599         0.718408          0.515697     0.002625 -0.386202   0.431748
 CenterTail_Cross_a100p0_s0p25        16            656   0.495014     0.479398    0.710608     0.503397            0.391210          0.650288         0.718644          0.506944    -0.004542 -0.378569   0.430237
   CenterTail_G1PCA_a10p0_s0p5        16            656   0.501987     0.490108    0.731996     0.510845            0.425154          0.629598         0.713335          0.443897    -0.025847 -0.330288   0.358120
  CenterTail_G1PCA_a100p0_s0p5        16            656   0.502079     0.489921    0.731983     0.510910            0.425300          0.629574         0.712975          0.444825    -0.026278 -0.328577   0.358551
 CenterTail_G1PCA_a1000p0_s0p5        16            656   0.502958     0.490407    0.732374     0.511693            0.427752          0.628478         0.711495          0.444731    -0.028544 -0.318677   0.356694
   CenterTail_Cross_a10p0_s0p5        16            656   0.505804     0.491935    0.731473     0.514222            0.426438          0.635495         0.713997          0.464820    -0.022026 -0.318637   0.362771
 CenterTail_Cross_a1000p0_s0p5        16            656   0.507094     0.497182    0.737263     0.515755            0.437040          0.626611         0.712322          0.434969    -0.031165 -0.300148   0.345205
  CenterTail_Cross_a100p0_s0p5        16            656   0.508160     0.497614    0.737999     0.516810            0.432869          0.633796         0.715088          0.455315    -0.029193 -0.311005   0.361260

## LOSO

{
  "center_rmse_improvement": -0.00011810712455573302,
  "incumbent_center_rmse": 0.36076982220030784,
  "incumbent_pooled_rmse": 0.49574142871073495,
  "incumbent_rmse_mean": 0.48784531674343823,
  "incumbent_tail_rmse": 0.6640489003338566,
  "n_splits": 16,
  "pooled_rmse_improvement": -0.0008261349331432633,
  "rmse_mean_improvement": -0.0007863913110580656,
  "selected_center_rmse": 0.3608879293248636,
  "selected_pooled_rmse": 0.4965675636438782,
  "selected_rmse_mean": 0.4886317080544963,
  "selected_tail_rmse": 0.6655982465856533,
  "selection_counts": {
    "Center_Cross_a1000p0": 3,
    "Local04D": 13
  },
  "tail_rmse_change": 0.0015493462517967016
}

## Decision

**06E_LOSO_GATE = FAIL**

The center/tail supervised mixture did not survive split-excluded selection.
Retain Local04D and close the planned Checkpoint 06 modeling path unless a new, predeclared hypothesis is introduced.
