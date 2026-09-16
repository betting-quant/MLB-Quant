# MLB 2026 Forward Test v0.1

Production models were trained on 2021-2025 only.

2026 outcomes were not used for model fitting, model selection, feature selection, or distribution fitting.

Starts evaluated: 4510

## Point-model performance

| target         | model                  |     mae |     rmse |      bias |   correlation |   poisson_deviance |    n |
|:---------------|:-----------------------|--------:|---------:|----------:|--------------:|-------------------:|-----:|
| strikeouts     | hist_gradient_boosting | 1.75665 |  2.20032 | 0.0261805 |      0.460603 |           1.11553  | 4510 |
| outs_recorded  | hist_gradient_boosting | 2.76718 |  3.6523  | 0.148349  |      0.575541 |           1.02477  | 4510 |
| batters_faced  | hist_gradient_boosting | 2.71697 |  3.73317 | 0.208149  |      0.68341  |           0.793096 | 4510 |
| pitches_thrown | hist_gradient_boosting | 9.35342 | 13.3811  | 0.595531  |      0.708039 |           2.7661   | 4510 |

## Probability performance

| target        | method                |    brier |   log_loss |   n_binary_observations |   expected_calibration_error |   probability_std |   monotonicity_violations |
|:--------------|:----------------------|---------:|-----------:|------------------------:|-----------------------------:|------------------:|--------------------------:|
| strikeouts    | poisson               | 0.124485 |   0.387126 |                   40590 |                   0.00211126 |          0.302692 |                         0 |
| outs_recorded | conditional_empirical | 0.142338 |   0.439514 |                   49610 |                   0.0145048  |          0.31404  |                         0 |

## Probability performance by line

| target        | method                |   line |     brier |   log_loss |    n |   mean_predicted_over |   actual_over_rate |   calibration_error |
|:--------------|:----------------------|-------:|----------:|-----------:|-----:|----------------------:|-------------------:|--------------------:|
| strikeouts    | poisson               |    2.5 | 0.133148  |  0.427884  | 4510 |             0.818932  |          0.807761  |         0.011171    |
| strikeouts    | poisson               |    3.5 | 0.194671  |  0.572656  | 4510 |             0.668236  |          0.66541   |         0.00282566  |
| strikeouts    | poisson               |    4.5 | 0.216985  |  0.620666  | 4510 |             0.501186  |          0.500222  |         0.000964752 |
| strikeouts    | poisson               |    5.5 | 0.198811  |  0.580006  | 4510 |             0.346049  |          0.351441  |        -0.00539188  |
| strikeouts    | poisson               |    6.5 | 0.156054  |  0.480206  | 4510 |             0.221043  |          0.228603  |        -0.00756017  |
| strikeouts    | poisson               |    7.5 | 0.102424  |  0.343063  | 4510 |             0.131424  |          0.131042  |         0.000381831 |
| strikeouts    | poisson               |    8.5 | 0.0678117 |  0.245076  | 4510 |             0.0731833 |          0.0802661 |        -0.00708273  |
| strikeouts    | poisson               |    9.5 | 0.035067  |  0.143245  | 4510 |             0.0383838 |          0.0388027 |        -0.00041888  |
| strikeouts    | poisson               |   10.5 | 0.0153993 |  0.0713292 | 4510 |             0.0190555 |          0.0161863 |         0.00286926  |
| outs_recorded | conditional_empirical |   11.5 | 0.0973874 |  0.337304  | 4510 |             0.846559  |          0.846563  |        -3.82172e-06 |
| outs_recorded | conditional_empirical |   12.5 | 0.140628  |  0.446989  | 4510 |             0.789631  |          0.772949  |         0.0166822   |
| outs_recorded | conditional_empirical |   13.5 | 0.159758  |  0.492539  | 4510 |             0.739863  |          0.731486  |         0.00837765  |
| outs_recorded | conditional_empirical |   14.5 | 0.182003  |  0.54343   | 4510 |             0.670001  |          0.678714  |        -0.00871264  |
| outs_recorded | conditional_empirical |   15.5 | 0.219256  |  0.623738  | 4510 |             0.529253  |          0.496674  |         0.0325793   |
| outs_recorded | conditional_empirical |   16.5 | 0.214387  |  0.613284  | 4510 |             0.454019  |          0.433703  |         0.0203159   |
| outs_recorded | conditional_empirical |   17.5 | 0.204814  |  0.591485  | 4510 |             0.368003  |          0.363415  |         0.00458807  |
| outs_recorded | conditional_empirical |   18.5 | 0.126039  |  0.4062    | 4510 |             0.199643  |          0.15765   |         0.0419929   |
| outs_recorded | conditional_empirical |   19.5 | 0.106743  |  0.35551   | 4510 |             0.149811  |          0.131264  |         0.018547    |
| outs_recorded | conditional_empirical |   20.5 | 0.0896791 |  0.310023  | 4510 |             0.110591  |          0.10643   |         0.00416089  |
| outs_recorded | conditional_empirical |   21.5 | 0.0250248 |  0.114154  | 4510 |             0.0404349 |          0.0257206 |         0.0147143   |

## strikeouts calibration

| probability_bucket   |     n |   predicted_probability |   observed_rate |   calibration_error | target     |
|:---------------------|------:|------------------------:|----------------:|--------------------:|:-----------|
| (-0.001, 0.1]        | 15360 |               0.0324436 |       0.0333333 |        -0.000889722 | strikeouts |
| (0.1, 0.2]           |  4898 |               0.145137  |       0.147815  |        -0.00267831  | strikeouts |
| (0.2, 0.3]           |  3416 |               0.24686   |       0.244145  |         0.00271472  | strikeouts |
| (0.3, 0.4]           |  2795 |               0.348408  |       0.352415  |        -0.00400694  | strikeouts |
| (0.4, 0.5]           |  2445 |               0.449791  |       0.451125  |        -0.00133404  | strikeouts |
| (0.5, 0.6]           |  2430 |               0.548875  |       0.547325  |         0.00155015  | strikeouts |
| (0.6, 0.7]           |  2416 |               0.651834  |       0.654387  |        -0.0025535   | strikeouts |
| (0.7, 0.8]           |  2552 |               0.750201  |       0.750392  |        -0.000190809 | strikeouts |
| (0.8, 0.9]           |  2802 |               0.848787  |       0.846181  |         0.00260587  | strikeouts |
| (0.9, 1.0]           |  1476 |               0.934588  |       0.922764  |         0.0118239   | strikeouts |

## outs_recorded calibration

| probability_bucket   |    n |   predicted_probability |   observed_rate |   calibration_error | target        |
|:---------------------|-----:|------------------------:|----------------:|--------------------:|:--------------|
| (-0.001, 0.1]        | 9297 |               0.0416452 |       0.0283963 |          0.013249   | outs_recorded |
| (0.1, 0.2]           | 7391 |               0.147038  |       0.125152  |          0.0218854  | outs_recorded |
| (0.2, 0.3]           | 3308 |               0.244168  |       0.225212  |          0.0189567  | outs_recorded |
| (0.3, 0.4]           | 3617 |               0.345585  |       0.301355  |          0.0442305  | outs_recorded |
| (0.4, 0.5]           | 3533 |               0.453329  |       0.430795  |          0.0225339  | outs_recorded |
| (0.5, 0.6]           | 3849 |               0.546755  |       0.527929  |          0.0188255  | outs_recorded |
| (0.6, 0.7]           | 3661 |               0.649497  |       0.647637  |          0.00185984 | outs_recorded |
| (0.7, 0.8]           | 4926 |               0.755077  |       0.749086  |          0.00599101 | outs_recorded |
| (0.8, 0.9]           | 6642 |               0.849042  |       0.847636  |          0.00140604 | outs_recorded |
| (0.9, 1.0]           | 3386 |               0.919896  |       0.924099  |         -0.004203   | outs_recorded |
