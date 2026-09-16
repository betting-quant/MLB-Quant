# MLB Distribution Model v0.1

Distribution selection used only chronological OOF predictions.

Meta-validation:
- 2022 residual history -> evaluate 2023
- 2022-2023 residual history -> evaluate 2024
- Freeze selected distribution
- Evaluate exactly once on untouched 2025 holdout

# strikeouts

Selected distribution: `poisson`

## Development results

| target     | method                |    brier |   log_loss |
|:-----------|:----------------------|---------:|-----------:|
| strikeouts | poisson               | 0.12836  |   0.39874  |
| strikeouts | negative_binomial     | 0.128505 |   0.399026 |
| strikeouts | empirical             | 0.128634 |   0.400008 |
| strikeouts | conditional_empirical | 0.128828 |   0.400075 |

## Untouched 2025 holdout

- Brier score: 0.128797
- Log loss: 0.399863
- Binary line observations: 43704

## 2025 calibration

| probability_bucket   |     n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|------:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 16692 |               0.0325705 |        0.038821 |         -0.00625047 |
| (0.1, 0.2]           |  5339 |               0.146095  |        0.161453 |         -0.0153582  |
| (0.2, 0.3]           |  3561 |               0.247251  |        0.267621 |         -0.02037    |
| (0.3, 0.4]           |  3101 |               0.348365  |        0.363754 |         -0.015389   |
| (0.4, 0.5]           |  2583 |               0.450226  |        0.469996 |         -0.0197701  |
| (0.5, 0.6]           |  2642 |               0.54881   |        0.565481 |         -0.0166709  |
| (0.6, 0.7]           |  2587 |               0.650571  |        0.670274 |         -0.0197033  |
| (0.7, 0.8]           |  2886 |               0.750371  |        0.749827 |          0.00054444 |
| (0.8, 0.9]           |  2885 |               0.848806  |        0.835009 |          0.0137972  |
| (0.9, 1.0]           |  1428 |               0.934815  |        0.929972 |          0.00484288 |

# outs_recorded

Selected distribution: `conditional_empirical`

## Development results

| target        | method                |    brier |   log_loss |
|:--------------|:----------------------|---------:|-----------:|
| outs_recorded | conditional_empirical | 0.148661 |   0.456852 |
| outs_recorded | empirical             | 0.150055 |   0.460936 |
| outs_recorded | poisson               | 0.151919 |   0.466331 |
| outs_recorded | negative_binomial     | 0.152639 |   0.468601 |

## Untouched 2025 holdout

- Brier score: 0.145430
- Log loss: 0.449351
- Binary line observations: 53416

## 2025 calibration

| probability_bucket   |    n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|-----:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 9373 |               0.0460305 |        0.039155 |          0.00687548 |
| (0.1, 0.2]           | 7741 |               0.151563  |        0.126857 |          0.0247062  |
| (0.2, 0.3]           | 3776 |               0.243837  |        0.22696  |          0.0168774  |
| (0.3, 0.4]           | 3907 |               0.344036  |        0.32173  |          0.0223062  |
| (0.4, 0.5]           | 3588 |               0.455012  |        0.420847 |          0.0341646  |
| (0.5, 0.6]           | 4550 |               0.546505  |        0.527473 |          0.0190323  |
| (0.6, 0.7]           | 4198 |               0.648291  |        0.643878 |          0.00441258 |
| (0.7, 0.8]           | 5515 |               0.755818  |        0.759565 |         -0.00374657 |
| (0.8, 0.9]           | 7617 |               0.850284  |        0.864776 |         -0.0144924  |
| (0.9, 1.0]           | 3151 |               0.921166  |        0.933037 |         -0.0118716  |
