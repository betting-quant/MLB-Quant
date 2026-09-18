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
| strikeouts | poisson               | 0.128353 |   0.398714 |
| strikeouts | negative_binomial     | 0.1285   |   0.399007 |
| strikeouts | empirical             | 0.128623 |   0.399954 |
| strikeouts | conditional_empirical | 0.128818 |   0.400079 |

## Untouched 2025 holdout

- Brier score: 0.128911
- Log loss: 0.400224
- Binary line observations: 43704

## 2025 calibration

| probability_bucket   |     n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|------:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 16721 |               0.0326374 |       0.0394115 |         -0.00677412 |
| (0.1, 0.2]           |  5294 |               0.146065  |       0.161692  |         -0.0156273  |
| (0.2, 0.3]           |  3615 |               0.247775  |       0.265284  |         -0.0175082  |
| (0.3, 0.4]           |  3042 |               0.348417  |       0.36785   |         -0.0194333  |
| (0.4, 0.5]           |  2634 |               0.450513  |       0.46773   |         -0.0172172  |
| (0.5, 0.6]           |  2603 |               0.549047  |       0.560891  |         -0.0118445  |
| (0.6, 0.7]           |  2602 |               0.650556  |       0.676787  |         -0.0262316  |
| (0.7, 0.8]           |  2854 |               0.750072  |       0.747022  |          0.00305053 |
| (0.8, 0.9]           |  2905 |               0.848472  |       0.833046  |          0.0154251  |
| (0.9, 1.0]           |  1434 |               0.934583  |       0.93166   |          0.00292351 |

# outs_recorded

Selected distribution: `conditional_empirical`

## Development results

| target        | method                |    brier |   log_loss |
|:--------------|:----------------------|---------:|-----------:|
| outs_recorded | conditional_empirical | 0.148626 |   0.456773 |
| outs_recorded | empirical             | 0.150031 |   0.460891 |
| outs_recorded | poisson               | 0.151901 |   0.466295 |
| outs_recorded | negative_binomial     | 0.152621 |   0.468566 |

## Untouched 2025 holdout

- Brier score: 0.145428
- Log loss: 0.449287
- Binary line observations: 53416

## 2025 calibration

| probability_bucket   |    n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|-----:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 9234 |               0.0455936 |       0.0383366 |          0.007257   |
| (0.1, 0.2]           | 7723 |               0.150341  |       0.123916  |          0.0264252  |
| (0.2, 0.3]           | 3885 |               0.243631  |       0.22677   |          0.0168615  |
| (0.3, 0.4]           | 3946 |               0.34291   |       0.321338  |          0.0215722  |
| (0.4, 0.5]           | 3511 |               0.457264  |       0.413273  |          0.0439913  |
| (0.5, 0.6]           | 4585 |               0.546649  |       0.529553  |          0.017096   |
| (0.6, 0.7]           | 4220 |               0.64852   |       0.645024  |          0.00349677 |
| (0.7, 0.8]           | 5385 |               0.755557  |       0.760446  |         -0.00488889 |
| (0.8, 0.9]           | 7807 |               0.849992  |       0.861278  |         -0.0112863  |
| (0.9, 1.0]           | 3120 |               0.921889  |       0.933333  |         -0.0114439  |
