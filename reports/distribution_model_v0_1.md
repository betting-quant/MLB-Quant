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
| strikeouts | poisson               | 0.12793  |   0.397534 |
| strikeouts | empirical             | 0.128129 |   0.399237 |
| strikeouts | conditional_empirical | 0.12839  |   0.399659 |
| strikeouts | negative_binomial     | 0.12877  |   0.400574 |

## Untouched 2025 holdout

- Brier score: 0.128641
- Log loss: 0.399849
- Binary line observations: 43704

## 2025 calibration

| probability_bucket   |     n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|------:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 16600 |                0.03302  |       0.0390964 |         -0.00607637 |
| (0.1, 0.2]           |  5269 |                0.145855 |       0.154489  |         -0.00863343 |
| (0.2, 0.3]           |  3704 |                0.24857  |       0.262689  |         -0.0141191  |
| (0.3, 0.4]           |  2937 |                0.349338 |       0.365339  |         -0.0160006  |
| (0.4, 0.5]           |  2719 |                0.448851 |       0.466348  |         -0.0174968  |
| (0.5, 0.6]           |  2568 |                0.550845 |       0.56581   |         -0.0149654  |
| (0.6, 0.7]           |  2606 |                0.649155 |       0.661167  |         -0.0120119  |
| (0.7, 0.8]           |  2847 |                0.750892 |       0.748507  |          0.00238499 |
| (0.8, 0.9]           |  2953 |                0.849745 |       0.83576   |          0.013985   |
| (0.9, 1.0]           |  1501 |                0.933482 |       0.92072   |          0.0127626  |

# outs_recorded

Selected distribution: `conditional_empirical`

## Development results

| target        | method                |    brier |   log_loss |
|:--------------|:----------------------|---------:|-----------:|
| outs_recorded | conditional_empirical | 0.146172 |   0.449694 |
| outs_recorded | empirical             | 0.147498 |   0.453723 |
| outs_recorded | poisson               | 0.149677 |   0.460093 |
| outs_recorded | negative_binomial     | 0.150099 |   0.46145  |

## Untouched 2025 holdout

- Brier score: 0.143070
- Log loss: 0.442149
- Binary line observations: 53416

## 2025 calibration

| probability_bucket   |     n |   predicted_probability |   observed_rate |   calibration_error |
|:---------------------|------:|------------------------:|----------------:|--------------------:|
| (-0.001, 0.1]        | 10149 |               0.0451791 |       0.0373436 |          0.0078355  |
| (0.1, 0.2]           |  7101 |               0.146217  |       0.12801   |          0.0182068  |
| (0.2, 0.3]           |  3726 |               0.244319  |       0.229737  |          0.0145822  |
| (0.3, 0.4]           |  4165 |               0.351107  |       0.327731  |          0.0233758  |
| (0.4, 0.5]           |  3458 |               0.454302  |       0.433198  |          0.0211033  |
| (0.5, 0.6]           |  4346 |               0.548926  |       0.530373  |          0.0185536  |
| (0.6, 0.7]           |  3848 |               0.648473  |       0.640333  |          0.00814077 |
| (0.7, 0.8]           |  5544 |               0.753645  |       0.75974   |         -0.00609481 |
| (0.8, 0.9]           |  8132 |               0.851421  |       0.864117  |         -0.012696   |
| (0.9, 1.0]           |  2947 |               0.926855  |       0.942314  |         -0.0154588  |
