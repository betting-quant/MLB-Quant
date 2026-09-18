# MLB Final Calibration Audit v0.1

Untouched 2025 holdout only.

## Sportsbook-relevant lines

| target        |    line |    n |   predicted |   observed |   calibration_error |
|:--------------|--------:|-----:|------------:|-----------:|--------------------:|
| strikeouts    |  3.5000 | 4856 |      0.6668 |     0.6748 |             -0.0080 |
| strikeouts    |  4.5000 | 4856 |      0.4960 |     0.5173 |             -0.0213 |
| strikeouts    |  5.5000 | 4856 |      0.3390 |     0.3598 |             -0.0208 |
| strikeouts    |  6.5000 | 4856 |      0.2141 |     0.2298 |             -0.0157 |
| strikeouts    |  7.5000 | 4856 |      0.1259 |     0.1419 |             -0.0160 |
| outs_recorded | 14.5000 | 4856 |      0.6833 |     0.6973 |             -0.0139 |
| outs_recorded | 15.5000 | 4856 |      0.5433 |     0.5056 |              0.0377 |
| outs_recorded | 16.5000 | 4856 |      0.4658 |     0.4446 |              0.0212 |
| outs_recorded | 17.5000 | 4856 |      0.3794 |     0.3783 |              0.0011 |
| outs_recorded | 18.5000 | 4856 |      0.2085 |     0.1637 |              0.0448 |

## Betting probability buckets

| target        | bucket   |     n |   predicted |   observed |   calibration_error |
|:--------------|:---------|------:|------------:|-----------:|--------------------:|
| outs_recorded | 55-60%   |  2230 |      0.5692 |     0.5435 |              0.0257 |
| outs_recorded | 60-65%   |  2024 |      0.6215 |     0.6082 |              0.0133 |
| outs_recorded | 65-70%   |  2196 |      0.6734 |     0.6790 |             -0.0056 |
| outs_recorded | 70%+     | 16312 |      0.8326 |     0.8418 |             -0.0092 |
| strikeouts    | 55-60%   |  1275 |      0.5744 |     0.5788 |             -0.0044 |
| strikeouts    | 60-65%   |  1255 |      0.6243 |     0.6534 |             -0.0291 |
| strikeouts    | 65-70%   |  1347 |      0.6750 |     0.6986 |             -0.0235 |
| strikeouts    | 70%+     |  7193 |      0.8266 |     0.8186 |              0.0080 |
