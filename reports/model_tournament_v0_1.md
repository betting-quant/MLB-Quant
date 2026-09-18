# MLB Model Tournament v0.1

## Protocol

2025 was held out untouched. Model selection used only chronological validation: train 2021 -> validate 2022, train 2021-2022 -> validate 2023, and train 2021-2023 -> validate 2024.

## Selected models

- batters_faced: `hist_gradient_boosting`
- outs_recorded: `hist_gradient_boosting`
- pitches_thrown: `hist_gradient_boosting`
- strikeouts: `hist_gradient_boosting`

## Walk-forward results

target          model                 
strikeouts      hist_gradient_boosting     1.803689
                ensemble                   1.809261
                poisson                    1.851584
                ridge                      1.856651
                decomposed                 1.863940
                naive                      1.915830
batters_faced   hist_gradient_boosting     2.806021
                ensemble                   2.900533
outs_recorded   hist_gradient_boosting     2.920697
                ensemble                   2.971155
batters_faced   naive                      3.087212
                poisson                    3.089855
outs_recorded   naive                      3.111364
batters_faced   ridge                      3.115019
outs_recorded   ridge                      3.118725
                poisson                    3.130801
                opportunity_aware          3.156562
pitches_thrown  hist_gradient_boosting     9.901140
                ensemble                  10.301239
                naive                     10.862900
                ridge                     11.206440
                poisson                   11.253867

## Baseline improvement (%)

{
  "batters_faced": 9.11,
  "outs_recorded": 6.13,
  "pitches_thrown": 8.85,
  "strikeouts": 5.85
}

## Feature-family ablation

feature_family      full  opponent  pitcher_history    recent  workload
target                                                                 
batters_faced   2.806021  2.798914         3.075166  2.798657  2.811371
outs_recorded   2.920697  2.924040         3.071862  2.924041  2.928723
pitches_thrown  9.901140  9.913781        11.039997  9.903454  9.876493
strikeouts      1.803689  1.816055         1.860486  1.816008  1.845065

## 2025 holdout

        target                  model      mae      rmse      bias  correlation  poisson_deviance    n
 batters_faced hist_gradient_boosting 2.626476  3.584528  0.048196     0.584626          0.695732 4856
 outs_recorded hist_gradient_boosting 2.755862  3.643853 -0.004151     0.473199          0.986712 4856
pitches_thrown hist_gradient_boosting 9.043843 12.699953  0.108048     0.617017          2.314283 4856
    strikeouts hist_gradient_boosting 1.793644  2.260710 -0.066601     0.404126          1.165242 4856

## Controls

- No random splits.
- 2025 outcomes were not used for feature selection, model selection, preprocessing, or weighting.
- IDs, dates, and current-game targets were excluded from model inputs.
- No sportsbook odds, EV, betting optimization, or subjective adjustments were used.

## Recommendation

READY FOR DISTRIBUTION MODELING
