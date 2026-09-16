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
strikeouts      hist_gradient_boosting     1.803655
                ensemble                   1.809212
                poisson                    1.851711
                ridge                      1.856663
                decomposed                 1.863957
                naive                      1.915830
batters_faced   hist_gradient_boosting     2.806025
                ensemble                   2.900540
outs_recorded   hist_gradient_boosting     2.920872
                ensemble                   2.971166
batters_faced   naive                      3.087212
                poisson                    3.089787
outs_recorded   naive                      3.111364
batters_faced   ridge                      3.115024
outs_recorded   ridge                      3.118593
                poisson                    3.130610
                opportunity_aware          3.156454
pitches_thrown  hist_gradient_boosting     9.901139
                ensemble                  10.301380
                naive                     10.862900
                ridge                     11.206564
                poisson                   11.253589

## Baseline improvement (%)

{
  "batters_faced": 9.11,
  "outs_recorded": 6.12,
  "pitches_thrown": 8.85,
  "strikeouts": 5.86
}

## Feature-family ablation

feature_family      full  opponent  pitcher_history    recent  workload
target                                                                 
batters_faced   2.806025  2.798914         3.075166  2.798657  2.811371
outs_recorded   2.920872  2.924040         3.071862  2.924041  2.928723
pitches_thrown  9.901139  9.913781        11.039997  9.903454  9.876493
strikeouts      1.803655  1.816055         1.860486  1.816008  1.845065

## 2025 holdout

        target                  model      mae      rmse      bias  correlation  poisson_deviance    n
 batters_faced hist_gradient_boosting 2.626470  3.584526  0.048168     0.584626          0.695731 4856
 outs_recorded hist_gradient_boosting 2.758535  3.644440 -0.015857     0.472951          0.986930 4856
pitches_thrown hist_gradient_boosting 9.045678 12.697560  0.075883     0.617286          2.315033 4856
    strikeouts hist_gradient_boosting 1.792230  2.259125 -0.066170     0.405513          1.163720 4856

## Controls

- No random splits.
- 2025 outcomes were not used for feature selection, model selection, preprocessing, or weighting.
- IDs, dates, and current-game targets were excluded from model inputs.
- No sportsbook odds, EV, betting optimization, or subjective adjustments were used.

## Recommendation

READY FOR DISTRIBUTION MODELING
