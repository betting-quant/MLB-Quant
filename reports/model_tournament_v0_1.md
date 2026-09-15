# MLB Model Tournament v0.1

## Protocol

2025 was held out untouched. Model selection used only chronological validation: train 2021 -> validate 2022, train 2021-2022 -> validate 2023, and train 2021-2023 -> validate 2024.

## Selected models

- batters_faced: `hist_gradient_boosting`
- outs_recorded: `hist_gradient_boosting`
- pitches_thrown: `hist_gradient_boosting`
- strikeouts: `ensemble`

## Walk-forward results

target          model                 
strikeouts      ensemble                   1.801125
                hist_gradient_boosting     1.803669
                ridge                      1.838492
                poisson                    1.839747
                decomposed                 1.845606
                naive                      1.915830
batters_faced   hist_gradient_boosting     2.774514
                ensemble                   2.866954
outs_recorded   hist_gradient_boosting     2.872331
                ensemble                   2.911657
                poisson                    3.018426
                ridge                      3.035432
batters_faced   poisson                    3.069667
outs_recorded   opportunity_aware          3.078712
batters_faced   ridge                      3.085113
                naive                      3.087212
outs_recorded   naive                      3.111364
pitches_thrown  hist_gradient_boosting     9.846806
                ensemble                  10.188531
                naive                     10.862900
                ridge                     11.055982
                poisson                   11.061360

## Baseline improvement (%)

{
  "batters_faced": 10.13,
  "outs_recorded": 7.68,
  "pitches_thrown": 9.35,
  "strikeouts": 5.99
}

## Feature-family ablation

feature_family      full  opponent  pitcher_history    recent  workload
target                                                                 
batters_faced   2.774514  2.772419         3.075166  2.763254  2.811371
outs_recorded   2.872331  2.882082         3.071862  2.877466  2.928723
pitches_thrown  9.846806  9.882294        11.039997  9.843727  9.876493
strikeouts      1.803669  1.809260         1.860486  1.811013  1.845065

## 2025 holdout

        target                  model      mae      rmse      bias  correlation  poisson_deviance    n
 batters_faced hist_gradient_boosting 2.609919  3.552255  0.026170     0.594405          0.684674 4856
 outs_recorded hist_gradient_boosting 2.722832  3.583677 -0.040928     0.499450          0.959919 4856
pitches_thrown hist_gradient_boosting 9.050750 12.683976  0.016938     0.618169          2.307925 4856
    strikeouts               ensemble 1.791880  2.257565 -0.044063     0.406572          1.163025 4856

## Controls

- No random splits.
- 2025 outcomes were not used for feature selection, model selection, preprocessing, or weighting.
- IDs, dates, and current-game targets were excluded from model inputs.
- No sportsbook odds, EV, betting optimization, or subjective adjustments were used.

## Recommendation

READY FOR DISTRIBUTION MODELING
