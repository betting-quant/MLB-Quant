# Point-in-Time Feature Engineering Report

## Output

- Dataset: `data/processed/model_features_2021_2025.parquet`
- Rows: 24,284
- Columns: 259 total
- Target columns: 4
- Feature/context columns: 255
- Date range: 2021-04-01 through 2025-09-28
- Duplicate `(season, game_id, pitcher_mlbam_id)` keys: 0
- Feature dictionary: `reports/feature_dictionary_2021_2025.csv`

Targets are preserved but excluded from feature inputs: strikeouts, outs recorded,
pitches thrown, and batters faced.

## Feature groups

The table includes pre-game pitcher season-to-date and rolling rates, workload
and rest features, pitch-type usage/velocity/whiff/strike/CSW metrics, prior
three-start pitch-mix and velocity changes versus season baseline, opponent
team and pitcher-hand splits, game context, and explicit insufficient-history
flags.

All historical aggregates are shifted or use `closed="left"` time windows. The
current appearance is excluded from every historical calculation.

## Missingness

- Total missing feature cells: 1,865,013
- Feature columns with at least one missing value: 212
- Missing values are retained rather than globally imputed.
- Expected early-history nulls include season-to-date and rolling features for
  first starts. For example, season K% and last-three K% are null for 4,844
  first appearances with no prior starts.
- `pitches_median_before_game` is null for 1,885 appearances without prior
  pitch-count history.
- Pitch-type features are null when that pitch type has not appeared in the
  pitcher's prior history; per-pitch-type insufficient-history flags are also
  provided.
- Opponent history is sparse only at the earliest games: opponent team K% is
  null for 150 rows.

## Leakage validation

Automated tests passed: **13 passed**.

The tests verify:

1. Game N's season and rolling features use only Games 1 through N-1.
2. Future-game metric mutation cannot change any prior game's feature row.
3. Early history is flagged and remains missing rather than filled with future
   or global averages.
4. Target columns remain present and are not included as input feature columns.
5. Real ingestion and scale regressions remain green.

## Recommendation

**READY FOR MODELING.** The feature table is point-in-time controlled, complete
at the requested row level, preserves both targets, and retains explicit
missingness/insufficient-history information. No model training, sportsbook
odds, or subjective adjustments were added.
