# Historical Statcast Scale Report

## Scope

- Seasons: 2021, 2022, 2023, 2024, 2025 regular seasons
- Chunk size: 7 days
- Source: `pybaseball.statcast`
- Raw layout: `data/raw/season=YYYY/statcast_<start>_<end>.parquet`
- 2026 data: not downloaded and not included in the scale CLI

## Season results

| Season | Raw pitch rows | Games | Starter appearances | Unique starters | Missing critical fields | Duplicate records | Failed checks | Two-starter games | Unusual/incomplete games |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 2021 | 712,320 | 2,429 | 4,858 | 396 | None | 0 | None | 2,429 | None |
| 2022 | 710,210 | 2,430 | 4,860 | 367 | None | 0 | None | 2,430 | None |
| 2023 | 720,684 | 2,430 | 4,860 | 383 | None | 0 | None | 2,430 | None |
| 2024 | 710,632 | 2,425 | 4,850 | 370 | None | 0 | None | 2,425 | None |
| 2025 | 711,897 | 2,428 | 4,856 | 369 | None | 0 | None | 2,428 | None |
| **Total** | **3,565,743** | **12,142** | **24,284** | **850 across all seasons** | **None** | **0** | **None** | **12,142** | **None** |

The raw source naturally contains missing noncritical values on non-terminal
pitch rows: `events` and `pitch_type`. These were reported but did not fail the
critical integrity gate. Critical identifiers, dates, inning sides, innings,
pitch numbers, and required schema fields passed validation for every season.

## Outputs

- `data/processed/starting_pitcher_games_2021.parquet`
- `data/processed/starting_pitcher_games_2022.parquet`
- `data/processed/starting_pitcher_games_2023.parquet`
- `data/processed/starting_pitcher_games_2024.parquet`
- `data/processed/starting_pitcher_games_2025.parquet`
- `data/processed/starting_pitcher_games_2021_2025.parquet`
- `data/processed/starting_pitcher_scale_report_2021_2025.json`

The consolidated table contains 24,284 rows and preserves game IDs, pitcher
MLBAM IDs, team/opponent, dates, starter metrics, source-row counts, and the
outs derivation method for later joins to pitch-level, batter, team, and future
market data.

## Validation

- Existing and scale tests: 10 passed
- Raw chunks: 134 total, partitioned by season and date
- Raw and large processed artifacts: excluded by `.gitignore`
- No 2026 paths or data created
- No predictive features, models, or sportsbook data added

## Recommendation

**READY FOR FEATURE ENGINEERING PENDING USER APPROVAL.** The historical table
passed the scale integrity gates, with exactly two identified starters for every
normal ingested game and no unusual/incomplete games silently forced into the
master table. Do not begin feature engineering until the user approves this
report.
