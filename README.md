# NFL Hackathon: yardage distributions

This repository contains two complementary pre-snap yardage-distribution analyses. They answer related questions with deliberately different input sources.

## Choose an analysis

| File | What it uses | What it is for |
| --- | --- | --- |
| [raw_tracking_yardage_analysis.ipynb](raw_tracking_yardage_analysis.ipynb) | **Tracking only:** player locations, speed, acceleration, direction, cross-team spacing, and shared game context. It does **not** use formation, coverage, PFF, or personnel labels. | Tests what can be predicted directly from the raw ball-snap geometry. |
| [formation_yardage_distribution.ipynb](formation_yardage_distribution.ipynb) | **Labels / formations:** offensive formation plus defensive-formation and personnel-style labels. | Tests what charted formation information says about the yardage distribution. |

The raw-tracking notebook links directly to the formation notebook and visualises the same three plays used there: game `2021090900`, plays `97`, `137`, and `187`. This makes the distribution charts easy to inspect side by side.

## Important comparison rule

The notebooks are complementary, but their headline scores are **not yet directly comparable**: they currently use different outcome fields and modelling setups. Before presenting a winner, standardise all of the following:

1. Target: use `prePenaltyPlayResult` in both notebooks.
2. Evaluation: use the same fixed held-out `gameId`s.
3. Reporting: compare MAE, RMSE, R², and predictive-interval coverage.

Until then, use the matched-play charts as a visual comparison, not as a model leaderboard.

## Data

Both analyses require the NFL regional-event tracking data locally. The raw-tracking notebook expects the data below `nfl-regional-data/data/`; its model can also be pointed at another data directory with `--data-dir`.
