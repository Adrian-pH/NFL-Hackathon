"""Snap-only yardage distributions from NFL tracking data.

The features in this script are calculated exclusively from the ball-snap
frame and basic game context.  In particular, it does not use PFF roles,
formation, coverage, personnel, or post-snap tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "nfl-regional-data" / "data"
QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90])


def choose_games(games: pd.DataFrame, n_games: int | None, seed: int) -> np.ndarray:
    """Choose games reproducibly, spread across the available weeks."""
    ids = np.sort(games["gameId"].unique())
    if n_games is None or n_games >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=n_games, replace=False))


def load_snap_frames(data_dir: Path, game_ids: np.ndarray) -> pd.DataFrame:
    """Return every player and ball row at the first ball-snap frame of a play."""
    columns = ["gameId", "playId", "nflId", "team", "frameId", "x", "y", "s", "a", "o", "dir", "event", "playDirection"]
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for game_id in game_ids:
        path = data_dir / "tracking" / f"tracking_{game_id}.csv"
        if not path.exists():
            missing.append(int(game_id))
            continue
        tracking = pd.read_csv(path, usecols=lambda column: column in columns)
        snaps = tracking.loc[tracking["event"].eq("ball_snap"), ["gameId", "playId", "frameId"]]
        snaps = snaps.groupby(["gameId", "playId"], as_index=False)["frameId"].min()
        frames.append(tracking.merge(snaps, on=["gameId", "playId", "frameId"], how="inner"))
    if missing:
        print(f"Skipped {len(missing)} requested games with no tracking file: {missing[:8]}")
    if not frames:
        raise FileNotFoundError(f"No tracking/tracking_*.csv files found below {data_dir}")
    return pd.concat(frames, ignore_index=True)


def normalize_snap(snap: pd.DataFrame) -> pd.DataFrame:
    """Make the offence move toward increasing x and centre y on the ball."""
    snap = snap.copy()
    left = snap["playDirection"].eq("left")
    snap.loc[left, "x"] = 120.0 - snap.loc[left, "x"]
    snap.loc[left, "y"] = 53.3 - snap.loc[left, "y"]
    for angle in ("o", "dir"):
        if angle in snap:
            snap.loc[left, angle] = (180.0 - snap.loc[left, angle]) % 360.0
    ball = snap.loc[snap["team"].eq("football"), ["gameId", "playId", "x", "y"]].drop_duplicates(["gameId", "playId"])
    ball = ball.rename(columns={"x": "ball_x", "y": "ball_y"})
    snap = snap.merge(ball, on=["gameId", "playId"], how="inner")
    snap["rel_x"] = snap["x"] - snap["ball_x"]
    snap["rel_y"] = snap["y"] - snap["ball_y"]
    return snap.loc[~snap["team"].eq("football")].copy()


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {f"{prefix}_{name}": np.nan for name in ("mean", "std", "min", "max", "q25", "q75")}
    return {
        f"{prefix}_mean": float(np.mean(values)), f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)), f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(np.quantile(values, .25)), f"{prefix}_q75": float(np.quantile(values, .75)),
    }


def graph_features(play: pd.DataFrame, possession_team: str) -> dict[str, float]:
    """Continuous team shape and cross-team distance features for one snap."""
    offence = play.loc[play["team"].eq(possession_team)]
    defence = play.loc[~play["team"].eq(possession_team)]
    result: dict[str, float] = {"offence_count": len(offence), "defence_count": len(defence)}
    for name, team in (("off", offence), ("def", defence)):
        for coordinate in ("rel_x", "rel_y", "s", "a"):
            result.update(_summary(f"{name}_{coordinate}", team[coordinate].dropna().to_numpy()))
        for angle in ("o", "dir"):
            radians = np.deg2rad(team[angle].dropna().to_numpy())
            result[f"{name}_{angle}_sin_mean"] = float(np.mean(np.sin(radians))) if len(radians) else np.nan
            result[f"{name}_{angle}_cos_mean"] = float(np.mean(np.cos(radians))) if len(radians) else np.nan
        # These are distances to the LOS, not arbitrary field bins.
        result[f"{name}_within_1yd_los"] = float((team["rel_x"].abs() <= 1).sum())
        result[f"{name}_within_3yd_los"] = float((team["rel_x"].abs() <= 3).sum())
        result[f"{name}_behind_los"] = float((team["rel_x"] < 0).sum())

    if len(offence) and len(defence):
        distances = cdist(offence[["rel_x", "rel_y"]], defence[["rel_x", "rel_y"]])
        nearest_off = distances.min(axis=1)
        nearest_def = distances.min(axis=0)
        result.update(_summary("off_nearest_defender", nearest_off))
        result.update(_summary("def_nearest_offender", nearest_def))
        result["cross_team_distance_mean"] = float(distances.mean())
        result["cross_team_pairs_under_2yd"] = float((distances < 2).sum())
        result["cross_team_pairs_under_4yd"] = float((distances < 4).sum())
    return result


def alignment_features(play: pd.DataFrame, possession_team: str, slots: int = 11) -> dict[str, float]:
    """Preserve the full snap alignment without formation, position, or PFF labels.

    Each team is ordered from the bottom to the top sideline.  This gives the
    tree model a fixed-width representation of individual players' raw
    coordinates and movement, rather than only losing them in aggregate stats.
    """
    offence = play.loc[play["team"].eq(possession_team)].sort_values("rel_y")
    defence = play.loc[~play["team"].eq(possession_team)].sort_values("rel_y")
    result: dict[str, float] = {}
    fields = ("rel_x", "rel_y", "s", "a")
    for name, team in (("off", offence), ("def", defence)):
        values = team.loc[:, fields].to_numpy(dtype=float)
        for slot in range(slots):
            for column, field in enumerate(fields):
                result[f"{name}_{field}_slot_{slot + 1:02d}"] = values[slot, column] if slot < len(values) else np.nan
        radians = np.deg2rad(team["dir"].to_numpy(dtype=float))
        for slot in range(slots):
            result[f"{name}_dir_sin_slot_{slot + 1:02d}"] = np.sin(radians[slot]) if slot < len(radians) else np.nan
            result[f"{name}_dir_cos_slot_{slot + 1:02d}"] = np.cos(radians[slot]) if slot < len(radians) else np.nan

    # For each fixed sideline-ordered player, retain its three closest opponents.
    if len(offence) and len(defence):
        distances = cdist(offence[["rel_x", "rel_y"]], defence[["rel_x", "rel_y"]])
        for name, matrix in (("off", distances), ("def", distances.T)):
            nearest = np.sort(matrix, axis=1)[:, :3]
            for slot in range(slots):
                for rank in range(3):
                    result[f"{name}_nearest_{rank + 1}_slot_{slot + 1:02d}"] = nearest[slot, rank] if slot < len(nearest) and rank < nearest.shape[1] else np.nan
    return result


def make_model_table(data_dir: Path, n_games: int | None, seed: int) -> pd.DataFrame:
    games = pd.read_csv(data_dir / "games.csv")
    plays = pd.read_csv(data_dir / "plays.csv")
    required = ["gameId", "playId", "possessionTeam", "prePenaltyPlayResult", "down", "yardsToGo", "absoluteYardlineNumber"]
    missing = set(required).difference(plays.columns)
    if missing:
        raise ValueError(f"plays.csv is missing required columns: {sorted(missing)}")
    available_ids = {
        int(path.stem.removeprefix("tracking_"))
        for path in (data_dir / "tracking").glob("tracking_*.csv")
    }
    games = games.loc[games["gameId"].isin(available_ids)]
    if games.empty:
        raise FileNotFoundError(f"No tracking files are checked out below {data_dir / 'tracking'}")
    selected = choose_games(games, n_games, seed)
    plays = plays.loc[plays["gameId"].isin(selected), required].dropna(subset=["possessionTeam", "prePenaltyPlayResult"])
    snap = normalize_snap(load_snap_frames(data_dir, selected))
    joined = snap.merge(plays[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"], how="inner")
    rows: list[dict[str, float]] = []
    for (game_id, play_id), play in joined.groupby(["gameId", "playId"], sort=False):
        possession = play["possessionTeam"].iloc[0]
        row = {"gameId": game_id, "playId": play_id}
        row.update(graph_features(play, possession))
        row.update(alignment_features(play, possession))
        rows.append(row)
    features = pd.DataFrame(rows)
    table = features.merge(plays, on=["gameId", "playId"], how="inner")
    table = table.replace([np.inf, -np.inf], np.nan)
    return table


def xgb_model(objective: str, alpha: float | None, seed: int) -> XGBRegressor:
    kwargs = dict(n_estimators=1200, max_depth=3, learning_rate=.025, subsample=.75,
                  colsample_bytree=.65, min_child_weight=25, reg_lambda=25, reg_alpha=.25,
                  gamma=.1, early_stopping_rounds=50, random_state=seed, n_jobs=-1,
                  objective=objective)
    if alpha is not None:
        kwargs["quantile_alpha"] = alpha
    return XGBRegressor(**kwargs)


def monotone_quantiles(predictions: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(predictions, axis=1)


def distribution_draws(quantile_predictions: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Interpolate quantile-XGBoost estimates into per-play Monte Carlo draws."""
    rng = np.random.default_rng(seed)
    uniforms = rng.uniform(0.01, 0.99, size=(len(quantile_predictions), n_draws))
    draws = np.empty_like(uniforms)
    for i, values in enumerate(quantile_predictions):
        iqr = max(values[3] - values[1], 1.0)
        support_q = np.r_[0.0, QUANTILES, 1.0]
        support_y = np.r_[values[0] - 1.5 * iqr, values, values[-1] + 1.5 * iqr]
        draws[i] = np.interp(uniforms[i], support_q, support_y)
    return draws


def save_distribution_plot(result: pd.DataFrame, draws: np.ndarray, output_dir: Path) -> None:
    """Save three representative, smooth yardage distributions for the held-out set."""
    percentiles = (0.20, 0.50, 0.80)
    profiles = ("Lower expected gain", "Typical expected gain", "Higher expected gain")
    chosen = [int(np.abs(result["yardage_mean"] - result["yardage_mean"].quantile(p)).argmin()) for p in percentiles]
    figure, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for axis, index, profile in zip(axes, chosen, profiles):
        row, samples = result.iloc[index], draws[index]
        x_grid = np.linspace(np.quantile(samples, .005), np.quantile(samples, .995), 400)
        density = gaussian_kde(samples, bw_method=.24)(x_grid)
        axis.plot(x_grid, density, color="#1f77b4", linewidth=2.5, label="XGBoost distribution")
        axis.fill_between(x_grid, density, color="#1f77b4", alpha=.20)
        axis.axvline(row["yardage_mean"], color="#0b3558", linewidth=2,
                    label=f"Predicted mean: {row['yardage_mean']:.1f} yd")
        axis.axvline(row["prePenaltyPlayResult"], color="#d62728", linestyle="--", linewidth=2,
                    label=f"Actual: {row['prePenaltyPlayResult']:.1f} yd")
        axis.set(title=f"{profile}\nGame {int(row['gameId'])}, play {int(row['playId'])}", xlabel="Yards gained")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Probability density")
    figure.suptitle("Raw-tracking XGBoost: held-out yardage distributions", y=1.03, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "heldout_yardage_distribution_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return

    """Save a readable predicted yardage density for one representative test play."""
    chosen = int(np.abs(result["yardage_mean"] - result["yardage_mean"].median()).argmin())
    row = result.iloc[chosen]
    samples = draws[chosen]
    figure, axis = plt.subplots(figsize=(9, 5))
    x_grid = np.linspace(np.quantile(samples, .005), np.quantile(samples, .995), 400)
    density = gaussian_kde(samples, bw_method=.24)(x_grid)
    axis.plot(x_grid, density, color="#1f77b4", linewidth=2.5, label="XGBoost distribution")
    axis.fill_between(x_grid, density, color="#1f77b4", alpha=.25)
    axis.axvline(row["yardage_p50"], color="#0b3558", linewidth=2, label=f"Median: {row['yardage_p50']:.1f} yd")
    axis.axvspan(row["yardage_p10"], row["yardage_p90"], color="#ffb000", alpha=.25,
                 label=f"10â€“90% interval: {row['yardage_p10']:.1f} to {row['yardage_p90']:.1f} yd")
    axis.axvline(row["prePenaltyPlayResult"], color="#d62728", linestyle="--", linewidth=2,
                 label=f"Actual: {row['prePenaltyPlayResult']:.1f} yd")
    axis.axvline(row["yardsToGo"], color="#333333", linestyle=":", linewidth=1.5,
                 label=f"First-down line: {row['yardsToGo']:.0f} yd")
    axis.set(title=f"Predicted pre-penalty yardage distribution\nGame {int(row['gameId'])}, play {int(row['playId'])}",
             xlabel="Yards gained", ylabel="Probability density")
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "heldout_yardage_distribution_example.png", dpi=180)
    plt.close(figure)


def plot_saved_distributions(output_dir: Path) -> None:
    """Render the PNG from previously saved draws, without refitting XGBoost."""
    result = pd.read_csv(output_dir / "heldout_yardage_distributions.csv")
    draws = np.vstack(result["distribution_samples"].map(json.loads).to_numpy())
    save_distribution_plot(result, draws, output_dir)
    print(f"Saved probability graphs to {output_dir / 'heldout_yardage_distribution_examples.png'}")


def train_and_save(table: pd.DataFrame, output_dir: Path, seed: int, n_draws: int) -> None:
    target = "prePenaltyPlayResult"
    controls = ["down", "yardsToGo", "absoluteYardlineNumber"]
    feature_columns = [column for column in table.columns if column not in {"gameId", "playId", "possessionTeam", target}]
    # Shared, universally known context is deliberately included; formation/PFF fields are not.
    assert set(controls).issubset(feature_columns)
    X = table[feature_columns].copy().fillna(table[feature_columns].median(numeric_only=True)).fillna(0)
    y = table[target].astype(float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=table["gameId"]))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed + 1)
    fit_local, validation_local = next(inner_splitter.split(X_train, y_train, groups=table.iloc[train_idx]["gameId"]))
    X_fit, X_validation = X_train.iloc[fit_local], X_train.iloc[validation_local]
    y_fit, y_validation = y_train.iloc[fit_local], y_train.iloc[validation_local]

    median_model = xgb_model("reg:quantileerror", .5, seed)
    try:
        median_model.fit(X_fit, y_fit, eval_set=[(X_validation, y_validation)], verbose=False)
    except Exception as error:
        raise RuntimeError("This script needs XGBoost 2.0+ for quantile distributions. Run: pip install -r requirements.txt") from error
    quantile_predictions = []
    for quantile in QUANTILES:
        model = median_model if quantile == .5 else xgb_model("reg:quantileerror", float(quantile), seed)
        if quantile != .5:
            model.fit(X_fit, y_fit, eval_set=[(X_validation, y_validation)], verbose=False)
        quantile_predictions.append(model.predict(X_test))
    predictions = monotone_quantiles(np.column_stack(quantile_predictions))
    draws = distribution_draws(predictions, n_draws, seed)
    median = predictions[:, 2]
    train_median = median_model.predict(X_fit)
    train_mean = float(y_fit.mean())
    train_target_median = float(y_fit.median())
    metrics = {
        "n_plays": int(len(table)), "n_train": int(len(fit_local)), "n_validation": int(len(validation_local)), "n_test": int(len(test_idx)),
        "n_train_games": int(table.iloc[train_idx[fit_local]]["gameId"].nunique()),
        "n_validation_games": int(table.iloc[train_idx[validation_local]]["gameId"].nunique()),
        "n_test_games": int(table.iloc[test_idx]["gameId"].nunique()),
        "train_mae_yards": float(mean_absolute_error(y_fit, train_median)),
        "train_rmse_yards": float(mean_squared_error(y_fit, train_median) ** .5),
        "train_r2": float(r2_score(y_fit, train_median)),
        "mae_yards": float(mean_absolute_error(y_test, median)),
        "rmse_yards": float(mean_squared_error(y_test, median) ** .5),
        "r2": float(r2_score(y_test, median)),
        "test_mean_baseline_mae": float(mean_absolute_error(y_test, np.full(len(y_test), train_mean))),
        "test_mean_baseline_rmse": float(mean_squared_error(y_test, np.full(len(y_test), train_mean)) ** .5),
        "test_median_baseline_mae": float(mean_absolute_error(y_test, np.full(len(y_test), train_target_median))),
        "test_prediction_median_mean": float(np.mean(median)),
        "test_prediction_median_std": float(np.std(median)),
        "test_target_mean": float(y_test.mean()),
        "test_target_std": float(y_test.std()),
        "p10_p90_coverage": float(((y_test.to_numpy() >= predictions[:, 0]) & (y_test.to_numpy() <= predictions[:, 4])).mean()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    result = table.iloc[test_idx][["gameId", "playId", "yardsToGo", target]].reset_index(drop=True).copy()
    for index, quantile in enumerate(QUANTILES):
        result[f"yardage_p{int(quantile * 100)}"] = predictions[:, index]
    result["yardage_mean"] = draws.mean(axis=1)
    result["yardage_std"] = draws.std(axis=1)
    result["prob_positive_yards"] = (draws > 0).mean(axis=1)
    result["prob_first_down"] = (draws >= result["yardsToGo"].to_numpy()[:, None]).mean(axis=1)
    result["distribution_samples"] = [json.dumps(np.round(row, 2).tolist()) for row in draws]
    result.to_csv(output_dir / "heldout_yardage_distributions.csv", index=False)
    save_distribution_plot(result, draws, output_dir)
    importance = pd.DataFrame({"feature": feature_columns, "gain_importance": median_model.feature_importances_}).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved held-out distributions to {output_dir / 'heldout_yardage_distributions.csv'}")
    print(f"Saved probability graphs to {output_dir / 'heldout_yardage_distribution_examples.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pre-penalty yardage distributions from snap tracking data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--games", type=int, default=20, help="Number of games; use --all-games for all available games.")
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true", help="Render the saved distribution PNG without retraining.")
    args = parser.parse_args()
    if args.plot_only:
        plot_saved_distributions(ROOT / "results")
        return
    table = make_model_table(args.data_dir, None if args.all_games else args.games, args.seed)
    print(f"Built {len(table):,} snap-only modelling rows across {table.gameId.nunique()} games.")
    train_and_save(table, ROOT / "results", args.seed, args.draws)


if __name__ == "__main__":
    main()


