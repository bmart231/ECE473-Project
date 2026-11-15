"""Charts for the Kalshi vs model comparison, shared by batch_compare.py and the notebook.

Each function takes the per-minute table from batch_compare.load_results() and returns
one matplotlib Figure. batch_compare.py saves them all as PNGs (save_all);
notebooks/kalshi_comparison.ipynb shows them inline with the discussion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

import batch_compare as bc

FIGURE_DIR = bc.ROOT / "figures" / "kalshi_comparison"
Q = bc.QUARTER_LABELS
MODELS = [name for name in bc.PREDICTORS if name != "market"]


def per_game(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game (date, teams, priors, result), indexed by game_id."""
    return df.groupby("game_id").first()


def _finish(ax, zero_line: bool = True) -> None:
    if zero_line:
        ax.axhline(0, color="gray", linewidth=1)
    ax.grid(alpha=0.3)


def log_loss_by_quarter(df: pd.DataFrame) -> Figure:
    """Model minus market log loss per quarter, for both priors, +/-95% across games."""
    by_quarter = bc.score_table(df, by="quarter")
    diffs = by_quarter[by_quarter["value"].str.endswith("- market")]
    games_per_q = by_quarter.groupby("group")["games"].first()

    fig, ax = plt.subplots(figsize=(9, 4))
    for offset, (name, grp) in zip((-0.1, 0.1), diffs.groupby("value", sort=False)):
        ax.errorbar(grp["group"] + offset, grp["mean"], yerr=grp["ci95"], fmt="o", capsize=4, label=name)
    ax.set_xticks(list(games_per_q.index), [f"{Q[q]}\n({n} games)" for q, n in games_per_q.items()])
    ax.set(ylabel="log loss, model - market",
           title="Model minus market log loss by quarter (below 0 = model better)")
    _finish(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def calibration(df: pd.DataFrame) -> Figure:
    """Predicted vs actual home win rate for the market and both models."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect")
    for offset, (name, column) in zip((-0.01, 0, 0.01), bc.PREDICTORS.items()):
        cal = bc.calibration(df, column)
        ax.errorbar(cal[("mean", "predicted")] + offset, cal[("mean", "actual")],
                    yerr=cal[("ci95", "actual")], marker="o", capsize=3, label=name)
    ax.set(xlabel="predicted P(home wins)", ylabel="actual home win rate", xlim=(0, 1), ylim=(0, 1),
           title="Calibration: model vs market")
    ax.set_aspect("equal")
    _finish(ax, zero_line=False)
    ax.legend()
    fig.tight_layout()
    return fig


def gap(df: pd.DataFrame) -> Figure:
    """Model - market, bucketed by the market's probability (left) and by quarter (right)."""
    gap_bucket = bc.gaps(df.assign(market_bucket=bc.bucket(df["mid"])), by="market_bucket")
    gap_quarter = bc.gaps(df, by="quarter")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for offset, (name, grp) in zip((-0.008, 0.008), gap_bucket.groupby("value", sort=False)):
        left.errorbar(grp["group"] + offset, grp["mean"], yerr=grp["ci95"], marker="o", capsize=3, label=name)
    left.set(xlabel="Kalshi mid (bucket centre)", ylabel="model - market", xlim=(0, 1),
             title="Gap by how lopsided the market says the game is")

    for offset, (name, grp) in zip((-0.1, 0.1), gap_quarter.groupby("value", sort=False)):
        right.errorbar(grp["group"] + offset, grp["mean"], yerr=grp["ci95"], fmt="o", capsize=4, label=name)
    right.set_xticks(list(Q), list(Q.values()))
    right.set(xlabel="quarter", title="Gap by quarter")

    for ax in (left, right):
        _finish(ax)
    left.legend()
    fig.tight_layout()
    return fig


def sigma_sweep(df: pd.DataFrame) -> Figure:
    """Distance to market and log loss vs sigma (tip-off prior), with our fit and the best here."""
    curve = bc.sigma_curve(df)
    best = bc.best_sigma(curve).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, value in zip(axes, ["distance to market", "log loss vs result"]):
        grp = curve[curve["value"] == value]
        ax.plot(grp["sigma"], grp["mean"])
        ax.axvline(bc.SIGMA, color="gray", linestyle="--", label=f"our fit (2015-23): {bc.SIGMA}")
        ax.axvline(best[value], color="C3", linestyle=":", label=f"best here: {best[value]}")
        ax.set(xlabel="sigma", ylabel=value, title=value)
        _finish(ax, zero_line=False)
        ax.legend()
    fig.tight_layout()
    return fig


def pregame(df: pd.DataFrame) -> Figure:
    """Elo pregame probability vs Kalshi's tip-off price, one dot per game."""
    games = per_game(df)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.scatter(games["prior_tipoff"], games["prior_elo"], c=games["home_won"], cmap="coolwarm_r", s=25)
    ax.set(xlabel="Kalshi mid at tip-off", ylabel="Elo P(home wins)", xlim=(0, 1), ylim=(0, 1),
           title="Pregame probability: Elo vs market (blue = home won)")
    ax.set_aspect("equal")
    _finish(ax, zero_line=False)
    fig.tight_layout()
    return fig


def example_games(df: pd.DataFrame, n: int = 6, seed: int = 1) -> Figure:
    """Kalshi mid and both models through n randomly chosen games."""
    games = per_game(df)
    examples = games.sample(min(n, len(games)), random_state=seed).index
    cols = 3
    rows = -(-len(examples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), sharey=True, squeeze=False)
    for ax, game_id in zip(axes.flat, examples):
        g = df[df["game_id"] == game_id]
        minutes = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        ax.plot(minutes, g["mid"], label="Kalshi mid", color="black")
        for name in MODELS:
            ax.plot(minutes, g[bc.PREDICTORS[name]], label=name, drawstyle="steps-post", alpha=0.85)
        info = games.loc[game_id]
        ax.set(title=f"{info['away']} @ {info['home']}, {info['date']:%b %d %Y} "
                     f"({'home' if info['home_won'] else 'away'} won)", ylim=(0, 1))
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(examples):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("minutes since tip-off (real time)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(home wins)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


CHARTS = {
    "1_log_loss_by_quarter": log_loss_by_quarter,
    "2_calibration": calibration,
    "3_gap": gap,
    "4_sigma_sweep": sigma_sweep,
    "5_pregame_elo_vs_kalshi": pregame,
    "6_example_games": example_games,
}


def save_all(df: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> list[Path]:
    """Render every chart in CHARTS to out_dir/<name>.png and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, chart in CHARTS.items():
        fig = chart(df)
        path = out_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
