"""Charts for the lead-lag study, shared by lead_lag.py and the notebook.

Each function takes the events table from lead_lag.load_events() and returns one matplotlib
Figure. lead_lag.py saves them all as PNGs (save_all); notebooks/lead_lag.ipynb shows them
inline with the discussion.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

import lead_lag as ll

FIGURE_DIR = ll.bc.ROOT / "figures" / "lead_lag"
MIN_GAMES = 10  # a group needs this many games to get its own curve
X_LABEL = "seconds from our play timestamp"


def _finish(ax, legend: bool = True) -> None:
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="our play timestamp")
    ax.grid(alpha=0.3)
    ax.set_xlabel(X_LABEL)
    if legend:
        ax.legend()


def response(events: pd.DataFrame) -> Figure:
    """Kalshi's average move around a scoring play: in cents (left), as a share (right)."""
    curve = ll.response_curve(events)
    t50 = ll.crossing(curve["share"].to_numpy())

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    left.plot(curve["offset"], 100 * curve["move"], color="C0")
    left.fill_between(curve["offset"], 100 * curve["move_lo"], 100 * curve["move_hi"], alpha=0.25)
    left.set(ylabel="price change (cents, toward the scoring team)",
             title="Kalshi's average move around a scoring play")
    _finish(left)

    right.plot(curve["offset"], curve["share"], color="C0")
    right.fill_between(curve["offset"], curve["share_lo"], curve["share_hi"], alpha=0.25)
    right.axhline(0.5, color="C3", linestyle=":", linewidth=1)
    right.axvline(t50, color="C3", linestyle=":", label=f"half done at {t50:+.1f} s")
    right.set(ylabel="share of the move done", ylim=(-0.1, 1.1),
              title="How much of the move is done by each second")
    _finish(right)
    fig.tight_layout()
    return fig


def activity(events: pd.DataFrame) -> Figure:
    """Trades per second around a scoring play, relative to the start of the window."""
    curve = ll.activity_curve(events)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curve["offset"], curve["ratio"], color="C1")
    ax.fill_between(curve["offset"], curve["lo"], curve["hi"], color="C1", alpha=0.25)
    ax.axhline(1, color="gray", linewidth=1)
    ax.set(ylabel=f"trades per second (x the rate {ll.WINDOW} s before)",
           title="Trading activity around a scoring play")
    _finish(ax)
    fig.tight_layout()
    return fig


def by_group(events: pd.DataFrame) -> Figure:
    """Share of the move done by each second, by quarter (left) and game state (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, column, labels in [(axes[0], "quarter", ll.bc.QUARTER_LABELS), (axes[1], "game_state", None)]:
        for name, group in events.groupby(column):
            if group["game_id"].nunique() < MIN_GAMES:
                continue
            curve = ll.response_curve(group)
            t50 = ll.crossing(curve["share"].to_numpy())
            label = labels[name] if labels else name
            ax.plot(curve["offset"], curve["share"], label=f"{label} (half at {t50:+.1f} s)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set(ylim=(-0.1, 1.1), title=f"By {column.replace('_', ' ')}")
        _finish(ax)
    axes[0].set_ylabel("share of the move done")
    fig.tight_layout()
    return fig


def examples(events: pd.DataFrame, n: int = 6, seed: int = 3) -> Figure:
    """Raw Kalshi trades around n randomly chosen scoring plays, to show what is averaged."""
    games = pd.read_parquet(ll.bc.GAMES_PATH).set_index("game_id")
    picks = events.sample(min(n, len(events)), random_state=seed)
    cols = 3
    rows = -(-len(picks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), squeeze=False)
    for ax, (_, play) in zip(axes.flat, picks.iterrows()):
        trades = pd.read_parquet(ll.CACHE_DIR / f"{play['game_id']}_trades.parquet")
        seconds = (trades["time"] - play["time"]).dt.total_seconds()
        near = seconds.between(-ll.WINDOW, ll.WINDOW)
        ax.scatter(seconds[near], 100 * trades.loc[near, "price"], s=8, alpha=0.6)
        info = games.loc[play["game_id"]]
        scorer = info["home"] if play["dp"] > 0 else info["away"]
        ax.set(title=f"{info['away']} @ {info['home']}: {scorer} scores, "
                     f"{ll.bc.QUARTER_LABELS[int(play['quarter'])]} (model {100 * play['dp']:+.1f}c)",
               ylabel="price, P(home wins) (cents)", xlim=(-ll.WINDOW, ll.WINDOW))
        _finish(ax, legend=False)
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


CHARTS = {
    "1_response": response,
    "2_activity": activity,
    "3_by_group": by_group,
    "4_examples": examples,
}


def save_all(events: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Render every chart in CHARTS to out_dir/<name>.png (default: one folder per run size,
    figures/lead_lag/<games>_games/) and return the paths."""
    if out_dir is None:
        out_dir = FIGURE_DIR / f"{events['game_id'].nunique()}_games"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, chart in CHARTS.items():
        fig = chart(events)
        path = out_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
