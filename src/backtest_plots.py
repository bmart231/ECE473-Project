"""Charts for the backtest, shared by backtest_batch.py and the notebook.

setting_chart compares the rule versions (one row each, a dot per feed delay with its 95%
interval); game_chart shows one stretch of one game: trades, our quotes and our fills.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import backtest as bt
import batch_compare as bc

FIGURE_DIR = bc.ROOT / "figures" / "backtest"
INK, MUTED = "#0b0b0b", "#52514e"
TRADES, QUOTES = "#2a78d6", "#eb6834"                                   # palette slots 1, 2
DELAY_COLORS = dict(zip(bt.DELAYS, ["#2a78d6", "#eb6834", "#1baf7a"]))  # slots 1-3
VERSION_KEYS = ["width", "center", "guard"]


def _version(row) -> str:
    return f"{row['width']} width, {row['center']} center, guard {row['guard']}"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def setting_chart(summary: pd.DataFrame, column: str, xlabel: str, title: str,
                  rule: str = "strict") -> Figure:
    """One row per rule version, a dot per feed delay; error bars from column_ci95 if present."""
    table = summary.xs(rule, level="rule").reset_index()
    table["version"] = table.apply(_version, axis=1)
    versions = list(dict.fromkeys(table["version"]))        # first-seen order, i.e. settings()
    row = {v: i for i, v in enumerate(versions)}
    ci = f"{column}_ci95"

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(versions) + 1.6))
    for k, delay in enumerate(sorted(table["delay"].unique())):
        group = table[table["delay"] == delay]
        y = np.array([row[v] for v in group["version"]]) + (k - 1) * 0.22
        ax.errorbar(group[column], y, xerr=group[ci] if ci in group else None, fmt="o", ms=6,
                    lw=1.5, capsize=0, color=DELAY_COLORS.get(delay, MUTED),
                    label=f"feed delay {delay:g} s")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(versions)), versions)
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, title=f"{title} ({rule} fills, {table['games'].max()} games)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def game_chart(frame: pd.DataFrame, trades: pd.DataFrame, fills: pd.DataFrame,
               path: pd.DataFrame, minutes: tuple[float, float], title: str) -> Figure:
    """Trades, the model's price, our bid and ask, and our fills from minutes[0] to
    minutes[1] after the first play (wall clock)."""
    lo, hi = minutes
    t_trades = (trades["time"] - frame["time"].iloc[0]).dt.total_seconds() / 60
    t_rows = np.arange(len(frame)) / 60
    in_trades, in_rows = t_trades.between(lo, hi), (t_rows >= lo) & (t_rows <= hi)
    shown = fills[(fills["second"] / 60).between(lo, hi)]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(t_trades[in_trades], 100 * trades.loc[in_trades, "price"], ".", color=TRADES,
            alpha=0.35, markersize=3, label="Kalshi trades")
    ax.plot(t_rows[in_rows], frame.loc[in_rows, "p_model"], color=MUTED, linewidth=1,
            linestyle="--", label="our model (as the feed sees it)")
    for side in ("bid", "ask"):
        ax.step(t_rows[in_rows], path.loc[in_rows, side], where="post", color=QUOTES,
                linewidth=1.5, label="our bid / ask" if side == "bid" else None)
    for side, marker, name in ((1, "^", "we buy"), (-1, "v", "we sell")):
        mine = shown[shown["side"] == side]
        ax.plot(mine["second"] / 60, mine["price"], marker, color=INK, markersize=8,
                linestyle="none", label=f"{name} ({len(mine)})")
    ax.set(xlabel="minutes after the first play", ylabel="P(home wins), cents", title=title)
    ax.legend(loc="best", frameon=False, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    return fig


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- unit tests

def test_setting_chart() -> None:
    index = pd.MultiIndex.from_tuples(
        [(w, "blend", "on", d, "strict") for w in ("sensitivity", "fixed") for d in bt.DELAYS],
        names=[*VERSION_KEYS, "delay", "rule"])
    summary = pd.DataFrame({"vs_baseline": np.arange(6.0), "vs_baseline_ci95": 0.5, "games": 3},
                           index=index)
    fig = setting_chart(summary, "vs_baseline", "x", "t")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "sensitivity width, blend center, guard on", "fixed width, blend center, guard on"]
    assert len(ax.containers) == len(bt.DELAYS)                  # one dot series per delay
    plt.close(fig)


def test_game_chart() -> None:
    plays, trades = bt._game([0.50] * 100 + [0.60] * 100, noise=0.05)
    params = bt.Params(w=0, cooldown=0)
    frame = bt.game_seconds(plays, trades, 3, params)
    fills = bt.simulate(frame, trades, params)
    fig = game_chart(frame, trades, fills, bt.quote_path(frame, fills, params), (1.0, 2.0), "t")
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    in_window = fills[(fills["second"] / 60).between(1, 2)]
    assert f"we buy ({(in_window['side'] == 1).sum()})" in labels
    assert f"we sell ({(in_window['side'] == -1).sum()})" in labels
    plt.close(fig)


TESTS = [test_setting_chart, test_game_chart]


def run_unit_tests() -> None:
    matplotlib.use("Agg")
    for test in TESTS:
        test()
    print(f"unit tests (backtest_plots): all {len(TESTS)} pass")


if __name__ == "__main__":
    run_unit_tests()
