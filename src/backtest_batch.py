"""The backtest over many games and settings (Backtest_Plan.md, steps 2-4).

Run:  uv run python src/backtest_batch.py --download   # plays + trades: training and held-out
      uv run python src/backtest_batch.py              # the comparison, training games only

Training games are batch_compare's list (Apr 2025 - Jan 2026), cached in lead_lag's folder.
Held-out games (Feb - Jun 2026) are listed and downloaded into their own folder, and nothing
here reads them back: they are for the one final run.
"""

import argparse
import contextlib
import io
import itertools
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (no stubs)

import backtest as bt
import backtest_plots as plots
import batch_compare as bc
import downloads
import lead_lag as ll
from quoting import Params

TRAIN_DIR = ll.CACHE_DIR
HELDOUT_DIR = bc.CACHE_DIR / "heldout"
RESULTS_PATH = bc.PROCESSED / "backtest_train.parquet"  # one row per game x setting x rule

# the explainer's 2x2 plus the guard: each knob's options, and what they change in Params()
VARIANTS = {
    "width": {"sensitivity": {}, "fixed": {"b": 0}},
    "center": {"blend": {}, "market": {"w": 0}},
    "guard": {"on": {}, "off": {"cooldown": 0}},
}
BASELINE = {"width": "fixed", "center": "market", "guard": "off"}  # = backtest.BASELINE
LABELS = [*VARIANTS, "delay", "rule"]
EXAMPLE_GAME = "0042400406"       # 2025 Finals Game 6, the explainer's game
EXAMPLE_MINUTES = (56.0, 66.0)    # a stretch where the 12.9 s delay adds fills


def outdated(path: Path) -> bool:
    """True for trades cached before 11/16, which lack taker_home (read from the schema only)."""
    return path.exists() and "taker_home" not in pq.read_schema(path).names


def download(games: pd.DataFrame, folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plays and trades for every game (columns game_id, ticker) into folder, refreshing
    outdated trades. Returns (rows and trades per game, failed games)."""
    def load_one(row: dict, allowed: bool) -> pd.DataFrame:
        game_id = row["game_id"]
        refresh = outdated(folder / f"{game_id}_trades.parquet")
        plays, trades = ll.load_game(game_id, row["ticker"], refresh, allowed, folder)
        return pd.DataFrame({"game_id": [game_id], "plays": [len(plays)], "trades": [len(trades)]})

    tables, failures = downloads.for_each_game(games, load_one)
    coverage = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["game_id", "plays", "trades"])
    return coverage, failures


def settings(delays: tuple[float, ...] = bt.DELAYS) -> list[tuple[dict, Params]]:
    """Every combination of VARIANTS at every feed delay: (labels, params)."""
    grid = []
    for picks in itertools.product(*(options.items() for options in VARIANTS.values())):
        labels = dict(zip(VARIANTS, (name for name, _ in picks)))
        params = replace(Params(), **{k: v for _, change in picks for k, v in change.items()})
        grid += [(labels | {"delay": delay}, params) for delay in delays]
    return grid


def run_one(plays: pd.DataFrame, trades: pd.DataFrame, home_won: int,
            grid: list[tuple[dict, Params]], rules: tuple[str, ...] = bt.RULES) -> pd.DataFrame:
    """One game under every setting and fill rule: labels plus backtest.score's columns."""
    return pd.DataFrame([{**labels, "rule": rule,
                          **bt.run_game(plays, trades, home_won, params, labels["delay"], rule)}
                         for labels, params in grid for rule in rules])


def collect(games: pd.DataFrame, folder: Path, grid: list[tuple[dict, Params]],
            rules: tuple[str, ...] = bt.RULES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """run_one for every game cached in folder (never downloads) -> (results, failed games)."""
    def load_one(row: dict, _allowed: bool) -> pd.DataFrame:
        plays, trades = ll.load_game(row["game_id"], row["ticker"], allowed=False, folder=folder)
        return run_one(plays, trades, int(row["home_won"]), grid, rules).assign(
            game_id=row["game_id"], date=row["date"])

    tables, failures = downloads.for_each_game(games, load_one)
    return pd.concat(tables, ignore_index=True), failures


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per setting: games, mean profit per game and its 95% interval, the same paired with
    the baseline on each game (vs_baseline: the H3 test), fills per game, contract-weighted
    markout, mean drawdown and the largest position. Money in dollars per game."""
    keys = ["game_id", "delay", "rule"]
    is_base = (results[list(BASELINE)] == pd.Series(BASELINE)).all(axis=1)
    paired = results.merge(results.loc[is_base, keys + ["pnl_dollars"]], on=keys, suffixes=("", "_base"))
    paired = paired.assign(vs_baseline=paired["pnl_dollars"] - paired["pnl_dollars_base"],
                           markout_x=paired["markout_cents"].fillna(0) * paired["contracts"])
    table = paired.groupby(LABELS).agg(
        games=("game_id", "nunique"), pnl=("pnl_dollars", "mean"), pnl_sd=("pnl_dollars", "std"),
        vs_baseline=("vs_baseline", "mean"), vs_sd=("vs_baseline", "std"),
        fills=("fills", "mean"), contracts=("contracts", "sum"), markout_x=("markout_x", "sum"),
        drawdown=("drawdown_dollars", "mean"), peak_position=("peak_position", "max"))
    root_n = np.sqrt(table["games"])
    return table.assign(pnl_ci95=1.96 * table["pnl_sd"] / root_n,
                        vs_baseline_ci95=1.96 * table["vs_sd"] / root_n,
                        markout_cents=table["markout_x"] / table["contracts"]
                        ).drop(columns=["pnl_sd", "vs_sd", "markout_x", "contracts"])


# ---------------------------------------------------------------- unit tests

def _cache_game(folder: Path, game_id: str, taker: bool = True, path: list[float] | None = None,
                noise: float = 0.0) -> None:
    """bt._game(path, noise) saved where ll.load_game looks for it (default: 2 seconds)."""
    plays, trades = bt._game(path or [0.5, 0.5], noise)
    plays.to_parquet(folder / f"{game_id}_plays.parquet")
    (trades if taker else trades.drop(columns="taker_home")).to_parquet(
        folder / f"{game_id}_trades.parquet")


def test_outdated() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        _cache_game(tmp, "new")
        _cache_game(tmp, "old", taker=False)
        assert outdated(tmp / "old_trades.parquet")          # cached before taker_home
        assert not outdated(tmp / "new_trades.parquet")
        assert not outdated(tmp / "missing_trades.parquet")  # nothing to refresh
    finally:
        shutil.rmtree(tmp)


def test_download() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        _cache_game(tmp, "g1")
        with contextlib.redirect_stdout(io.StringIO()):  # hide the "1/1 games" progress line
            coverage, failures = download(pd.DataFrame({"game_id": ["g1"], "ticker": ["T"]}), tmp)
        assert coverage.to_dict("records") == [{"game_id": "g1", "plays": 1, "trades": 32}]
        assert failures.empty
    finally:
        shutil.rmtree(tmp)


def test_settings() -> None:
    grid = settings()
    assert len(grid) == 8 * len(bt.DELAYS)
    by_name = {tuple(labels[k] for k in VARIANTS): params for labels, params in grid}
    assert by_name[("sensitivity", "blend", "on")] == Params()   # the full rule
    assert by_name[tuple(BASELINE.values())] == bt.BASELINE       # plain mid-quoting
    assert by_name[("fixed", "blend", "on")] == Params(b=0)


def test_run_one() -> None:
    plays, trades = bt._game(bt.FLAT, noise=0.05)
    grid = [g for g in settings((3.0,)) if g[1] in (Params(), bt.BASELINE)]
    rows = run_one(plays, trades, 1, grid, rules=("strict",))
    pnl = rows.set_index("width")["pnl_dollars"]
    # full rule quotes 48/52 (as in backtest's test); the baseline's 1c quotes are 49/51:
    # 599 s x (2c x 10 contracts - two 5c fees) = $59.90
    assert np.isclose(pnl["sensitivity"], 179.70) and np.isclose(pnl["fixed"], 59.90)
    assert set(rows.columns) >= set(LABELS) and (rows["rule"] == "strict").all()


def test_collect() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        _cache_game(tmp, "g1", path=bt.FLAT, noise=0.05)
        games = pd.DataFrame({"game_id": ["g1", "g2"], "ticker": "T", "home_won": 1,
                              "date": pd.Timestamp("2026-01-01")})
        grid = [g for g in settings((3.0,)) if g[1] == Params()]
        with contextlib.redirect_stdout(io.StringIO()):
            results, failures = collect(games, tmp, grid, rules=("strict",))
        assert results["game_id"].tolist() == ["g1"] and np.isclose(results["pnl_dollars"].item(), 179.70)
        # g2 isn't cached: it fails without touching the network
        assert failures["game_id"].tolist() == ["g2"] and "not downloaded" in failures["error"].item()
    finally:
        shutil.rmtree(tmp)


def test_summarize() -> None:
    full = dict(width="sensitivity", center="blend", guard="on", delay=12.9, rule="strict")
    rows = [dict(BASELINE, delay=12.9, rule="strict", game_id=g, pnl_dollars=pnl, contracts=0.0,
                 markout_cents=np.nan, fills=0, drawdown_dollars=0.0, peak_position=0.0)
            for g, pnl in [("a", 1.0), ("b", 3.0)]]
    rows += [dict(full, game_id=g, pnl_dollars=pnl, contracts=c, markout_cents=m, fills=f,
                  drawdown_dollars=1.0, peak_position=p)
             for g, pnl, c, m, f, p in [("a", 2.0, 10.0, 1.0, 1, 10.0), ("b", 6.0, 30.0, 3.0, 3, 30.0)]]
    table = summarize(pd.DataFrame(rows))
    row = table.loc[tuple(full[k] for k in LABELS)]
    assert row["games"] == 2 and row["pnl"] == 4 and row["fills"] == 2
    # paired with the baseline game by game: differences 1 and 3 -> 2 +- 1.96 * sd / sqrt(2)
    assert row["vs_baseline"] == 2 and np.isclose(row["vs_baseline_ci95"], 1.96)
    assert row["markout_cents"] == 2.5                                 # contract-weighted
    assert row["peak_position"] == 30 and row["drawdown"] == 1
    assert table.loc[tuple(dict(BASELINE, delay=12.9, rule="strict")[k] for k in LABELS),
                     "vs_baseline"] == 0


TESTS = [test_outdated, test_download, test_settings, test_run_one, test_collect, test_summarize]


def run_unit_tests() -> None:
    for test in TESTS:
        test()
    print(f"unit tests (backtest_batch): all {len(TESTS)} pass")


def download_all() -> None:
    """Step 2: plays and trades for the training games and the held-out games."""
    lists = [(bc.build_game_list(), TRAIN_DIR, "training"),
             (bc.build_game_list(window=bc.HELDOUT_WINDOW, seasons=bc.HELDOUT_SEASONS,
                                 path=bc.HELDOUT_PATH), HELDOUT_DIR, "held-out")]
    for games, folder, name in lists:
        print(f"{name}: {len(games)} games, {games['date'].min():%Y-%m-%d} to "
              f"{games['date'].max():%Y-%m-%d}")
        coverage, failures = download(games, folder)
        print(f"{name}: {len(coverage)} downloaded, {len(failures)} failed")
        if not failures.empty:
            print(failures.to_string(index=False))


def save_charts(summary: pd.DataFrame, example: pd.Series) -> list[Path]:
    """The comparison charts, and the example game under the full rule at each delay."""
    folder = plots.FIGURE_DIR / f"{summary['games'].max()}_games"
    paths = [plots.save(plots.setting_chart(summary, column, xlabel, title, rule),
                        folder / f"{column}_{rule}.png")
             for column, xlabel, title in [
                 ("vs_baseline", "profit per game minus the baseline's, same game ($)",
                  "Each version vs plain mid-quoting"),
                 ("pnl", "profit per game after fees ($)", "Profit per game")]
             for rule in bt.RULES]
    paths.append(plots.save(plots.setting_chart(summary, "markout_cents", "average 60 s markout "
                                                "(cents per contract; < 0 = picked off)",
                                                "Adverse selection"), folder / "markout.png"))
    plays, trades = ll.load_game(str(example["game_id"]), str(example["ticker"]), allowed=False,
                                 folder=TRAIN_DIR)
    for delay in (bt.DELAYS[0], bt.DELAYS[-1]):
        frame = bt.game_seconds(plays, trades, delay, Params())
        fills = bt.simulate(frame, trades, Params())
        title = (f"{example['away']} @ {example['home']}, {example['date']:%Y-%m-%d}: full rule, "
                 f"feed delay {delay:g} s")
        fig = plots.game_chart(frame, trades, fills, bt.quote_path(frame, fills, Params()),
                               EXAMPLE_MINUTES, title)
        paths.append(plots.save(fig, folder / f"example_delay_{delay:g}.png"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the quoting rule over many games.")
    parser.add_argument("--download", action="store_true",
                        help="step 2: download plays and trades (training and held-out)")
    parser.add_argument("--saved", action="store_true",
                        help="summarize the saved results instead of running the games again")
    args = parser.parse_args()

    bt.run_unit_tests()
    run_unit_tests()
    if args.download:
        download_all()
        return

    games = bc.build_game_list()
    if not args.saved:
        results, failures = collect(games, TRAIN_DIR, settings())
        results.to_parquet(RESULTS_PATH)
        print(f"saved {RESULTS_PATH} ({results['game_id'].nunique()} games, {len(results):,} rows)")
        if not failures.empty:
            print(f"{len(failures)} games failed:\n{failures['error'].value_counts().to_string()}")
    summary = summarize(pd.read_parquet(RESULTS_PATH))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(2).to_string())
    example = games.set_index("game_id").loc[EXAMPLE_GAME].rename(EXAMPLE_GAME)
    paths = save_charts(summary, pd.concat([example, pd.Series({"game_id": EXAMPLE_GAME})]))
    print(f"saved {len(paths)} charts to {paths[0].parent}")


if __name__ == "__main__":
    main()
