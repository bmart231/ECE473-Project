"""Does Kalshi move before or after our play-by-play timestamps? (background: LeadLag_Plan.md)

Run:  uv run python src/lead_lag.py               # first 150 games of the batch sample
      uv run python src/lead_lag.py --n 300       # a superset of the 150 (same seed)

Needs batch_compare.py to have run first: it supplies the game list and each game's
Kalshi tip-off price. Writes data/processed/lead_lag_events.parquet (one row per scoring
play used, with the Kalshi price and trade count at every second around it) and the charts
from lead_lag_plots.py to figures/lead_lag/<games>_games/.
Discussion and conclusions: notebooks/lead_lag.ipynb (reads the parquet, no API calls).

Method, an event study:
1. Scoring plays come from the NBA live feed. Only "isolated" ones are used: no other score
   change within ISOLATION seconds either side, so each window holds a single event.
2. Around each play: the last Kalshi trade price at every second from -WINDOW to +WINDOW
   of its timestamp, and the number of trades in each of those seconds.
3. Price changes are measured from -WINDOW and signed so the scoring team's direction is
   up. Averaged per game, then across games with each game counted once, they give the
   market's response curve. Dividing by the full move gives the share done by each second:
   if half is done before 0, the market moves before our timestamp.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import batch_compare as bc
import downloads
import live_prototype as lp
import stern

CACHE_DIR = bc.CACHE_DIR / "lead_lag"                     # per-game plays and trades (git-ignored)
EVENTS_PATH = bc.PROCESSED / "lead_lag_events.parquet"    # the result
TRADES_URL = f"{lp.g6.KALSHI_BASE}/historical/trades"

N_GAMES = 150
SIGMA = 22.0          # the sigma Kalshi prices at (notebooks/kalshi_comparison.ipynb), for model jumps
WINDOW = 20           # seconds either side of a play
ISOLATION = 20        # no other score change this close (seconds) on either side
OFFSETS = np.arange(-WINDOW, WINDOW + 1)                  # seconds relative to the play
PRICE_RANGE = (0.03, 0.97)  # price before the play must leave room to move
TRADE_PAD = pd.Timedelta(minutes=2)                       # trades fetched this far past the plays
CLOSE_GAME = (0.3, 0.7)     # price before the play in this range -> "close game"
N_BOOT = 1000               # bootstrap resamples (of games) for intervals


# ---------- 1. data ----------

def fetch_trades(ticker: str, start: pd.Timestamp, end: pd.Timestamp, url: str = TRADES_URL) -> list[dict]:
    """Every Kalshi trade in ticker between start and end (pages of 1000, newest first).
    Settled games are under /historical/trades (the default); recent ones under /markets/trades."""
    params = {"ticker": ticker, "limit": 1000,
              "min_ts": int(start.timestamp()), "max_ts": int(end.timestamp())}
    trades = []
    while True:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            response.raise_for_status()  # throttled or down: temporary, retried by downloads
        if not response.ok:
            raise ValueError(f"Kalshi trades {response.status_code}: {response.text[:200]}")
        body = response.json()
        trades += body["trades"]
        if not body.get("cursor") or not body["trades"]:
            return trades
        params["cursor"] = body["cursor"]


def trades_to_frame(trades: list[dict], home: bool) -> pd.DataFrame:
    """Raw trades -> columns time (UTC), price (as P(home wins)), contracts, taker_home
    (the taker bought the home team); oldest first."""
    if not trades:
        raise ValueError("no Kalshi trades in the game window")
    price = pd.to_numeric(pd.Series([t["yes_price_dollars"] for t in trades]))
    taker_yes = pd.Series([t["taker_side"] == "yes" for t in trades])
    frame = pd.DataFrame({
        "time": pd.to_datetime([t["created_time"] for t in trades], utc=True, format="ISO8601"),
        "price": price if home else 1 - price,
        "contracts": pd.to_numeric(pd.Series([t["count_fp"] for t in trades])),
        "taker_home": taker_yes if home else ~taker_yes,
    })
    return frame.sort_values("time", kind="stable").reset_index(drop=True)


def scoring_events(plays: pd.DataFrame) -> pd.DataFrame:
    """Every score change in a game, from live_prototype.load_plays, in feed order.

    Columns: time, period, secs_left, margin_before, margin, in_order. in_order is False
    where the timestamp steps back behind an earlier play (a hand-logging slip): those plays
    still count when checking isolation, but their own timing can't be trusted.
    """
    before = plays["margin"].shift()
    changed = plays["margin"].ne(before) & before.notna()
    events = plays.assign(margin_before=before, in_order=plays["time"] >= plays["time"].cummax())
    columns = ["time", "period", "secs_left", "margin_before", "margin", "in_order"]
    return events.loc[changed, columns].reset_index(drop=True)


def usable(scores: pd.DataFrame, isolation: float = ISOLATION) -> pd.Series:
    """Score changes we can time: in order, before the buzzer, with no other score change
    within isolation seconds either side."""
    gap = scores["time"].diff().dt.total_seconds()
    isolated = (gap.fillna(np.inf) > isolation) & (gap.shift(-1).fillna(np.inf) > isolation)
    return scores["in_order"] & (scores["secs_left"] > 0) & isolated


def price_paths(times: pd.Series, trades: pd.DataFrame, offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """Last trade price at or before time + offset seconds, for every time (rows) and offset
    (columns). NaN where there is no earlier trade."""
    query = times.to_numpy("datetime64[ns]")[:, None] + offsets.astype("timedelta64[s]")
    index = np.searchsorted(trades["time"].to_numpy("datetime64[ns]"), query, side="right") - 1
    prices = trades["price"].to_numpy()[np.maximum(index, 0)]
    return np.where(index >= 0, prices, np.nan)


def trade_counts(times: pd.Series, trades: pd.DataFrame, offsets: np.ndarray = OFFSETS) -> np.ndarray:
    """Number of trades in [time + offset, time + offset + 1 s) for every time and offset."""
    start = times.to_numpy("datetime64[ns]")[:, None] + offsets.astype("timedelta64[s]")
    trade_times = trades["time"].to_numpy("datetime64[ns]")
    after = np.searchsorted(trade_times, start + np.timedelta64(1, "s"), side="left")
    return after - np.searchsorted(trade_times, start, side="left")


def game_events(game_id: str, scores: pd.DataFrame, trades: pd.DataFrame, p_pre: float,
                sigma: float = SIGMA) -> pd.DataFrame:
    """One row per usable scoring play: game state, the model's jump dp, and the Kalshi price
    (price_<k>) and trade count (trades_<k>) at each offset k in OFFSETS."""
    events = scores[usable(scores)].reset_index(drop=True)
    secs = events["secs_left"].to_numpy()
    dp = (stern.stern_p(events["margin"].to_numpy(), secs, p_pre, sigma)
          - stern.stern_p(events["margin_before"].to_numpy(), secs, p_pre, sigma))
    prices = price_paths(events["time"], trades)
    counts = trade_counts(events["time"], trades)

    table = pd.concat([
        events.drop(columns="in_order").assign(game_id=game_id, dp=dp),
        pd.DataFrame(prices, columns=[f"price_{k}" for k in OFFSETS]),
        pd.DataFrame(counts, columns=[f"trades_{k}" for k in OFFSETS]),
    ], axis=1)
    start = table[f"price_{-WINDOW}"]
    return table[start.between(*PRICE_RANGE)].reset_index(drop=True)


def load_game(game_id: str, ticker: str, refresh: bool = False, allowed: bool = True,
              folder: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every play, trades) for one game, each cached in folder. Also the backtest's loader."""
    plays = downloads.cached(folder / f"{game_id}_plays.parquet",
                             lambda: lp.load_plays(game_id), refresh, allowed)
    start, end = plays["time"].min() - TRADE_PAD, plays["time"].max() + TRADE_PAD
    trades = downloads.cached(
        folder / f"{game_id}_trades.parquet",
        lambda: trades_to_frame(fetch_trades(ticker, start, end), lp.ticker_is_home(ticker)),
        refresh, allowed)
    return plays, trades


def tipoff_price(game_id: str) -> float:
    """Kalshi's tip-off price for the game, from batch_compare's cache."""
    path = bc.CACHE_DIR / f"{game_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run batch_compare.py on this game first")
    return float(pd.read_parquet(path)["prior_tipoff"].iloc[0])


def collect(games: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """game_events for every game -> (all events, failed games)."""
    def load_one(row: dict, allowed: bool) -> pd.DataFrame:
        plays, trades = load_game(row["game_id"], row["ticker"], refresh, allowed)
        return game_events(row["game_id"], scoring_events(plays), trades, tipoff_price(row["game_id"]))

    tables, failures = downloads.for_each_game(games, load_one)
    return pd.concat(tables, ignore_index=True), failures


def load_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    """The saved events with quarter and game-state labels added."""
    events = pd.read_parquet(path)
    start = events[f"price_{-WINDOW}"]
    return events.assign(
        quarter=events["period"].clip(upper=5),
        game_state=np.where(start.between(*CLOSE_GAME), "close game", "lopsided game"),
    )


# ---------- 2. analysis ----------

def matrix(events: pd.DataFrame, prefix: str) -> np.ndarray:
    """The <prefix>_<k> columns as an (events x OFFSETS) array."""
    return events[[f"{prefix}_{k}" for k in OFFSETS]].to_numpy(dtype=float)


def per_game(events: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Mean of values over each game's events -> (games x OFFSETS); every game counts once."""
    return pd.DataFrame(values).groupby(events["game_id"].to_numpy()).mean().to_numpy()


def signed_moves(events: pd.DataFrame) -> np.ndarray:
    """Price change since -WINDOW at each offset, signed so the scoring team's way is up."""
    prices = matrix(events, "price")
    return np.sign(events["dp"].to_numpy())[:, None] * (prices - prices[:, :1])


def crossing(share: np.ndarray, level: float = 0.5) -> float:
    """First offset (seconds, interpolated) where share reaches level; NaN if it never does."""
    above = np.flatnonzero(share >= level)
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(OFFSETS[0])
    x0, x1, y0, y1 = OFFSETS[i - 1], OFFSETS[i], share[i - 1], share[i]
    return float(x0 + (level - y0) / (y1 - y0) * (x1 - x0))


def _bootstrap(game_rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """Mean over games for N_BOOT resamples of the games -> (N_BOOT x OFFSETS)."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(game_rows), size=(N_BOOT, len(game_rows)))
    return np.nanmean(game_rows[picks], axis=1)


def response_curve(events: pd.DataFrame) -> pd.DataFrame:
    """The market's average response around a play, with 95% intervals across games.

    Columns: offset, move (signed price change since -WINDOW, in probability), share (move
    as a share of the full move at +WINDOW), and lo/hi for each.
    """
    games = per_game(events, signed_moves(events))
    move, boot = np.nanmean(games, axis=0), _bootstrap(games)
    share, boot_share = move / move[-1], boot / boot[:, -1:]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    share_lo, share_hi = np.percentile(boot_share, [2.5, 97.5], axis=0)
    return pd.DataFrame({"offset": OFFSETS, "move": move, "move_lo": lo, "move_hi": hi,
                         "share": share, "share_lo": share_lo, "share_hi": share_hi})


def activity_curve(events: pd.DataFrame) -> pd.DataFrame:
    """Trades per second around a play, relative to the first 5 seconds of the window."""
    games = per_game(events, matrix(events, "trades"))
    rate = games / games[:, :5].mean(axis=1, keepdims=True).clip(min=1e-9)
    boot = _bootstrap(rate, seed=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"offset": OFFSETS, "ratio": np.nanmean(rate, axis=0), "lo": lo, "hi": hi,
                         "per_second": np.nanmean(games, axis=0)})


def summary(events: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Headline numbers per group of `by`: games, plays, the model's average jump and the
    market's full move (cents), the share of that move done by our timestamp, and the second
    at which half is done (negative = before our timestamp), with 95% intervals."""
    rows = []
    groups = events.groupby(by) if by else [("all", events)]
    zero = int(np.flatnonzero(OFFSETS == 0)[0])
    for name, group in groups:
        games = per_game(group, signed_moves(group))
        if len(games) < 2:
            continue
        move, boot = np.nanmean(games, axis=0), _bootstrap(games)
        boot_share = boot / boot[:, -1:]
        boot_t50 = np.array([crossing(s) for s in boot_share])
        rows.append({
            "group": name, "games": len(games), "plays": len(group),
            "model jump (c)": 100 * np.abs(group["dp"]).mean(),
            "market move (c)": 100 * move[-1],
            "share by 0": move[zero] / move[-1],
            "share by 0 lo": np.percentile(boot_share[:, zero], 2.5),
            "share by 0 hi": np.percentile(boot_share[:, zero], 97.5),
            "half done at (s)": crossing(move / move[-1]),
            "half lo": np.nanpercentile(boot_t50, 2.5),
            "half hi": np.nanpercentile(boot_t50, 97.5),
        })
    return pd.DataFrame(rows).set_index("group")


# ---------- checks ----------

def run_unit_tests() -> None:
    """Offline checks on hand-made plays and trades with a known answer (no network)."""
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")

    def at(seconds) -> pd.Series:
        return pd.Series(t0 + pd.to_timedelta(seconds, unit="s"))

    # trades_to_frame: dollar strings -> floats, away ticker flipped, oldest first
    raw = [{"created_time": "2026-01-01T00:00:02.5Z", "yes_price_dollars": "0.6000", "count_fp": "3.00",
            "taker_side": "yes"},
           {"created_time": "2026-01-01T00:00:01Z", "yes_price_dollars": "0.5500", "count_fp": "1.00",
            "taker_side": "no"}]
    home, away = trades_to_frame(raw, home=True), trades_to_frame(raw, home=False)
    assert home["price"].tolist() == [0.55, 0.60] and home["contracts"].tolist() == [1, 3]
    assert np.allclose(away["price"], [0.45, 0.40])
    assert home["taker_home"].tolist() == [False, True]   # a YES taker on the away ticker
    assert away["taker_home"].tolist() == [True, False]   # is selling the home team
    assert lp.g6._raises(ValueError, trades_to_frame, [], True)

    # load_game: reads both cached tables; with downloads off, a missing one is an error
    tmp = Path(tempfile.mkdtemp())
    try:
        plays = pd.DataFrame({"time": at([0, 10]), "period": 1, "margin": [0, 2],
                              "secs_left": [2880.0, 2870.0]})
        plays.to_parquet(tmp / "g_plays.parquet")
        home.to_parquet(tmp / "g_trades.parquet")
        got_plays, got_trades = load_game("g", "T", allowed=False, folder=tmp)
        assert got_plays.equals(plays) and got_trades.equals(home)
        assert lp.g6._raises(RuntimeError, load_game, "other", "T", False, False, tmp)
    finally:
        shutil.rmtree(tmp)

    # scoring_events + usable: free-throw pair too close together, a timestamp that steps
    # back, and a buzzer-beater are all excluded; the isolated basket is kept
    plays = pd.DataFrame({
        "time": at([0, 50, 100, 108, 200, 195, 300, 400]),
        "period": 4, "secs_left": [600, 550, 500, 492, 400, 395, 300, 0],
        "margin": [0, 2, 3, 4, 4, 1, 1, 3],
    })
    scores = scoring_events(plays)
    assert scores["margin"].tolist() == [2, 3, 4, 1, 3]
    assert scores["in_order"].tolist() == [True, True, True, False, True]
    assert scores.loc[usable(scores), "margin"].tolist() == [2]

    # price_paths / trade_counts
    trades = pd.DataFrame({"time": at([-1, 0, 0.5, 2]), "price": [0.40, 0.50, 0.55, 0.60]})
    paths = price_paths(at([0]), trades, np.array([-2, -1, 0, 1, 2]))
    assert np.allclose(paths[0, 1:], [0.40, 0.50, 0.55, 0.60]) and np.isnan(paths[0, 0])
    assert trade_counts(at([0]), trades, np.array([-1, 0, 1, 2])).tolist() == [[1, 2, 0, 1]]

    # end to end: the market jumps 3 s before each timestamp -> half done at -3.5 s
    lead = 3
    event_times = np.arange(100, 4000, 100)
    signs = np.where(np.arange(len(event_times)) % 2 == 0, 1, -1)
    trade_times = np.arange(0, 4100, 0.25)
    jumps = (trade_times[:, None] >= event_times - lead) * (0.04 * signs)
    fake = pd.DataFrame({"time": at(trade_times), "price": 0.5 + jumps.sum(axis=1)})
    scores = pd.DataFrame({"time": at(event_times), "period": 2, "secs_left": 2000.0,
                           "margin_before": -signs + 0.0, "margin": signs + 0.0, "in_order": True})
    events = pd.concat([game_events(g, scores, fake, 0.5) for g in ["a", "b", "c"]], ignore_index=True)
    curve = response_curve(events)
    assert np.allclose(curve["move"].iloc[-1], 0.04) and curve["share"].iloc[0] == 0
    assert crossing(curve["share"].to_numpy()) == -lead - 0.5
    assert summary(events)["half done at (s)"].item() == -lead - 0.5

    print("unit tests (lead_lag): all pass")


def first_games(n: int, seed: int = bc.SEED) -> pd.DataFrame:
    """The first n games of the batch sample's order that batch_compare has cached."""
    order = bc.draw_sample(bc.build_game_list(), None, seed, by_date=False)
    have = order["game_id"].map(lambda g: (bc.CACHE_DIR / f"{g}.parquet").exists())
    return order[have].head(n).sort_values("date")


def main() -> None:
    parser = argparse.ArgumentParser(description="Does Kalshi move before our play timestamps?")
    parser.add_argument("--n", type=int, default=N_GAMES, help=f"games (default {N_GAMES})")
    parser.add_argument("--seed", type=int, default=bc.SEED, help=f"sampling seed (default {bc.SEED})")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    args = parser.parse_args()

    downloads.run_unit_tests()
    run_unit_tests()
    games = first_games(args.n, args.seed)
    print(f"{len(games)} games, {games['date'].min():%Y-%m-%d} to {games['date'].max():%Y-%m-%d}")

    events, failures = collect(games, args.refresh)
    events.to_parquet(EVENTS_PATH)
    print(f"saved {EVENTS_PATH} ({events['game_id'].nunique()} games, {len(events):,} scoring plays)")
    if not failures.empty:
        print(f"\n{len(failures)} games failed:\n{failures.to_string(index=False)}")

    events = load_events()
    columns = ["games", "plays", "model jump (c)", "market move (c)", "share by 0", "half done at (s)"]
    print("\nKalshi's response to a scoring play (half done at < 0: market moves before our timestamp):")
    for by in (None, "quarter", "game_state"):
        print(summary(events, by)[columns].round(2).to_string(), "\n")

    import lead_lag_plots  # imported here because lead_lag_plots imports this module
    paths = lead_lag_plots.save_all(events)
    print(f"saved {len(paths)} charts to {paths[0].parent} (discussed in notebooks/lead_lag.ipynb)")


if __name__ == "__main__":
    main()
