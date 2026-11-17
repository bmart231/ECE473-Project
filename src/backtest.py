"""Replay past games second by second through quoting.quote() and score the fills.
Plan and assumptions: Backtest_Plan.md. The rule itself: quoting_rule.md.

Run:  uv run python src/backtest.py GAME_ID            # full rule vs baseline, 3 feed delays
      uv run python src/backtest.py GAME_ID --rule touch --refresh

Each second i of a game is a row of what the bot knows at that instant (game_seconds):
  - p_market: Kalshi's price from trades already printed (median of the last few seconds)
  - margin, secs_left: the play-by-play as of `delay` seconds ago (our feed is late)
  - p_model: the Stern model from those, None while an unexplained jump is younger than delay
The quote decided from row i is live during second i + LATENCY. A trade that prints through
it is our fill, at our price (simulate). Prices are in cents, like quoting.py.
"""

import argparse
import math

import numpy as np
import pandas as pd

import batch_compare as bc
import lead_lag as ll
import stern
from quoting import MarketState, Params, Quote, kalshi_fee, quote, seconds_since_jump

PRICE_WINDOW = 5                # seconds of trades in p_market
LATENCY = 1                     # seconds from deciding a quote to it being live
MARKOUT_SECS = 60               # how long after a fill we judge it
DELAYS = (0.0, 3.0, 12.9)       # feed delays: ideal, paid low-latency feed, measured free feed
RULES = ("strict", "touch")     # fills: only prints through our price / also prints at it
BASELINE = Params(w=0, b=0, cooldown=0)  # plain mid-quoting: market only, 1c fixed, no guard
FILL_COLUMNS = {"second": "int64", "side": "int64", "price": "float64",  # side +1 = we bought
                "contracts": "float64", "fee": "float64"}


def _cents(trades: pd.DataFrame) -> np.ndarray:
    """Trade prices in cents, rounded: 100 * 0.58 is 57.99999999999999."""
    return np.round(trades["price"].to_numpy(dtype=float) * 100, 2)


def _grid(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return start + pd.to_timedelta(np.arange(n), unit="s")


def market_price(trades: pd.DataFrame, start: pd.Timestamp, n: int,
                 window: int = PRICE_WINDOW) -> np.ndarray:
    """Kalshi's price known at the start of each of n seconds: the median over the last
    `window` seconds of each second's median trade (NaN before the first trade)."""
    per_second = pd.Series(_cents(trades), index=trades["time"]).resample("1s").median().ffill()
    known = per_second.rolling(window, min_periods=1).median().shift(1, freq="1s")
    return known.reindex(_grid(start, n), method="ffill").to_numpy()


def feed_as_of(plays: pd.DataFrame, start: pd.Timestamp, n: int, delay: float) -> pd.DataFrame:
    """period, margin, secs_left from the latest play at least `delay` seconds old, for each
    of n seconds. Before the feed delivers anything, the pregame state (the first play)."""
    seen = pd.DataFrame({"time": _grid(start, n) - pd.to_timedelta(delay, unit="s")})
    feed = pd.merge_asof(seen.astype({"time": plays["time"].dtype}),
                         plays.sort_values("time", kind="stable"), on="time")
    return feed.drop(columns="time").bfill()


def game_seconds(plays: pd.DataFrame, trades: pd.DataFrame, delay: float, params: Params
                 ) -> pd.DataFrame:
    """One row per second from the first play to the last: what the bot knows then.

    Columns: time, period, margin, secs_left, p_market, since_jump, p_model. The model's
    pregame prior is the market's price at tip-off.
    """
    start = plays["time"].min().floor("s")
    n = int((plays["time"].max() - start).total_seconds()) + 1
    p_market = market_price(trades, start, n)
    if np.isnan(p_market[0]):
        raise ValueError("no Kalshi trade before tip-off")
    feed = feed_as_of(plays, start, n, delay)
    since = seconds_since_jump(p_market, params)
    p_model = 100 * stern.stern_p(feed["margin"], feed["secs_left"], p_market[0] / 100, params.sigma)
    return feed.assign(time=_grid(start, n), p_market=p_market, since_jump=since,
                       p_model=np.where(since < delay, np.nan, p_model))


def matched(q: Quote, cents: np.ndarray, contracts: np.ndarray, taker_buys: np.ndarray,
            position: float, params: Params, rule: str) -> tuple[float, float]:
    """(bought, sold) by quote q against one second's trades. A trade fills us if it prints
    through our price, or ("touch") at our price from a taker on the other side. Each side
    fills at most q.size and never past the position limit."""
    touch = rule == "touch"
    bought = sold = 0.0
    if q.bid is not None:
        hit = (cents < q.bid) | (touch & (cents == q.bid) & ~taker_buys)
        bought = min(q.size, params.q_max - position, contracts[hit].sum())
    if q.ask is not None:
        hit = (cents > q.ask) | (touch & (cents == q.ask) & taker_buys)
        sold = min(q.size, params.q_max + position, contracts[hit].sum())
    return bought, sold


def fill_table(fills: list[tuple]) -> pd.DataFrame:
    """(second, side, price, contracts, fee) tuples -> FILL_COLUMNS, typed even when empty."""
    return pd.DataFrame(fills, columns=list(FILL_COLUMNS)).astype(FILL_COLUMNS)


def _rows(frame: pd.DataFrame) -> list[tuple]:
    """The columns a quote needs, as plain tuples (fast to read one at a time)."""
    return list(frame[["p_market", "p_model", "secs_left", "since_jump"]].itertuples(index=False, name=None))


def _state(row: tuple, position: float) -> MarketState:
    """A row of _rows as the bot sees it: no order book, so the price stands in for both
    sides; p_model NaN means stale."""
    p_market, p_model, secs_left, since_jump = row
    return MarketState(p_market=p_market, best_bid=p_market, best_ask=p_market,
                       p_model=None if math.isnan(p_model) else p_model, secs_left=secs_left,
                       position=position, seconds_since_jump=since_jump)


def simulate(frame: pd.DataFrame, trades: pd.DataFrame, params: Params, rule: str = "strict"
             ) -> pd.DataFrame:
    """Our fills over a game (columns FILL_COLUMNS). trades must be oldest first.

    Only seconds with trades can fill, so only those are quoted: the quote for second s
    comes from row s - LATENCY, with the position after every earlier fill.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    second = ((trades["time"] - frame["time"].iloc[0]) // pd.Timedelta(seconds=1)).to_numpy()
    live = (second >= LATENCY) & (second < len(frame))
    second = second[live]
    cents = _cents(trades)[live]
    contracts = trades["contracts"].to_numpy()[live]
    taker_buys = trades["taker_home"].to_numpy()[live]
    rows = _rows(frame)

    seconds, first = np.unique(second, return_index=True)
    position, fills = 0.0, []
    for s, lo, hi in zip(seconds, first, np.r_[first[1:], len(second)]):
        q = quote(_state(rows[s - LATENCY], position), params)
        if q is None:
            continue
        bought, sold = matched(q, cents[lo:hi], contracts[lo:hi], taker_buys[lo:hi],
                               position, params, rule)
        for side, price, size in ((1, q.bid, bought), (-1, q.ask, sold)):
            if price is not None and size > 0:
                fills.append((s, side, price, size, kalshi_fee(price, size)))
                position += side * size
    return fill_table(fills)


def quote_path(frame: pd.DataFrame, fills: pd.DataFrame, params: Params) -> pd.DataFrame:
    """The bid and ask live in every second (NaN: not quoting), given simulate's fills:
    the same quotes simulate posted, for charts. Columns: bid, ask, position."""
    position = np.zeros(len(frame))
    np.add.at(position, fills["second"].to_numpy(), (fills["side"] * fills["contracts"]).to_numpy())
    before = np.r_[0.0, position.cumsum()[:-1]]  # held when the second starts
    rows = _rows(frame)
    path = []
    for s in range(len(frame)):
        q = quote(_state(rows[s - LATENCY], before[s]), params) if s >= LATENCY else None
        path.append((np.nan, np.nan) if q is None else (q.bid, q.ask))
    return pd.DataFrame(path, columns=["bid", "ask"], dtype=float).assign(position=before)


def score(fills: pd.DataFrame, p_market: np.ndarray, home_won: int) -> dict:
    """One game's result. Money in dollars (marked to p_market each second, settled at the
    end); markout in cents per contract, against the price MARKOUT_SECS later (or the
    settlement, if the game ends first)."""
    n, settle = len(p_market), 100.0 * home_won
    second, side, price, size, fee = (fills[c].to_numpy() for c in FILL_COLUMNS)
    position, cash = np.zeros(n), np.zeros(n)
    np.add.at(position, second, side * size)
    np.add.at(cash, second, -side * price * size - fee)
    position, cash = position.cumsum(), cash.cumsum()
    value = np.r_[0.0, cash + position * p_market, cash[-1] + position[-1] * settle]
    later = np.where(second + MARKOUT_SECS < n, p_market[np.minimum(second + MARKOUT_SECS, n - 1)], settle)
    return {
        "pnl_dollars": float(value[-1]) / 100,
        "fees_dollars": float(fee.sum()) / 100,
        "fills": len(fills),
        "contracts": float(size.sum()),
        "markout_cents": float((side * (later - price) * size).sum() / size.sum()) if len(fills) else math.nan,
        "drawdown_dollars": float((np.maximum.accumulate(value) - value).max()) / 100,
        "peak_position": float(np.abs(position).max()),
    }


def run_game(plays: pd.DataFrame, trades: pd.DataFrame, home_won: int, params: Params = Params(),
             delay: float = DELAYS[-1], rule: str = "strict") -> dict:
    """score() for one game under one setting."""
    frame = game_seconds(plays, trades, delay, params)
    return score(simulate(frame, trades, params, rule), frame["p_market"].to_numpy(), home_won)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the quoting rule on one game.")
    parser.add_argument("game_id", help="NBA game id, e.g. 0022500001 (must be in games.parquet)")
    parser.add_argument("--rule", choices=RULES, default="strict", help="fill rule (default strict)")
    parser.add_argument("--refresh", action="store_true", help="download the plays and trades again")
    args = parser.parse_args()

    run_unit_tests()
    game = pd.read_parquet(bc.GAMES_PATH).set_index("game_id").loc[args.game_id]
    plays, trades = ll.load_game(args.game_id, game["ticker"], args.refresh)
    if "taker_home" not in trades:
        raise SystemExit("these trades were cached before 11/16 without taker_home: add --refresh")

    settings = [("full rule", Params(), delay) for delay in DELAYS] + [("baseline", BASELINE, 0.0)]
    rows = [{"setting": name, "feed delay (s)": delay,
             **run_game(plays, trades, int(game["home_won"]), params, delay, args.rule)}
            for name, params, delay in settings]
    print(f"\n{game['away']} @ {game['home']}, {game['date']:%Y-%m-%d}, {args.rule} fills, "
          f"{'home' if game['home_won'] else 'away'} won")
    print(pd.DataFrame(rows).round(2).to_string(index=False))



# ---------------------------------------------------------------- unit tests

T0 = pd.to_datetime("2026-01-01T00:00:00Z")


def _at(seconds) -> pd.Series:
    return pd.Series(T0 + pd.to_timedelta(np.asarray(seconds, dtype=float), unit="s"))


def _game(path: list[float], noise: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A made-up tied game: a play every 10 s, 30 s of trades at path[0] before tip-off, then
    each second a trade at that second's price (taker buying) and, if noise, a taker selling
    at price - noise and one buying at price + noise. The per-second median is the path."""
    n = len(path)
    t = np.arange(0, n + 1, 10)
    plays = pd.DataFrame({"time": _at(t), "period": 1, "margin": 0, "secs_left": 2880.0 - t})
    prices = np.r_[[path[0]] * 30, path]
    seconds = np.arange(-30, n) + 0.5
    offsets = [0.0, -noise, noise] if noise else [0.0]
    trades = pd.concat([pd.DataFrame({"time": _at(seconds + 0.1 * k), "price": prices + off,
                                      "contracts": 50.0, "taker_home": off >= 0})
                        for k, off in enumerate(offsets)])
    return plays, trades.sort_values("time", kind="stable").reset_index(drop=True)


FLAT = [0.50] * 600
SLIDE = [0.50] * 100 + list(0.50 - 0.02 * np.arange(1, 11)) + [0.30] * 200  # 2c a second


def test_market_price() -> None:
    trades = pd.DataFrame({"time": _at([-3, 0.2, 0.5, 0.7, 1.5, 4.0]),
                           "price": [0.40, 0.58, 0.58, 0.62, 0.70, 0.50]})
    # per second: -3: 40, 0: 58 (not 57.999... from 100 * 0.58), 1: 70, 4: 50;
    # a second's trades are known when it ends
    assert market_price(trades, T0, 6, window=1).tolist() == [40, 58, 70, 70, 70, 50]
    assert market_price(trades, T0, 6, window=3).tolist() == [40, 40, 58, 70, 70, 70]
    late = market_price(trades.iloc[4:], T0, 4, window=1)  # first trade at 1.5 s
    assert np.isnan(late[:2]).all() and late[2] == 70


def test_feed_as_of() -> None:
    plays = pd.DataFrame({"time": _at([0, 10, 20]), "period": 1, "margin": [0, 2, 5],
                          "secs_left": [2880.0, 2870.0, 2860.0]})
    feed = feed_as_of(plays, T0, 25, delay=5)
    assert (feed["margin"].iloc[:15] == 0).all() and (feed["margin"].iloc[15:] == 2).all()
    assert feed["secs_left"].iloc[0] == 2880                       # before the feed: pregame
    assert feed["margin"].iloc[20] == 2 and feed_as_of(plays, T0, 25, 0)["margin"].iloc[20] == 5
    assert feed.equals(feed_as_of(plays.iloc[::-1], T0, 25, 5))   # play order doesn't matter


def test_game_seconds() -> None:
    flat = game_seconds(*_game(FLAT, noise=0.05), delay=3, params=Params())
    assert len(flat) == 601 and (flat["p_market"] == 50).all()
    assert np.allclose(flat["p_model"], 50) and np.isinf(flat["since_jump"]).all()

    # the model is ignored exactly while a jump is younger than the feed delay
    slide = game_seconds(*_game(SLIDE), delay=12.9, params=Params())
    assert (slide["since_jump"] < 12.9).any()
    assert (slide["p_model"].isna() == (slide["since_jump"] < 12.9)).all()

    plays, trades = _game(FLAT)
    no_pregame = trades[trades["time"] > T0 + pd.Timedelta(seconds=5)]
    assert bc.lp.g6._raises(ValueError, game_seconds, plays, no_pregame, 3, Params())


def test_matched() -> None:
    q, p = Quote(bid=48, ask=52, size=10), Params()
    cents = np.array([47, 48, 50, 53, 52.0])
    contracts = np.array([3, 4, 100, 20, 5.0])
    taker_buys = np.array([False, False, True, True, True])
    assert matched(q, cents, contracts, taker_buys, 0, p, "strict") == (3, 10)
    assert matched(q, cents, contracts, taker_buys, 0, p, "touch") == (7, 10)
    # a buyer trading at our bid price never hit our bid
    assert matched(q, np.array([48.0]), np.array([5.0]), np.array([True]), 0, p, "touch") == (0, 0)
    # never past the position limit; no bid, no buying
    assert matched(q, np.array([40.0]), np.array([50.0]), np.array([False]), 95, p, "strict") == (5, 0)
    no_bid = Quote(bid=None, ask=52, size=10)
    assert matched(no_bid, np.array([40.0]), np.array([50.0]), np.array([False]), 0, p, "strict") == (0, 0)


def test_simulate() -> None:
    # flat market with sellers at 45c and buyers at 55c: every second we buy 10 at our 48c bid
    # and sell 10 at our 52c ask, so the position stays flat
    plays, trades = _game(FLAT, noise=0.05)
    fills = simulate(game_seconds(plays, trades, 3, Params()), trades, Params())
    assert len(fills) == 2 * 599 and set(fills["price"]) == {48, 52}
    assert (fills["contracts"] == 10).all() and fills["second"].min() == LATENCY

    # the market steps from 50c to 60c at second 100. p_market (a 5 s median) shows 60 from
    # row 103, so the quotes of seconds 100..102 + LATENCY are stale and buyers lift our ask;
    # each sale makes us shorter, which leans the ask up: 52, 52, 52, then 53
    plays, trades = _game([0.50] * 100 + [0.60] * 100, noise=0.05)
    market_only = Params(w=0, cooldown=0)
    fills = simulate(game_seconds(plays, trades, 3, market_only), trades, market_only)
    stale = fills[(fills["second"] >= 100) & (fills["price"] < 55)]
    assert stale["second"].tolist() == list(range(100, 103 + LATENCY))
    assert (stale["side"] == -1).all() and stale["price"].tolist() == [52, 52, 52, 53]

    # the market slides 2c a second: the guard stops the fills; with it off we keep buying
    plays, trades = _game(SLIDE)
    frame = game_seconds(plays, trades, 12.9, Params())
    guard_on = simulate(frame, trades, Params())
    guard_off = simulate(frame, trades, Params(cooldown=0))
    rows = guard_on["second"].to_numpy() - LATENCY
    assert (frame["since_jump"].to_numpy()[rows] >= Params().cooldown).all()
    assert guard_off["contracts"].sum() > guard_on["contracts"].sum() > 0
    assert bc.lp.g6._raises(ValueError, simulate, frame, trades, Params(), "maybe")


def test_quote_path() -> None:
    # every fill happened at the quote the path shows for that second
    plays, trades = _game([0.50] * 100 + [0.60] * 100, noise=0.05)
    params = Params(w=0, cooldown=0)
    frame = game_seconds(plays, trades, 3, params)
    fills = simulate(frame, trades, params)
    path = quote_path(frame, fills, params)
    assert len(path) == len(frame) and path[["bid", "ask"]].iloc[:LATENCY].isna().all().all()
    shown = np.where(fills["side"] == 1, path["bid"].to_numpy()[fills["second"]],
                     path["ask"].to_numpy()[fills["second"]])
    assert (shown == fills["price"]).all()
    # the guard's pause shows as no quote
    slide = game_seconds(*_game(SLIDE), delay=3, params=Params())
    pulled = quote_path(slide, simulate(slide, _game(SLIDE)[1], Params()), Params())["bid"].isna()
    assert pulled.to_numpy()[LATENCY:].tolist() == (slide["since_jump"] < 15).to_numpy()[:-LATENCY].tolist()


def test_score() -> None:
    p_market = np.full(200, 50.0)
    p_market[100:] = 40
    round_trip = fill_table([(10, 1, 48, 10, 5), (20, -1, 52, 10, 5)])
    s = score(round_trip, p_market, home_won=0)
    assert math.isclose(s["pnl_dollars"], 0.30) and math.isclose(s["fees_dollars"], 0.10)
    assert s["markout_cents"] == 2 and s["fills"] == 2 and s["contracts"] == 20
    assert s["drawdown_dollars"] == 0 and s["peak_position"] == 10

    # long 10 into a loss: marked at 50, then 40, settled at 0
    held = fill_table([(10, 1, 48, 10, 5)])
    s = score(held, p_market, home_won=0)
    assert math.isclose(s["pnl_dollars"], -4.85)
    assert math.isclose(s["drawdown_dollars"], 5.00)                 # from +0.15 down to -4.85
    # a fill too close to the end is judged against the settlement
    late = fill_table([(190, 1, 48, 10, 5)])
    assert score(late, p_market, home_won=1)["markout_cents"] == 52
    assert math.isnan(score(late.iloc[:0], p_market, 1)["markout_cents"])
    # a game with no fills at all: every trade at the mid never reaches our quotes
    plays, trades = _game(FLAT)
    frame = game_seconds(plays, trades, 3, Params())
    none = score(simulate(frame, trades, Params()), frame["p_market"].to_numpy(), 1)
    assert none["fills"] == 0 and none["pnl_dollars"] == 0


def test_run_game() -> None:
    # flat game: 599 seconds x (4c x 10 contracts - two 5c maker fees) = $179.70
    plays, trades = _game(FLAT, noise=0.05)
    result = run_game(plays, trades, home_won=1, delay=3)
    assert math.isclose(result["pnl_dollars"], 179.70) and result["peak_position"] == 0


TESTS = [test_market_price, test_feed_as_of, test_game_seconds, test_matched, test_simulate,
         test_quote_path, test_score, test_run_game]


def run_unit_tests() -> None:
    for test in TESTS:
        test()
    print(f"unit tests (backtest): all {len(TESTS)} pass")


if __name__ == "__main__":
    main()
