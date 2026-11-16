"""One night of live logging (src/feed_delay.py), analyzed for notebooks/live_lag.ipynb.

For every basket in the logged games we have three clocks:
- the NBA feed: when our logger received the play (feed_delay.scoring_delays),
- Kalshi trades: exchange timestamps, downloaded afterwards from the public API,
- Kalshi's top of book as our logger polled it once a second from the public REST API.
Times are seconds after the play's timeActual, as in the lead-lag study.

The polled book turns out to be ~13 s stale (book_staleness lines its volume up with the
trade log), so market timing comes from the trades; the book is shown only to make the point.

Run:  uv run python src/live_lag.py               # download trades (once) and print the numbers
      uv run python src/live_lag.py 2026-11-15
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import downloads
import feed_delay as fd
import lead_lag as ll
import live_prototype as lp

TRADES_URL = f"{lp.g6.KALSHI_BASE}/markets/trades"  # games too recent for /historical/
OFFSETS = np.arange(-20, 41)     # seconds around a basket
FULL_MOVE = slice(-11, None)     # the full move: average of +30..+40 s
ISOLATION = 20                   # no other score change this close (s), as in the lead-lag study
PRICE_RANGE = (0.03, 0.97)       # price before the basket must leave room to move
N_BOOT = 1000                    # bootstrap resamples of games
TICKER_RE = re.compile(r"KXNBAGAME-\d{2}[A-Z]{3}\d{2}[A-Z]{6}-[A-Z]{3}")
TRADE_COLOR, BOOK_COLOR = fd.MARKET_COLOR, "#7b5bd6"
FEED_COLOR, INK, MUTED = fd.FEED_COLOR, "#0b0b0b", "#52514e"


@dataclass
class Night:
    day: str
    games: pd.DataFrame    # game_id, away, home, ticker (the home team's market)
    delays: pd.DataFrame   # made field goals timed live: feed / published / entered delays
    scores: pd.DataFrame   # every score change: game_id, ticker, time, side (+1 home), points
    book: pd.DataFrame     # ticker, time, bid, ask, mid (P(home wins)), volume: our 1 s poll
    trades: pd.DataFrame   # ticker, time, price (P(home wins)), contracts


# ---------- data ----------

def load_trades(ticker: str, start: pd.Timestamp, end: pd.Timestamp, folder: Path) -> pd.DataFrame:
    """Every Kalshi trade in the window, downloaded once and cached next to the log."""
    if not TICKER_RE.fullmatch(ticker):  # it becomes part of a file name
        raise ValueError(f"unexpected Kalshi ticker {ticker!r}")
    frame = downloads.cached(
        folder / f"trades_{ticker}.parquet",
        lambda: ll.trades_to_frame(ll.fetch_trades(ticker, start, end, TRADES_URL), lp.ticker_is_home(ticker)))
    return frame.assign(ticker=ticker)


def score_changes(final: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Every score change in the final play-by-play, with the side that scored."""
    f = final.assign(time=pd.to_datetime(final["time_actual"], utc=True, format="ISO8601"),
                     n=final["action_number"].astype(int),
                     home=pd.to_numeric(final["score_home"]), away=pd.to_numeric(final["score_away"]))
    f = f.sort_values(["game_id", "n"])
    home_pts = f.groupby("game_id")["home"].diff().fillna(0)
    away_pts = f.groupby("game_id")["away"].diff().fillna(0)
    scored = (home_pts > 0) | (away_pts > 0)
    out = pd.DataFrame({"game_id": f["game_id"], "action_number": f["n"], "time": f["time"],
                        "side": np.where(home_pts > 0, 1, -1), "points": home_pts + away_pts})[scored]
    return out.merge(games[["game_id", "ticker"]], on="game_id").reset_index(drop=True)


def load(day: str | None = None, root: Path = fd.LOG_ROOT) -> Night:
    day = day or fd.latest_day(root)
    if not fd.DATE_RE.fullmatch(day):
        raise ValueError(f"date must look like 2026-11-15, got {day!r}")
    folder = root / day
    games, plays, final, book = (fd.read_csv(folder / f"{name}.csv") for name in ["games", "plays", "final", "book"])
    book = pd.DataFrame({"ticker": book["ticker"],
                         "time": pd.to_datetime(book["received_at"], utc=True, format="ISO8601"),
                         "bid": pd.to_numeric(book["yes_bid"]), "ask": pd.to_numeric(book["yes_ask"]),
                         "volume": pd.to_numeric(book["volume"])})
    book = book.assign(mid=(book["bid"] + book["ask"]) / 2).sort_values("time", kind="stable")
    windows = book.groupby("ticker")["time"].agg(["min", "max"])
    trades = pd.concat([load_trades(t, w["min"] - pd.Timedelta(minutes=1), w["max"] + pd.Timedelta(minutes=1), folder)
                        for t, w in windows.iterrows()], ignore_index=True)
    return Night(day, games[games["ticker"].isin(windows.index)].reset_index(drop=True),
                 fd.scoring_delays(plays, final), score_changes(final, games), book, trades)


def coverage(night: Night) -> pd.DataFrame:
    """Per game: when we logged it and how much of each source we have."""
    rows = []
    for g in night.games.itertuples():
        book = night.book[night.book["ticker"] == g.ticker]
        rows.append({"game": f"{g.away} @ {g.home}",
                     "logged (ET)": f"{book['time'].min().tz_convert('America/New_York'):%H:%M}"
                                    f"-{book['time'].max().tz_convert('America/New_York'):%H:%M}",
                     "baskets timed live": int((night.delays["game_id"] == g.game_id).sum()),
                     "book rows": len(book),
                     "Kalshi trades": int((night.trades["ticker"] == g.ticker).sum())})
    return pd.DataFrame(rows).set_index("game")


# ---------- the market's response around a basket ----------

def last_value(times: pd.Series, series: pd.DataFrame, column: str) -> np.ndarray:
    """series[column] at or before time + offset, for every time (rows) and OFFSETS (columns)."""
    query = times.to_numpy("datetime64[ns]")[:, None] + OFFSETS.astype("timedelta64[s]")
    index = np.searchsorted(series["time"].to_numpy("datetime64[ns]"), query, side="right") - 1
    return np.where(index >= 0, series[column].to_numpy()[np.maximum(index, 0)], np.nan)


def basket_events(night: Night) -> pd.DataFrame:
    """Isolated made field goals with the Kalshi price path around each, signed toward the
    scoring team: trade_<k> (last trade) and book_<k> (our polled mid) at each offset k.

    Uses every basket in the logged window, including ones the feed missed live (the 403
    gap), since the market side doesn't depend on our feed."""
    s = night.scores.sort_values(["game_id", "time"])
    gap = s.groupby("game_id")["time"].diff().dt.total_seconds()
    next_gap = -s.groupby("game_id")["time"].diff(-1).dt.total_seconds()
    s = s[(s["points"] >= 2) & gap.fillna(np.inf).gt(ISOLATION) & next_gap.fillna(np.inf).gt(ISOLATION)]

    parts = []
    for ticker, events in s.groupby("ticker"):
        book = night.book[night.book["ticker"] == ticker]
        trades = night.trades[night.trades["ticker"] == ticker]
        inside = events[(events["time"] + pd.Timedelta(seconds=int(OFFSETS[0])) >= book["time"].min())
                        & (events["time"] + pd.Timedelta(seconds=int(OFFSETS[-1])) <= book["time"].max())]
        if inside.empty:
            continue
        side = inside["side"].to_numpy()[:, None]
        paths = {"trade": last_value(inside["time"], trades, "price"),
                 "book": last_value(inside["time"], book, "mid")}
        moves = {k: side * (v - v[:, :1]) for k, v in paths.items()}
        keep = (paths["book"][:, 0] > PRICE_RANGE[0]) & (paths["book"][:, 0] < PRICE_RANGE[1])
        cols = {f"{k}_{o}": moves[k][keep, i] for k in moves for i, o in enumerate(OFFSETS)}
        parts.append(pd.concat([inside[keep].reset_index(drop=True), pd.DataFrame(cols)], axis=1))
    return pd.concat(parts, ignore_index=True)


def _moves(events: pd.DataFrame, kind: str) -> np.ndarray:
    return 100 * events[[f"{kind}_{o}" for o in OFFSETS]].to_numpy()


def reach(offsets: np.ndarray, share: np.ndarray, level: float = 0.5) -> float:
    """First offset at or after -5 s (interpolated) where share reaches level."""
    above = np.flatnonzero((share >= level) & (offsets >= -5))
    if above.size == 0:
        return np.nan
    i = above[0]
    return float(np.interp(level, share[i - 1:i + 1], offsets[i - 1:i + 1]))


def response(events: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average signed move (cents) at each offset for trades and our polled book, each also
    as a share of its own full move, with 95% intervals from resampling games.

    Returns (curve, half): curve by offset; half = when each is half done, with interval."""
    rng = np.random.default_rng(seed)
    games = events["game_id"].unique()
    picks = rng.integers(0, len(games), size=(N_BOOT, len(games)))
    curve, half = {"offset": OFFSETS}, {}
    for kind in ["trade", "book"]:
        m = _moves(events, kind)
        rows = [m[(events["game_id"] == g).to_numpy()] for g in games]
        boot = np.array([np.nanmean(np.vstack([rows[i] for i in p]), axis=0) for p in picks])
        mean = np.nanmean(m, axis=0)
        curve[kind] = mean
        curve[f"{kind}_lo"], curve[f"{kind}_hi"] = np.percentile(boot, [2.5, 97.5], axis=0)
        curve[f"{kind}_share"] = mean / mean[FULL_MOVE].mean()
        boot_half = [reach(OFFSETS, b / b[FULL_MOVE].mean()) for b in boot]
        half[kind] = [reach(OFFSETS, curve[f"{kind}_share"]), *np.nanpercentile(boot_half, [2.5, 97.5])]
    return pd.DataFrame(curve), pd.DataFrame(half, index=["half done at (s)", "lo", "hi"]).T


def book_staleness(night: Night, lags: np.ndarray = np.arange(0, 30.5, 0.5)) -> pd.DataFrame:
    """How old the polled book was, per game: the lag L for which the volume it reported best
    matches the trade log's cumulative volume L seconds before we received it.
    (Games with too few trades to line up are left out.)"""
    rows = []
    for g in night.games.itertuples():
        book = night.book[night.book["ticker"] == g.ticker]
        trades = night.trades[night.trades["ticker"] == g.ticker]
        if len(trades) < 1000:
            continue
        cum = trades["contracts"].cumsum().to_numpy()
        trade_t = trades["time"].to_numpy("datetime64[ns]")
        errors = []
        for lag in lags:
            query = (book["time"] - pd.Timedelta(seconds=float(lag))).to_numpy("datetime64[ns]")
            i = np.searchsorted(trade_t, query, side="right") - 1
            gap = book["volume"].to_numpy()[i >= 0] - cum[i[i >= 0]]
            errors.append(np.mean(np.abs(gap - np.median(gap))))
        best = int(np.argmin(errors))
        rows.append({"game": f"{g.away} @ {g.home}", "book is this many s old": lags[best],
                     "volume mismatch at that lag": errors[best], "mismatch if fresh (0 s)": errors[0]})
    return pd.DataFrame(rows).set_index("game")


def race(night: Night, curve: pd.DataFrame, half: pd.DataFrame) -> pd.DataFrame:
    """When each clock gets the news, and how much of Kalshi's move is done by then."""
    feed = night.delays["feed"]
    times = {"Kalshi trades: half the move done": half.loc["trade"],
             "Kalshi book as we polled it: half the move done": half.loc["book"],
             "NBA feed delivers the basket (median, 10th-90th pct)":
                 pd.Series([feed.median(), feed.quantile(0.1), feed.quantile(0.9)], index=half.columns)}
    out = pd.DataFrame(times).T
    out["Kalshi move already done"] = np.interp(out["half done at (s)"], curve["offset"], curve["trade_share"])
    return out.rename(columns={"half done at (s)": "seconds after basket"})


# ---------- charts ----------

def _style(ax) -> None:
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def by_game(night: Night) -> Figure:
    """Feed delay per game: median dot, 10th-90th percentile bar, every basket as a faint tick."""
    d = night.delays.merge(night.games, on="game_id")
    d["game"] = d["away"] + " @ " + d["home"]
    order = d.groupby("game")["feed"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.4))
    for row, game in enumerate(order):
        f = d.loc[d["game"] == game, "feed"]
        ax.plot(f, np.full(len(f), row), "|", color=FEED_COLOR, alpha=0.35, markersize=10)
        ax.plot([f.quantile(0.1), f.quantile(0.9)], [row, row], color=FEED_COLOR, linewidth=2, alpha=0.5)
        ax.plot(f.median(), row, "o", color=FEED_COLOR, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"n={len(f)}", (1, row), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.axvline(fd.KALSHI_HALF_MOVE, color=TRADE_COLOR, linewidth=1.5)
    ax.annotate(f"Kalshi half done (+{fd.KALSHI_HALF_MOVE} s, lead-lag study)", (fd.KALSHI_HALF_MOVE, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order)
    ax.set(xlabel="seconds from the basket to our logger receiving it", xlim=(0, None),
           title="Feed delay by game (dot = median, bar = 10th-90th percentile)")
    _style(ax)
    fig.tight_layout()
    return fig


def race_chart(night: Night, events: pd.DataFrame, curve: pd.DataFrame) -> Figure:
    """Kalshi's average move after a basket (trades and our polled book) against when our
    feed delivers the basket."""
    feed = night.delays["feed"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(feed.quantile(0.1), feed.quantile(0.9), color=FEED_COLOR, alpha=0.12, linewidth=0)
    ax.axvline(feed.median(), color=FEED_COLOR, linewidth=2)
    ax.annotate(f"our feed delivers the basket\n(median +{feed.median():.1f} s; band = 10th-90th pct)",
                (feed.median(), 0.04), xycoords=("data", "axes fraction"), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK)
    for kind, color, label in [("trade", TRADE_COLOR, "Kalshi trades (exchange time)"),
                               ("book", BOOK_COLOR, "Kalshi mid as our logger polled it (REST, 1 s)")]:
        ax.fill_between(curve["offset"], curve[f"{kind}_lo"], curve[f"{kind}_hi"], color=color, alpha=0.15, linewidth=0)
        ax.plot(curve["offset"], curve[kind], color=color, linewidth=2, label=label)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("basket\n(timeActual)", (0, 0.04), xycoords=("data", "axes fraction"), xytext=(-4, 0),
                textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set(xlabel="seconds after the basket", ylabel="price move toward the scoring team (cents)",
           title=f"The race after a basket ({len(events)} isolated made field goals, "
                 f"{events['game_id'].nunique()} games, {night.day})")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.14), frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def example(night: Night, events: pd.DataFrame, minutes: float = 4) -> Figure:
    """One stretch of one game: Kalshi trades and our polled mid, with each basket's time and
    when our feed delivered it. Picks the stretch where the baskets moved the price most."""
    timed = night.delays
    big = events.assign(size=events["trade_20"].abs()).merge(
        timed[["game_id", "action_number", "feed"]], on=["game_id", "action_number"])
    window = pd.Timedelta(minutes=minutes)
    score = [(big[(big["game_id"] == r.game_id) & big["time"].between(r.time, r.time + window)]["size"].sum(), r)
             for r in big.itertuples()]
    _, start = max(score, key=lambda pair: pair[0])
    t0, t1 = start.time - pd.Timedelta(seconds=30), start.time + window
    ticker = start.ticker
    game = night.games[night.games["ticker"] == ticker].iloc[0]

    def clip(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame[(frame["ticker"] == ticker) & frame["time"].between(t0, t1)]
        return f.assign(s=(f["time"] - t0).dt.total_seconds())

    trades, book = clip(night.trades), clip(night.book)
    baskets = clip(night.scores.merge(timed[["game_id", "action_number", "feed"]], how="left",
                                      on=["game_id", "action_number"]))
    baskets = baskets[baskets["points"] >= 2]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(trades["s"], 100 * trades["price"], ".", color=TRADE_COLOR, alpha=0.35, markersize=3, label="Kalshi trades")
    ax.step(book["s"], 100 * book["mid"], where="post", color=BOOK_COLOR, linewidth=1.8, label="Kalshi mid as polled (1 s)")
    for b in baskets.itertuples():
        ax.axvline(b.s, color=MUTED, linestyle="--", linewidth=1)
        team = game["home"] if b.side > 0 else game["away"]
        ax.annotate(f"{team} +{b.points:g}", (b.s, 1), xycoords=("data", "axes fraction"), xytext=(2, -10),
                    textcoords="offset points", fontsize=8, color=MUTED, rotation=90, va="top")
        if pd.notna(b.feed):
            ax.annotate("", xy=(b.s + b.feed, 0.06), xytext=(b.s, 0.06), xycoords=("data", "axes fraction"),
                        arrowprops={"arrowstyle": "->", "color": FEED_COLOR, "linewidth": 1.8})
    ax.plot([], [], color=MUTED, linestyle="--", linewidth=1, label="basket (timeActual)")
    ax.plot([], [], color=FEED_COLOR, linewidth=1.8, label="basket until our feed receives it")
    ax.set(xlabel=f"seconds from {t0.tz_convert('America/New_York'):%H:%M:%S} ET",
           ylabel=f"P({game['home']} wins), cents",
           title=f"{game['away']} @ {game['home']}: Kalshi moves on the basket, our feed arrives later")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    night = load(sys.argv[1] if len(sys.argv) > 1 else None)
    events = basket_events(night)
    curve, half = response(events)
    with pd.option_context("display.width", 160):
        print(coverage(night).to_string(), end="\n\n")
        print(f"{len(events)} isolated made field goals in {events['game_id'].nunique()} games")
        print(race(night, curve, half).round(2).to_string(), end="\n\n")
        print(book_staleness(night).round(1).to_string())


if __name__ == "__main__":
    main()
