"""Any game from the 2025 playoffs onward: Kalshi mid vs Stern model, against UTC time.

Same idea as `src/prototype_game_6.py`, but the game is an input instead of a constant.
Kalshi's NBA game markets start in April 2025, so that is the earliest game this works on.

Run:  uv run python src/live_prototype.py GAME_ID KALSHI_TICKER
    uv run python src/live_prototype.py 0042400406 KXNBAGAME-25JUN19OKCIND-IND
    uv run python src/live_prototype.py 0042400406 KXNBAGAME-25JUN19OKCIND-IND --prior 0.345 --no-plot

From other code (a loop over many games, the notebook):
    table = live_prototype.compare_game("0042400406", "KXNBAGAME-25JUN19OKCIND-IND")

What changed from the Game 6 script, and why:
1. candle window  - worked out from the play timestamps instead of typed in
2. ticker side    - a ticker can be the away team; then the home probability is 1 - mid
3. prior          - Elo for the game if we have it, else Kalshi's own tip-off price
4. overtime       - the clock restarts each overtime period instead of going negative
5. checks         - only the ones true of every game (no "final margin is +17")
"""

import argparse
from functools import cache

import matplotlib.pyplot as plt
import pandas as pd
import requests
from nba_api.live.nba.endpoints import playbyplay

import play_by_play
import prior
import prototype_game_6 as g6  # reuse the pieces that were never Game 6 specific
import stern

SIGMA = g6.SIGMA                      # 19.64, pooled fit (sigma_results.md)
ELO_SEASONS = list(range(2015, 2025))  # seasons we have play-by-play for in data/raw/pbp/
WINDOW_PAD = pd.Timedelta(minutes=10)  # candles from this long before tip-off to this long after the buzzer


# ---------- step 1: plays with real time ----------

def actions_to_plays(actions: list[dict]) -> pd.DataFrame:
    """Raw live play-by-play actions -> table with columns: time, period, margin, secs_left.

    Differs from the Game 6 version in one way: overtime. There, period 5 gave a negative
    secs_left. Here an overtime period counts only its own clock (5:00 down to 0:00), the
    same rule as play_by_play.clock_to_secs_left, so the model keeps working in overtime.
    """
    raw = pd.DataFrame(actions, columns=["timeActual", "period", "clock", "scoreHome", "scoreAway"])
    clock = raw["clock"].str.extract(g6.CLOCK_RE.pattern).astype(float)
    if clock.isna().any().any():
        bad = raw.loc[clock.isna().any(axis=1), "clock"].iloc[0]
        raise ValueError(f"unexpected clock format: {bad!r}")

    period = raw["period"].astype(int)
    this_period = clock[0] * 60 + clock[1]                      # seconds left in the current period
    later_quarters = (4 - period).clip(lower=0) * g6.PERIOD_SECS  # 0 in Q4 and in every overtime

    return pd.DataFrame({
        "time": pd.to_datetime(raw["timeActual"], utc=True),
        "period": period,
        "margin": raw["scoreHome"].astype(int) - raw["scoreAway"].astype(int),
        "secs_left": later_quarters + this_period,
    })


def load_plays(game_id: str) -> pd.DataFrame:
    """Live play-by-play for one game -> table with columns: time, period, margin, secs_left."""
    actions = playbyplay.PlayByPlay(game_id, timeout=30).get_dict()["game"]["actions"]
    return actions_to_plays(actions)


# ---------- step 2: Kalshi mid per minute, as P(home wins) ----------

def ticker_is_home(ticker: str) -> bool:
    """True if the ticker's team is the home team.

    'KXNBAGAME-25JUN19OKCIND-IND' = series - date + AWAY + HOME - team this market is about.
    So the market is the home team's when the last part matches the end of the middle part.
    """
    _, matchup, team = ticker.split("-")
    if not (matchup.endswith(team) or matchup[7:].startswith(team)):
        raise ValueError(f"team {team!r} is not in the matchup part of {ticker!r}")
    return matchup.endswith(team)


def load_kalshi_home_p(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """1-minute candles for ticker between start and end -> table with columns: time, mid.

    mid is always P(home wins): an away-team ticker is flipped to 1 - mid here, once, so
    nothing downstream has to care which side the ticker was.
    """
    params = {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()), "period_interval": 1}

    # settled games live under /historical/; a game too recent to be archived yet is still
    # under the normal series path, so try that second
    for path in (f"historical/markets/{ticker}/candlesticks",
                f"series/{ticker.split('-')[0]}/markets/{ticker}/candlesticks"):
        response = requests.get(f"{g6.KALSHI_BASE}/{path}", params=params, timeout=30)
        if response.ok and response.json().get("candlesticks"):
            break
    else:
        raise ValueError(f"no Kalshi candles for {ticker} between {start} and {end}")

    kalshi = g6.candles_to_mid(response.json()["candlesticks"])
    if not ticker_is_home(ticker):
        kalshi["mid"] = 1 - kalshi["mid"]
    return kalshi


# ---------- step 3: pregame prior ----------

@cache
def elo_priors() -> pd.Series:
    """Elo P(home wins) for every game in data/raw/pbp/, indexed by game_id.

    Reads all ten seasons (Elo has to run through them in order), so it takes a few
    seconds. @cache means a loop over many games only pays that once.
    """
    games = play_by_play.games_table(ELO_SEASONS)
    return prior.build_prior(games).set_index("game_id")["p_pre"]


def pregame_prior(game_id: str, kalshi: pd.DataFrame, tip_off: pd.Timestamp) -> tuple[float, str]:
    """(p_pre, where it came from) for one game.

    Elo if the game is in our downloaded seasons (through the 2025 playoffs). Otherwise
    the last Kalshi mid before tip-off: a stand-in until 2025-26 data is downloaded, and
    it makes the model start exactly on the market, so the early gap means less.
    """
    try:
        priors = elo_priors()
    except FileNotFoundError:  # data/raw/pbp/ not downloaded on this machine
        priors = pd.Series(dtype=float)

    if game_id in priors.index:
        return float(priors[game_id]), "Elo"

    before = kalshi[kalshi["time"] <= tip_off]
    if before.empty:
        raise ValueError(f"no Elo prior for {game_id} and no Kalshi quote before tip-off; pass p_pre")
    return float(before["mid"].iloc[-1]), "Kalshi mid at tip-off"


# ---------- step 4: put it together ----------

def compare_game(game_id: str, ticker: str, p_pre: float | None = None,
                sigma: float = SIGMA) -> pd.DataFrame:
    """One row per in-game Kalshi minute: time, mid, p, gap, plus the game state.

    mid and p are both P(home wins); gap = p - mid (model minus market).
    The full series are kept in .attrs["kalshi"] and .attrs["plays"] for plotting.
    """
    plays = load_plays(game_id)
    tip_off, buzzer = plays["time"].min(), plays["time"].max()
    kalshi = load_kalshi_home_p(ticker, tip_off - WINDOW_PAD, buzzer + WINDOW_PAD)

    if p_pre is None:
        p_pre, source = pregame_prior(game_id, kalshi, tip_off)
    else:
        source = "given"

    plays["p"] = stern.stern_p(plays["margin"].to_numpy(), plays["secs_left"].to_numpy(), p_pre, sigma)
    check_game(kalshi, plays, p_pre)

    # for each Kalshi minute, the model's value as of the most recent play
    state = plays.sort_values("time", kind="stable")
    in_game = kalshi[kalshi["time"].between(tip_off, buzzer)]
    table = pd.merge_asof(in_game, state, on="time", direction="backward")
    table["gap"] = table["p"] - table["mid"]
    table.insert(0, "game_id", game_id)

    table.attrs.update(kalshi=kalshi, plays=plays, p_pre=p_pre, prior_source=source, ticker=ticker)
    return table


def check_game(kalshi: pd.DataFrame, plays: pd.DataFrame, p_pre: float) -> None:
    """Checks that hold for every game. A failure means the inputs don't belong together."""
    home_won = plays["margin"].iloc[-1] > 0

    assert plays["margin"].iloc[-1] != 0, "game ends tied: play-by-play is incomplete"
    assert abs(plays["p"].iloc[0] - p_pre) < 1e-6, "model does not start at the prior"
    assert plays["p"].iloc[-1] == float(home_won), "model does not end at 0 or 1"
    # timeActual can step back ~1 s where a sub is logged between free throws; nothing bigger
    assert plays["time"].diff().dt.total_seconds().min() > -5, "play timestamps jump backwards"
    # the market should end agreeing with the result. If it ends on the wrong side, the
    # ticker is for a different game or ticker_is_home got the side wrong.
    final_mid = kalshi["mid"].iloc[-1]
    assert (final_mid > 0.9) == home_won and (final_mid < 0.1) == (not home_won), (
        f"Kalshi ends at {final_mid:.3f} but home_won is {home_won}: wrong ticker or wrong side"
    )


def summarize(table: pd.DataFrame) -> None:
    """Print the headline gap numbers and the per-period breakdown."""
    gap = table["gap"]
    print(f"prior   {table.attrs['p_pre']:.3f} ({table.attrs['prior_source']})")
    print(f"minutes {len(table)}   mean |gap| {gap.abs().mean():.3f}   "
        f"mean gap {gap.mean():+.3f}   max |gap| {gap.abs().max():.3f}")
    by_period = table.groupby("period")["gap"].agg(
        minutes="size", mean_gap="mean", mean_abs_gap=lambda g: g.abs().mean())
    print(by_period.round(3).to_string())


def plot(table: pd.DataFrame) -> None:
    """Kalshi mid and Stern model against UTC time on one chart."""
    kalshi, plays = table.attrs["kalshi"], table.attrs["plays"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(kalshi["time"], kalshi["mid"], label="Kalshi mid")
    ax.plot(plays["time"], plays["p"], drawstyle="steps-post", label="Stern model")
    ax.set(ylim=(0, 1), xlabel="time (UTC)", ylabel="P(home wins)",
        title=f"{table.attrs['ticker']}: Kalshi vs Stern model")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------- checks ----------

def run_unit_tests() -> None:
    """Offline checks on the new pieces, using small hand-made inputs (no network)."""
    # ticker side
    assert ticker_is_home("KXNBAGAME-25JUN19OKCIND-IND")
    assert not ticker_is_home("KXNBAGAME-25JUN19OKCIND-OKC")
    assert g6._raises(ValueError, ticker_is_home, "KXNBAGAME-25JUN19OKCIND-NYK")

    # clock: regulation matches the Game 6 parser; overtime counts only its own period
    def action(period: int, clock: str, home: int, away: int) -> dict:
        return {"timeActual": "2025-06-20T00:40:00Z", "period": period, "clock": clock,
                "scoreHome": str(home), "scoreAway": str(away)}

    plays = actions_to_plays([
        action(1, "PT12M00.00S", 0, 0), action(2, "PT05M30.00S", 50, 40),
        action(4, "PT00M00.00S", 100, 100), action(5, "PT05M00.00S", 100, 100),
        action(5, "PT00M00.00S", 110, 108),
    ])
    assert list(plays.columns) == ["time", "period", "margin", "secs_left"]
    assert plays["secs_left"].tolist() == [2880, 1770, 0, 300, 0]
    assert plays["margin"].tolist() == [0, 10, 0, 0, 2]

    # the model through overtime: 0.5 when tied at the buzzer, live again in OT, then 0/1
    p = stern.stern_p(plays["margin"].to_numpy(), plays["secs_left"].to_numpy(), 0.5, SIGMA)
    assert p[2] == 0.5 and abs(p[3] - 0.5) < 1e-9 and p[4] == 1.0

    print("unit tests: all pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi mid vs Stern model for one NBA game.")
    parser.add_argument("game_id", help="NBA game id, e.g. 0042400406")
    parser.add_argument("ticker", help="Kalshi market ticker, e.g. KXNBAGAME-25JUN19OKCIND-IND")
    parser.add_argument("--prior", type=float, help="pregame P(home wins); default is Elo, else Kalshi at tip-off")
    parser.add_argument("--no-plot", action="store_true", help="print the numbers only")
    args = parser.parse_args()

    run_unit_tests()
    table = compare_game(args.game_id, args.ticker, p_pre=args.prior)
    summarize(table)
    if not args.no_plot:
        plot(table)


if __name__ == "__main__":
    main()
