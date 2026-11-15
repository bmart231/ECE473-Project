"""Stern model vs Kalshi over a random sample of games: who is better calibrated, where do they
disagree, and what sigma is the market acting as if it uses?

Run:  uv run python src/batch_compare.py                 # 100 games, seed 473
      uv run python src/batch_compare.py --n 200         # a superset of the 100 (same seed)
      uv run python src/batch_compare.py --refresh       # re-download everything

Writes data/processed/kalshi_vs_model.parquet, one row per in-game Kalshi minute, and the
charts from compare_plots.py to figures/kalshi_comparison/*.png.
Discussion and conclusions: notebooks/kalshi_comparison.ipynb (reads the parquet, no API calls).

Design:
- Games come from the tuning period only (WINDOW). Feb-Jun 2026 is the market maker's
  held-out test set, so nothing here looks at it.
- Each game is fetched once and cached. The cache holds the game state and the market,
  not model output, so priors and sigma can change without re-downloading.
- Two priors per game: Elo (the full pipeline) and Kalshi's own tip-off price. With the
  tip-off prior both sides start at the same number, so any gap is purely in-game updating.
- Rows within a game are strongly correlated, so every average and interval here treats
  the game, not the minute, as the unit (each game's rows are averaged first).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder

import live_prototype as lp
import play_by_play
import prior
import stern

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = PROCESSED / "cache"                       # one parquet per game (git-ignored)
GAMES_PATH = PROCESSED / "games.parquet"              # matched game list with Elo priors
MINUTES_PATH = PROCESSED / "kalshi_vs_model.parquet"  # the batch result

WINDOW = (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-01-31"))  # tuning period, inclusive
N_GAMES = 100
SEED = 473
SIGMA = lp.SIGMA  # 19.64

# NBA seasons covering WINDOW. The downloaded play-by-play stops at the 2025 playoffs, so
# 2025-26 results come from nba_api; they extend Elo and are matched to Kalshi markets.
NBA_SEASONS = [("2024-25", "Playoffs"), ("2025-26", "Regular Season")]

# 'KXNBAGAME-25JUN19OKCIND' -> date '25JUN19', away 'OKC', home 'IND'
EVENT_RE = r"^KXNBAGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]{3})([A-Z]{3})$"

# label -> column holding that predictor's P(home wins)
PREDICTORS = {
    "market": "mid",
    "model, Elo prior": "p_model_elo",
    "model, tip-off prior": "p_model_tipoff",
}
QUARTER_LABELS = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "OT"}
EPS = 1e-6  # keeps log() finite for p of exactly 0 or 1


# ---------- 1. game list ----------

def fetch_kalshi_markets() -> list[dict]:
    """Every settled KXNBAGAME market (two per game, one per team)."""
    markets, cursor = [], None
    while True:
        response = requests.get(
            f"{lp.g6.KALSHI_BASE}/historical/markets",
            params={"series_ticker": "KXNBAGAME", "limit": 1000, "cursor": cursor},
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        markets += page["markets"]
        cursor = page.get("cursor")
        if not cursor or not page["markets"]:
            return markets


def kalshi_events(markets: list[dict]) -> pd.DataFrame:
    """Markets -> one row per game: date, away, home, ticker (the home team's market),
    home_yes (that market settled YES, i.e. Kalshi says the home team won).

    Drops games that settled at a fractional price (result 'scalar') and tickers that don't
    parse. Exhibitions against non-NBA teams parse fine; they drop out later, at the match
    to NBA game ids.
    """
    df = pd.DataFrame(markets, columns=["ticker", "event_ticker", "result"])
    df = df[df["result"].isin(["yes", "no"])]

    parts = df["event_ticker"].str.extract(EVENT_RE).dropna()
    events = pd.DataFrame({
        "event_ticker": df.loc[parts.index, "event_ticker"],
        "date": pd.to_datetime(parts[0], format="%y%b%d"),
        "away": parts[1],
        "home": parts[2],
    }).drop_duplicates("event_ticker")

    events["ticker"] = events["event_ticker"] + "-" + events["home"]
    events["home_yes"] = events["ticker"].map(df.set_index("ticker")["result"]) == "yes"
    events = events[events["ticker"].isin(df["ticker"])]  # home market must exist
    return events.drop(columns="event_ticker").reset_index(drop=True)


def finder_to_games(rows: pd.DataFrame, season: int) -> pd.DataFrame:
    """LeagueGameFinder rows (two per game) -> one row per game.

    MATCHUP is 'IND vs. OKC' on the home team's row and 'OKC @ IND' on the away team's.
    """
    home = rows[rows["MATCHUP"].str.contains(" vs. ", regex=False)]
    return pd.DataFrame({
        "game_id": home["GAME_ID"],
        "date": pd.to_datetime(home["GAME_DATE"]),
        "season": season,
        "home": home["TEAM_ABBREVIATION"],
        "away": home["MATCHUP"].str[-3:],
        "home_won": (home["WL"] == "W").astype(int),
    })


def fetch_nba_games() -> pd.DataFrame:
    """Results for NBA_SEASONS, in date order: game_id, date, season, home, away, home_won."""
    frames = []
    for season, season_type in NBA_SEASONS:
        rows = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, season_type_nullable=season_type,
            league_id_nullable="00", timeout=60,
        ).get_data_frames()[0]
        frames.append(finder_to_games(rows, int(season[:4])))
    return pd.concat(frames).sort_values(["date", "game_id"]).reset_index(drop=True)


def elo_priors(recent: pd.DataFrame) -> pd.Series:
    """Elo P(home wins) by game_id, run from 2015 through the games in `recent`.

    Elo needs every game in order, so the play-by-play seasons go first and the nba_api
    games not already in them (2025-26) are appended.
    """
    history = play_by_play.games_table(lp.ELO_SEASONS)
    new = recent.loc[~recent["game_id"].isin(history["game_id"]), history.columns]
    games = pd.concat([history, new], ignore_index=True)
    return prior.build_prior(games).set_index("game_id")["p_pre"]


def build_game_list(refresh: bool = False) -> pd.DataFrame:
    """Kalshi games in WINDOW matched to NBA game ids, with Elo priors. Cached to GAMES_PATH.

    Columns: date, away, home, ticker, game_id, season, home_won, prior_elo
    """
    if GAMES_PATH.exists() and not refresh:
        return pd.read_parquet(GAMES_PATH)

    events = kalshi_events(fetch_kalshi_markets())
    events = events[events["date"].between(*WINDOW)]
    nba = fetch_nba_games()

    games = events.merge(nba, on=["date", "away", "home"], how="inner", validate="one_to_one")
    games["prior_elo"] = games["game_id"].map(elo_priors(nba))
    # Kalshi's settlement must agree with the NBA result: confirms the match and the side
    settled_right = games["home_yes"] == games["home_won"].astype(bool)
    games = games[settled_right].dropna(subset=["prior_elo"]).drop(columns="home_yes")
    games = games.reset_index(drop=True)

    print(f"Kalshi games in window: {len(events)}   matched to an NBA game: {settled_right.size}   "
          f"settlement disagrees with result: {(~settled_right).sum()}   usable (with Elo): {len(games)}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    games.to_parquet(GAMES_PATH)
    return games


# ---------- 2. collect minutes ----------

def game_minutes(game_id: str, ticker: str, prior_elo: float, refresh: bool = False) -> pd.DataFrame:
    """In-game Kalshi minutes for one game with the game state at each. Cached per game.

    Columns: game_id, time, period, secs_left, margin, mid, prior_tipoff
    compare_game also runs live_prototype's per-game checks, so a bad match fails here.
    """
    path = CACHE_DIR / f"{game_id}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    table = lp.compare_game(game_id, ticker, p_pre=prior_elo)
    tip_off = table.attrs["plays"]["time"].min()
    minutes = pd.DataFrame({
        "game_id": game_id,
        "time": table["time"],
        "period": table["period"],
        "secs_left": table["secs_left"],
        "margin": table["margin"],
        "mid": table["mid"],
        "prior_tipoff": lp.kalshi_tipoff_mid(table.attrs["kalshi"], tip_off),
    })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    minutes.to_parquet(path)
    return minutes


def collect(games: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """game_minutes for every game -> (minutes joined to the game list, failed games)."""
    frames, failures = [], []
    # plain lists rather than itertuples(), whose fields are untyped (see prior.build_prior)
    rows = zip(games["game_id"].tolist(), games["ticker"].tolist(), games["prior_elo"].tolist())
    for i, (game_id, ticker, prior_elo) in enumerate(rows, start=1):
        try:
            frames.append(game_minutes(game_id, ticker, prior_elo, refresh))
        except Exception as error:  # one bad game must not stop the batch; the reason is kept
            failures.append({"game_id": game_id, "ticker": ticker,
                             "error": f"{type(error).__name__}: {error}"})
        if i % 10 == 0 or i == len(games):
            print(f"  {i}/{len(games)} games ({len(failures)} failed)")

    columns = ["game_id", "date", "home", "away", "home_won", "prior_elo"]
    minutes = pd.concat(frames, ignore_index=True).merge(games[columns], on="game_id")
    return minutes, pd.DataFrame(failures, columns=["game_id", "ticker", "error"])


def load_results(path: Path = MINUTES_PATH, sigma: float = SIGMA) -> pd.DataFrame:
    """The saved batch, ready to analyse: model predictions and quarter added.

    Rows with secs_left == 0 are dropped. That is the final buzzer, where the model is
    certain (0 or 1) and the market is not yet settled, which would flatter the model.
    """
    df = pd.read_parquet(path)
    df = df[df["secs_left"] > 0].reset_index(drop=True)
    margin, secs_left = df["margin"].to_numpy(), df["secs_left"].to_numpy()
    return df.assign(
        quarter=df["period"].clip(upper=5),
        p_model_elo=stern.stern_p(margin, secs_left, df["prior_elo"].to_numpy(), sigma),
        p_model_tipoff=stern.stern_p(margin, secs_left, df["prior_tipoff"].to_numpy(), sigma),
    )


# ---------- 3. analysis ----------

def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """-log(probability given to what happened), one value per row. Lower is better."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Squared error of the probability, one value per row. Lower is better."""
    return (p - y) ** 2


LOSSES = {"log_loss": log_loss_rows, "brier": brier_rows}


def bucket(p: pd.Series, width: float = 0.1) -> pd.Series:
    """Probability -> centre of its bucket (0.05, 0.15, ..., 0.95 for width 0.1)."""
    n = round(1 / width)
    return (np.minimum(np.floor(p * n), n - 1) + 0.5) / n


def clustered(df: pd.DataFrame, values: dict, by: str | None = None) -> pd.DataFrame:
    """Game-clustered mean and 95% interval of each value, per group of `by`.

    Each game's rows are averaged first, then games are averaged with equal weight; the
    interval is 1.96 standard errors across games. Returns one row per (group, value):
    columns group, value, games, mean, ci95.
    """
    data = pd.DataFrame(values, index=df.index)
    data["game_id"] = df["game_id"]
    data["group"] = df[by] if by else "all"

    per_game = data.groupby(["group", "game_id"]).mean().reset_index()
    long = per_game.melt(id_vars=["group", "game_id"], var_name="value", value_name="_game_mean")
    stats = long.groupby(["group", "value"], sort=False)["_game_mean"].agg(["count", "mean", "std"])
    stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    return stats.rename(columns={"count": "games"}).drop(columns="std").reset_index()


def score_table(df: pd.DataFrame, by: str | None = None, metric: str = "log_loss") -> pd.DataFrame:
    """Each predictor's score, and each model's score minus the market's (negative = model
    better), per group of `by`."""
    y = df["home_won"].to_numpy()
    loss = {name: LOSSES[metric](y, df[col].to_numpy()) for name, col in PREDICTORS.items()}
    values = dict(loss)
    for name in PREDICTORS:
        if name != "market":
            values[f"{name} - market"] = loss[name] - loss["market"]
    return clustered(df, values, by)


def calibration(df: pd.DataFrame, column: str, width: float = 0.1) -> pd.DataFrame:
    """Predicted vs actual home win rate, bucketed by the predictor in `column`."""
    df = df.assign(bucket=bucket(df[column], width))
    table = clustered(df, {"predicted": df[column], "actual": df["home_won"]}, by="bucket")
    return table.pivot(index="group", columns="value", values=["mean", "ci95", "games"])


def gaps(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean model - market gap for each model, per group of `by`."""
    values = {name: df[col] - df["mid"] for name, col in PREDICTORS.items() if name != "market"}
    return clustered(df, values, by)


def sigma_curve(df: pd.DataFrame, sigmas: np.ndarray | None = None, by: str | None = None) -> pd.DataFrame:
    """For each sigma, how far the model sits from the market and how well it predicts.

    Uses the tip-off prior, so model and market start from the same number and sigma is
    the only thing that differs. Values per row: squared distance to the Kalshi mid, and
    log loss against the result.
    """
    if sigmas is None:
        sigmas = np.arange(10, 40.01, 0.5)
    y = df["home_won"].to_numpy()
    margin, secs_left = df["margin"].to_numpy(), df["secs_left"].to_numpy()
    prior_tipoff, mid = df["prior_tipoff"].to_numpy(), df["mid"].to_numpy()

    frames = []
    for sigma in sigmas:
        p = stern.stern_p(margin, secs_left, prior_tipoff, sigma)
        values = {"distance to market": (p - mid) ** 2, "log loss vs result": log_loss_rows(y, p)}
        frames.append(clustered(df, values, by).assign(sigma=sigma))
    return pd.concat(frames, ignore_index=True)


def best_sigma(curve: pd.DataFrame) -> pd.DataFrame:
    """The sigma minimising each value of a sigma_curve, per group."""
    best = curve.loc[curve.groupby(["group", "value"])["mean"].idxmin()]
    return best.pivot(index="group", columns="value", values="sigma")


# ---------- checks ----------

def run_unit_tests() -> None:
    """Offline checks on the pure functions, using small hand-made inputs (no network)."""
    # kalshi_events: home market picked; scalar settlement and unparseable ticker dropped
    markets = [
        {"ticker": "KXNBAGAME-25JUN19OKCIND-IND", "event_ticker": "KXNBAGAME-25JUN19OKCIND", "result": "yes"},
        {"ticker": "KXNBAGAME-25JUN19OKCIND-OKC", "event_ticker": "KXNBAGAME-25JUN19OKCIND", "result": "no"},
        {"ticker": "KXNBAGAME-26JAN08MIACHI-CHI", "event_ticker": "KXNBAGAME-26JAN08MIACHI", "result": "scalar"},
        {"ticker": "KXNBAGAME-NOTAGAME-X", "event_ticker": "KXNBAGAME-NOTAGAME", "result": "yes"},
    ]
    events = kalshi_events(markets)
    assert events.to_dict("records") == [{"date": pd.Timestamp("2025-06-19"), "away": "OKC", "home": "IND",
                                          "ticker": "KXNBAGAME-25JUN19OKCIND-IND", "home_yes": True}]

    # finder_to_games: one row per game from the home team's row
    rows = pd.DataFrame({
        "GAME_ID": ["0042400406", "0042400406"], "GAME_DATE": ["2025-06-19", "2025-06-19"],
        "TEAM_ABBREVIATION": ["IND", "OKC"], "MATCHUP": ["IND vs. OKC", "OKC @ IND"], "WL": ["W", "L"],
    })
    games = finder_to_games(rows, 2024)
    assert games.to_dict("records") == [{"game_id": "0042400406", "date": pd.Timestamp("2025-06-19"),
                                         "season": 2024, "home": "IND", "away": "OKC", "home_won": 1}]

    # losses and buckets
    assert np.isclose(log_loss_rows(np.array([1]), np.array([0.5]))[0], np.log(2))
    assert np.allclose(brier_rows(np.array([1, 0]), np.array([0.8, 0.8])), [0.04, 0.64])
    assert bucket(pd.Series([0.0, 0.05, 0.99, 1.0])).round(2).tolist() == [0.05, 0.05, 0.95, 0.95]

    # clustered: each game counts once however many rows it has
    df = pd.DataFrame({"game_id": ["a"] * 9 + ["b"], "x": [0.0] * 9 + [1.0]})
    out = clustered(df, {"x": df["x"]})
    assert out["games"].item() == 2 and out["mean"].item() == 0.5

    # sigma_curve recovers the sigma that generated the market prices
    rng = np.random.default_rng(0)
    n = 4000
    fake = pd.DataFrame({
        "game_id": rng.integers(0, 200, n).astype(str),
        "margin": rng.integers(-15, 16, n), "secs_left": rng.uniform(60, 2880, n),
        "prior_tipoff": rng.uniform(0.3, 0.7, n), "home_won": rng.integers(0, 2, n),
    })
    fake["mid"] = stern.stern_p(fake["margin"], fake["secs_left"], fake["prior_tipoff"], 25.0)
    assert best_sigma(sigma_curve(fake))["distance to market"].item() == 25.0

    print("unit tests: all pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stern model vs Kalshi over a random sample of games.")
    parser.add_argument("--n", type=int, default=N_GAMES, help=f"games to sample (default {N_GAMES})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"sampling seed (default {SEED})")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    args = parser.parse_args()

    run_unit_tests()
    games = build_game_list(args.refresh)
    # numpy's sample is a permutation prefix, so a larger n with the same seed is a superset
    sample = games.sample(n=min(args.n, len(games)), random_state=args.seed).sort_values("date")
    print(f"sampled {len(sample)} of {len(games)} games, "
          f"{sample['date'].min():%Y-%m-%d} to {sample['date'].max():%Y-%m-%d}")

    minutes, failures = collect(sample, args.refresh)
    minutes.to_parquet(MINUTES_PATH)
    print(f"saved {MINUTES_PATH} ({minutes['game_id'].nunique()} games, {len(minutes):,} minutes)")
    if not failures.empty:
        print(f"\n{len(failures)} games failed:\n{failures.to_string(index=False)}")

    df = load_results()
    print("\nlog loss and Brier, each game weighted equally (lower is better):")
    for metric in LOSSES:
        table = score_table(df, metric=metric)
        print(table[["value", "mean", "ci95"]].round(4).to_string(index=False, header=[metric, "mean", "+/-95%"]))
    print("\nsigma that best matches the market vs our fit of", SIGMA)
    print(best_sigma(sigma_curve(df)).to_string())

    import compare_plots  # imported here because compare_plots itself imports this module
    paths = compare_plots.save_all(df)
    print(f"\nsaved {len(paths)} charts to {paths[0].parent} "
          "(discussed in notebooks/kalshi_comparison.ipynb)")


if __name__ == "__main__":
    main()
