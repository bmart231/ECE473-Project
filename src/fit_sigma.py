"""Fit sigma for the Stern model on 2015-23 play-by-play; check on 2024-25.

Run: uv run python src/fit_sigma.py
Outputs: fitted sigma (pooled and per season), train/validation log loss and Brier,
and the same scores for the pregame-only baseline so we can see what the live update adds.
"""
import pandas as pd 
import numpy as np 
from scipy.optimize import minimize_scalar

import play_by_play
import prior
import stern

TRAIN_SEASONS   = list(range(2015, 2024)) # train these seaons, will use 2025 as testing
VALID_SEASONS   = [2024]                  # 2024-25 season
GRID_SECS       = 15                      # one row every 15 s of game clock

EPS = 1e-6  # keeps log() finite when p is exactly 0 or 1 (the secs_left == 0 rows)



def build_rows(plays: pd.DataFrame, priors: pd.Series) -> pd.DataFrame:
    """One row per game per GRID_SECS of regulation clock, with the margin as of the most recent scorig play.
    Column: game_id, season, secs_left, margin, home_won, p_pre"""
    plays = plays[plays["period"] <=4 ] # withing regulation
    
    grid = np.arange(stern.GAME_SECS - GRID_SECS, -1, -GRID_SECS, dtype=float)
    ids = plays["game_id"].unique()
    rows = pd.DataFrame({"game_id": np.repeat(ids, len(grid)), "secs_left": np.tile(grid, len(ids))})
    
    rows["t"] = - rows["secs_left"]
    p = plays[["game_id", "secs_left", "margin"]].assign(t=lambda d: -d["secs_left"]).sort_values("t")
    rows = pd.merge_asof(rows.sort_values("t"), p, on="t", by="game_id", direction="backward", suffixes=("", "_play"))
    
    rows["margin"] = rows["margin"].fillna(0)
    
    rows = rows.drop(columns=["t", "secs_left_play"])
    
    per_game = plays.groupby("game_id")[["season", "home_won"]].first()
    rows["season"] = rows["game_id"].map(per_game["season"])
    rows["home_won"] = rows["game_id"].map(per_game["home_won"])
    rows["p_pre"] = rows["game_id"].map(priors)
    return rows


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Average -log(probability given to what actually happened). Lower is better."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def brier(y: np.ndarray, p: np.ndarray) -> float:
    """It works by calculating the mean squared difference between
    the predicted probabilities and the actual binary outcomes."""
    return float(np.mean((p - y) ** 2))


def fit_sigma(rows: pd.DataFrame) -> float:
    """The sigma that minimizes log loss on these rows."""
    y = rows["home_won"].to_numpy()
    
    def loss(sigma: float) -> float:
        p = stern.stern_p(rows["margin"], rows["secs_left"], rows["p_pre"], sigma)
        return log_loss(y, p)
    
    result = minimize_scalar(loss, bounds=(5, 30), method="bounded")
    return float(result.x)   

def report(name: str, rows: pd.DataFrame, sigma: float) -> None:
    """Print log loss and Brier for the live model and the pregame-only baseline."""
    y = rows["home_won"].to_numpy()
    p_live = stern.stern_p(rows["margin"], rows["secs_left"], rows["p_pre"], sigma)
    p_base = rows["p_pre"].to_numpy()  # never updates during the game
    print(f"{name:<6} live:     log loss {log_loss(y, p_live):.4f}  Brier {brier(y, p_live):.4f}")
    print(f"{name:<6} baseline: log loss {log_loss(y, p_base):.4f}  Brier {brier(y, p_base):.4f}")
    
    
# function to see whether the sigma values found are actaully well calibrated
def calibration(rows: pd.DataFrame, sigma: float) -> None:
    """Predicted probability vs actual home win rate, by quarter and probability bucket."""
    df = rows.assign(
        p=stern.stern_p(rows["margin"], rows["secs_left"], rows["p_pre"], sigma),
        quarter=4 - (rows["secs_left"] // 720).clip(upper=3).astype(int),
    )
    df["p_bin"] = pd.cut(df["p"], np.linspace(0, 1, 11).tolist(), include_lowest=True)

    table = df.groupby(["quarter", "p_bin"], observed=True).agg(
        n=("p", "size"), predicted=("p", "mean"), actual=("home_won", "mean")
    )
    table["gap"] = table["actual"] - table["predicted"]
    print(table.round(3).to_string())

def main() -> None:
    seasons = TRAIN_SEASONS + VALID_SEASONS

    # Elo has to run through every season in order, so build priors on all of them
    games = play_by_play.games_table(seasons)
    priors = prior.build_prior(games).set_index("game_id")["p_pre"]

    plays = play_by_play.load_plays(seasons)
    rows = build_rows(plays, priors)
    assert rows["p_pre"].notna().all(), "some games have no prior"

    train = rows[rows["season"].isin(TRAIN_SEASONS)]
    valid = rows[rows["season"].isin(VALID_SEASONS)]

    sigma = fit_sigma(train)
    print(f"\npooled sigma (2015-23): {sigma:.2f}")
    for season, grp in train.groupby("season"):
        print(f"  {season}: {fit_sigma(grp):.2f}")

    print()
    report("train", train, sigma)
    report("valid", valid, sigma)
    
    print()
    calibration(valid, sigma)


if __name__ == "__main__":
    main()





