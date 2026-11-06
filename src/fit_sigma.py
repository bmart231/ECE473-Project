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
    
    rows["margins"] = rows["margin"].fillna(0)
    
    rows = rows.drop(columns=["t", "secs_left_play"])
    
    per_game = plays.groupby("game_id")[["season", "home_won"]].first()
    rows["season"] = rows["game_id"].map(per_game["season"])
    rows["home_won"] = rows["game_id"].map(per_game["home_won"])
    rows["p_pre"] = rows["game_id"].map(priors)
    return rows


def log_loss(y: np.ndarray)



    
    