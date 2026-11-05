"""Pregame prior: P(home wins) before tip-off, one number per game.
Output table (`prior`):  game_id (str, 10 chars) | p_pre (float in (0, 1))
"""

import pandas as pd

START_ELO = 1500.0          # default elo coming into game
K = 20.0                    # learning rate, got this from internet so idk
HOME_ADV = 40               # added boost for home court advantage, subject to change 
SEASON_CARRYOVER = 0.75     # new-season rating = carryover * old + (1 - carryover) * START_ELO


def expected_home_win(elo_home: float, elo_away: float) -> float:
    """Simple calculation to return prediction for a home game win
    Args:
        elo_home (float): team's elo in home games
        elo_away (float): team's elo in away games

    Returns:
        float: the expected value of a home win
    """
    return 1 / (1 + 10 ** ((elo_away - (elo_home + HOME_ADV)) / 400))
    
    


def update(elo_home: float, elo_away: float, home_won: int) -> tuple[float, float]:
    # updating elo using previous method
    delta = K * (home_won - expected_home_win(elo_home, elo_away))
    return elo_home + delta, elo_away - delta


def build_prior(games: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}  # team -> elo
    rows: list[tuple[str, float]] = []
    current_season: int | None = None  # so the season reset fires once per boundary

    # Pull columns out as plain Python lists: itertuples() is untyped (every field is a
    # pandas "Scalar"), which makes Pylance complain downstream. Also faster.
    for game_id, season, home, away, home_won in zip(
        games["game_id"].tolist(), games["season"].tolist(),
        games["home"].tolist(), games["away"].tolist(), games["home_won"].tolist(),
    ):
        # new season: pull every team part of the way back toward average
        # (on the very first game `ratings` is empty, so this loop does nothing)
        if season != current_season:
            for team in ratings:
                ratings[team] = SEASON_CARRYOVER * ratings[team] + (1 - SEASON_CARRYOVER) * START_ELO
            current_season = season

        # a team not seen yet starts at START_ELO
        elo_home = ratings.get(home, START_ELO)
        elo_away = ratings.get(away, START_ELO)

        # the prior is the prediction made before the result is known
        p_pre = expected_home_win(elo_home, elo_away)
        rows.append((game_id, p_pre))

        # now learn from the result
        ratings[home], ratings[away] = update(elo_home, elo_away, home_won)

    return pd.DataFrame(rows, columns=["game_id", "p_pre"])


if __name__ == "__main__":  
    p_even = expected_home_win(1500, 1500)
    assert 0.5 < p_even < 0.6, p_even