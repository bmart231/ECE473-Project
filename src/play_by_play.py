"""
Play-by-play: turn the raw nbastats CSVs in data/raw/pbp/into two clean tables

plays(one row per scoring event):
    game_id
    sesson
    period
    secs_left
    margin
    home_won
    
games (one row per game): the input prior.buid_prior() 
    game_id, season, home, away, home_won
"""

from pathlib import Path
import pandas as pd 

# basically js getting -> /Users/bmart231/ECE473-Project/data/raw/pbp, for users 
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pbp"
PERIOD_SECS = 720   # 12 min quarters (seconds)
OT_SECS     = 300   # 5 min OT (seconds)

def clock_to_secs_left(period: int, clock: str) -> float:
    """ 10:50 in period 2 -> seconds left in regulation
    Using period and not quarter because if OT quarter 5 doens't make sense
    and pylance will throw some errors ig

    Args:
        period (int): period that it is in
        clock (str): turn to number

    Returns:
        returns float value of time left
    """
    
    mins, secs = clock.split(":")   # "10:50" -> ["10", "50"]
    minutes = float(mins)
    seconds = float(secs)
    
    if period <= 4:                 # in regulation
        return (4 - period) * PERIOD_SECS + minutes * 60 + seconds
    return minutes * 60 + seconds   # overtime, only this period counts
    
    

def load_plays(seasons:list[int]) -> pd.DataFrame:
    """ Read nbastats_{year}.csv and nbastats_po_{year}.csv for each seasons.
    
    

    Args:
        seasons (list[int]): get season from nbastats api

    Returns:
        columns: game_id, season, period, secs_left, margin, home_won
    print one line count per season (games and rows) -> since 2020 season cut short
    """
    frames = [] # one df per file, concatenate at the end
    
    for year in seasons:
        for name in (f"nbastats_{year}", f"nbastats_po_{year}"):
            # getting this /Users/bmart231/ECE473-Project/data/raw/pbp/nbastats_2015.csv for users
            path = RAW_DIR / f"{name}.csv"
            
            # read only cols we need
            df = pd.read_csv(
                path,
                usecols=["GAME_ID", "PERIOD", "PCTIMESTRING", "SCORE"], 
                dtype={"GAME_ID": str} # read as text, not number
            )
            # the CSVs themselves dropped the leading zeros (22400001); NBA ids are 10 chars
            df["GAME_ID"] = df["GAME_ID"].str.zfill(10)

            # scoring plays only 
            df = df.dropna(subset=["SCORE"])
            
            # "away - home" text -> margin
            # two cols of strings -> astype(int) turns them to numbers; column 0 is away, 1 is home
            score = df["SCORE"].str.split(" - ", expand=True).astype(int) 
            # margin is home - away
            df["margin"] = score[1] - score[0]
            
            # seconds left, one value per row
            # zip two cols + call function on each pair
            df["secs_left"] = [
                clock_to_secs_left(p, c) for p, c in zip(df["PERIOD"], df["PCTIMESTRING"])
            ]
            
            # check who won
            final_margin = df.groupby("GAME_ID")["margin"].last()
            df["home_won"] = (df["GAME_ID"].map(final_margin) > 0).astype(int)
            
            df["season"] = year
            df = df.rename(columns={"GAME_ID": "game_id", "PERIOD": "period"})
            frames.append(df[["game_id", "season", "period", "secs_left", "margin", "home_won"]])
    
    plays = pd.concat(frames, ignore_index=True)
    
    for year, grp in plays.groupby("season"):
        print(f"{year}: {grp.game_id.nunique():>5} games  {len(grp):>8,} scoring rows")
    return plays

def games_table(seasons: list[int]) -> pd.DataFrame:
    """One row per game: game_id, season, home, away, home_won. In play order.

    The raw files have no home/away team column, so we infer it from which description
    column is filled: HOMEDESCRIPTION rows belong to the home team, VISITORDESCRIPTION
    rows to the away team. PLAYER1_TEAM_ABBREVIATION on those rows is (usually) that
    team, so take the most common value per game. (About 6% of rows carry the other
    team, e.g. blocks and fouls drawn, so "first row" would sometimes be wrong.)
    """
    frames = []

    for year in seasons:
        for name in (f"nbastats_{year}", f"nbastats_po_{year}"):
            path = RAW_DIR / f"{name}.csv"

            #read the id, both description columns, and the team abbreviation
            df = pd.read_csv(
                path,
                usecols=["GAME_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "PLAYER1_TEAM_ABBREVIATION"],
                dtype={"GAME_ID": str},
            )
            df["GAME_ID"] = df["GAME_ID"].str.zfill(10)

            # rows that have a team abbreviation at all (timeouts, period starts don't)
            df = df.dropna(subset=["PLAYER1_TEAM_ABBREVIATION"])

            # team = most common abbreviation on home-description rows, per game
            # mode() returns the most common value(s) as a Series; .iloc[0] takes the first
            home = (
                df[df["HOMEDESCRIPTION"].notna()]
                .groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"]
                .agg(lambda s: s.mode().iloc[0])
            )

            # same for the away team, using VISITORDESCRIPTION
            away = df[df["VISITORDESCRIPTION"].notna()].groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(lambda s: s.mode().iloc[0])

            # one row per game. home and away are both indexed by GAME_ID,
            games = pd.DataFrame({"home": home, "away": away})
            games.index.name = "game_id"
            games = games.reset_index()
            games["season"] = year
            frames.append(games)

    games = pd.concat(frames, ignore_index=True)

    # home_won from load_plays(). It's per-row there; take one per game.
    # plays.groupby("game_id")["home_won"].first() -> Series indexed by game_id
    # games["home_won"] = games["game_id"].map(that_series)
    plays = load_plays(seasons)
    won = plays.groupby("game_id")["home_won"].first()
    games["home_won"] = games["game_id"].map(won)

    # order. Sort by season, then game_id. Within a season, regular-season ids
    # 002...) sort before playoffs (004...), and ids are sequential within each.
    games = games.sort_values(["season", "game_id"]).reset_index(drop=True)

    return games[["game_id", "season", "home", "away", "home_won"]]


if __name__ == "__main__":
    plays = load_plays([2024])
    print(plays.head())
    print("home win rate:", plays.groupby("game_id")["home_won"].first().mean())

    games = games_table([2024])
    print(games.head())
    assert games.game_id.is_unique
    assert (games.home != games.away).all()
    print(len(games), "games")  # expect 1314 = 1230 regular season + 84 playoffs