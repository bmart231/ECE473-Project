"""
Luh calm api calling to nba api to make sure it works

Output: 2025-2026 Seaon (including finals)
"""

from nba_api.live.nba.endpoints import playbyplay
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

SEASON = "2025-26" # only using this for now to keep data set small 

# team records 25-26 -> col
teams = leaguedashteamstats.LeagueDashTeamStats(season=SEASON, timeout=30).get_data_frames()[0]
teams = teams.sort_values("W_PCT", ascending=False) # winning teams
print(f"{SEASON} team records (top 10):")
# P-M = plus minus lmk whether to change it to whole thing. head(10) -> top 10 teams
print(teams[["TEAM_NAME", "GP", "W", "L", "W_PCT", "PTS", "PLUS_MINUS"]].head(10).to_string(index=False))

# regualr season games: one row per team per game, 2 rows per game
games = leaguegamefinder.LeagueGameFinder(
    season_nullable=SEASON, league_id_nullable="00",
    season_type_nullable="Regular Season", 
    timeout=30, 
).get_data_frames()[0]

print(f"\nRegular-Season Games: {games['GAME_ID'].nunique()}")
print(games[["GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]].head(4).to_string(index=False))

print()
# play by play -> 2025 finals game 6
actions = playbyplay.PlayByPlay("0042400406", timeout=30).get_dict()["game"]["actions"]
for action in actions[:3] + actions[-2:]:
    print(f"  {action['timeActual']}  Q{action['period']} {action['clock']}  "
            f"{action['scoreHome']}-{action['scoreAway']}  {action['description']}")

