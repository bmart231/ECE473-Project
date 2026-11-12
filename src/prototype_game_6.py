"""This is the file where game 6 finals for pacers vs okc will be
Use this to test against Kalshi predictions

Run: uv run python src/prototype_game_6.py
Output: one chart, Kalshi mid and Stern model probability against UTC time.
Notes and checks for each step: Finals2025_Prototype.md
"""

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from nba_api.live.nba.endpoints import playbyplay

import stern

GAME_ID = "0042400406" # game id
KALSHI_TICKER = "KXNBAGAME-25JUN19OKCIND-IND" # indy wins

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CANDLE_START = "2025-06-20T00:30:00"  # UTC, a bit before tip-off
CANDLE_END = "2025-06-20T03:20:00"    # UTC, just before the market closed

SIGMA = 19.64   # pooled fit from what we calculated
P_PRE = 0.345   # ind pregame probability from the explainer -> swap in Elo later
PERIOD_SECS = 720

def to_ts(text: str) -> int:
    """'2025-06-20T00:30:00' (UTC) -> unix seconds, which the candle endpoint takes.
    """
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())
    

def load_kalshi_mid() -> pd.DataFrame:
    """1-minute candles for KALSHI_TICKER -> table with columns: time, mid.

    Steps:
    1. GET {KALSHI_BASE}/historical/markets/{KALSHI_TICKER}/candlesticks
        with params start_ts, end_ts (use to_ts) and period_interval=1
    2. the list is under the "candlesticks" key
    3. for each candle:
        time = end_period_ts (unix seconds) as a UTC datetime
        mid  = (float(yes_bid close) + float(yes_ask close)) / 2
    4. return pd.DataFrame with columns ["time", "mid"]

    Check: about 170 rows, mid ends near 1.00.
    """
    
    

def clock_to_secs_left(period: int, clock: str) -> float:
    """Period 1, 'PT11M44.00S' -> 2864.0 seconds left in regulation.

    The live feed's clock is an ISO duration, not '11:44':
    minutes = the number between 'PT' and 'M'
    seconds = the number between 'M' and 'S'
    secs_left = (4 - period) * PERIOD_SECS + minutes * 60 + seconds
    """
    # TODO
    raise NotImplementedError


def load_plays() -> pd.DataFrame:
    """Live play-by-play for GAME_ID -> table with columns: time, margin, secs_left.

    Steps:
    1. actions = playbyplay.PlayByPlay(GAME_ID, timeout=30).get_dict()["game"]["actions"]
    2. put them in a DataFrame; keep timeActual, period, clock, scoreHome, scoreAway
    3. time      = pd.to_datetime(timeActual, utc=True)
        margin    = int(scoreHome) - int(scoreAway)
        secs_left = clock_to_secs_left(period, clock) for each row
    4. return the columns ["time", "margin", "secs_left"]

    Check: last row has margin +17 and secs_left 0.
    """
    # TODO
    raise NotImplementedError


# ---------- step 3: model probability ----------

def add_model_p(plays: pd.DataFrame) -> pd.DataFrame:
    """Add column p = Stern P(Indiana wins) for every play.

    One call to stern.stern_p with the margin and secs_left columns, P_PRE and SIGMA.

    Check: first p is about 0.345, last p is 1.0.
    """
    # TODO
    raise NotImplementedError


# ---------- step 4: plot ----------

def plot(kalshi: pd.DataFrame, plays: pd.DataFrame) -> None:
    """Both series against UTC time on one chart.

    Steps:
    1. fig, ax = plt.subplots(figsize=(11, 5))
    2. Kalshi: ax.plot(time, mid, label="Kalshi mid")
    3. model:  ax.plot(time, p, drawstyle="steps-post", label="Stern model")
    4. y-axis from 0 to 1, axis labels, title, legend
    5. plt.show()   (or fig.savefig(...) to keep a copy)
    """
    # TODO
    raise NotImplementedError


def main() -> None:
    kalshi = load_kalshi_mid()
    print(f"kalshi: {len(kalshi)} candles, last mid {kalshi['mid'].iloc[-1]:.3f}")

    plays = add_model_p(load_plays())
    print(f"plays:  {len(plays)} rows, last margin {plays['margin'].iloc[-1]:+d}, "
          f"first p {plays['p'].iloc[0]:.3f}, last p {plays['p'].iloc[-1]:.3f}")

    plot(kalshi, plays)


if __name__ == "__main__":
    # clock parser checks (these fail until clock_to_secs_left is written)
    assert clock_to_secs_left(1, "PT12M00.00S") == 2880
    assert clock_to_secs_left(1, "PT11M44.00S") == 2864
    assert clock_to_secs_left(2, "PT00M00.00S") == 1440
    assert clock_to_secs_left(4, "PT00M30.50S") == 30.5
    assert clock_to_secs_left(4, "PT00M00.00S") == 0

    main()
