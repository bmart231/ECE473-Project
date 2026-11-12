"""2025 Finals Game 6 (OKC at IND): Kalshi mid vs Stern model, against UTC time.

First time the model meets the market it will trade against. Tests the join between
game clock and real time and shows how far apart the model and the market are.

Run: uv run python src/prototype_game_6.py
Output: offline unit tests, live-data checks, then one chart (also saved to figures/).
Notes and checks for each step: Finals2025_Prototype.md
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from nba_api.live.nba.endpoints import playbyplay

import stern

GAME_ID = "0042400406"
KALSHI_TICKER = "KXNBAGAME-25JUN19OKCIND-IND"  # "Indiana wins"; IND is home, so YES = P(home wins)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CANDLE_START = "2025-06-20T00:30:00"  # UTC, a bit before tip-off
CANDLE_END = "2025-06-20T03:20:00"    # UTC, just before the market closed

SIGMA = 19.64   # pooled fit, sigma_results.md
P_PRE = 0.345   # IND pregame probability from the explainer -> swap in Elo later
PERIOD_SECS = 720

FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "game6_kalshi_vs_stern.png"

# live feed clock is an ISO duration: 'PT11M44.00S' -> minutes '11', seconds '44.00'
CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")


# ---------- step 1: Kalshi mid per minute ----------

def to_ts(text: str) -> int:
    """'2025-06-20T00:30:00' (UTC) -> unix seconds, which the candle endpoint takes."""
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def candles_to_mid(candles: list[dict]) -> pd.DataFrame:
    """Raw Kalshi candles -> table with columns: time (UTC), mid.

    Prices come back as dollar strings ('0.3300'). A minute with no bid or no ask
    has no mid, so that row is dropped rather than guessed.
    """
    df = pd.DataFrame({
        "time": pd.to_datetime([c["end_period_ts"] for c in candles], unit="s", utc=True),
        "bid": pd.to_numeric([c["yes_bid"].get("close") for c in candles], errors="coerce"),
        "ask": pd.to_numeric([c["yes_ask"].get("close") for c in candles], errors="coerce"),
    })
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df.dropna(subset=["mid"])[["time", "mid"]].reset_index(drop=True)


def load_kalshi_mid() -> pd.DataFrame:
    """1-minute candles for KALSHI_TICKER -> table with columns: time, mid."""
    response = requests.get(
        f"{KALSHI_BASE}/historical/markets/{KALSHI_TICKER}/candlesticks",
        params={"start_ts": to_ts(CANDLE_START), "end_ts": to_ts(CANDLE_END), "period_interval": 1},
        timeout=30,
    )
    response.raise_for_status()
    return candles_to_mid(response.json()["candlesticks"])


# ---------- step 2: plays with real time ----------

def _secs_left(period, minutes, seconds):
    """Seconds left in regulation. Works on scalars and pandas columns alike."""
    return (4 - period) * PERIOD_SECS + minutes * 60 + seconds


def clock_to_secs_left(period: int, clock: str) -> float:
    """Period 1, 'PT11M44.00S' -> 2864.0 seconds left in regulation."""
    match = CLOCK_RE.fullmatch(clock)
    if match is None:
        raise ValueError(f"unexpected clock format: {clock!r}")
    return _secs_left(period, float(match[1]), float(match[2]))


def actions_to_plays(actions: list[dict]) -> pd.DataFrame:
    """Raw live play-by-play actions -> table with columns: time, margin, secs_left.

    Vectorized: the clock column is parsed in one str.extract instead of a row loop.
    Overtime (period 5+) gives secs_left <= 0, which stern_p treats as "margin decides".
    """
    raw = pd.DataFrame(actions, columns=["timeActual", "period", "clock", "scoreHome", "scoreAway"])
    clock = raw["clock"].str.extract(CLOCK_RE.pattern).astype(float)
    if clock.isna().any().any():
        bad = raw.loc[clock.isna().any(axis=1), "clock"].iloc[0]
        raise ValueError(f"unexpected clock format: {bad!r}")

    return pd.DataFrame({
        "time": pd.to_datetime(raw["timeActual"], utc=True),
        "margin": raw["scoreHome"].astype(int) - raw["scoreAway"].astype(int),
        "secs_left": _secs_left(raw["period"].astype(int), clock[0], clock[1]),
    })


def load_plays() -> pd.DataFrame:
    """Live play-by-play for GAME_ID -> table with columns: time, margin, secs_left."""
    actions = playbyplay.PlayByPlay(GAME_ID, timeout=30).get_dict()["game"]["actions"]
    return actions_to_plays(actions)


# ---------- step 3: model probability ----------

def add_model_p(plays: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of plays with column p = Stern P(home wins), one vectorized call."""
    p = stern.stern_p(plays["margin"].to_numpy(), plays["secs_left"].to_numpy(), P_PRE, SIGMA)
    return plays.assign(p=p)


# ---------- step 4: plot ----------

def plot(kalshi: pd.DataFrame, plays: pd.DataFrame, save_to: Path | None = FIGURE_PATH) -> None:
    """Kalshi mid and Stern model against UTC time on one chart."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(kalshi["time"], kalshi["mid"], label="Kalshi mid")
    ax.plot(plays["time"], plays["p"], drawstyle="steps-post", label="Stern model")

    ax.set_ylim(0, 1)
    ax.set_xlabel("time (UTC, Jun 20 2025)")
    ax.set_ylabel("P(Indiana wins)")
    ax.set_title("2025 Finals Game 6, OKC at IND: Kalshi vs Stern model")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_to), dpi=150)
        print(f"saved   {save_to}")
    plt.show()


# ---------- checks ----------

def _raises(error: type[Exception], func, *args) -> bool:
    """True if func(*args) raises error. Lets the tests below assert on bad input."""
    try:
        func(*args)
    except error:
        return True
    return False


def run_unit_tests() -> None:
    """Offline checks on every function, using small hand-made inputs (no network)."""
    # to_ts
    assert to_ts("1970-01-01T00:00:00") == 0
    assert to_ts(CANDLE_START) == 1750379400

    # clock parser
    assert clock_to_secs_left(1, "PT12M00.00S") == 2880
    assert clock_to_secs_left(1, "PT11M44.00S") == 2864
    assert clock_to_secs_left(2, "PT00M00.00S") == 1440
    assert clock_to_secs_left(4, "PT00M30.50S") == 30.5
    assert clock_to_secs_left(4, "PT00M00.00S") == 0
    assert _raises(ValueError, clock_to_secs_left, 1, "11:44")

    # candles: strings -> floats, missing quote row dropped, times in UTC
    candles = [
        {"end_period_ts": 1750379400, "yes_bid": {"close": "0.3300"}, "yes_ask": {"close": "0.3500"}},
        {"end_period_ts": 1750379460, "yes_bid": {"close": None}, "yes_ask": {"close": "0.3600"}},
        {"end_period_ts": 1750379520, "yes_bid": {}, "yes_ask": {"close": "0.3600"}},
        {"end_period_ts": 1750379580, "yes_bid": {"close": "0.9900"}, "yes_ask": {"close": "1.0000"}},
    ]
    kalshi = candles_to_mid(candles)
    assert list(kalshi.columns) == ["time", "mid"]
    assert len(kalshi) == 2
    assert np.allclose(kalshi["mid"], [0.34, 0.995])
    assert str(kalshi["time"].dt.tz) == "UTC"
    assert kalshi["time"].iloc[0] == pd.Timestamp("2025-06-20T00:30:00Z")

    # actions: string scores -> int margin, vectorized clock agrees with scalar parser
    actions = [
        {"timeActual": "2025-06-20T00:40:49.9Z", "period": 1, "clock": "PT12M00.00S",
         "scoreHome": "0", "scoreAway": "0", "description": "extra keys are ignored"},
        {"timeActual": "2025-06-20T01:30:00.0Z", "period": 2, "clock": "PT05M30.00S",
         "scoreHome": "50", "scoreAway": "40"},
        {"timeActual": "2025-06-20T03:11:21.8Z", "period": 4, "clock": "PT00M00.00S",
         "scoreHome": "108", "scoreAway": "91"},
    ]
    plays = actions_to_plays(actions)
    assert list(plays.columns) == ["time", "margin", "secs_left"]
    assert plays["margin"].tolist() == [0, 10, 17]
    assert plays["secs_left"].tolist() == [clock_to_secs_left(a["period"], a["clock"]) for a in actions]
    assert plays["secs_left"].tolist() == [2880, 1770, 0]
    assert str(plays["time"].dt.tz) == "UTC"
    assert _raises(ValueError, actions_to_plays, [{**actions[0], "clock": "12:00"}])

    # model: tip-off returns the prior, a lead pushes p up, final buzzer is 0/1
    modeled = add_model_p(plays)
    assert "p" not in plays.columns  # input left untouched
    assert abs(modeled["p"].iloc[0] - P_PRE) < 1e-9
    assert P_PRE < modeled["p"].iloc[1] < 1
    assert modeled["p"].iloc[-1] == 1.0

    print("unit tests: all pass")


def main() -> None:
    kalshi = load_kalshi_mid()
    plays = add_model_p(load_plays())
    print(f"kalshi: {len(kalshi)} candles, last mid {kalshi['mid'].iloc[-1]:.3f}")
    print(f"plays:  {len(plays)} rows, last margin {plays['margin'].iloc[-1]:+d}, "
          f"first p {plays['p'].iloc[0]:.3f}, last p {plays['p'].iloc[-1]:.3f}")

    # live-data checks from Finals2025_Prototype.md
    assert 160 <= len(kalshi) <= 175, len(kalshi)
    assert kalshi["mid"].iloc[-1] > 0.95
    assert plays["margin"].iloc[-1] == 17 and plays["secs_left"].iloc[-1] == 0
    assert abs(plays["p"].iloc[0] - P_PRE) < 1e-6
    assert plays["p"].iloc[-1] == 1.0
    # feed order is game order (secs_left never goes up); timeActual can step back by ~1s
    # where a sub was logged between free throws, so allow that but nothing bigger
    assert plays["secs_left"].is_monotonic_decreasing
    assert plays["time"].diff().dt.total_seconds().min() > -5
    # both series cover the same evening (a 4-hour offset means a time-zone bug)
    assert kalshi["time"].min() <= plays["time"].min() and plays["time"].max() <= kalshi["time"].max()
    print("live checks: all pass")

    plot(kalshi, plays)


if __name__ == "__main__":
    run_unit_tests()
    main()
