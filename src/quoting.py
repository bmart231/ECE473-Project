"""The quoting rule (QuotingRule_Explainer.pdf): one bid and one ask per game per second.

Pure functions, no network or files, so the backtester and a live bot call the same code.
Every price in this module is in cents (0-100), the unit Kalshi ticks in.

  1. center         c  = w * p_model + (1 - w) * p_market   (market only while the model is stale)
  2. half_spread    h  = max(h_min, a + b * basket sensitivity)
  3. skewed_center  c' = c - (position / q_max) * h          (lean the quotes back toward flat)
  4. jump guard     post nothing for `cooldown` s after the market moves >= `jump` in `lookback` s
  5. quote          bid = floor(c' - h), ask = ceil(c' + h), never cross the book, stop near 0/100

The baselines are the same rule with other Params: w = 0 (market only), b = 0 (fixed width),
cooldown = 0 (guard off).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stern import basket_sensitivity

MAKER_FEE = 0.0175  # Kalshi fee rates: rate * contracts * P * (1 - P), rounded up per order
TAKER_FEE = 0.07
EPS = 1e-9          # float slack: 100 * 0.58 - 100 * 0.55 is 2.99999999999998, not 3


@dataclass(frozen=True)
class Params:
    """The knobs (explainer's starting guesses; the backtest tunes them)."""
    w: float = 0.2          # weight on our model in the center
    a: float = 0.5          # base half-spread
    b: float = 0.3          # extra half-spread per cent of basket sensitivity
    h_min: float = 1.0      # smallest half-spread (one tick)
    q_max: int = 100        # position limit, contracts; also sets the skew strength
    jump: float = 3.0       # a move this big within `lookback` s trips the guard
    lookback: int = 2       # seconds
    cooldown: float = 15.0  # seconds quotes stay pulled after a jump (0 = guard off)
    edge: float = 3.0       # no quotes when the market is below edge or above 100 - edge
    size: int = 10          # contracts per quote
    sigma: float = 22.0     # volatility Kalshi prices at (799-game comparison)


@dataclass(frozen=True)
class MarketState:
    """What the bot knows at one moment."""
    p_market: float                   # Kalshi mid, or the recent trade price when the book is thin
    best_bid: float
    best_ask: float
    p_model: float | None             # Stern P(win); None while our feed is behind
    secs_left: float                  # game clock (overtime: seconds left in the period)
    position: int = 0                 # contracts held, + = long YES
    seconds_since_jump: float = math.inf  # from seconds_since_jump()


@dataclass(frozen=True)
class Quote:
    bid: int | None  # None: not bidding (long at the position limit)
    ask: int | None  # None: not offering (short at the position limit)
    size: int


def center(state: MarketState, params: Params) -> float:
    """Step 1: fair value, mostly the market (as accurate as our model and 13 s faster)."""
    if state.p_model is None:
        return state.p_market
    return params.w * state.p_model + (1 - params.w) * state.p_market


def half_spread(state: MarketState, params: Params) -> float:
    """Step 2: quote wider when one basket would move the price more (late, close games)."""
    sensitivity = 100 * basket_sensitivity(state.p_market / 100, state.secs_left, params.sigma)
    return max(params.h_min, params.a + params.b * float(sensitivity))


def skewed_center(c: float, h: float, position: int, params: Params) -> float:
    """Step 3: shift both quotes to unload a position; a full h at the limit."""
    lean = max(-1.0, min(1.0, position / params.q_max))
    return c - lean * h


def seconds_since_jump(prices: ArrayLike, params: Params) -> np.ndarray:
    """Step 4's detector, over a whole game at once: for a price sampled every second,
    seconds since it last moved >= params.jump over params.lookback s (inf: never yet)."""
    p = np.asarray(prices, dtype=float)
    n, k = len(p), params.lookback
    jumped = np.zeros(n, dtype=bool)
    jumped[k:] = np.abs(p[k:] - p[:-k]) >= params.jump - EPS
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(jumped, t, -1))
    return np.where(last >= 0, t - last, np.inf)


def quote(state: MarketState, params: Params = Params()) -> Quote | None:
    """Steps 1-5: bid and ask in whole cents, or None to post nothing (guard, or a price
    near 0 or 100). Rounding goes outward, so it never narrows the spread."""
    if state.seconds_since_jump < params.cooldown:
        return None
    if not params.edge <= state.p_market <= 100 - params.edge:
        return None
    h = half_spread(state, params)
    c = skewed_center(center(state, params), h, state.position, params)
    bid = min(math.floor(c - h + EPS), math.ceil(state.best_ask - EPS) - 1)  # below best ask
    ask = max(math.ceil(c + h - EPS), math.floor(state.best_bid + EPS) + 1)  # above best bid
    return Quote(bid=None if state.position >= params.q_max else max(bid, 1),
                 ask=None if state.position <= -params.q_max else min(ask, 99),
                 size=params.size)


def kalshi_fee(price: float, contracts: int, rate: float = MAKER_FEE) -> int:
    """Cents for one order of `contracts` at `price`: rate * C * P * (1 - P), rounded up."""
    return math.ceil(rate * contracts * price * (100 - price) / 100 - EPS)


# ---------------------------------------------------------------- unit tests

def _state(**changes) -> MarketState:
    """The explainer's worked example (4th quarter, 5:00 left, long 30), with changes."""
    example = dict(p_market=62, best_bid=61, best_ask=63, p_model=66, secs_left=300, position=30)
    return MarketState(**(example | changes))


def test_center() -> None:
    assert math.isclose(center(_state(), Params()), 62.8)               # 0.2 * 66 + 0.8 * 62
    assert center(_state(p_model=None), Params()) == 62                 # model stale: market only
    assert center(_state(), Params(w=0)) == 62                          # baseline ignores the model


def test_half_spread() -> None:
    assert abs(half_spread(_state(), Params()) - 3.68) < 0.01           # 0.5 + 0.3 * 10.6
    assert abs(half_spread(_state(p_market=50, secs_left=2700), Params()) - 1.62) < 0.01  # 3.74c
    assert half_spread(_state(p_market=95, secs_left=2880), Params()) == 1.0  # floor: one tick
    assert half_spread(_state(secs_left=10), Params(a=2, b=0)) == 2     # fixed-width baseline


def test_skewed_center() -> None:
    p = Params()
    assert skewed_center(60, 3, 0, p) == 60
    assert math.isclose(skewed_center(60, 3, 30, p), 59.1)              # long: lean down
    assert skewed_center(60, 3, -100, p) == 63                          # short limit: up a full h
    assert skewed_center(60, 3, 250, p) == 57                           # past the limit: capped


def test_seconds_since_jump() -> None:
    p = Params()  # jump 3c within 2 s
    flat_then_up = [50, 50, 50, 53, 53, 53, 53]
    # 53 vs 50 two seconds back at t = 3 and again at t = 4, which restarts the timer
    assert seconds_since_jump(flat_then_up, p).tolist() == [math.inf] * 3 + [0, 0, 1, 2]
    assert np.isinf(seconds_since_jump([50, 51, 52, 51, 50], p)).all()  # drift is not a jump
    # cents converted from dollars carry float error; a 3c move must still count
    assert seconds_since_jump(100 * np.array([0.55, 0.55, 0.58]), p)[-1] == 0


def test_quote() -> None:
    # the worked example: center 62.8, skew -1.1, h 3.7 -> 58 / 66
    assert quote(_state()) == Quote(bid=58, ask=66, size=10)

    # guard: pulled for `cooldown` s after a jump; cooldown 0 turns it off
    assert quote(_state(seconds_since_jump=14)) is None
    assert quote(_state(seconds_since_jump=15)) is not None
    assert quote(_state(seconds_since_jump=0), Params(cooldown=0)) is not None

    # no quotes near 0 or 100
    assert quote(_state(p_market=2.5)) is None and quote(_state(p_market=97.5)) is None
    assert quote(_state(p_market=3, p_model=None, position=0)) is not None

    # never cross the book; never quote outside 1..99
    tied = dict(p_market=50, best_bid=49, best_ask=51, secs_left=2700, position=0)
    assert quote(_state(**tied, p_model=100), Params(w=1)) == Quote(bid=50, ask=99, size=10)
    assert quote(_state(**tied, p_model=0), Params(w=1)) == Quote(bid=1, ask=50, size=10)

    # at the position limit stop adding; the other quote sits at fair value
    long_max = quote(_state(p_model=None, position=100))
    assert long_max is not None and long_max.bid is None and long_max.ask == 62
    short_max = quote(_state(p_model=None, position=-120))
    assert short_max is not None and short_max.ask is None and short_max.bid == 62

    # baseline: market only, fixed 2c width, guard off -> same quote early and late
    base = Params(w=0, a=2, b=0, cooldown=0)
    for secs in (2700, 60):
        assert quote(_state(**tied | dict(secs_left=secs), p_model=70), base) == Quote(48, 52, 10)


def test_kalshi_fee() -> None:
    assert kalshi_fee(50, 100, TAKER_FEE) == 175  # exactly 175, not rounded up to 176
    assert kalshi_fee(50, 1, TAKER_FEE) == 2      # 1.75c rounds up per order
    assert kalshi_fee(60, 10) == 5                # maker: 4.2c -> 5c
    assert kalshi_fee(99, 1) == 1


TESTS = [test_center, test_half_spread, test_skewed_center, test_seconds_since_jump,
         test_quote, test_kalshi_fee]


def run_unit_tests() -> None:
    for test in TESTS:
        test()
    print(f"unit tests (quoting): all {len(TESTS)} pass")


if __name__ == "__main__":
    run_unit_tests()
