"""Stern (1994) Brownian-motion win probability. Pure math, no file I/O.

The score margin is treated as a random walk with drift mu and noise sigma.
With margin m and fraction tau of the game left:
    P(home wins) = Phi( (m + mu * tau) / (sigma * sqrt(tau)) )
mu is set so the formula returns p_pre at tip-off:  mu = sigma * Phi^-1(p_pre)

Everything takes numpy arrays (or scalars) and returns arrays, so it runs on the
whole training table at once instead of one row at a time.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

GAME_SECS = 2880.0  # 4 quarters x 12 minutes


def stern_p(margin: ArrayLike, secs_left: ArrayLike, p_pre: ArrayLike, sigma: float) -> np.ndarray:
    """P(home wins) for each row.

    Steps:
      1. np.asarray each input (so scalars and pandas columns both work)
      2. tau = secs_left / GAME_SECS
      3. mu = sigma * norm.ppf(p_pre)            # drift, in points per full game
      4. z = (margin + mu * tau) / (sigma * np.sqrt(tau))
      5. p = norm.cdf(z)

    Edge case: tau == 0 (game over) divides by zero. Handle it with np.where:
      p = 1.0 if margin > 0, 0.0 if margin < 0, 0.5 if tied (goes to OT).
      Compute z with tau clipped to a tiny positive number (np.maximum(tau, 1e-9)) so
      step 4 never raises, then overwrite the tau == 0 rows.
    """
    # 1. accept scalars, lists, numpy arrays or pandas columns alike
    margin = np.asarray(margin, dtype=float)
    secs_left = np.asarray(secs_left, dtype=float)
    p_pre = np.asarray(p_pre, dtype=float)

    # 2. fraction of the game left
    tau = secs_left / GAME_SECS

    # 3. drift: how many points the home team is expected to win by over a full game
    mu = sigma * norm.ppf(p_pre)

    # 4-5. clip tau so the division never blows up; fix the tau == 0 rows afterwards
    safe_tau = np.maximum(tau, 1e-9)
    z = (margin + mu * tau) / (sigma * np.sqrt(safe_tau))
    p = norm.cdf(z)

    # game over: margin decides (sign gives -1/0/1 -> 0/0.5/1)
    p = np.where(tau <= 0, np.sign(margin) * 0.5 + 0.5, p)
    return p


def p_volatility(p: ArrayLike, secs_left: ArrayLike, dt: float = 60.0) -> np.ndarray:
    """How much p is likely to move over the next dt seconds (one standard deviation).

        SD(dp) ~ phi(z) * sqrt(dt / secs_left),   z = Phi^-1(p)

    phi is the normal density (norm.pdf), largest at p = 0.5. Biggest late in close
    games, which is when the market maker should widen its spread. Used by the quoting
    rule, not by the fit; it's here because it comes from the same model.

    Guard: secs_left <= dt means "less than dt left", cap the ratio at 1.
    """
    p = np.asarray(p, dtype=float)
    secs_left = np.asarray(secs_left, dtype=float)

    z = norm.ppf(p)
    ratio = np.minimum(dt / np.maximum(secs_left, 1e-9), 1.0)
    return norm.pdf(z) * np.sqrt(ratio)


if __name__ == "__main__":
    # tip-off returns the prior exactly
    assert abs(stern_p(0, 2880, 0.345, 16) - 0.345) < 1e-9

    # 2025 Finals G6 at halftime: IND +22, p_pre 0.345, sigma 16 -> ~0.95 (ProjectExplainer)
    half = stern_p(22, 1440, 0.345, 16)
    assert 0.94 < half < 0.96, half

    # a late lead is nearly certain; game over goes to 0/1; tie at the buzzer is 0.5
    assert stern_p(5, 30, 0.5, 16) > 0.99
    assert stern_p(-1, 0, 0.5, 16) == 0.0
    assert stern_p(3, 0, 0.5, 16) == 1.0
    assert stern_p(0, 0, 0.5, 16) == 0.5

    # vectorized: arrays in, array out, no loops
    out = stern_p(np.array([0, 5]), np.array([2880, 720]), 0.5, 16)
    assert out.shape == (2,) and out[1] > out[0]

    # volatility: coin flip at halftime moves ~0.08/min, with a minute left ~0.40
    assert 0.07 < p_volatility(0.5, 1440) < 0.09
    assert 0.39 < p_volatility(0.5, 60) < 0.41
    assert p_volatility(0.95, 60) < p_volatility(0.5, 60)  # lopsided games move less

    print(f"stern.py: all checks pass  (G6 halftime p = {half:.3f})")
