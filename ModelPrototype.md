# Model Prototype: Brownian Motion Win Probability

How the first forecast model works and how we're building it. This is the
baseline every later model has to beat. Source: Stern (1994), "A Brownian Motion
Model for the Progress of Sports Scores."

---

## 1. The idea

Treat the score margin (home − away) as a random walk over the game:

- It drifts toward the better team (the **drift**, μ).
- It wiggles randomly as teams score (the **noise**, σ).

If the home team leads by m with a fraction τ of the game left, the final margin
is m plus a normal random amount with mean μτ and standard deviation σ√τ. The
home team wins if the final margin is above 0, so:

```
P(home wins) = Φ( (m + μτ) / (σ√τ) )
```

Φ is the standard normal CDF (`scipy.stats.norm.cdf`).

## 2. The pieces

| Symbol | Meaning                                    | Where it comes from                                         |
| ------ | ------------------------------------------ | ----------------------------------------------------------- |
| m      | Home margin right now                      | `scoreHome − scoreAway` from play-by-play                   |
| τ      | Fraction of the game left                  | seconds left / 2880 (48 min)                                |
| σ      | How noisy a full game's margin is (points) | Start with 12; fit on 2015–24 later                         |
| p_pre  | Pregame P(home wins)                       | Prototype: Kalshi mid at tip-off. Later: Elo + closing line |
| μ      | Expected full-game margin (points)         | μ = σ·Φ⁻¹(p_pre), so the formula gives p_pre at tip-off     |

Φ⁻¹ is `scipy.stats.norm.ppf`. Example: p_pre = 0.345 → μ = 12 × (−0.40) =
−4.8, so the home team is expected to lose by about 5.

## 3. Checks that it behaves

- At tip-off (m = 0, τ = 1) it gives exactly p_pre.
- As time runs out (τ → 0) it goes to 1 if the home team leads and 0 if it
  trails.
- The same lead is worth more later in the game (even teams, σ = 12):

| Situation          | P(leader wins) |
| ------------------ | -------------- |
| Tied at tip-off    | 0.500          |
| +5 at halftime     | 0.722          |
| +5 with 6 min left | 0.881          |
| +5 with 2 min left | 0.979          |
| +5 with 30 s left  | 1.000          |

## 4. How we'll implement it (one-game prototype)

Planned file: `testing_docs/one_game_prototype.py`, on 2025 Finals Game 6
(`0042400406`, Kalshi `KXNBAGAME-25JUN19OKCIND-IND`, Indiana at home).

1. **Play-by-play** from `nba_api` (as in `nba_api_demo.py`): keep `timeActual`,
   `period`, `clock`, `scoreHome`, `scoreAway`.
2. **Time left** for each play: parse the clock and convert to seconds left.
3. **Stern p** for each play, with p_pre = Kalshi mid at tip-off.
4. **Kalshi mid** from the 1-minute candles (as in `kalshi_api_demo.py`):
   (YES bid + YES ask) / 2.
5. **Plot** Stern p and Kalshi mid against UTC time, with ticks at made baskets.

The three helper functions (tested on this game):

```python
from scipy.stats import norm

GAME_SECONDS = 2880  # 4 quarters x 12 minutes


def clock_seconds(clock: str) -> float:
    """'PT11M44.00S' -> 704.0"""
    minutes, seconds = clock[2:-1].split("M")
    return int(minutes) * 60 + float(seconds)


def seconds_left(period: int, clock: str) -> float:
    """Seconds left in the game; in overtime, only the current 5-minute period."""
    if period <= 4:
        return (4 - period) * 720 + clock_seconds(clock)
    return clock_seconds(clock)


def stern_p(margin: float, secs_left: float, p_pre: float, sigma: float = 12.0) -> float:
    """P(home wins) from the Brownian motion model."""
    tau = secs_left / GAME_SECONDS
    if tau <= 0:  # game over (a tie at 0 goes to overtime)
        return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    mu = sigma * norm.ppf(p_pre)
    return float(norm.cdf((margin + mu * tau) / (sigma * tau ** 0.5)))
```

What to look for in the plot:

- Does Stern p follow the Kalshi mid?
- Does Kalshi move before or after the play-by-play timestamps? (This sets the
  delay we use in the backtest.)
- Halftime and timeouts: the clock stops but the market keeps trading.

## 5. First look: 2025 Finals Game 6 (σ = 12)

Kalshi mid at tip-off: 0.345 for Indiana, so μ = −4.8.

| End of | Score (IND–OKC) | τ    | Stern p | Kalshi mid |
| ------ | --------------- | ---- | ------- | ---------- |
| Q1     | 28–25           | 0.75 | 0.477   | 0.425      |
| Q2     | 64–42           | 0.50 | 0.990   | 0.870      |
| Q3     | 90–60           | 0.25 | 1.000   | 0.995      |
| Q4     | 108–91          | 0.00 | 1.000   | 0.995      |

The model moves in the right direction, but at halftime it's much more confident
than the market (0.99 vs. 0.87). To match the market there, σ would have to be
about 22. Two possible reasons:

- σ = 12 is too small, so we should fit it on real games (section 6).
- The market overprices comebacks. That's the favorite-longshot bias Bürgi et
  al. found on Kalshi, and it could be an edge.

One game proves nothing, so check this across many games.

## 6. Next: fitting σ

1. Build the training table from 2015–24 play-by-play (`shufinskiy/nba_data`).
   Use one row every 15 s of game clock, with margin, seconds left, p_pre and
   whether the home team won.
2. Pick the σ that gives the lowest log loss on those rows
   (`scipy.optimize.minimize_scalar`, searching σ from 5 to 25).
3. Check it on the 2024–25 season, keeping all rows from one game on the same
   side of the split.

This needs a p_pre for every training game (Arjav's Elo). A rough p_pre is fine
for testing the code, but fit the final σ with the real prior. Otherwise team
strength the prior misses gets counted as noise and σ comes out too big.

## 7. Known limits (why there are models 2 and 3)

- Treats scoring as steady all game: no end-of-game fouling, no garbage time.
- Ignores who has the ball, which is worth about a point late in a game.
- Uses one σ for every game and every moment.

Model 2 (logistic regression, the Yeh et al. benchmark) adds possession and
lets the data set the weights. Model 3 is a flexible model, kept only if it
beats model 2.

## 8. Bonus: volatility for the quoting rule

The same model says how much p is likely to move soon. Over the next Δt seconds:

```
SD of p ≈ φ(z) · √(Δt / seconds left),   z = Φ⁻¹(p)
```

φ is the normal density (`norm.pdf`). It's largest late in close games, which is
when the market maker should widen its spread.
