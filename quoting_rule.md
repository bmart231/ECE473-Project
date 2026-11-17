# Quoting rule: implementation notes (11/16/2026)

Short summary of what was built. For the full explanation (why each step exists, charts,
worked example), see `QuotingRule_Explainer.pdf`.

## What was built (11/16)

`src/quoting.py`: the quoting rule as pure functions (no network, no files). All prices are in
cents. Every function has a unit test (`python src/quoting.py`).

| Function | Step | What it does |
|---|---|---|
| `center` | 1 | `0.2 × model + 0.8 × Kalshi price`; market only while the model is behind |
| `half_spread` | 2 | `max(1¢, A + B × basket sensitivity)`: wider late in close games |
| `skewed_center` | 3 | shifts both quotes to unload a position (a full half-spread at the ±100 limit) |
| `seconds_since_jump` | 4 | jump detector over a whole game: seconds since the price moved ≥ 3¢ in 2 s |
| `quote` | 1–5 | returns `Quote(bid, ask, size)` or `None` (guard tripped, or price below 3¢ / above 97¢) |
| `kalshi_fee` | 5 | Kalshi fee per order, rounded up |

`src/stern.py` gained `basket_sensitivity`: how many cents one basket would move the price,
from just the price and the clock (σ = 22).

- **Baselines use the same code with different settings:** `Params(w=0, b=0, cooldown=0)` is
  plain mid-quoting with a fixed width and no guard.
- **Checked against the explainer:** the worked example gives 58¢ / 66¢.
- **Small tolerance for float error:** `100*0.58 − 100*0.55` comes out as 2.99999999999998, which
  would miss a 3¢ jump, and a 175¢ fee would round up to 176¢. A small tolerance fixes both, and
  tests cover them.

## Where it fits

`quoting.py` decides prices. It does **not** compute our probability, read data, or place orders.

```
stern.stern_p(...)  → our P(win), × 100 → p_model
MarketState(p_market, best_bid, best_ask, p_model, secs_left, position, seconds_since_jump)
quoting.quote(state, params) → Quote(bid, ask, size) or None
```

- **Quotes are centered on Kalshi's price, not ours.** Over 799 games our model was no more
  accurate than Kalshi, and Kalshi reacts about 13 s faster than our feed. So our model only
  nudges the center (w = 0.2).
- **Our probability** still comes from `stern.py`.
- **The backtester (next)** builds a `MarketState` every second from Kalshi trades and the
  play-by-play, then scores fills, markout, and profit after fees.
- **A live bot** would build the state from Kalshi's WebSocket and the NBA feed and send real
  orders. None is planned before the report.

## Open decisions

1. **No order book in the backtest:** the trade history has no best bid or best ask, so the
   plan is to pass the recent trade price as both.
2. **Overtime:** `secs_left` is the seconds left in the period, so the rule quotes in OT with
   very wide spreads. Whether to quote in OT at all is still open.
