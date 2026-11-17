# Backtest plan (11/16/2026)

The goal is to replay past games second by second with `quoting.quote()`, count
the fills it would have gotten, and score them: profit after fees, markout,
risk. This answers H3: does the full rule beat plain mid-quoting on held-out
games? The rule itself is described in `quoting_rule.md` and
`QuotingRule_Explainer.pdf`.

## What data we have (checked 11/16)

| Data                                                                  | Games                  | Enough for the backtest?                                                                    |
| --------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------- |
| Every Kalshi trade, lead-lag cache (`data/processed/cache/lead_lag/`) | 150, Apr 2025–Jan 2026 | Yes for the trades. Median 6,655 trades per game; 13 games have fewer than 1,000            |
| Score changes, same cache                                             | same 150               | Only partly: the clock is known only at baskets                                             |
| Kalshi minute candles plus plays (`data/processed/cache/`)            | 799                    | No: a one-minute price can't tell us which trades went through our quote                    |
| Live log from 11/15 (`data/raw/live/`)                                | 8                      | Current season, so only good for a demo                                                     |
| **Held-out test, Feb–Jun 2026**                                       | **0**                  | **Not downloaded:** `games.parquet` stops at 2026-01-31 on purpose (`batch_compare.WINDOW`) |

**Gaps to fill before the backtest:**

1. **Trades for the other ~650 training games.** `lead_lag.fetch_trades` already
   downloads them, so this only needs a run. The 150-game lead-lag run took
   about 5 min, so this should take about 25 min.
2. **The game clock between baskets.** The step-2 spread needs `secs_left` every
   second, but the cache only stores score changes. `live_prototype.load_plays`
   already returns every play, so we cache that table instead (at most a few
   seconds between plays).
3. **Which side started each trade.** `trades_to_frame` drops `taker_side`. The
   conservative fill rule doesn't need it: a print below our bid means a seller
   would have hit us. The optimistic rule (fills at our exact price) does need
   it, so the new cache keeps it.
4. **The held-out games (Feb–Jun 2026):** game list, plays and trades. We
   download them, but nothing reads them until the final run.

## Design: `src/backtest.py`

Pure functions, unit-tested first (the same TDD approach as `quoting.py`).
Downloading goes through `downloads.cached` and `with_retries`, so the downloads
are cached and retried like the other scripts.

| Function                                     | Job                                                                                                                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `load_game(game_id)`                         | cached plays (every play) and trades (with taker side)                                                                                    |
| `per_second(plays, trades)`                  | one row per second: `p_market` (median of the last few seconds of trades, in cents), `secs_left`, `margin`                                |
| `model_price(frame, p_pre, delay)`           | Stern P(win) from the score as of `t − delay` (delay 0 / 3 / 12.9 s), set to None while stale (see below)                                 |
| `simulate(frame, trades, params, fill_rule)` | loop over seconds: build a `MarketState` → `quote()` → match that second's trades against the quote → update position and cash            |
| `score(fills, frame, home_won)`              | profit after maker fees, 60 s markout per fill, fills, largest drawdown, peak position, results by quarter and by close vs. lopsided game |

**Assumptions** (these go in the report):

- **Order book:** trades have no best bid or ask, so
  `best_bid = best_ask = p_market`. Our bid stays below the recent price and our
  ask above it.
- **Our latency:** a quote decided at second t is live from t + 1 s, because we
  see Kalshi about 1 s late.
- **Fills:** conservative means a trade strictly through our price, size
  `min(our remaining, trade size)`. Optimistic also counts trades at our price
  from a taker on the other side. Conservative is the headline result;
  optimistic is the upper bound.
- **Stale model:** `p_model = None` while `seconds_since_jump < delay`. That is
  what a live bot can know: a jump it has seen that its feed hasn't explained
  yet.
- **Settlement:** at the end, the position pays 100¢ or 0¢ from `home_won`.

**Speed:** about 9,000 s per game (wall clock) × 800 games × many settings is a
lot of scalar `quote()` calls, and `basket_sensitivity` calls scipy. We time one
game first. If it's too slow, we work out sensitivity for the whole game at once
(it doesn't depend on position) before changing anything else.

**Tests:** each function gets a test, plus three made-up games with known
answers:

- the price stays flat while trades print either side → we earn the spread
  exactly
- a 10¢ jump with the guard on → no fills after the trip
- the same jump with the guard off → we're picked off for the jump size

## Held-out games: what they are and why (added 11/17)

Held-out games are games we set aside and **don't look at** while we build and
tune the rule. We use them once at the end, as the final exam.

**Why:** tuning means trying many settings on the same games and keeping the
best. Part of that best score is luck: settings that happen to fit those
particular games. A number reported from those games would overstate how good
the rule is. The held-out games have never been seen, so their result is an
honest estimate of how the rule does on new games.

|              | Games | Dates               | Used for                                                                             |
| ------------ | ----- | ------------------- | ------------------------------------------------------------------------------------ |
| **Training** | 803   | Apr 2025 – Jan 2026 | everything so far (σ fit, model vs. Kalshi, lead-lag, the comparison) and the tuning |
| **Held-out** | 582   | Feb – Jun 2026      | **one** final run of the locked settings vs. the baseline                            |

- **Kept apart in the code:** the held-out games have their own list
  (`data/processed/games_heldout.parquet`) and their own download folder
  (`data/processed/cache/heldout/`). Nothing in the comparison or tuning reads
  either one.
- **Split by date:** the held-out games are later than everything we trained on,
  just like the games a real bot would face.
- **Run them once.** If we look at the held-out result, change settings and run
  again, it's no longer a fair test.

## Experiments

1. **Check on one game:** plot quotes, trades and fills for one game (like
   `figures/quoting/q_guard.png`).
2. **The 2×2 from the explainer:** fixed vs. sensitivity-based width, crossed
   with market-only vs. blended center. Each is run with the guard on and off,
   at delays 0 / 3 / 12.9 s, on the training games.
3. **Tuning on training games only,** at the 12.9 s delay: w ∈ {0, 0.1, 0.2,
   0.4}, B ∈ {0.1 … 0.6}, J ∈ {2, 3, 5}¢, T ∈ {10, 15, 20} s. We pick by mean
   profit per game after fees, with a bootstrap interval over games.
4. **Held-out run:** the chosen settings plus the baseline on Feb–Jun 2026,
   **once.** H3 is supported if the full rule beats the baseline with an
   interval that excludes 0.
