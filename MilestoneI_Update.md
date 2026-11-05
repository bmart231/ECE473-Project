# ECE 473 Milestone I Update

**Domain-Specialized Market Making for NBA Prediction Markets** · Angel Chang
Liu, Arjav Akkole, Bryan Martinez · October 29, 2026

## Planned vs. actual progress

**Bryan Martinez** (live win probability model, calibration, quoting rule)

- _Planned:_ train the live model; report Brier score, log loss and calibration
  against the logistic benchmark and the market price.
- _Actual:_
  - Set up the shared repo and environment (Python 3.13, `uv` with locked
    dependencies) so all three of us run identical code.
  - Verified both main data sources on real games. **Kalshi:** no API key
    needed; 1,449 NBA games (Apr 2025 – Jun 2026) with 1-minute bid/ask candles
    and every trade (millisecond timestamps, taker side). **`nba_api`:**
    play-by-play with UTC timestamps for 2024–25 and 2025–26. The two line up on
    2025 Finals Game 6.
  - Implemented the Stern Brownian motion model and tested it on one game.
  - **Not done yet:** training on 2015–24 play-by-play and the Brier / log loss
    / calibration report.

**Angel Chang Liu** (market data scrapers and matching, fill simulation)

- _Planned:_ scrape all Kalshi and Polymarket NBA game markets and match at
  least 90% to NBA game IDs.
- _Actual:_ **[TODO Angel]**

**Arjav Akkole** (play-by-play alignment, Elo prior, baselines)

- _Planned:_ align play-by-play to UTC; tag scoring runs and news windows; build
  the Elo pregame prior.
- _Actual:_ **[TODO Arjav]**

**All:** _Planned:_ backtest engine and mid-quoting baseline. _Actual:_ design
written up (plug-in price and quoting functions, two fill rules, delay tests);
not built yet.

**Challenges so far**

- Kalshi serves past games only from separate `/historical/` endpoints; the
  standard endpoints return nothing for them.
- The NBA live feed (with UTC times) doesn't cover old seasons, so 2015–24
  training data must come from the stats endpoints or a public pre-scraped dump.
- Data cleaning: 43 preseason games and 3 games that settled at fractional
  prices (not 0/1) must be dropped.
- First model result: at halftime of Finals Game 6 (Indiana +22) the model gave
  0.99 vs. the market's 0.87. Either the noise parameter σ must be fitted rather
  than assumed, or the market overprices comebacks (favorite-longshot bias). We
  will test this across all games.

## Remaining work for Milestone II (due 11/12)

| Task                                                                     | Owner              | Target |
| ------------------------------------------------------------------------ | ------------------ | ------ |
| Download Kalshi candles and trades for all games; match to NBA game IDs  | Angel              | 11/5   |
| Play-by-play for Kalshi-era games; tag scoring runs, final minutes, news | Arjav              | 11/5   |
| Pregame prior: Elo + de-vigged closing line                              | Arjav              | 11/5   |
| Fit σ on 2015–24; logistic benchmark; Brier / log loss / calibration     | Bryan              | 11/5   |
| Backtest engine and mid-quoting baseline                                 | All (Angel: fills) | 11/7   |
| Quoting rule (inventory skew, volatility-based spreads); tune            | Bryan              | 11/10  |
| Pregame-only and Pinnacle baselines                                      | Arjav              | 11/10  |
| Out-of-sample test: PnL, spread earned, 60 s markout, max drawdown       | All                | 11/12  |

## Refinements to the plan

1. **Polymarket becomes optional.** Its 2024–25 data ends before our Feb–Jun
   2026 test window, so we drop it from the 90% matching target and use it only
   as a robustness check if time allows.
2. **Added a validation season.** The 2024–25 regular season (unused in the
   one-pager) is used to choose and calibrate the live model, keeping the Kalshi
   tuning period for the market maker only.
3. **Pinnacle odds depend on access.** If historical odds are paid-only or start
   in 2020, we use closing lines where available and Elo alone before that; the
   Pinnacle baseline becomes the pregame closing line.
4. **Injury news windows are lowest priority;** scoring-run and final-minute
   windows come first.
5. **Timeline:** model training and the backtest engine move into the first week
   of Milestone II. Milestone II scope is otherwise unchanged.
