# ECE473 Project History

A running log of experiments, results, and decisions. Newest entries go at the
bottom.

---

## October 29, 2026: Project setup and plan

### The project

Build a market maker for Kalshi NBA game-winner contracts (`KXNBAGAME`) that
quotes around a calibrated live win probability. Test whether it beats a market
maker quoting around the market mid on out-of-sample calibration and PnL. Full
proposal: `ECE473_Project.pdf`.

| Person          | Responsibilities                                                       |
| --------------- | ---------------------------------------------------------------------- |
| Angel Chang Liu | Market data scrapers and matching; fill simulation and trading metrics |
| Arjav Akkole    | Play-by-play alignment, Elo and pregame prior; baselines               |
| Bryan Martinez  | Live win probability model and calibration; quoting rule and tuning    |
| All             | Backtest design, reports and presentation                              |

### Data

| #   | Source              | What                                            | Period                       |
| --- | ------------------- | ----------------------------------------------- | ---------------------------- |
| 1   | Kalshi API          | `KXNBAGAME` trades and 1-minute bid/ask candles | 2025 playoffs to 2026 Finals |
| 2   | Polymarket Data API | NBA trades                                      | 2024–25                      |
| 3   | `nba_api`           | Play-by-play with timestamps                    | 2015–16 to 2025–26           |
| 4   | The Odds API        | Historical Pinnacle odds                        | Not set yet                  |
| 5   | NBA                 | Official injury report PDFs                     | Not set yet                  |

All data is joined on game ID and UTC timestamp.

### Data splits

| Period                    | Data                  | Use                                   |
| ------------------------- | --------------------- | ------------------------------------- |
| 2015–16 to 2023–24        | Play-by-play          | Fit the live model                    |
| 2024–25 regular season    | Play-by-play          | Choose between live models, calibrate |
| 2025 playoffs to Jan 2026 | Kalshi + play-by-play | Tune the market maker                 |
| Feb to Jun 2026           | Kalshi + play-by-play | Test (run once)                       |

The 2024–25 regular season isn't in the proposal. Using it here keeps the Kalshi
tuning period for the market maker only.

### Plan

The work splits into two layers that share one backtest:

- **Forecast:** outputs P(home wins) at any moment. Scored with Brier score, log
  loss and calibration.
- **Quoting:** given a center price, inventory and game state, outputs a bid,
  ask and size. Scored with PnL in the backtest.
- **Backtest engine:** takes the center and the quoting rule as plug-in
  functions, so every baseline runs with the same fill rules and fees.

**Forecast models** (simplest first; keep a model only if it beats the one
before it on validation log loss):

1. Stern Brownian motion: P = Φ((m + μτ) / (σ√τ)), with m = home margin, τ =
   fraction of the game left, μ = σ·Φ⁻¹(p_pre). Only σ is fitted.
2. Logistic regression (the Yeh et al. benchmark): margin/√τ, logit(p_pre)·τ, τ,
   possession.
3. One flexible model (gradient boosting or a GAM), calibrated with isotonic
   regression.

Pregame prior p_pre: Elo (538's NBA settings) combined with the de-vigged
closing line.

**Quoting rule** (add one piece at a time, check each on the tuning period):

1. Fixed half-spread around the center, fixed size, inventory limit
2. Inventory skew: center r = c − γ·q·p(1−p) (Avellaneda–Stoikov for a contract
   that settles at 0 or 1)
3. Wider spreads when p is volatile (Stern: SD of p over Δt ≈ φ(z)·√(Δt/τ)) and
   for a few seconds after scoring plays
4. Blended center: w·p_model + (1−w)·mid, in log-odds

**Baselines** (same engine, same tuning budget):

| Baseline                 | Center price                      |
| ------------------------ | --------------------------------- |
| Mid-quoting market maker | Kalshi mid                        |
| Pregame-only             | Pregame prior, no in-game updates |
| Pinnacle                 | Vig-free Pinnacle odds            |

**Backtest rules:**

- Fills: optimistic (trade at or through our price) and conservative (strictly
  through). Only claim results that survive the conservative one.
- Delay: the model sees plays 0, 5, 15 and 30 s late.
- Include Kalshi fees; keep quote sizes small.
- 2×2 design: center (mid vs. model) × spread (fixed vs. model volatility).

**Evaluation:**

- Main results: (1) test log loss, model vs. Kalshi mid; (2) PnL per game,
  forecast market maker vs. mid market maker.
- Confidence intervals by resampling games, not rows.
- Windows, fixed before looking at test results: volatile (2 min after a run of
  6+ unanswered points), final minutes (last 5 min, margin ≤ 5), calm (the
  rest).
- PnL split: spread earned, adverse selection (60-second markout), inventory and
  settlement.

### Shared table formats (to agree on)

| Table    | Columns                                                 |
| -------- | ------------------------------------------------------- |
| `games`  | game_id, tip-off UTC, home, away, Kalshi ticker, result |
| `pbp`    | game_id, UTC time, period, clock, scores, possession    |
| `book`   | ticker, minute, bid, ask                                |
| `trades` | ticker, time, price, size, taker side                   |
| `prior`  | game_id, pregame P(home)                                |

### Data checks (do these first, on one game each)

- [ ] Kalshi: trades and 1-minute candles still downloadable for settled games
      back to the 2025 playoffs; one market per team? (if so, always use home
      YES)
- [ ] `nba_api`: the live play-by-play endpoint gives UTC times per play
      (`timeActual`); check on one 2025–26 game
- [ ] Bulk play-by-play for 2015–24: scraping is slow and rate-limited; check
      pre-scraped dumps (e.g. `shufinskiy/nba_data` on GitHub)
- [ ] The Odds API: historical odds may only start in mid-2020 and need a paid
      plan. If so, there are no closing lines for the 2015–2020 prior
- [ ] Polymarket: the 2024–25 window ends before the test window; decide its
      role or drop it
- [ ] Injury PDFs: lowest priority; use report release times as news windows

### Proposal fixes

- [ ] Page 1 Motivation says "We study this in weather markets"; the rest is
      about the NBA
- [ ] LaTeX setup text ("[10pt]article [margin=0.6in]geometry …") is printed on
      page 1

### Next steps

- [ ] Run the data checks above
- [ ] Agree on the shared table formats
- [ ] One-game prototype: plot Kalshi mid and Stern p against real time, with
      scoring plays marked
- [ ] Bryan: build the training table from 2015–24 play-by-play and fit the
      Stern model
