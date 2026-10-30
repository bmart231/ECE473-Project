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

| Person   | Responsibilities                                                       |
| -------- | ---------------------------------------------------------------------- |
| ACL      | Market data scrapers and matching; fill simulation and trading metrics |
| Arjav    | Play-by-play alignment, Elo and pregame prior; baselines               |
| Bryan    | Live win probability model and calibration; quoting rule and tuning    |
| Everyone | Backtest design, reports and presentation                              |

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

**Big Math**

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

- [x] Kalshi: trades and 1-minute candles still downloadable for settled games
      back to the 2025 playoffs; one market per team? (if so, always use home
      YES)
- [x] `nba_api`: the live play-by-play endpoint gives UTC times per play
      (`timeActual`); check on one 2025–26 game
- [ ] Bulk play-by-play for 2015–24: scraping is slow and rate-limited; check
      pre-scraped dumps (e.g. `shufinskiy/nba_data` on GitHub)
- [ ] The Odds API: historical odds may only start in mid-2020 and need a paid
      plan. If so, there are no closing lines for the 2015–2020 prior
- [ ] Polymarket: the 2024–25 window ends before the test window; decide its
      role or drop it
- [ ] Injury PDFs: lowest priority; use report release times as news windows

### Proposal fixes

- [ ] Page 1 Motivation says "We study this in weather markets" -> the rest is
      about the NBA

### Next steps

- [ ] Run the data checks above
- [ ] Agree on the shared table formats
- [ ] One-game prototype: plot Kalshi mid and Stern p against real time, with
      scoring plays marked
- [ ] Bryan: build the training table from 2015–24 play-by-play and fit the
      Stern model

---

## October 29, 2026: Environment setup and API checks (`testing_docs/`)

### What was set up

| Item         | Value                                                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Python       | 3.13 (uv-managed), in `.venv`                                                                                          |
| Dependencies | `pyproject.toml` + `uv.lock`; teammates run `uv sync`                                                                  |
| Packages     | pandas, numpy, pyarrow, requests, nba-api, scipy, scikit-learn, matplotlib, python-dotenv, pdfplumber; ipykernel (dev) |
| Test scripts | `testing_docs/nba_api_demo.py`, `testing_docs/kalshi_api_demo.py`                                                      |
| `.gitignore` | `__pycache__/`, `.venv/`, `.env`, `data/raw/`, `.DS_Store`                                                             |

`lightgbm` was left out: on a Mac it needs Homebrew's `libomp`. scikit-learn's
`HistGradientBoostingClassifier` covers the flexible model instead.

### Kalshi findings (`testing_docs/kalshi_api_demo.py`)

- **No API key needed** to read market data (a key is only for trading or your
  own account). There is a rate limit, so pause between requests when
  downloading every game.
- **Use the `/historical/` endpoints.** Markets settled before 2026-08-30 (see
  `historical/cutoff`) only come back from `historical/markets`,
  `historical/trades` and `historical/markets/{ticker}/candlesticks`. The normal
  endpoints return nothing for them.
- **Two markets per game**, one per team (e.g. `...-IND` = "Indiana wins").
  Ticker format: `KXNBAGAME-25JUN19OKCIND-IND` = date + away team + home team +
  the team the market is for.
- **Prices are strings in dollars** (`"0.3300"`), so convert with `float()`.

Games on Kalshi, by period:

| Period                                               | Games     | Use                            |
| ---------------------------------------------------- | --------- | ------------------------------ |
| 2025 playoffs (Apr–Jun 2025)                         | 86        | Tuning                         |
| 2025–26 preseason (Oct 10–20, 2025)                  | 43        | Drop                           |
| 2025–26 regular season (Oct 21, 2025 – Apr 12, 2026) | 1,233     | Tuning (to Jan), test (Feb on) |
| 2026 play-in and playoffs                            | 87        | Test                           |
| **Total**                                            | **1,449** |                                |

- 1,446 games settled with exactly one team YES. **3 games settled at a
  fractional price** instead of 0/1 and should be dropped:
  `KXNBAGAME-26JAN08MIACHI` (0.69/0.31), `KXNBAGAME-26JAN25DALMIL` (0.56/0.44),
  `KXNBAGAME-26JAN25DENMEM` (0.62/0.38).
- By market close date: about 854 games in the tuning period (2025 playoffs to
  Jan 2026, including the preseason) and 595 in the test period (Feb–Jun 2026).

One game in detail, 2025 Finals Game 6 (`KXNBAGAME-25JUN19OKCIND-IND`):

| Data             | What came back                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| 1-minute candles | 171 over the game; YES bid/ask open/high/low/close, volume, open interest                                          |
| Trades           | 22,604 (about 11.8M contracts); time to the millisecond, YES price, size, taker side; newest first, 1,000 per page |
| Price path       | Indiana mid 0.34 at tip-off, 0.53 at 01:30 UTC, 0.87 at 02:00, 0.995 at 03:00                                      |
| Result           | Indiana YES; market closed 03:21 UTC                                                                               |

### `nba_api` findings (`testing_docs/nba_api_demo.py`)

- **No API key needed**; it's a pip package.
- **Column names are uppercase** (`TEAM_NAME`, `GP`, `PLUS_MINUS`, `GAME_ID`)
  and must match exactly when selecting columns.
- `LeagueDashTeamStats` gives **season totals** by default (`PTS`,
  `PLUS_MINUS`); pass `per_mode_detailed="PerGame"` for per-game numbers.
  2025–26 best record: OKC 64–18 (+914 total, about +11 per game).
- `LeagueGameFinder`: 1,230 regular-season games in 2025–26, **2 rows per game**
  (one per team). In `MATCHUP`, `vs.` = home team, `@` = away team.
- **Live play-by-play** (`nba_api.live.nba.endpoints.playbyplay`) has a UTC
  `timeActual` per play, to 0.1 s:

| Game                                   | Plays | First → last play (UTC)     | Final           |
| -------------------------------------- | ----- | --------------------------- | --------------- |
| 2025 Finals Game 6 (`0042400406`)      | 598   | 00:40 → 03:11, Jun 20, 2025 | IND 108, OKC 91 |
| CHA @ NYK, Apr 12, 2026 (`0022501190`) | 546   | 22:11 → 00:16               | CHA 110, NYK 96 |
| 2015–16 opener (`0021500001`)          | —     | Returned nothing            | —               |

- **The live feed doesn't cover old seasons.** Training data for 2015–24 has to
  come from the stats endpoints or a pre-scraped dump (`shufinskiy/nba_data` on
  GitHub, updated July 2026). Training only needs game clock and score, not UTC
  time, so this is fine.
- **The game clock is an ISO duration** (`PT11M44.00S`), so it needs parsing to
  seconds.

### Kalshi and `nba_api` line up

For 2025 Finals Game 6, the play-by-play ends at 03:11 UTC with Indiana winning
108–91, and the Kalshi Indiana market closed at 03:21 UTC and settled YES. The
two sources can be joined on time, at least for this game.

### Next steps

- [ ] One-game prototype: Kalshi mid and Stern p against UTC time for 2025
      Finals Game 6, scoring plays marked
- [ ] Download 2015–24 play-by-play (`shufinskiy/nba_data`) and build the
      training table
- [ ] Tell ACL and Arjav: run `uv sync`; use Kalshi's `/historical/` endpoints;
      drop preseason and the 3 fractional-price games
- [ ] Remaining data checks: The Odds API (paid plan?), Polymarket, injury PDFs
