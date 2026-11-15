# ECE473 Project

## Domain-Specialized Market Making for NBA Prediction Markets

### What we're doing:

We are building a market maker for Kalshi NBA game-winner contracts. Instead of
quoting around the market price, it quotes around our own live **win**
probability, which updates after every play. Essentially we are trying to answer
whether a calibrated basketball model can beat a market maker that just follows
the market mid, on calibration and on profit.

Full proposal $\rightarrow$ `ECE473_Project.pdf`. Running log of results and
decisions: `ProgressLog.md`.

## How it works

1. **Pregame prior:** Elo ratings give each game a starting $\mathbb{P}$(home
   wins).
2. **Live model:** the Stern (1994) random-walk model turns the score margin and
   the time left into $\mathbb{P}$(home wins). Its one free parameter, $\sigma$,
   sets how much a lead is worth.
3. **Quoting (still need to do):** center bid and ask on the model, with a
   spread that widens when the price is likely to move fast.

## What we've done so far (updated 11/15/2026)

- **Data:** 10 seasons of NBA play-by-play (2015-16 to 2024-25), plus the Kalshi
  and `nba_api` feeds checked on real games.
- **Model:** Stern model and Elo prior implemented.
- **Fit:** $\sigma$ = **19.64**, fitted on 2015-23 and checked on 2024-25. The
  live model cuts log loss from 0.61 (Elo alone) to 0.46 and is well calibrated
  (`sigma_results.md`).
- **First market test:** 2025 Finals Game 6, model vs Kalshi. They track closely
  (average gap about 3 cents), but the model trusts leads more than the market
  does.
- **799-game comparison** (Apr 2025 to Jan 2026, `notebooks/kalshi_comparison.pdf`):
  - The model predicts the winner **no better than Kalshi** (log loss +0.0003 ±
    0.0037), so any profit has to come from the spread.
  - Starting from **Elo is worse** than starting from Kalshi's pregame price.
  - The model is **too extreme**: the market acts as if $\sigma \approx 22$, not
    19.64, and is better calibrated.
- **Lead-lag** (150 games, 8,105 scoring plays, `notebooks/lead_lag.pdf`; method
  in `LeadLag_Plan.md`):
  - Kalshi reacts **after** the NBA feed's play timestamp, not before: 20% of the
    move is done by the timestamp, half by +3.1 s, and trading peaks at +1 s.
  - The market reacts fastest in Q4 (half by +2.2 s) and moves about half as
    much as the model's jump.

## Repo layout

| Path                                | What it is                                                |
| ----------------------------------- | --------------------------------------------------------- |
| `src/download_pbp.py`               | Downloads the play-by-play data into `data/raw/pbp/`      |
| `src/play_by_play.py`               | Cleans play-by-play into scoring plays and a games table  |
| `src/prior.py`                      | Elo pregame prior                                         |
| `src/stern.py`                      | Stern win probability model                               |
| `src/fit_sigma.py`                  | Fits σ and reports log loss, Brier and calibration        |
| `src/prototype_game_6.py`           | Game 6: Kalshi vs model chart                             |
| `src/live_prototype.py`             | Kalshi vs model for any game from Apr 2025 on             |
| `src/batch_compare.py`              | Kalshi vs model over many games; saves data and charts    |
| `src/compare_plots.py`              | Charts for the batch comparison                           |
| `src/lead_lag.py`                   | Kalshi trades around scoring plays: who moves first       |
| `src/lead_lag_plots.py`             | Charts for the lead-lag study                             |
| `src/downloads.py`                  | Shared retries, per-game cache and batch download loop    |
| `notebooks/project_so_far.ipynb`    | Walkthrough of the model, fit and Game 6                  |
| `notebooks/kalshi_comparison.ipynb` | Batch comparison results and conclusions (also as `.pdf`) |
| `notebooks/lead_lag.ipynb`          | Lead-lag results and conclusions (also as `.pdf`)         |
| `figures/kalshi_comparison/`        | Charts from each batch run, one folder per run size       |
| `figures/lead_lag/`                 | Charts from each lead-lag run, one folder per run size    |
| `testing_docs/`                     | Small demos of the Kalshi and `nba_api` APIs              |

## Setup

### using uv for faster rust but i think pip should work fine too

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync                                        # install dependencies
uv run python src/download_pbp.py              # download play-by-play (once)
uv run python src/fit_sigma.py                 # fit sigma
uv run python src/prototype_game_6.py          # Game 6 chart
uv run python src/batch_compare.py --n 803     # all games vs Kalshi (cached after the first run)
uv run python src/lead_lag.py                  # lead-lag on 150 games (needs batch_compare first)
```

## Next steps

- **Feed delay:** log when each play arrives from the live NBA feed during real
  games. If it is more than about 2 s behind the play, we can't keep up.
- **Quoting rule:** start from Kalshi's pregame price with $\sigma \approx 22$,
  a spread floor of about 3 cents, and pull or widen quotes for a few seconds
  after every score.
- **Backtest:** simulate fills and PnL against a market maker that follows the
  mid, then test once on the held-out Feb-Jun 2026 games.
