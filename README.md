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

## What we've done so far (updated 11/12/2026)

- **Data:** 10 seasons of NBA play-by-play (2015-16 to 2024-25), plus the Kalshi
  and `nba_api` feeds checked on real games.
- **Model:** Stern model and Elo prior implemented.
- **Fit:** $\sigma$ = **19.64**, fitted on 2015-23 and checked on 2024-25. The
  live model cuts log loss from 0.61 (Elo alone) to 0.46 and is well calibrated
  (`sigma_results.md`).
- **First market test:** 2025 Finals Game 6, model vs Kalshi. They track closely
  (average gap about 2.6 cents), but the model trusts leads more than the market
  does.

## Repo layout

| Path                             | What it is                                               |
| -------------------------------- | -------------------------------------------------------- |
| `src/download_pbp.py`            | Downloads the play-by-play data into `data/raw/pbp/`     |
| `src/play_by_play.py`            | Cleans play-by-play into scoring plays and a games table |
| `src/prior.py`                   | Elo pregame prior                                        |
| `src/stern.py`                   | Stern win probability model                              |
| `src/fit_sigma.py`               | Fits σ and reports log loss, Brier and calibration       |
| `src/prototype_game_6.py`        | Game 6: Kalshi vs model chart                            |
| `notebooks/project_so_far.ipynb` | Walkthrough of everything above                          |
| `testing_docs/`                  | Small demos of the Kalshi and `nba_api` APIs             |

## Setup

### using uv for faster rust but i think pip should work fine too

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync                                        # install dependencies
uv run python src/download_pbp.py              # download play-by-play (once)
uv run python src/fit_sigma.py                 # fit sigma
uv run python src/prototype_game_6.py          # Game 6 chart
```

## Next steps

- Use the Elo prior (not a hand-set number) for Game 6, and run more games.
- Check whether Kalshi moves before our play timestamps.
- Build the quoting rule and simulate fills and PnL.
