# Sigma fit results

Output of `python ./src/fit_sigma.py`, run 2026-11-10. Stern model fitted on
2015–23 play-by-play (regular season and playoffs), checked on 2024–25. One row
per game per 15 s of regulation clock.

## What sigma is and why we fit it

The Stern model treats the score margin (home minus away) as a random walk: it
drifts toward the better team and gets knocked around by noise. **Sigma is the
size of that noise**: how many points the margin typically swings over a full
game, as one standard deviation. With a fraction `tau` of the game left, the
margin can still move by about `sigma * sqrt(tau)` points.

**Why we need it.** The model has to turn a lead in points into a probability.
It does that by comparing the lead to how far the margin could still move:

    P(home wins) = Phi( (margin + mu * tau) / (sigma * sqrt(tau)) )

A 10-point lead is safe if the margin can only move 3 more points and shaky if
it can move 15. Sigma sets that scale, so it decides how much the model trusts a
lead. It also sets the drift, since `mu = sigma * Phi^-1(p_pre)`.

Example: home team up 10 at halftime, pregame prior 0.50.

| Sigma | P(home wins) |
| ----- | ------------ |
| 12    | 0.88         |
| 16    | 0.81         |
| 19.64 | 0.76         |

A sigma that is too low makes us overconfident in leads; too high makes us
underconfident. Either way our quotes are off and someone trades against us.

**Why we fit it.** Sigma is the model's only free parameter, and there is no
official value to look up. So we try values of sigma on 2015–23 games where we
already know who won, and keep the one whose probabilities match the outcomes
best (lowest log loss). Then we score it on 2024–25, which the fit never saw, to
check the number holds up on new games.

## Data

| Season | Games | Scoring rows |
| ------ | ----- | ------------ |
| 2015   | 1,316 | 156,409      |
| 2016   | 1,309 | 158,460      |
| 2017   | 1,312 | 159,184      |
| 2018   | 1,312 | 165,829      |
| 2019   | 1,139 | 144,047      |
| 2020   | 1,165 | 145,889      |
| 2021   | 1,317 | 162,948      |
| 2022   | 1,314 | 169,770      |
| 2023   | 1,312 | 166,646      |
| 2024   | 1,314 | 166,264      |

## Fitted sigma

Pooled (2015–23): **19.64**

| Season | Sigma |
| ------ | ----- |
| 2015   | 17.85 |
| 2016   | 19.74 |
| 2017   | 19.62 |
| 2018   | 19.84 |
| 2019   | 20.31 |
| 2020   | 19.21 |
| 2021   | 20.32 |
| 2022   | 19.98 |
| 2023   | 19.87 |

## Scores (pooled sigma)

Lower is better for both. Baseline is the pregame Elo probability, never updated
during the game.

| Set             | Model    | Log loss | Brier  |
| --------------- | -------- | -------- | ------ |
| Train (2015–23) | Live     | 0.4693   | 0.1575 |
| Train (2015–23) | Baseline | 0.6289   | 0.2196 |
| Valid (2024–25) | Live     | 0.4603   | 0.1546 |
| Valid (2024–25) | Baseline | 0.6145   | 0.2136 |
