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

## Calibration (2024–25, pooled sigma)

Calibration asks: when the model says 70%, does the home team win about 70% of
the time? Rows are grouped by quarter and by predicted probability. `Predicted`
is the average model probability in the bucket, `Actual` is the share of those
rows where the home team won, and `Gap` is actual minus predicted.

| Quarter | Predicted bucket | Rows   | Predicted | Actual | Gap    |
| ------- | ---------------- | ------ | --------- | ------ | ------ |
| 1       | 0.0–0.1          | 943    | 0.067     | 0.037  | -0.030 |
| 1       | 0.1–0.2          | 2,784  | 0.157     | 0.131  | -0.026 |
| 1       | 0.2–0.3          | 5,181  | 0.254     | 0.295  | +0.041 |
| 1       | 0.3–0.4          | 7,390  | 0.351     | 0.368  | +0.017 |
| 1       | 0.4–0.5          | 8,850  | 0.452     | 0.437  | -0.015 |
| 1       | 0.5–0.6          | 10,071 | 0.550     | 0.530  | -0.020 |
| 1       | 0.6–0.7          | 10,086 | 0.650     | 0.619  | -0.031 |
| 1       | 0.7–0.8          | 8,887  | 0.747     | 0.758  | +0.011 |
| 1       | 0.8–0.9          | 6,797  | 0.845     | 0.830  | -0.015 |
| 1       | 0.9–1.0          | 2,083  | 0.934     | 0.932  | -0.002 |
| 2       | 0.0–0.1          | 3,047  | 0.054     | 0.018  | -0.036 |
| 2       | 0.1–0.2          | 4,632  | 0.153     | 0.160  | +0.008 |
| 2       | 0.2–0.3          | 6,000  | 0.251     | 0.306  | +0.054 |
| 2       | 0.3–0.4          | 6,660  | 0.351     | 0.375  | +0.024 |
| 2       | 0.4–0.5          | 6,819  | 0.450     | 0.431  | -0.019 |
| 2       | 0.5–0.6          | 7,009  | 0.551     | 0.545  | -0.007 |
| 2       | 0.6–0.7          | 7,429  | 0.649     | 0.625  | -0.024 |
| 2       | 0.7–0.8          | 7,223  | 0.750     | 0.724  | -0.026 |
| 2       | 0.8–0.9          | 7,444  | 0.852     | 0.824  | -0.027 |
| 2       | 0.9–1.0          | 6,809  | 0.944     | 0.957  | +0.013 |
| 3       | 0.0–0.1          | 7,617  | 0.038     | 0.028  | -0.010 |
| 3       | 0.1–0.2          | 5,014  | 0.150     | 0.140  | -0.010 |
| 3       | 0.2–0.3          | 5,064  | 0.250     | 0.295  | +0.046 |
| 3       | 0.3–0.4          | 5,300  | 0.351     | 0.348  | -0.002 |
| 3       | 0.4–0.5          | 5,156  | 0.451     | 0.433  | -0.018 |
| 3       | 0.5–0.6          | 4,972  | 0.549     | 0.549  | 0.000  |
| 3       | 0.6–0.7          | 5,003  | 0.650     | 0.649  | -0.001 |
| 3       | 0.7–0.8          | 5,507  | 0.751     | 0.745  | -0.007 |
| 3       | 0.8–0.9          | 6,759  | 0.852     | 0.851  | -0.001 |
| 3       | 0.9–1.0          | 12,680 | 0.961     | 0.954  | -0.007 |
| 4       | 0.0–0.1          | 15,948 | 0.018     | 0.020  | +0.002 |
| 4       | 0.1–0.2          | 3,738  | 0.148     | 0.142  | -0.006 |
| 4       | 0.2–0.3          | 3,094  | 0.248     | 0.250  | +0.001 |
| 4       | 0.3–0.4          | 2,790  | 0.350     | 0.351  | 0.000  |
| 4       | 0.4–0.5          | 2,604  | 0.452     | 0.446  | -0.006 |
| 4       | 0.5–0.6          | 2,630  | 0.548     | 0.524  | -0.024 |
| 4       | 0.6–0.7          | 2,961  | 0.649     | 0.645  | -0.004 |
| 4       | 0.7–0.8          | 3,009  | 0.751     | 0.723  | -0.027 |
| 4       | 0.8–0.9          | 4,043  | 0.853     | 0.839  | -0.014 |
| 4       | 0.9–1.0          | 22,255 | 0.982     | 0.979  | -0.003 |

### What this shows

- **The model is well calibrated.** 35 of the 40 buckets have a gap of 0.03 or
  less, and the largest is 0.054. When the model says 75%, the home team wins
  about 75% of the time.
- **It gets better as the game goes on.** Quarters 3 and 4 are the tightest,
  with most gaps at 0.01 or less. Those are the quarters where prices move fastest,
  so that is where accuracy matters most for quoting.
- **One sigma works for the whole game.** If sigma were wrong for part of the
  game, the gaps in that quarter would all lean the same way (leads trusted too
  much or too little). They don't, so 19.64 stays.
- **The 0.2–0.3 bucket is the one repeat miss.** In quarters 1–3, home teams
  given about 25% actually win about 30%. It is gone by quarter 4. This could be
  real or noise (see the caveat below).
- **Slight lean toward negative gaps.** 28 of the 40 gaps are negative, so the
  model may overstate the home team a little. That would come from the Elo
  home-court bonus, not from sigma.
- **The row counts show how certainty builds.** In quarter 1 most rows sit
  between 0.4 and 0.7. By quarter 4 most sit below 0.1 or above 0.9, because
  most games are decided by then.

Caveat: rows from the same game are not independent (48 rows per game per
quarter), so each bucket holds far fewer independent games than its row count
suggests. Gaps of 0.02–0.03 are within noise, and the 0.2–0.3 pattern is not
proven. This is also one season of validation data.
