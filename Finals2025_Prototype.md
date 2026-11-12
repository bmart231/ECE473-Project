# One-game prototype: 2025 Finals Game 6

Working notes for building `src/prototype_game_6.py`.

**Goal:** one chart with two lines against real (UTC) time for 2025 Finals Game
6, Oklahoma City at Indiana:

- the Kalshi mid price for "Indiana wins"
- our Stern model's probability that the home team (Indiana) wins

**Why:** it is the first time the model meets the market it will trade against.
It tests the join between game clock and real time, and it shows how far apart
the model and the market usually are, which the quoting rule needs.

## Reference values

| Thing            | Value                                                 |
| ---------------- | ----------------------------------------------------- |
| NBA game id      | `0042400406`                                          |
| Kalshi ticker    | `KXNBAGAME-25JUN19OKCIND-IND` ("Indiana wins")        |
| Home team        | Indiana, so Kalshi YES = our home-win probability     |
| Final score      | Indiana 108, Oklahoma City 91 (margin +17)            |
| Plays (UTC)      | 00:40 → 03:11, Jun 20, 2025 (598 plays)               |
| Kalshi close     | 03:21 UTC, settled YES                                |
| Candle window    | `2025-06-20T00:30:00` to `2025-06-20T03:20:00`        |
| Sigma            | 19.64 (from `sigma_results.md`)                       |
| Pregame prior    | 0.345 to start (from the explainer); swap in Elo later |

## Steps

Build one piece at a time and run its check before starting the next.

### 1. Kalshi mid per minute

- Copy the candle call from `testing_docs/kalshi_api_demo.py` (the `get(...)`
  helper, `to_ts`, and the `historical/markets/{TICKER}/candlesticks` request
  with `period_interval=1`).
- Build a table with two columns:
  - `time`: from `end_period_ts` (unix seconds), converted to UTC
  - `mid`: `(yes_bid close + yes_ask close) / 2`
- Prices come back as strings in dollars (`"0.3300"`), so wrap them in
  `float()`.

**Check:** about 170 rows, and `mid` ends near 1.00.

### 2. Plays with real time

- Copy the `playbyplay.PlayByPlay("0042400406", timeout=30)` call from
  `testing_docs/nba_api_demo.py`. The plays are in
  `.get_dict()["game"]["actions"]`.
- For each action keep `timeActual`, `period`, `clock`, `scoreHome`,
  `scoreAway`.
- Add two columns:
  - `margin = scoreHome - scoreAway`. The scores may be strings, so convert
    with `int()` first.
  - `secs_left`: seconds left in regulation, from `period` and `clock`.
- `timeActual` is a UTC timestamp string; `pd.to_datetime(..., utc=True)` turns
  it into something that plots.

**The fiddly part: the clock format.** The live feed gives the clock as an ISO
duration, `PT11M44.00S`, not the `11:44` that
`play_by_play.clock_to_secs_left` expects. Write a small parser:

- minutes are the number between `PT` and `M`
- seconds are the number between `M` and `S`
- then `secs_left = (4 - period) * 720 + minutes * 60 + seconds`

Test the parser on these before using it:

| Period | Clock         | secs_left |
| ------ | ------------- | --------- |
| 1      | `PT12M00.00S` | 2880      |
| 1      | `PT11M44.00S` | 2864      |
| 2      | `PT00M00.00S` | 1440      |
| 4      | `PT00M30.50S` | 30.5      |
| 4      | `PT00M00.00S` | 0         |

(Game 6 had no overtime, so period 5+ can be ignored here.)

**Check:** the last row has margin +17 and `secs_left` 0.

### 3. Model probability

One line:

```python
p = stern.stern_p(margin, secs_left, 0.345, 19.64)
```

**Check:** the first value is about 0.345 and the last is 1.0.

### 4. Plot

- Both series against UTC time with matplotlib.
- Kalshi mid as a normal line.
- Model as a step line (`drawstyle="steps-post"`), since it only changes when a
  play happens.
- y-axis from 0 to 1; label the lines "Kalshi mid" and "Stern model".
- Optional: mark scoring plays (rows where the margin changed).

## What to look for in the chart

- **Do the two lines track each other at all?** If not, the time join is wrong
  (check time zones first).
- **How big is the typical gap?** This sets a floor on our spread.
- **Which line moves first after a scoring run?** If Kalshi moves before our
  play timestamp, faster traders can pick off our quotes.
- **Does the gap close or persist?** A persistent gap means one side is wrong,
  and it is worth finding out which.

## Things that can go wrong

- **Time zones.** Both sources are UTC. If the lines are offset by 4 hours,
  something got converted to local time.
- **Candles are per minute, plays are per event.** Don't try to merge them into
  one table for the plot; just draw both on the same axis.
- **A candle with no quotes.** If a bid or ask is missing for a minute, skip
  that row or carry the last mid forward.
- **Other games.** If the Kalshi market is for the away team, compare it to
  `1 - p`, not `p`.

## After this works

- Replace the 0.345 prior with the Elo `p_pre` for this game.
- Run the same script on a few more games.
- Start the quoting rule.
