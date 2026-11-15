# Lead-lag: what it is and how we'll test it

## The question

When a basket is scored, two things react:

1. **Our model**, once the NBA play-by-play feed logs the play.
2. **The Kalshi market**, once traders see it (often on TV or a faster data
   feed).

Lead-lag asks: **which one moves first, and by how many seconds?**

```
         basket in the arena
               │
   ────────────┼──────────────────────────► time
               │
   market moves at ?    our feed logs the play at ?
```

## Why it matters

The 799-game comparison (`notebooks/kalshi_comparison.pdf`) showed the model
predicts winners **no better than Kalshi**. So any edge comes from timing, not
prediction. Example: a basket moves fair value from 50¢ to 55¢.

- **Market moves first:** for a few seconds our quotes still say 50¢ while
  everyone else knows it's 55¢. Fast traders buy from us at 50¢ and we lose 5¢ a
  contract. This is **adverse selection** ("getting picked off"), the main way
  market makers lose money.
- **We keep up:** quoting around the model is safe, and speed is an advantage.

## How we'll measure it

An **event study** on Kalshi's trade history (every individual trade, to the
second):

1. For each scoring play, take Kalshi trade prices from about 60 s before to 60
   s after the play's timestamp.
2. Line all plays up on that timestamp, flipped so "good for home" always points
   up, and average them.
3. Measure what fraction of the eventual price move has happened at each second.

The output is one curve. If half the move happens 5 s before our timestamp, the
market leads by about 5 s. Split it by quarter and by how close the game is to
see where the risk is highest.

## Caveats

1. **Play timestamps are noisy.** They are logged by hand and can step back by
   up to about 2 minutes (seen in the batch run). If typical errors are a few
   seconds, they are the same size as what we're measuring. Use scoring plays
   only, and check how sensitive the result is to timestamp error.
2. **History shows when a play happened, not when we'd have received it.** A
   live feed adds its own delay of a few seconds. The backtest measures the
   market against the event itself. Our true delay needs one live game where we
   log the time each play arrives from the feed.
3. **Trades only, no order book.** Kalshi's history has trades but not order
   book snapshots, so prices are seen only when someone trades. Trading is
   sparse between plays, which adds noise; averaging over many plays handles it.

## After implementation

Results: `notebooks/lead_lag.pdf`. Kalshi reacts 1-3 s **after** the play's
timestamp, so we move on to the quoting rule and backtest rather than improving
the model first.

### Why not improve the model first

- **It can't create an edge.** At best a better model predicts winners as well as
  Kalshi, which is where it already is: +0.0003 log loss over 799 games. Getting
  even closer to the market doesn't make money by itself.
- **The model's job has changed.** We aren't trying to out-predict Kalshi any
  more. We need a quick estimate of where the price _should_ go in the 1-3 s
  after a play, before the market gets there. What matters is how big the model's
  jump is, not how well it predicts winners.
- **Without a backtest we can't tell which improvement pays.** Improving the
  model now would be guessing which error costs money. The backtest will show
  where we lose: stale quotes after plays, jumps that are too big, or quotes that
  are too wide.

### Two adjustments, as quoting-rule settings rather than changes to `stern.py`

| Setting    | What the data says                                                                                     | Starting value                          |
| ---------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| σ          | The market prices as if σ ≈ 22, not 19.64. By quarter it runs from about 26 in Q1 to about 21.5 in Q4. | 22, with a per-quarter version to try   |
| Jump scale | Within 20 s the market moves about half the model's jump (2.4¢ vs 4.6¢)                                | Between 0.5 and 1, chosen by the backtest |

`stern_p` already takes σ as an argument, so neither needs a change to
`stern.py`. Both are tuned on the tuning-period games inside the backtest, never
on the held-out Feb-Jun 2026 games.
