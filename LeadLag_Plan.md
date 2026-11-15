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

## Python vs C++

**Stay in Python.**

- **Data is small.** At a guessed few thousand trades a game, 800 games is about
  1-10 million rows. pandas handles that in seconds with the same tools as the
  batch run (`merge_asof`, `groupby`).
- **The bottleneck is the network.** The slow part is paging through Kalshi's
  trade history under rate limits. C++ can't speed that up; caching and retries
  (already in `batch_compare.py`) do.
- **C++ costs more than it saves.** It would mean rewriting the model, loaders
  and checks, plus a second toolchain.

C++ (or `numba`) only pays off for a full tick-by-tick backtest with many
millions of simulated events and parameter sweeps. Even then, optimize the inner
loop only, after vectorized pandas proves too slow.

## Plan

1. Download trades for a sample of about 100-200 games, cached per game like the
   batch run.
2. Line them up against scoring plays, and build the lead-lag curve and the
   by-quarter split.
3. Add a notebook section with the chart and what it means for quoting:
   - **Market leads:** quote around the market mid, and pull or widen quotes
     right after plays.
   - **We keep up:** quote around the model.
