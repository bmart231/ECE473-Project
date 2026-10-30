"""luh calm demo of kalshi api: works every settled NBA game market, 1 min candles for one game. 
"""

from datetime import datetime, timezone

import requests

# dont need api key for this -> scalable not so much?
BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAGAME"  # NBA game-winner markets, one market per team per game
TICKER = "KXNBAGAME-25JUN19OKCIND-IND"  # 2025 Finals Game 6, "Indiana wins" (same game as nba_api_demo)
def get(path: str, **params) -> dict:
    """GET one Kalshi endpoint and return the JSON."""
    response = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def to_ts(text: str) -> int:
    """'2025-06-20T00:30:00' (UTC) -> unix seconds, which the candle endpoint takes."""
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


cutoff = get("historical/cutoff")
print(f"settled before {cutoff['market_settled_ts']} -> use /historical/ endpoints\n")
markets, cursor = [], None
while True:
    page = get("historical/markets", series_ticker=SERIES, limit=1000, cursor=cursor)
    markets += page["markets"]
    cursor = page.get("cursor")
    if not cursor or not page["markets"]:
        break
closes = sorted(m["close_time"] for m in markets)
print(f"settled markets: {len(markets)}  games: {len({m['event_ticker'] for m in markets})}")
print(f"first close: {closes[0][:10]}  last close: {closes[-1][:10]}")
for m in markets[:2]:
    print(f"  {m['ticker']}  {m['title']}  result: {m['result']}")
candles = get(
    f"historical/markets/{TICKER}/candlesticks",
    start_ts=to_ts("2025-06-20T00:30:00"), end_ts=to_ts("2025-06-20T03:20:00"),
    period_interval=1,
)["candlesticks"]
print(f"\n1-minute candles for {TICKER}: {len(candles)}")
for c in candles[::30]:  # every 30 minutes
    bid, ask = float(c["yes_bid"]["close"]), float(c["yes_ask"]["close"])
    minute = datetime.fromtimestamp(c["end_period_ts"], timezone.utc).strftime("%H:%M")
    print(f"  {minute} UTC  bid {bid:.2f}  ask {ask:.2f}  mid {(bid + ask) / 2:.3f}  volume {float(c['volume']):,.0f}")

trades = get("historical/trades", ticker=TICKER, limit=1000)["trades"]
print(f"\nfirst page of trades: {len(trades)} (more pages via cursor)")
for t in trades[:3]:
    print(f"  {t['created_time']}  YES {float(t['yes_price_dollars']):.2f}  "
        f"size {float(t['count_fp']):,.0f}  taker {t['taker_side']}")