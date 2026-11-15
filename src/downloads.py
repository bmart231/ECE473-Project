"""Downloading per-game data politely: retries, a per-game parquet cache, and a batch loop.

Shared by batch_compare.py (Kalshi candles + plays) and lead_lag.py (Kalshi trades + plays).

- with_retries:  retry a call that failed for a temporary reason (throttling, network)
- cached:        read a game's table from disk, or download it once and save it
- for_each_game: run a loader over many games, keep going past bad games, and stop
                 downloading (but keep reading the cache) if the APIs keep failing
"""

import contextlib
import io
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

# After ~700 games in a row the NBA feed started answering with empty bodies, which nba_api
# parses without checking the status, so they surface as JSONDecodeError. Kalshi throttling
# surfaces as HTTPError, network drops as ConnectionError/Timeout.
TRANSIENT_ERRORS = (json.JSONDecodeError, requests.RequestException)
REQUEST_PAUSE = 1.0           # seconds to wait after each download
RETRY_WAITS = (5, 20, 60)     # seconds before each retry of a temporary failure
MAX_TRANSIENT_STREAK = 3      # this many games in a row still failing -> stop downloading


def with_retries(func, *args, waits: tuple[float, ...] | None = None):
    """func(*args), retried after each wait in waits (default RETRY_WAITS) if it fails with
    a TRANSIENT_ERROR.

    Anything else (a failed check, a missing quote) is a property of the game and is
    raised at once: retrying would only give the same answer.
    """
    for wait in RETRY_WAITS if waits is None else waits:
        try:
            return func(*args)
        except TRANSIENT_ERRORS as error:
            print(f"    {type(error).__name__}, retrying in {wait:g}s")
            time.sleep(wait)
    return func(*args)


def cached(path: Path, download: Callable[[], pd.DataFrame], refresh: bool = False,
           allowed: bool = True) -> pd.DataFrame:
    """The table saved at path, or download() (with retries) saved there first.

    allowed=False reads the cache only; for_each_game passes that once the APIs keep failing.
    """
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if not allowed:
        raise RuntimeError("not downloaded: the APIs kept failing earlier in this run; rerun to fetch")

    table = with_retries(download)
    time.sleep(REQUEST_PAUSE)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return table


def for_each_game(games: pd.DataFrame, load_one: Callable[[dict, bool], pd.DataFrame]
                  ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """load_one(row, allowed) for every row of games -> (tables that loaded, failed games).

    games needs game_id and ticker columns; row is that game's row as a dict. One bad game
    must not stop the batch, so failures are recorded with their reason. If
    MAX_TRANSIENT_STREAK games in a row still fail with temporary API errors after their
    retries, the APIs are down or blocking us: downloading stops for the rest of the run
    (hammering them would only extend the block), but cached games still load.
    """
    tables, failures = [], []
    allowed, streak = True, 0
    for i, row in enumerate(games.to_dict("records"), start=1):
        try:
            tables.append(load_one(row, allowed))
            streak = 0
        except Exception as error:
            failures.append({"game_id": row["game_id"], "ticker": row["ticker"],
                             "error": f"{type(error).__name__}: {error}"})
            streak = streak + 1 if isinstance(error, TRANSIENT_ERRORS) else 0
            if allowed and streak >= MAX_TRANSIENT_STREAK:
                allowed = False
                print(f"  {streak} games in a row failed with API errors even after retries; "
                      "skipping the remaining downloads (cached games still load). Rerun later.")
        if i % 10 == 0 or i == len(games):
            print(f"  {i}/{len(games)} games ({len(failures)} failed)")
    return tables, pd.DataFrame(failures, columns=["game_id", "ticker", "error"])


def _raises(error: type[Exception], func, *args) -> bool:
    try:
        func(*args)
    except error:
        return True
    return False


def run_unit_tests() -> None:
    """Offline checks with fake downloads (no network, no waiting)."""
    throttled = json.JSONDecodeError("Expecting value", "", 0)

    def flaky(errors: list[Exception]):
        calls = []

        def func():
            calls.append(1)
            if errors:
                raise errors.pop(0)
            return pd.DataFrame({"x": [1]})
        return func, calls

    tmp = Path(tempfile.mkdtemp())
    global REQUEST_PAUSE
    pause, REQUEST_PAUSE = REQUEST_PAUSE, 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # hide the "retrying in 0s" lines
            # with_retries: temporary errors are retried, a game's own errors are raised at once
            func, calls = flaky([throttled, requests.ConnectionError()])
            assert with_retries(func, waits=(0, 0, 0))["x"].item() == 1 and len(calls) == 3
            func, calls = flaky([ValueError("no Kalshi quote before tip-off")])
            assert _raises(ValueError, with_retries, func) and len(calls) == 1
            func, calls = flaky([throttled] * 4)
            assert _raises(json.JSONDecodeError, lambda: with_retries(func, waits=(0, 0, 0)))
            assert len(calls) == 4

            # cached: downloads once, then reads the file; allowed=False never downloads
            func, calls = flaky([])
            cached(tmp / "a.parquet", func)
            cached(tmp / "a.parquet", func)
            assert len(calls) == 1
            assert _raises(RuntimeError, cached, tmp / "b.parquet", func, False, False)

            # for_each_game: after MAX_TRANSIENT_STREAK throttled games it stops downloading,
            # but a cached game still loads
            downloads = []

            def load_one(row: dict, allowed: bool) -> pd.DataFrame:
                def download():
                    downloads.append(row["game_id"])
                    raise throttled
                return cached(tmp / f"{row['game_id']}.parquet", download, allowed=allowed)

            ids = ["g1", "g2", "g3", "a", "g5"]
            games = pd.DataFrame({"game_id": ids, "ticker": ids})
            global RETRY_WAITS
            waits, RETRY_WAITS = RETRY_WAITS, (0, 0, 0)
            try:
                tables, failures = for_each_game(games, load_one)
            finally:
                RETRY_WAITS = waits
    finally:
        REQUEST_PAUSE = pause
        shutil.rmtree(tmp)

    assert len(tables) == 1 and failures["game_id"].tolist() == ["g1", "g2", "g3", "g5"]
    assert set(downloads) == {"g1", "g2", "g3"}  # g5 never touched the API
    assert failures["error"].iloc[-1].startswith("RuntimeError: not downloaded")
    print("unit tests (downloads): all pass")


if __name__ == "__main__":
    run_unit_tests()
