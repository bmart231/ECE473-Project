"""Live feed-delay logger: how many seconds after a play does the NBA live feed deliver it?

The lead-lag study (notebooks/lead_lag.pdf) found Kalshi does half its move on a scoring
play by +3.1 s after the play's timestamp (timeActual). Whether our quotes can keep up
depends on when the live feed hands us the play, which history can't show. This script
logs it during real games.

Run:  uv run python src/feed_delay.py                      # log today's games until they all end
      uv run python src/feed_delay.py --summary            # delay numbers for the latest log
      uv run python src/feed_delay.py --summary 2026-11-15
      uv run python src/feed_delay.py --test               # offline unit tests only

Safe to stop (Ctrl+C) and restart: rows are appended as they arrive, and a restart carries
on where it stopped. Plays found on a game's first poll after a start are flagged as
backlog and left out of the delay numbers, since we did not see them arrive.

About once a second, for each live game:
- NBA play-by-play (cdn.nba.com, the file nba_api's live endpoint reads), every 2 s per
  game. By default this is the plain, CDN-cached URL that nba_api uses. --bypass-cache adds
  a unique query string to skip the cache (5-7 s fresher in testing), but the CDN blocked
  that (403) after ~10 minutes of 1 request/s per game, so it is off by default. Each play
  is logged once, the first time we see it, with
    received_at    when our response arrived (clock-corrected, see below)
    seen_after     when the previous poll of this game went out: the play reached the feed
                   between seen_after and received_at
    last_modified  the feed file's Last-Modified header (server clock, whole seconds)
    edited         the play's own 'edited' stamp in the feed
- Kalshi top of book for every live game's home-team market, in one request.

Our clock is checked against NTP every 10 minutes and received_at / seen_after are
corrected by the offset, so a PC clock that is a second off doesn't show up as feed delay.

Output, one folder per game day (git-ignored): data/raw/live/<date>/
    games.csv  plays.csv  book.csv  final.csv  clock.csv  log.txt
"""

import argparse
import csv
import re
import socket
import struct
import tempfile
import time
import traceback
from http.cookiejar import DefaultCookiePolicy
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from requests.structures import CaseInsensitiveDict

from prototype_game_6 import KALSHI_BASE, _raises

ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "data" / "raw" / "live"
FIGURE_DIR = ROOT / "figures" / "feed_delay"

NBA_CDN = "https://cdn.nba.com/static/json/liveData"
SCOREBOARD_URL = f"{NBA_CDN}/scoreboard/todaysScoreboard_00.json"
PBP_URL = NBA_CDN + "/playbyplay/playbyplay_{game_id}.json"
MARKETS_URL = f"{KALSHI_BASE}/markets"
# the CDN answers 403 without browser-like headers; these match what nba_api sends
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:158.0) Gecko/20100101 Firefox/158.0",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
}

SCHEDULED, LIVE, FINAL = 1, 2, 3   # scoreboard gameStatus
POLL_SECS = 1.0                    # main loop and Kalshi book: once a second
NBA_POLL_SECS = 2.0                # each game's feed: the file only changes every ~7 s
SCOREBOARD_SECS = 15               # how often to refresh which games are live
CLOCK_SECS = 600                   # how often to re-check our clock against NTP
STATUS_SECS = 60                   # how often to print a progress line
TIMEOUT = (3, 3)                   # connect, read (s): a stuck request can't stall the loop long
BACKOFF_SECS = (2, 5, 15, 30, 60, 120, 300)  # pause a failing source, one step more per failure
MAX_HOURS = 10.0                   # stop even if a game never reports final
NTP_SERVERS = ("time.google.com", "time.windows.com", "pool.ntp.org")
NTP_EPOCH = 2_208_988_800          # seconds from 1900 (NTP) to 1970 (Unix)
MAX_CLOCK_OFFSET = 300             # an NTP answer further off than this is ignored as bogus
MAX_POLL_GAP = 3.0                 # summary: only plays whose arrival window is this tight
MAX_DELAY = 60.0                   # summary: a play first seen later than this is a correction
FIELD_GOALS = ("2pt", "3pt")       # summary: made shots timed (free throws carry the foul's time)
KALSHI_HALF_MOVE = 3.1             # s after timeActual, from the lead-lag study
KALSHI_PEAK = 1.0                  # s: when trading peaks after a scoring play (lead-lag study)
MARKET_COLOR, FEED_COLOR = "#2a78d6", "#eb6834"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

GAME_ID_RE = re.compile(r"\d{10}")
TRICODE_RE = re.compile(r"[A-Z]{3}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

GAME_COLUMNS = ["game_id", "away", "home", "ticker", "game_et"]
PLAY_COLUMNS = ["game_id", "action_number", "period", "clock", "action_type", "shot_result",
                "score_home", "score_away", "time_actual", "edited", "last_modified",
                "seen_after", "received_at", "backlog"]
FINAL_COLUMNS = ["game_id", "action_number", "time_actual", "edited", "score_home", "score_away"]
BOOK_COLUMNS = ["received_at", "ticker", "yes_bid", "yes_ask", "last_price", "volume"]
CLOCK_COLUMNS = ["checked_at", "offset_s", "server"]


# ---------- small pieces ----------

def cache_buster() -> dict:
    """A query string no one has asked for before, so the CDN fetches a fresh copy."""
    return {"cb": str(time.time_ns())}


def iso(t: float) -> str:
    """Unix seconds -> ISO-8601 UTC with milliseconds."""
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")


def http_time(header: str | None) -> str:
    """HTTP date header ('Sun, 15 Nov 2026 22:00:49 GMT') -> ISO-8601 UTC, or '' if missing/bad."""
    try:
        return parsedate_to_datetime(header).astimezone(UTC).isoformat() if header else ""
    except (TypeError, ValueError):
        return ""


def kalshi_ticker(game_et: str, away: str, home: str) -> str:
    """The home team's Kalshi market, e.g. 'KXNBAGAME-26NOV15LALPHX-PHX', so YES = P(home wins).

    game_et is the scoreboard's tip-off in US Eastern time (its 'Z' suffix is wrong), and
    Kalshi names games by that Eastern date. Months are spelled out here rather than with
    strftime('%b'), which depends on the PC's language.
    """
    if not (TRICODE_RE.fullmatch(away) and TRICODE_RE.fullmatch(home)):
        raise ValueError(f"bad team codes {away!r}, {home!r}")
    if not DATE_RE.fullmatch(game_et[:10]):
        raise ValueError(f"bad game date {game_et!r}")
    year, month, day = (int(part) for part in game_et[:10].split("-"))
    return f"KXNBAGAME-{year % 100:02d}{MONTHS[month - 1]}{day:02d}{away}{home}-{home}"


def parse_scoreboard(body: dict) -> tuple[str, list[dict]]:
    """Scoreboard JSON -> (game date, [{game_id, away, home, game_et, status}]).

    Ids and team codes are checked before they go into URLs and file names; a game that
    fails (an exhibition with odd codes) is skipped.
    """
    board = body["scoreboard"]
    date = str(board["gameDate"])
    if not DATE_RE.fullmatch(date):
        raise ValueError(f"bad scoreboard date {date!r}")
    games = []
    for game in board["games"]:
        game_id = str(game["gameId"])
        away, home = game["awayTeam"]["teamTricode"], game["homeTeam"]["teamTricode"]
        if GAME_ID_RE.fullmatch(game_id) and TRICODE_RE.fullmatch(away) and TRICODE_RE.fullmatch(home):
            games.append({"game_id": game_id, "away": away, "home": home,
                          "game_et": str(game["gameEt"]), "status": int(game["gameStatus"])})
    return date, games


def game_over(actions: list[dict]) -> bool:
    """True once the feed has its end-of-game action (does not wait for the scoreboard)."""
    return any(a.get("actionType") == "game" and a.get("subType") == "end" for a in actions)


def safe_text(value) -> str:
    """Feed text for a CSV cell. A cell starting with = + - @ (or a tab / line break) would run
    as a formula if the file were opened in Excel, so such text gets a leading quote."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") or text[:1].isspace() else text


def play_row(game_id: str, action: dict, last_modified: str, seen_after: float | None,
             received_at: float, backlog: bool) -> dict:
    """One feed action, the first time we see it -> a plays.csv row."""
    return {
        "game_id": game_id, "action_number": int(action["actionNumber"]),
        "period": action.get("period"), "clock": safe_text(action.get("clock")),
        "action_type": safe_text(action.get("actionType")), "shot_result": safe_text(action.get("shotResult")),
        "score_home": action.get("scoreHome"), "score_away": action.get("scoreAway"),
        "time_actual": action.get("timeActual"), "edited": action.get("edited"),
        "last_modified": last_modified, "seen_after": "" if seen_after is None else iso(seen_after),
        "received_at": iso(received_at), "backlog": backlog,
    }


def final_rows(game_id: str, actions: list[dict]) -> list[dict]:
    """The feed's final version of every action: timestamps can be edited after first sight,
    and the lead-lag study used the final ones."""
    return [{"game_id": game_id, "action_number": int(a["actionNumber"]), "time_actual": a.get("timeActual"),
             "edited": a.get("edited"), "score_home": a.get("scoreHome"), "score_away": a.get("scoreAway")}
            for a in actions]


def book_rows(body: dict, tickers: set[str], received_at: float) -> list[dict]:
    """Kalshi /markets answer -> one book.csv row per market we asked about."""
    return [{"received_at": iso(received_at), "ticker": m["ticker"], "yes_bid": m.get("yes_bid_dollars"),
             "yes_ask": m.get("yes_ask_dollars"), "last_price": m.get("last_price_dollars"),
             "volume": m.get("volume_fp", m.get("volume"))}
            for m in body.get("markets", []) if m.get("ticker") in tickers]


def append_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows to a CSV (header first if the file is new). Appending as we go means a
    crash loses at most the current second."""
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> pd.DataFrame:
    """A log CSV with every column as text (game ids keep their leading zeros), or empty."""
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


# ---------- clock ----------

def ntp_time(raw: bytes) -> float:
    """8-byte NTP timestamp -> Unix seconds."""
    seconds, fraction = struct.unpack("!II", raw)
    return seconds - NTP_EPOCH + fraction / 2**32


def parse_sntp(data: bytes, sent: float, received: float) -> float:
    """SNTP reply -> offset (s) to add to our clock: ((t1 - t0) + (t2 - t3)) / 2."""
    if len(data) < 48 or data[1] == 0:  # too short, or stratum 0 ("kiss of death")
        raise ValueError("bad NTP reply")
    offset = ((ntp_time(data[32:40]) - sent) + (ntp_time(data[40:48]) - received)) / 2
    if abs(offset) > MAX_CLOCK_OFFSET:
        raise ValueError(f"NTP offset {offset:.1f}s is implausible")
    return offset


def ntp_offset(server: str, timeout: float = 2.0) -> float:
    """Our clock's offset from one NTP server, in seconds (one SNTP request over UDP)."""
    request = b"\x1b" + 47 * b"\0"  # version 3, client mode
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sent = time.time()
        sock.sendto(request, (server, 123))
        data, _ = sock.recvfrom(48)
        received = time.time()
    return parse_sntp(data, sent, received)


# ---------- the logger ----------

@dataclass
class Backoff:
    """Pauses a source that keeps failing (throttled, blocked, down) instead of hammering it."""
    failures: int = 0
    next_ok: float = 0.0

    def ready(self) -> bool:
        return time.monotonic() >= self.next_ok

    def failed(self) -> float:
        wait = BACKOFF_SECS[min(self.failures, len(BACKOFF_SECS) - 1)]
        self.failures += 1
        self.next_ok = time.monotonic() + wait
        return wait

    def ok(self) -> None:
        self.failures, self.next_ok = 0, 0.0


@dataclass
class Game:
    game_id: str
    away: str
    home: str
    game_et: str
    ticker: str
    status: int = SCHEDULED
    seen: set[int] = field(default_factory=set)  # action numbers already logged
    etag: str = ""                               # lets the CDN answer 304 when nothing changed
    next_poll: float = 0.0                       # monotonic time this game is next due
    last_request: float | None = None            # corrected send time of the last answered poll
    polled: bool = False                         # answered at least once in this run
    logged: bool = False                         # has plays in plays.csv (this run or earlier)
    finalized: bool = False                      # final.csv written
    session: requests.Session = field(default_factory=requests.Session)


class FeedLogger:
    """Polls the live feeds and appends what arrives to data/raw/live/<date>/.

    Network calls run in a thread pool (one game per thread); everything that touches state
    or files runs in the main thread, so there are no races.
    """

    def __init__(self, out_root: Path = LOG_ROOT, poll_secs: float = POLL_SECS,
                 nba_poll_secs: float = NBA_POLL_SECS, bypass_cache: bool = False,
                 session_factory: Callable[[], Any] = requests.Session,
                 clock: Callable[[str], float] = ntp_offset, quiet: bool = False):
        self.out_root, self.poll_secs, self.quiet = out_root, poll_secs, quiet
        self.nba_poll_secs, self.bypass_cache = nba_poll_secs, bypass_cache
        self.session_factory, self.clock = session_factory, clock
        self.board_session = self._nba_session()
        self.kalshi_session = session_factory()
        self.backoff: dict[str, Backoff] = {}
        self.games: dict[str, Game] = {}
        self.day: str | None = None
        self.earlier: dict[str, set[int]] = {}  # action numbers logged by an earlier run today
        self.finished: set[str] = set()         # games whose finals an earlier run saved
        self.offset = 0.0
        self.counts = {"plays": 0, "scores": 0, "book": 0, "errors": 0}
        self.next_board = self.next_clock = self.next_status = 0.0

    # --- helpers ---

    def _nba_session(self) -> requests.Session:
        """A connection to the NBA CDN that keeps no cookies. With cookies kept, the CDN's bot
        check flagged each long-lived session after ~20 minutes (403) while fresh requests
        still got through; nba_api keeps none either."""
        session = self.session_factory()
        session.headers.update(NBA_HEADERS)
        session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        return session

    def now(self) -> float:
        return time.time() + self.offset

    def path(self, name: str) -> Path:
        assert self.day is not None, "no game day yet"
        return self.out_root / self.day / name

    def log(self, message: str) -> None:
        line = f"{datetime.now():%H:%M:%S}  {message}"
        if not self.quiet:
            print(line, flush=True)
        if self.day is not None:
            with self.path("log.txt").open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def failed(self, source: str, error: Exception) -> None:
        wait = self.backoff.setdefault(source, Backoff()).failed()
        self.counts["errors"] += 1
        self.log(f"{source}: {type(error).__name__}: {str(error)[:200]} (pausing it {wait}s)")

    def ready(self, source: str) -> bool:
        return self.backoff.setdefault(source, Backoff()).ready()

    def fetch(self, session: requests.Session, url: str, params: dict | None = None,
              etag: str = "") -> tuple[dict | None, CaseInsensitiveDict, float, float]:
        """GET url -> (JSON body or None if unchanged (304), headers, sent, received), with
        sent/received on the corrected clock. Raises on any HTTP error or non-JSON body."""
        headers = {"If-None-Match": etag} if etag else None
        sent = self.now()
        response = session.get(url, params=params, headers=headers, timeout=TIMEOUT)
        received = self.now()
        if response.status_code == 304:
            return None, response.headers, sent, received
        response.raise_for_status()
        return response.json(), response.headers, sent, received

    # --- game day ---

    def start_day(self, day: str) -> None:
        """Use data/raw/live/<day>/, picking up plays and finals already logged there."""
        self.day = day
        self.path("").mkdir(parents=True, exist_ok=True)
        plays, final = read_csv(self.path("plays.csv")), read_csv(self.path("final.csv"))
        self.earlier = {str(gid): set(group["action_number"].astype(int))
                        for gid, group in plays.groupby("game_id")} if len(plays) else {}
        self.finished = set(final["game_id"]) if len(final) else set()
        self.log(f"logging to {self.path('')}")

    def refresh_games(self) -> None:
        body, *_ = self.fetch(self.board_session, SCOREBOARD_URL, self.nba_params())
        if body is None:
            raise ValueError("empty scoreboard")
        day, games = parse_scoreboard(body)
        if self.day is None:
            self.start_day(day)
        elif day != self.day:
            return  # the scoreboard rolled over to tomorrow: keep finishing today's games

        for info in games:
            game = self.games.get(info["game_id"])
            if game is None:
                game = Game(info["game_id"], info["away"], info["home"], info["game_et"],
                            kalshi_ticker(info["game_et"], info["away"], info["home"]),
                            session=self._nba_session())
                game.seen = self.earlier.get(game.game_id, set())
                game.logged = bool(game.seen)
                game.finalized = game.game_id in self.finished
                self.games[game.game_id] = game
            game.status = max(game.status, info["status"])  # never step back from final

        rows = [{c: getattr(g, c) for c in GAME_COLUMNS} for g in self.games.values()]
        pd.DataFrame(rows, columns=GAME_COLUMNS).to_csv(self.path("games.csv"), index=False)

    def check_clock(self) -> None:
        for server in NTP_SERVERS:
            try:
                self.offset = self.clock(server)
            except (OSError, ValueError):
                continue
            append_rows(self.path("clock.csv"),
                        [{"checked_at": iso(time.time()), "offset_s": round(self.offset, 4), "server": server}],
                        CLOCK_COLUMNS)
            return
        self.log(f"NTP unreachable; keeping clock offset {self.offset:+.3f}s")

    # --- one poll ---

    def targets(self) -> list[Game]:
        """Games to poll now: live ones, plus finished ones we logged but haven't saved finals for."""
        wanted = [g for g in self.games.values()
                  if g.status == LIVE or (g.status == FINAL and g.logged and not g.finalized)]
        now = time.monotonic()
        return [g for g in wanted if now >= g.next_poll and self.ready(g.game_id)]

    def nba_params(self) -> dict | None:
        return cache_buster() if self.bypass_cache else None

    def record_game(self, game: Game, body: dict | None, headers, sent: float, received: float) -> None:
        if body is None:  # 304: nothing new since the last poll
            game.last_request = sent
            return
        actions = body["game"]["actions"]
        game.etag = headers.get("ETag", "")
        last_modified = http_time(headers.get("Last-Modified"))
        backlog = not game.polled
        rows = [play_row(game.game_id, a, last_modified, game.last_request, received, backlog)
                for a in actions if int(a["actionNumber"]) not in game.seen]
        append_rows(self.path("plays.csv"), rows, PLAY_COLUMNS)
        game.seen.update(row["action_number"] for row in rows)
        game.last_request, game.polled = sent, True
        game.logged = game.logged or bool(rows)
        if not backlog:
            self.counts["plays"] += len(rows)
            self.counts["scores"] += sum(row["shot_result"] == "Made" for row in rows)

        if game_over(actions) or game.status == FINAL:
            game.status = FINAL
            if not game.finalized:
                append_rows(self.path("final.csv"), final_rows(game.game_id, actions), FINAL_COLUMNS)
                game.finalized = True
                self.log(f"{game.away} @ {game.home} final: {len(game.seen)} plays logged")

    def poll(self, pool: ThreadPoolExecutor) -> None:
        """One round: every target game's feed and the Kalshi book, fetched in parallel."""
        games = self.targets()
        for game in games:
            game.next_poll = time.monotonic() + self.nba_poll_secs
        jobs = {g.game_id: pool.submit(self.fetch, g.session, PBP_URL.format(game_id=g.game_id),
                                       self.nba_params(), g.etag)
                for g in games}
        tickers = {g.ticker for g in self.games.values() if g.status == LIVE}
        book = (pool.submit(self.fetch, self.kalshi_session, MARKETS_URL, {"tickers": ",".join(sorted(tickers))})
                if tickers and self.ready("kalshi") else None)

        for game in games:
            try:
                self.record_game(game, *jobs[game.game_id].result())
                self.backoff[game.game_id].ok()
            except Exception as error:  # noqa: BLE001 - one bad answer must not stop the night
                self.failed(game.game_id, error)
        if book is not None:
            try:
                body, _, _, received = book.result()
                rows = book_rows(body or {}, tickers, received)
                append_rows(self.path("book.csv"), rows, BOOK_COLUMNS)
                self.counts["book"] += len(rows)
                self.backoff["kalshi"].ok()
            except Exception as error:  # noqa: BLE001
                self.failed("kalshi", error)

    def done(self) -> bool:
        """Every game today is final and saved (scheduled games can't be: they haven't played)."""
        games = self.games.values()
        return bool(self.games) and all(g.status == FINAL and (g.finalized or not g.logged) for g in games)

    def status_line(self) -> str:
        live = [f"{g.away}@{g.home}" for g in self.games.values() if g.status == LIVE]
        left = sum(g.status == SCHEDULED for g in self.games.values())
        c = self.counts
        return (f"live {len(live)} {' '.join(live)} | later {left} | new plays {c['plays']} "
                f"(made shots {c["scores"]}) | book rows {c['book']} | errors {c['errors']} | "
                f"clock {self.offset:+.3f}s")

    # --- main loop ---

    def step(self, pool: ThreadPoolExecutor) -> None:
        tick = time.monotonic()
        if tick >= self.next_board and self.ready("scoreboard"):
            try:
                self.refresh_games()
                self.backoff["scoreboard"].ok()
                self.next_board = tick + SCOREBOARD_SECS
            except Exception as error:  # noqa: BLE001
                self.failed("scoreboard", error)
        if self.day is None:
            return
        if tick >= self.next_clock:
            self.check_clock()
            self.next_clock = tick + CLOCK_SECS
        self.poll(pool)
        if tick >= self.next_status:
            self.log(self.status_line())
            self.next_status = tick + STATUS_SECS

    def run(self, max_hours: float = MAX_HOURS) -> None:
        stop = time.monotonic() + max_hours * 3600
        with ThreadPoolExecutor(max_workers=16) as pool:
            while time.monotonic() < stop and not self.done():
                started = time.monotonic()
                try:
                    self.step(pool)
                except Exception as error:  # noqa: BLE001 - log it and keep going
                    self.counts["errors"] += 1
                    self.log(f"unexpected {type(error).__name__}: {error}")
                    if self.day is not None:
                        with self.path("log.txt").open("a", encoding="utf-8") as f:
                            f.write(traceback.format_exc())
                time.sleep(max(0.0, self.poll_secs - (time.monotonic() - started)))
        self.log("all games final" if self.done() else "stopped (time limit)")
        if self.day is not None:
            self.log(self.status_line())


# ---------- summary ----------

def latest_day(root: Path = LOG_ROOT) -> str:
    days = sorted(p.name for p in root.glob("*") if p.is_dir() and DATE_RE.fullmatch(p.name))
    if not days:
        raise FileNotFoundError(f"no logs in {root}; run the logger first")
    return days[-1]


def scoring_delays(plays: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """One row per made field goal we saw arrive live, with delays in seconds after the play:

    feed       our response arrived (what a live market maker would get)
    published  the feed file changed (server clock: no PC clock or polling in it)
    entered    the play's 'edited' stamp
    feed is measured from the final timeActual when final.csv has it (the timestamps the
    lead-lag study used), else from the one we first saw.

    Free throws are left out: they carry the foul's timestamp, so they look 20-40 s late
    when the feed isn't. Plays first seen more than MAX_DELAY after they happened are
    scorer corrections added later (seen: Q1 free throws inserted in Q2), not live reports.
    """
    made = plays[(plays["backlog"] != "True") & (plays["shot_result"] == "Made")
                 & plays["action_type"].isin(FIELD_GOALS)].copy()
    keys = ["game_id", "action_number", "time_actual"]
    final_time = {(g, n): t for g, n, t in final[keys].itertuples(index=False) if t} if len(final) else {}
    made["time_play"] = [final_time.get((g, n), t) for g, n, t in made[keys].itertuples(index=False)]

    def t(column: str) -> pd.Series:
        return pd.to_datetime(made[column].where(made[column] != ""), utc=True, format="ISO8601")

    play = t("time_play")
    out = pd.DataFrame({
        "game_id": made["game_id"], "action_number": made["action_number"].astype(int),
        "period": made["period"].astype(int), "time_play": play,
        "feed": (t("received_at") - play).dt.total_seconds(),
        "published": (t("last_modified") - play).dt.total_seconds(),
        "entered": (t("edited") - play).dt.total_seconds(),
        "window": (t("received_at") - t("seen_after")).dt.total_seconds(),
        "edit_shift": (play - t("time_actual")).dt.total_seconds(),
    })
    return out[(out["window"] <= MAX_POLL_GAP) & (out["feed"] <= MAX_DELAY)].reset_index(drop=True)


def describe(delays: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Median, 10th/90th percentile and share under 2 s / under Kalshi's half-move time."""
    def stats(d: pd.DataFrame) -> pd.Series:
        f = d["feed"]
        return pd.Series({"games": d["game_id"].nunique(), "plays": len(d), "feed median": f.median(),
                          "feed p10": f.quantile(0.1), "feed p90": f.quantile(0.9),
                          "share <= 2s": (f <= 2).mean(), f"share <= {KALSHI_HALF_MOVE}s": (f <= KALSHI_HALF_MOVE).mean(),
                          "published median": d["published"].median(), "entered median": d["entered"].median()})
    if by is None:
        return stats(delays).to_frame("all").T
    return pd.DataFrame({name: stats(group) for name, group in delays.groupby(by)}).T


def summarize(day: str | None = None, root: Path = LOG_ROOT) -> pd.DataFrame:
    day = day or latest_day(root)
    if not DATE_RE.fullmatch(day):
        raise ValueError(f"date must look like 2026-11-15, got {day!r}")
    folder = root / day
    plays, final = read_csv(folder / "plays.csv"), read_csv(folder / "final.csv")
    if plays.empty:
        raise FileNotFoundError(f"no plays logged in {folder}")
    delays = scoring_delays(plays, final)
    print(f"{day}: {delays['game_id'].nunique()} games, {len(delays)} made field goals seen arriving live "
          f"(arrival known to within {MAX_POLL_GAP:g} s)")
    if delays.empty:
        return delays
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.concat([describe(delays), describe(delays, "period")]).round(2).to_string())
    median = delays["feed"].median()
    verdict = "before" if median < KALSHI_HALF_MOVE else "after"
    print(f"\nmedian feed delay {median:.1f} s: we usually see a basket {verdict} Kalshi has made half "
          f"its move (+{KALSHI_HALF_MOVE} s in the lead-lag study)")
    out = FIGURE_DIR / day
    out.mkdir(parents=True, exist_ok=True)
    for name, chart in {"1_timeline": timeline, "2_distribution": distribution}.items():
        fig = chart(delays)
        fig.savefig(str(out / f"{name}.png"), dpi=150)
        plt.close(fig)
    print(f"charts saved to {out}")
    return delays


# ---------- charts ----------

def timeline(delays: pd.DataFrame) -> Figure:
    """When each thing happens after a basket: Kalshi's reaction (lead-lag study) against
    each step of the NBA feed (median, with a 10th-90th percentile bar)."""
    steps: list[tuple[str, float, tuple[float, float] | None, str]] = [
             ("Kalshi: trading peaks", KALSHI_PEAK, None, MARKET_COLOR),
             ("Kalshi: half its move done", KALSHI_HALF_MOVE, None, MARKET_COLOR)]
    for label, column in [("NBA scorer logs the play", "entered"),
                          ("NBA feed file updated", "published"), ("our logger receives it", "feed")]:
        d = delays[column].dropna()
        steps.append((label, float(d.median()), (float(d.quantile(0.1)), float(d.quantile(0.9))), FEED_COLOR))

    fig, ax = plt.subplots(figsize=(10, 4))
    for row, (label, x, spread, color) in enumerate(reversed(steps)):
        if spread is not None:
            ax.plot(spread, [row, row], color=color, linewidth=2, alpha=0.45, solid_capstyle="round")
        ax.plot(x, row, "o", color=color, markersize=9, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{x:+.1f} s", (x, row), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(range(len(steps)), [step[0] for step in reversed(steps)])
    ax.axvline(0, color="#52514e", linestyle="--", linewidth=1)
    ax.set(xlabel="seconds after the play (its timeActual)", ylim=(-0.6, len(steps) - 0.3),
           title=f"A basket's timeline: market vs NBA feed ({len(delays)} made field goals, "
                 f"{delays['game_id'].nunique()} games)")
    ax.plot([], [], "o", color=MARKET_COLOR, label="Kalshi (lead-lag study, 150 games)")
    ax.plot([], [], "o", color=FEED_COLOR, label="NBA feed (median; bar = 10th-90th pct)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def distribution(delays: pd.DataFrame) -> Figure:
    """How late each basket reached us, against the time Kalshi is half done reacting."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    top = max(15, int(delays["feed"].quantile(0.98)) + 1)
    ax.hist(delays["feed"].clip(upper=top), bins=list(range(top + 2)),
            color=FEED_COLOR, edgecolor="white", linewidth=1)
    ax.axvline(KALSHI_HALF_MOVE, color=MARKET_COLOR, linewidth=2)
    ax.annotate(f"Kalshi half done (+{KALSHI_HALF_MOVE} s)", (KALSHI_HALF_MOVE, 1),
                xycoords=("data", "axes fraction"), xytext=(-4, -14), textcoords="offset points",
                ha="right", fontsize=9, color="#0b0b0b")
    share = (delays["feed"] <= KALSHI_HALF_MOVE).mean()
    ax.set(xlabel="seconds from the play to our logger receiving it", ylabel="made field goals",
           title=f"Feed delay per made field goal: median {delays['feed'].median():.1f} s, "
                 f"{share:.0%} arrive before Kalshi is half done")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------- checks ----------

class FakeResponse:
    def __init__(self, status: int, body=None, headers: dict | None = None):
        self.status_code, self.body = status, body
        self.headers = CaseInsensitiveDict(headers or {})

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self.body is None:
            raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)
        return self.body


class FakeSession:
    """Stands in for requests.Session: answers from a script of responses per URL."""
    script: dict[str, list] = {}

    def __init__(self):
        self.headers: dict = {}
        self.cookies = requests.Session().cookies

    def get(self, url, params=None, headers=None, timeout=None):
        replies = FakeSession.script[url.split("/")[-1].split("_")[0]]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def run_unit_tests() -> None:
    """Offline checks (no network): parsing, clock maths, files, and a scripted game night."""
    # tickers and parsing
    assert kalshi_ticker("2026-11-15T16:00:00Z", "LAL", "PHX") == "KXNBAGAME-26NOV15LALPHX-PHX"
    assert kalshi_ticker("2027-01-05T19:30:00Z", "NYK", "BOS") == "KXNBAGAME-27JAN05NYKBOS-BOS"
    assert _raises(ValueError, kalshi_ticker, "2026-11-15T16:00:00Z", "../", "PHX")
    assert _raises(ValueError, kalshi_ticker, "garbage", "LAL", "PHX")

    def board_game(gid, away, home, status):
        return {"gameId": gid, "gameStatus": status, "gameEt": "2026-11-15T19:00:00Z",
                "awayTeam": {"teamTricode": away}, "homeTeam": {"teamTricode": home}}

    board = {"scoreboard": {"gameDate": "2026-11-15", "games": [
        board_game("0022600237", "LAL", "PHX", LIVE), board_game("../../x", "LAL", "PHX", LIVE),
        board_game("0022600238", "lal", "PHX", LIVE)]}}
    day, games = parse_scoreboard(board)
    assert day == "2026-11-15" and [g["game_id"] for g in games] == ["0022600237"]
    assert _raises(ValueError, parse_scoreboard, {"scoreboard": {"gameDate": "../etc", "games": []}})

    assert http_time("Sun, 15 Nov 2026 22:00:49 GMT") == "2026-11-15T22:00:49+00:00"
    assert http_time(None) == "" and http_time("not a date") == ""
    assert safe_text("=HYPERLINK(x)") == "'=HYPERLINK(x)" and safe_text("2pt") == "2pt" and safe_text(None) == ""
    assert safe_text(chr(9) + "=x").startswith("'")  # tab-led

    # NTP: a server 2.5 s ahead of us, 0.1 s round trip each way -> offset +2.5
    def ntp_bytes(t: float) -> bytes:
        seconds = int(t) + NTP_EPOCH
        return struct.pack("!II", seconds, int((t - int(t)) * 2**32))
    sent, received = 1_700_000_000.0, 1_700_000_000.2
    reply = b"\x1c\x02" + 30 * b"\0" + ntp_bytes(sent + 0.1 + 2.5) + ntp_bytes(sent + 0.1 + 2.5)
    assert abs(parse_sntp(reply, sent, received) - 2.5) < 1e-6
    assert _raises(ValueError, parse_sntp, reply[:40], sent, received)
    assert _raises(ValueError, parse_sntp, b"\x1c\x00" + reply[2:], sent, received)  # stratum 0
    assert _raises(ValueError, parse_sntp, reply, sent - 10_000, received - 10_000)  # implausible

    # a scripted night: game live, first poll is backlog, a 429 is survived, a basket arrives,
    # an unchanged feed answers 304, the game ends and its finals are saved, and a restart
    # does not log anything twice
    def action(n, t, home, away, kind="2pt", result="Made", sub=""):
        return {"actionNumber": n, "clock": "PT05M00.00S", "timeActual": t, "edited": t, "period": 1,
                "actionType": kind, "subType": sub, "shotResult": result,
                "scoreHome": str(home), "scoreAway": str(away)}

    first = [action(1, "2026-11-15T22:00:00.000Z", 0, 0, "period", "", "start"),
             action(2, "2026-11-15T22:00:10.000Z", 2, 0)]
    basket = first + [action(3, "2026-11-15T22:00:20.000Z", 2, 3, "3pt")]
    ended = basket + [action(4, "2026-11-15T22:00:30.000Z", 2, 3, "game", "", "end")]
    lm = {"Last-Modified": "Sun, 15 Nov 2026 22:00:21 GMT", "ETag": '"a"'}
    market = {"markets": [{"ticker": "KXNBAGAME-26NOV15LALPHX-PHX", "yes_bid_dollars": "0.40",
                           "yes_ask_dollars": "0.42", "last_price_dollars": "0.41", "volume_fp": "10"},
                          {"ticker": "KXNBAGAME-26NOV20AAABBB-BBB"}]}
    FakeSession.script = {
        "todaysScoreboard": [FakeResponse(200, board)],
        "playbyplay": [FakeResponse(200, {"game": {"actions": first}}, lm),
                       FakeResponse(429), requests.ConnectionError("network down"),
                       FakeResponse(200, None),  # empty body (how a throttled CDN looked before)
                       FakeResponse(200, {"game": {"actions": basket}}, lm),
                       FakeResponse(304, None, lm),
                       FakeResponse(200, {"game": {"actions": ended}}, lm)],
        "markets": [FakeResponse(200, market)],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logger = FeedLogger(root, poll_secs=0, nba_poll_secs=0, session_factory=FakeSession, clock=lambda server: 0.25, quiet=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(30):
                for b in logger.backoff.values():
                    b.next_ok = 0.0  # skip the real backoff pauses in the test
                logger.step(pool)
                if logger.done():
                    break
        assert logger.done() and logger.counts["errors"] == 3 and logger.offset == 0.25
        plays = read_csv(root / "2026-11-15" / "plays.csv")
        assert plays["action_number"].tolist() == ["1", "2", "3", "4"]
        assert plays["backlog"].tolist() == ["True", "True", "False", "False"]
        assert plays.loc[2, "last_modified"] == "2026-11-15T22:00:21+00:00" and plays.loc[2, "seen_after"]
        final = read_csv(root / "2026-11-15" / "final.csv")
        assert final["action_number"].tolist() == ["1", "2", "3", "4"]
        book = read_csv(root / "2026-11-15" / "book.csv")
        assert set(book["ticker"]) == {"KXNBAGAME-26NOV15LALPHX-PHX"} and book.loc[0, "yes_bid"] == "0.40"
        assert (root / "2026-11-15" / "games.csv").exists() and (root / "2026-11-15" / "clock.csv").exists()

        # restart: same day, game already final and saved -> done at once, nothing re-logged
        FakeSession.script["todaysScoreboard"] = [FakeResponse(200, {"scoreboard": {
            "gameDate": "2026-11-15", "games": [board_game("0022600237", "LAL", "PHX", FINAL)]}})]
        again = FeedLogger(root, poll_secs=0, nba_poll_secs=0, session_factory=FakeSession, clock=lambda server: 0.0, quiet=True)
        again.run(max_hours=0.01)
        assert again.done() and len(read_csv(root / "2026-11-15" / "plays.csv")) == 4

        # summary: only the live-seen made field goal counts, timed against the final timestamp;
        # free throws and late corrections are left out
        plays.loc[2, ["seen_after", "received_at"]] = ["2026-11-15T22:00:25+00:00", "2026-11-15T22:00:26+00:00"]
        delays = scoring_delays(plays, final)
        assert len(delays) == 1 and delays.loc[0, "published"] == 1.0 and delays.loc[0, "entered"] == 0.0
        assert delays.loc[0, "feed"] == 6.0 and delays.loc[0, "window"] == 1.0
        assert scoring_delays(plays.assign(action_type="freethrow"), final).empty
        late = plays.assign(received_at="2026-11-15T22:30:00+00:00", seen_after="2026-11-15T22:29:59+00:00")
        assert scoring_delays(late, final).empty

    print("unit tests (feed_delay): all pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log when the NBA live feed delivers each play.")
    parser.add_argument("--summary", nargs="?", const="", metavar="DATE",
                        help="print delay numbers for a logged day (default: the latest)")
    parser.add_argument("--test", action="store_true", help="run the offline unit tests only")
    parser.add_argument("--bypass-cache", action="store_true",
                        help="skip the CDN cache (fresher, but the CDN blocked it after ~10 min)")
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS, help="stop after this long")
    args = parser.parse_args()

    run_unit_tests()
    if args.test:
        return
    if args.summary is not None:
        summarize(args.summary or None)
        return
    logger = FeedLogger(bypass_cache=args.bypass_cache)
    try:
        logger.run(args.max_hours)
    except KeyboardInterrupt:
        logger.log("stopped by Ctrl+C; rerun the same command to carry on")
    print("summary: uv run python src/feed_delay.py --summary")


if __name__ == "__main__":
    main()
