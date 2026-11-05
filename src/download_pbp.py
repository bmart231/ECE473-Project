"""Download 2015-16 to 2024-25 NBA play-by-play from shufinskiy/nba_data into data/raw/pbp/.

Run once from anywhere: `uv run python src/download_pbp.py`
Safe to re-run: seasons that are already unpacked are skipped.
"""

import tarfile
import time
from pathlib import Path

import requests

LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pbp"

# file year = season start year: nbastats_2015 is 2015-16, nbastats_2024 is 2024-25
# 2015-2023 -> fit the live model, 2024 -> validation season
SEASONS = range(2015, 2025)
NAMES = [f"nbastats_{y}" for y in SEASONS] + [f"nbastats_po_{y}" for y in SEASONS]  # regular season + playoffs


def file_urls() -> dict[str, str]:
    """list_data.txt has one 'name=url' line per data file."""
    response = requests.get(LIST_URL, timeout=30)
    response.raise_for_status()
    return dict(line.split("=", 1) for line in response.text.split() if "=" in line)


def download(url: str, path: Path) -> None:
    """Stream one file to disk (write to .part first so a failed download never looks finished)."""
    part = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part.rename(path)


def unpack(archive: Path) -> list[str]:
    """Extract the CSV(s) in a .tar.xz next to it, then delete the archive."""
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getnames()
        tar.extractall(OUT_DIR, filter="data")  # 'data' filter blocks paths that escape OUT_DIR
    archive.unlink()
    return members


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = file_urls()
    missing = [n for n in NAMES if n not in urls]
    if missing:
        raise SystemExit(f"not in list_data.txt: {missing}")

    for name in NAMES:
        if (OUT_DIR / f"{name}.csv").exists():
            print(f"{name}: already here, skipping")
            continue
        archive = OUT_DIR / f"{name}.tar.xz"
        download(urls[name], archive)
        members = unpack(archive)
        print(f"{name}: {', '.join(members)}")
        time.sleep(0.5)  # be polite to GitHub

    csvs = sorted(OUT_DIR.glob("*.csv"))
    size_mb = sum(p.stat().st_size for p in csvs) / 1e6
    print(f"\n{len(csvs)} CSVs in {OUT_DIR} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
