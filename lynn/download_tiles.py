"""Download clean reference tile images from FluffyStuff/riichi-mahjong-tiles."""

from __future__ import annotations

import urllib.request
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/FluffyStuff/riichi-mahjong-tiles/master/Export/Regular"
OUT_DIR = Path(__file__).parent / "reference"

TILE_MAP = {
    "Man1": "1m", "Man2": "2m", "Man3": "3m", "Man4": "4m", "Man5": "5m",
    "Man6": "6m", "Man7": "7m", "Man8": "8m", "Man9": "9m",
    "Pin1": "1p", "Pin2": "2p", "Pin3": "3p", "Pin4": "4p", "Pin5": "5p",
    "Pin6": "6p", "Pin7": "7p", "Pin8": "8p", "Pin9": "9p",
    "Sou1": "1s", "Sou2": "2s", "Sou3": "3s", "Sou4": "4s", "Sou5": "5s",
    "Sou6": "6s", "Sou7": "7s", "Sou8": "8s", "Sou9": "9s",
    "Ton": "east", "Nan": "south", "Shaa": "west", "Pei": "north",
    "Haku": "haku", "Hatsu": "hatsu", "Chun": "chun",
    "Man5-Dora": "0m", "Pin5-Dora": "0p", "Sou5-Dora": "0s",
}


def main():
    OUT_DIR.mkdir(exist_ok=True)

    for repo_name, shorthand in TILE_MAP.items():
        url = f"{REPO_BASE}/{repo_name}.png"
        out_path = OUT_DIR / f"{shorthand}.png"

        if out_path.exists():
            print(f"  skip {shorthand} (already exists)")
            continue

        print(f"  {repo_name}.png -> {shorthand}.png ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, str(out_path))
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} tiles in {OUT_DIR}/")


if __name__ == "__main__":
    main()
