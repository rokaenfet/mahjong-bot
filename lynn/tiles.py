from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class Suit(Enum):
    MAN = "man"        # Characters (萬)
    PIN = "pin"        # Circles (筒)
    SOU = "sou"        # Bamboo (索)
    WIND = "wind"      # Wind tiles (風牌)
    DRAGON = "dragon"  # Dragon tiles (三元牌)


SUIT_SHORT = {
    Suit.MAN: "m",
    Suit.PIN: "p",
    Suit.SOU: "s",
}

WIND_VALUES = {"east": 1, "south": 2, "west": 3, "north": 4}
DRAGON_VALUES = {"haku": 1, "hatsu": 2, "chun": 3}

HONOR_NAMES = {
    (Suit.WIND, 1): "East",
    (Suit.WIND, 2): "South",
    (Suit.WIND, 3): "West",
    (Suit.WIND, 4): "North",
    (Suit.DRAGON, 1): "Haku",
    (Suit.DRAGON, 2): "Hatsu",
    (Suit.DRAGON, 3): "Chun",
}


@dataclass(frozen=True)
class Tile:
    suit: Suit
    value: int  # 1-9 for numbered suits; 1-4 for winds; 1-3 for dragons
    is_red: bool = False  # True for red-five variants (0m/0p/0s)

    def __str__(self) -> str:
        if self.suit in SUIT_SHORT:
            prefix = "0" if self.is_red and self.value == 5 else str(self.value)
            return f"{prefix}{SUIT_SHORT[self.suit]}"
        return HONOR_NAMES.get((self.suit, self.value), "??")

    def __repr__(self) -> str:
        return f"Tile({self})"

    @classmethod
    def from_shorthand(cls, code: str) -> "Tile":
        """Parse shorthand notation like '1m', '5p', '3s', 'east', 'chun'."""
        code = code.strip().lower()

        if code in WIND_VALUES:
            return cls(Suit.WIND, WIND_VALUES[code])
        if code in DRAGON_VALUES:
            return cls(Suit.DRAGON, DRAGON_VALUES[code])

        if len(code) == 2 and code[0].isdigit():
            value = int(code[0])
            suit_char = code[1]
            suit_map = {"m": Suit.MAN, "p": Suit.PIN, "s": Suit.SOU}
            if suit_char in suit_map and 1 <= value <= 9:
                return cls(suit_map[suit_char], value)

        raise ValueError(f"Invalid tile shorthand: '{code}'")


# All 34 unique tile types
ALL_TILES: list[Tile] = (
    [Tile(Suit.MAN, v) for v in range(1, 10)]
    + [Tile(Suit.PIN, v) for v in range(1, 10)]
    + [Tile(Suit.SOU, v) for v in range(1, 10)]
    + [Tile(Suit.WIND, v) for v in range(1, 5)]
    + [Tile(Suit.DRAGON, v) for v in range(1, 4)]
)

# Lookup: shorthand string -> Tile
TILE_LOOKUP: dict[str, Tile] = {str(t): t for t in ALL_TILES}

# Red-five shorthands
TILE_LOOKUP["0m"] = Tile(Suit.MAN, 5, is_red=True)
TILE_LOOKUP["0p"] = Tile(Suit.PIN, 5, is_red=True)
TILE_LOOKUP["0s"] = Tile(Suit.SOU, 5, is_red=True)
