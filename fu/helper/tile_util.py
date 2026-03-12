from helper import config

from typing import List, Optional, Tuple, Union, Set, Dict, Any
from dataclasses import dataclass


class Tile:
    """
    Represents a single Mahjong tile.
    Compatible with mahjong-python library conventions.
    
    Attributes:
        id_136 (int): Unique ID in 0-135 range
        type_34 (int): Tile type in 0-33 range (for 34-array indexing)
        copy_idx (int): Which of the 4 copies (0-3)
        suit (str): 'm', 'p', 's', or 'z'
        rank (int): 1-9 for suits, 1-7 for honors
        is_red_dora (bool): Whether this copy is a red dora
    """
    
    SUITS = ['m', 'p', 's', 'z']
    
    def __init__(self, id_136: int):
        if not 0 <= id_136 < 136:
            raise ValueError(f"Tile ID must be 0-135, got {id_136}")
        
        self.id_136 = id_136
        self.type_34 = id_136 // 4
        self.copy_idx = id_136 % 4
        self.is_red_dora = id_136 in config.RED_DORA_IDS
        
        # Calculate suit and rank from 34-type
        if self.type_34 < 9:  # Manzu 1-9
            self.suit = 'm'
            self.rank = self.type_34 + 1
        elif self.type_34 < 18:  # Pinzu 1-9
            self.suit = 'p'
            self.rank = self.type_34 - 9 + 1
        elif self.type_34 < 27:  # Souzu 1-9
            self.suit = 's'
            self.rank = self.type_34 - 18 + 1
        else:  # Honors (27-33)
            self.suit = 'z'
            self.rank = self.type_34 - 27 + 1  # 1=E, 2=S, ..., 7=Chun
    
    def to_34_type(self) -> int:
        """Return the 34-format tile type (0-33)."""
        return self.type_34
    
    def to_mspzd_char(self) -> str:
        """Return MSPZD notation for this tile (e.g., '5m', 'Haku')."""
        if self.suit == 'z':
            return config.HONOR_NAMES.get(self.type_34, str(self.rank))
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        red_mark = "*" if self.is_red_dora else ""
        return f"Tile({self.to_mspzd_char()}{red_mark}, id={self.id_136})"
    
    def __eq__(self, other):
        if isinstance(other, Tile):
            return self.id_136 == other.id_136
        if isinstance(other, int):
            return self.id_136 == other
        return False
    
    def __hash__(self):
        return hash(self.id_136)
    


@dataclass
class MahjongMeld:
    """
    Represents an exposed meld (Pon, Chi, Kan).
    Compatible with mahjong-python library conventions.
    """
    MELD_PON = 0
    MELD_CHI = 1
    MELD_KAN_OPEN = 2
    MELD_KAN_CLOSED = 3  # Ankan: exposed but keeps menzen
    
    meld_type: int          # MELD_PON, MELD_CHI, etc.
    tiles: List[int]        # List of 136-format tile IDs in the meld
    from_player: int        # Seat index of player who discarded (for Pon/Chi/Kan)
    called_tile: int        # The specific tile ID that was called/discarded
    
    @property
    def is_closed(self) -> bool:
        """Ankan is 'exposed' but retains closed-hand status for scoring."""
        return self.meld_type == MahjongMeld.MELD_KAN_CLOSED
    
    @property
    def tile_34_types(self) -> List[int]:
        """Return 34-format tile types for library compatibility."""
        return [t // 4 for t in self.tiles]
