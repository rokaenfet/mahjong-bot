from helper import config
from helper.tile_util import Tile

from typing import List, Optional, Tuple, Union, Set, Dict, Any
import re


class MahjongConverter:
    """
    Utility class for Mahjong tile data conversions.
    Fully compatible with mahjong-python library.
    """

    @staticmethod
    def to_34_array(tiles_136: List[int]) -> List[int]:
        """Convert list of 136-format tile IDs to 34-format count array."""
        tiles_34 = [0] * 34
        for t in tiles_136:
            tile_type = t // 4
            if 0 <= tile_type < 34:
                tiles_34[tile_type] += 1
        return tiles_34

    @staticmethod
    def from_34_to_136(tiles_34: List[int]) -> List[int]:
        """Convert 34-format count array to list of 136-format tile IDs."""
        tiles_136 = []
        for tile_type, count in enumerate(tiles_34):
            base_id = tile_type * 4
            for i in range(min(count, 4)):
                tiles_136.append(base_id + i)
        return tiles_136

    @staticmethod
    def to_str(tiles_136: List[int], use_red_zero: bool = False) -> str:
        """
        Convert 136-format tile list to MSPZD string notation.
        
        Args:
            tiles_136: List of tile IDs (0-135)
            use_red_zero: If True, represent red 5s as '0' (e.g., '0m' = red 5-man)
        """
        if not tiles_136:
            return ""
        
        tiles_34 = MahjongConverter.to_34_array(tiles_136)
        sections = [(0, 9, "m"), (9, 18, "p"), (18, 27, "s"), (27, 34, "z")]
        res = ""
        
        for start, end, suffix in sections:
            group = ""
            for i in range(start, end):
                count = tiles_34[i]
                if count > 0:
                    if suffix == "z":
                        # Use library honor names
                        group += config.HONOR_NAMES.get(i, str(i - 27 + 1)) * count
                    else:
                        rank = i - start + 1
                        # Handle red dora representation
                        if use_red_zero and rank == 5 and i in config.RED_DORA_BY_TYPE:
                            red_id = config.RED_DORA_BY_TYPE[i]
                            red_count = 1 if red_id in tiles_136 else 0
                            if red_count > 0:
                                group += "0" + str(rank) * (count - 1)
                            else:
                                group += str(rank) * count
                        else:
                            group += str(rank) * count
            
            if group:
                res += f"{group}{suffix}"
        
        return res

    @staticmethod
    def to_136(hand_str: str) -> List[int]:
        """
        Parse MSPZD string notation to list of 136-format tile IDs.
        
        Supports:
        - Standard: "123m456p789s123z"
        - Honor names: "ESWNHakuHatsuChun" or "1234567z"
        - Red dora as '0': "0m" = red 5-man (FIVE_RED_MAN=16)
        """
        tiles_34 = [0] * 34
        # Match digits, '0' for red, or honor names followed by suit
        pattern = r"(0|\d+|E|S|W|N|Haku|Hatsu|Chun)([mpsz])"
        matches = re.findall(pattern, hand_str)
        
        suit_offset = {"m": 0, "p": 9, "s": 18, "z": 27}
        honor_name_to_type = {"E": config.EAST, "S": config.SOUTH, "W": config.WEST, "N": config.NORTH, 
                             "Haku": config.HAKU, "Hatsu": config.HATSU, "Chun": config.CHUN}
        
        for value, suffix in matches:
            if suffix == "z":
                if value in honor_name_to_type:
                    tile_type = honor_name_to_type[value]
                else:
                    tile_type = suit_offset["z"] + int(value) - 1
            else:
                if value == "0":
                    # Red 5: use the red dora ID directly
                    rank = 5
                    tile_type = suit_offset[suffix] + rank - 1
                    # Add the specific red dora ID later
                else:
                    rank = int(value)
                    tile_type = suit_offset[suffix] + rank - 1
            
            if 0 <= tile_type < 34:
                if suffix != "z" and value == "0" and tile_type in config.RED_DORA_BY_TYPE:
                    # Special handling: add red dora ID instead of regular 5
                    tiles_34[tile_type] += 1  # Still count it in 34-array
                else:
                    tiles_34[tile_type] += 1
        
        # Convert to 136 IDs, substituting red dora IDs where appropriate
        tiles_136 = []
        for tile_type, count in enumerate(tiles_34):
            if tile_type in config.RED_DORA_BY_TYPE and count > 0:
                # Check if we need to use red dora ID (when '0' was in input)
                # For simplicity, we use red dora ID for the first copy if count > 0
                # In practice, parsing should track which were specified as '0'
                base_id = tile_type * 4
                red_id = config.RED_DORA_BY_TYPE[tile_type]
                added_red = False
                for i in range(count):
                    if not added_red and red_id not in tiles_136:
                        tiles_136.append(red_id)
                        added_red = True
                    else:
                        # Add non-red copy
                        for copy in range(4):
                            candidate = base_id + copy
                            if candidate != red_id and candidate not in tiles_136:
                                tiles_136.append(candidate)
                                break
            else:
                base_id = tile_type * 4
                for i in range(min(count, 4)):
                    tiles_136.append(base_id + i)
        
        return tiles_136

    @staticmethod
    def get_red_dora_ids() -> set:
        """Return set of 136-format IDs that represent red dora tiles."""
        return config.RED_DORA_IDS.copy()
    
    @staticmethod
    def is_red_dora(tile_id: int) -> bool:
        """Check if a 136-format tile ID is a red dora."""
        return tile_id in config.RED_DORA_IDS
    
    @staticmethod
    def get_red_dora_for_type(tile_type_34: int) -> Optional[int]:
        """Get the red dora 136-ID for a given 34-format tile type, or None."""
        return config.RED_DORA_BY_TYPE.get(tile_type_34)


class MahjongBase:
    """Provides shared comparison and arithmetic logic for tile collections."""
    
    def to_34(self) -> List[int]:
        raise NotImplementedError

    def to_ids(self) -> List[int]:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """Equality comparison with flexible type coercion."""
        if isinstance(other, str):
            other = MSPZD(other)
        elif isinstance(other, list) and all(isinstance(x, int) for x in other):
            other = Hand136(other)
        elif isinstance(other, int):
            other = Hand136([other])
        
        if hasattr(other, 'to_34'):
            return self.to_34() == other.to_34()
        return False

    def __add__(self, other: Union['MahjongBase', str, list, int]) -> 'Hand136':
        """Combine tile collections using + operator."""
        self_ids = self.to_ids()
        
        if isinstance(other, MahjongBase):
            other_ids = other.to_ids()
        elif isinstance(other, str):
            other_ids = MahjongConverter.to_136(other)
        elif isinstance(other, list):
            other_ids = other
        elif isinstance(other, int):
            other_ids = [other]
        else:
            raise TypeError(f"Unsupported operand type for +: '{type(other).__name__}'")
            
        return Hand136(self_ids + other_ids)

    def __radd__(self, other: Union[str, list, int]) -> 'Hand136':
        """Handle addition when raw types are on the left side."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['MahjongBase', str, list, int]) -> 'Hand136':
        """Remove tiles using - operator."""
        self_ids = self.to_ids().copy()
        
        if isinstance(other, MahjongBase):
            remove_ids = other.to_ids()
        elif isinstance(other, str):
            remove_ids = MahjongConverter.to_136(other)
        elif isinstance(other, list):
            remove_ids = other
        elif isinstance(other, int):
            remove_ids = [other]
        else:
            raise TypeError(f"Unsupported operand type for -: '{type(other).__name__}'")
        
        for rid in remove_ids:
            if rid in self_ids:
                self_ids.remove(rid)
                
        return Hand136(self_ids)


class Hand136(MahjongBase):
    """
    Represents a collection of tiles in 136-format ID list.
    Fully compatible with mahjong-python library.
    """
    
    def __init__(self, ids: Union[int, List[int]]):
        if isinstance(ids, int):
            self.ids = [ids]
        else:
            self.ids = sorted(ids) if ids else []

    def to_34(self) -> List[int]:
        return MahjongConverter.to_34_array(self.ids)

    def to_ids(self) -> List[int]:
        return self.ids.copy()  # Return copy to prevent external mutation

    def to_mspzd(self, use_red_zero: bool = False) -> 'MSPZD':
        return MSPZD(MahjongConverter.to_str(self.ids, use_red_zero))
    
    def to_tiles(self) -> List[Tile]:
        """Convert to list of Tile objects."""
        return [Tile(tid) for tid in self.ids]
    
    def contains(self, tile_id: int) -> bool:
        """Check if hand contains a specific tile ID."""
        return tile_id in self.ids
    
    def count(self, tile_id: int) -> int:
        """Count occurrences of a specific tile ID."""
        return self.ids.count(tile_id)
    
    def count_type(self, type_34: int) -> int:
        """Count tiles of a specific 34-format type."""
        return sum(1 for tid in self.ids if tid // 4 == type_34)
    
    def has_red_dora(self) -> bool:
        """Check if hand contains any red dora tiles."""
        return any(MahjongConverter.is_red_dora(tid) for tid in self.ids)
    
    def add(self, tile_id: int) -> 'Hand136':
        """Add a tile and return new Hand136 (immutable-style)."""
        return Hand136(self.ids + [tile_id])
    
    def remove(self, tile_id: int) -> 'Hand136':
        """Remove one instance of a tile and return new Hand136."""
        new_ids = self.ids.copy()
        if tile_id in new_ids:
            new_ids.remove(tile_id)
        return Hand136(new_ids)
    
    def draw(self, tile_id: int) -> None:
        """In-place draw: add tile to hand."""
        self.ids.append(tile_id)
        self.ids.sort()
    
    def discard(self, tile_id: int) -> bool:
        """In-place discard: remove tile from hand. Returns True if successful."""
        if tile_id in self.ids:
            self.ids.remove(tile_id)
            return True
        return False
    
    def __repr__(self):
        return f"Hand136({self.ids})"
    
    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)
    
    def __getitem__(self, position):
        return self.ids[position]
    
    def __copy__(self):
        return Hand136(self.ids.copy())


class MSPZD(MahjongBase):
    """
    Represents tiles in MSPZD string notation.
    Immutable wrapper compatible with mahjong-python conventions.
    """
    
    def __init__(self, notation: str):
        self.notation = notation.strip()

    def to_34(self) -> List[int]:
        return MahjongConverter.to_34_array(MahjongConverter.to_136(self.notation))

    def to_ids(self) -> List[int]:
        return MahjongConverter.to_136(self.notation)

    def to_136(self) -> Hand136:
        return Hand136(self.to_ids())
    
    def __repr__(self):
        return f"MSPZD('{self.notation}')"
    
    def __str__(self):
        return self.notation
    