from helper import config
from helper.utility import Hand136
from helper.tile_util import MahjongMeld

from typing import List, Optional, Tuple, Union, Set, Dict, Any
from enum import Enum, auto

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.agari import Agari
from mahjong.meld import Meld as LibraryMeld

import random

class MahjongPlayer:
    """
    Represents a player in a Riichi Mahjong game.
    
    Manages hand state, discards, melds, and Riichi-specific flags.
    Decision methods are stubs to be overridden by AI implementations.
    """
    
    def __init__(self, player_id: int, name: str, initial_score: int = 25000):
        self.player_id = player_id  # 0-3, fixed seat index
        self.name = name
        self.score = initial_score
        
        # Hand state (reset each kyoku)
        self.hand: Optional[Hand136] = None
        self.discards: List[int] = []  # List of 136-format tile IDs discarded this hand
        self.melds: List[MahjongMeld] = []    # Exposed melds (Pon/Chi/Kan)
        
        # Riichi-specific flags
        self.is_riichi = False
        self.is_menzen = True          # True if no open melds (Ankan keeps this True)
        self.furiten = False           # Permanent furiten (cannot Ron)
        self.temp_furiten = False      # Temporary furiten (resets on next draw)
        self.riichi_ippatsu = False    # True if can win on ippatsu after riichi
        self.riichi_stick_paid = False # Track if 1000pt was paid
        
        # Round state
        self.seat_wind: int = 0        # 0=East, 1=South, 2=West, 3=North (relative to dealer)
        self.is_dealer = False         # True if this player is Oya this kyoku
        self.is_tenpai_last = False    # Track tenpai status for ryukyoku settlement
        
        # Cached calculations (invalidate on hand change)
        self._wait_cache: Optional[Set[int]] = None
        self._shanten_cache: Optional[int] = None
        
        # Library instances (reused for efficiency)
        self._shanten_calc = Shanten()
        self._agari_calc = Agari()
    
    # =========================================================================
    # HAND MANAGEMENT
    # =========================================================================
    
    def set_hand(self, tile_ids: List[int]) -> None:
        """Initialize hand at start of kyoku."""
        self.hand = Hand136(tile_ids)
        self._invalidate_cache()
    
    def draw_tile(self, tile_id: int) -> None:
        """Add drawn tile to hand. Resets temporary furiten."""
        if self.hand is None:
            raise RuntimeError("Cannot draw: hand not initialized")
        self.hand.draw(tile_id)
        self.temp_furiten = False  # Temporary furiten clears on draw
        self._invalidate_cache()
    
    def discard_tile(self, tile_id: int) -> bool:
        """
        Discard a tile from hand. Updates furiten status.
        Returns True if successful, False if tile not in hand.
        """
        if self.hand is None or not self.hand.discard(tile_id):
            return False
        
        # Record discard
        self.discards.append(tile_id)
        
        # Update furiten: if discarded tile is in current waits, set permanent furiten
        waits = self.get_wait_tiles()
        if tile_id in waits:
            self.furiten = True
        
        # Temporary furiten: cannot Ron until next self-draw
        self.temp_furiten = True
        
        self._invalidate_cache()
        return True
    
    def add_exposed_meld(self, meld: MahjongMeld) -> None:
        """Add a called meld. Updates menzen status."""
        self.melds.append(meld)
        # Ankan (closed kan) keeps menzen, all others break it
        if not meld.is_closed:
            self.is_menzen = False
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Clear cached shanten/wait calculations."""
        self._shanten_cache = None
        self._wait_cache = None
    
    # =========================================================================
    # RIICHI STATE & FURITEN
    # =========================================================================
    
    def declare_riichi(self, discard_tile_id: int) -> bool:
        """
        Declare Riichi. Must be called during discard phase.
        Returns True if successful, False if conditions not met.
        """
        if not self.is_menzen:
            return False
        if self.is_riichi:
            return False
        if self.get_shanten() != 0:
            return False
        if self.score < 1000:
            return False
        
        # Pay 1000pt stick (handled by game engine, but track here)
        self.riichi_stick_paid = True
        self.is_riichi = True
        self.riichi_ippatsu = True  # Ippatsu chance starts now
        
        # Auto-discard mode: must discard drawn tile unless it wins
        # (Enforced in game loop, not here)
        
        return True
    
    def clear_ippatsu_chance(self) -> None:
        """Called when any call (Pon/Chi/Kan) occurs after riichi."""
        self.riichi_ippatsu = False
    
    def reset_furiten_on_draw(self) -> None:
        """Clear temporary furiten at start of draw phase."""
        self.temp_furiten = False
    
    # =========================================================================
    # HAND EVALUATION (Library Integration)
    # =========================================================================
    
    def get_shanten(self) -> int:
        """Calculate current shanten number (0 = tenpai)."""
        if self._shanten_cache is not None:
            return self._shanten_cache
        
        if self.hand is None:
            return 8  # Invalid state
        
        tiles_34 = self.hand.to_34()
        melds_34 = [m.tile_34_types for m in self.melds]
        
        self._shanten_cache = self._shanten_calc.calculate_shanten(tiles_34, melds_34)
        return self._shanten_cache
    
    def is_tenpai(self) -> bool:
        """Check if hand is in tenpai (shanten == 0)."""
        return self.get_shanten() == 0
    
    def get_wait_tiles(self) -> Set[int]:
        """Return set of 34-format tile types that complete the hand."""
        if self._wait_cache is not None:
            return self._wait_cache.copy()
        
        if self.hand is None:
            return set()
        
        tiles_34 = self.hand.to_34()
        melds_for_library = self._to_library_meld_tuples()
        
        waits = set()
        for tile_type in range(34):
            if tiles_34[tile_type] >= 4:
                continue
            
            test_tiles = tiles_34.copy()
            test_tiles[tile_type] += 1
            
            if sum(test_tiles) != 14:
                continue
            
            try:
                if self._agari_calc.is_agari(test_tiles, melds_for_library):
                    waits.add(tile_type)
            except Exception as e:
                print(f"Warning: is_agari failed: {e}")
                continue
        
        self._wait_cache = waits
        return waits.copy()
    
    def can_win_on_tile(self, tile_34_type: int, is_tsumo: bool) -> bool:
        """
        Check if hand can win on a specific tile.
        Args:
            tile_34_type: 34-format tile type (0-33)
            is_tsumo: True if self-draw, False if ron
        Returns:
            True if win is valid (tenpai + yaku + no furiten for ron)
        Note: Yaku validation must be done separately by game engine.
        """
        if not self.is_tenpai():
            return False
        
        waits = self.get_wait_tiles()
        if tile_34_type not in waits:
            return False
        
        # Furiten check: cannot Ron if furiten (tsumo is allowed)
        if not is_tsumo and (self.furiten or self.temp_furiten):
            return False
        
        return True
    
    # =========================================================================
    # CALL VALIDATION (Pon/Chi/Kan)
    # =========================================================================
    
    def can_call_pon(self, discarded_tile_34: int) -> bool:
        """Check if Pon is possible on the discarded tile."""
        if self.hand is None:
            return False
        return self.hand.count_type(discarded_tile_34) == 2
    
    def can_call_chi(self, discarded_tile_34: int, is_from_left: bool) -> bool:
        """
        Check if Chi is possible. 
        Chi is ONLY allowed from the player to immediate left.
        """
        if not is_from_left:
            return False
        if self.hand is None or discarded_tile_34 >= 27:  # Honors cannot chi
            return False
        
        hand_34 = self.hand.to_34()
        suit_start = (discarded_tile_34 // 9) * 9
        rank = discarded_tile_34 % 9
        
        # Check three possible sequences
        patterns = [
            (rank >= 2, discarded_tile_34 - 2, discarded_tile_34 - 1),
            (1 <= rank <= 7, discarded_tile_34 - 1, discarded_tile_34 + 1),
            (rank <= 6, discarded_tile_34 + 1, discarded_tile_34 + 2),
        ]
        
        for valid, t1, t2 in patterns:
            if valid and hand_34[t1] > 0 and hand_34[t2] > 0:
                return True
        return False
    
    def can_call_kan(self, discarded_tile_34: int, is_closed: bool) -> bool:
        """
        Check if Kan is possible.
        Args:
            discarded_tile_34: 34-format tile type
            is_closed: True for Ankan (4 in hand), False for Daiminkan (3 in hand + discard)
        """
        if self.hand is None:
            return False
        count = self.hand.count_type(discarded_tile_34)
        return count == 4 if is_closed else count == 3
    
    def can_call_ron(self, discarded_tile_34: int) -> bool:
        """Check if Ron is possible (tenpai + not furiten). Yaku check separate."""
        return self.can_win_on_tile(discarded_tile_34, is_tsumo=False)
    
    # =========================================================================
    # DECISION STUBS (Override for AI)
    # =========================================================================
    
    def decide_discard(self) -> int:
        """
        Select a tile to discard. Returns 136-format tile ID.
        Override with AI logic. Default: random legal discard.
        """
        if self.hand is None or len(self.hand) == 0:
            raise RuntimeError("No tiles to discard")
        
        # If riichi, must discard drawn tile (handled by game, but safety check)
        if self.is_riichi and len(self.hand) == 14:
            return self.hand.ids[-1]  # Last drawn tile
        
        # Simple random discard (replace with AI)
        return random.choice(self.hand.ids)
    
    def decide_call(self, discarded_tile: int, is_from_left: bool) -> Optional[str]:
        """
        Decide whether to call on a discard.
        Returns: 'ron', 'kan', 'pon', 'chi', or None.
        Override with AI logic. Default: prioritize Ron > Kan > Pon > Chi.
        """
        tile_34 = discarded_tile // 4
        
        # Priority: Ron (if valid)
        if self.can_call_ron(tile_34):
            # Note: Yaku validation must be done by game engine before confirming Ron
            return 'ron'
        
        # Kan (higher priority than Pon/Chi per rules)
        if self.can_call_kan(tile_34, is_closed=False):
            return 'kan'
        # Note: Ankan is handled in draw phase, not on discard
        
        # Pon
        if self.can_call_pon(tile_34):
            return 'pon'
        
        # Chi (only from left)
        if self.can_call_chi(tile_34, is_from_left):
            return 'chi'
        
        return None
    
    def decide_ankan(self) -> Optional[int]:
        """
        Check for Ankan opportunity during draw phase.
        Returns 34-format tile type if Ankan should be declared, None otherwise.
        Override with AI logic. Default: declare if possible.
        """
        if self.hand is None:
            return None
        
        for tile_type in range(34):
            if self.hand.count_type(tile_type) == 4:
                # Simple AI: always ankan if possible (replace with strategy)
                return tile_type
        return None
    
    # =========================================================================
    # KYOKU RESET & UTILITIES
    # =========================================================================
    
    def reset_for_kyoku(self, seat_wind: int, is_dealer: bool) -> None:
        """Reset player state for new kyoku."""
        self.hand = None
        self.discards = []
        self.melds = []
        self.is_riichi = False
        self.is_menzen = True
        self.furiten = False
        self.temp_furiten = False
        self.riichi_ippatsu = False
        self.riichi_stick_paid = False
        self.seat_wind = seat_wind
        self.is_dealer = is_dealer
        self.is_tenpai_last = False
        self._invalidate_cache()
    
    def add_score(self, points: int) -> None:
        """Add points to score (can be negative for payments)."""
        self.score += points
    
    def get_discard_history_34(self) -> List[int]:
        """Return discard history as 34-format tile types."""
        return [t // 4 for t in self.discards]
    
    def _to_library_meld_tuples(self) -> List[tuple]:
        """
        Convert internal Meld objects to mahjong-python library tuple format.
        
        Returns:
            List of tuples: [(meld_type, tile1, tile2, tile3[, tile4]), ...]
        
        Library meld type constants:
            0 = CHI (sequence)
            1 = PON (triplet)
            2 = KAN (quad)
            3 = KAN_CLOSED (closed quad)
        """
        from mahjong.meld import Meld as LibraryMeld
        
        result = []
        for meld in self.melds:
            if meld.meld_type == MahjongMeld.MELD_CHI:
                lib_type = LibraryMeld.CHI
                tiles = sorted(meld.tile_34_types)  # CHI must be sorted
                result.append((lib_type, tiles[0], tiles[1], tiles[2]))
            
            elif meld.meld_type == MahjongMeld.MELD_PON:
                lib_type = LibraryMeld.PON
                tiles = meld.tile_34_types
                result.append((lib_type, tiles[0], tiles[1], tiles[2]))
            
            elif meld.meld_type == MahjongMeld.MELD_KAN_OPEN:
                lib_type = LibraryMeld.KAN
                tiles = meld.tile_34_types
                result.append((lib_type, tiles[0], tiles[1], tiles[2], tiles[3]))
            
            elif meld.meld_type == MahjongMeld.MELD_KAN_CLOSED:
                lib_type = LibraryMeld.KAN  # Some library versions use 3 for closed
                tiles = meld.tile_34_types
                result.append((lib_type, tiles[0], tiles[1], tiles[2], tiles[3]))
        
        return result
    
    def __repr__(self):
        dealer_mark = " [DEALER]" if self.is_dealer else ""
        riichi_mark = " [RIICHI]" if self.is_riichi else ""
        furiten_mark = " [FURITEN]" if self.furiten else ""
        hand_count = len(self.hand) if self.hand else 0
        return (f"{self.name}{dealer_mark}{riichi_mark}{furiten_mark} | "
                f"Score: {self.score} | Hand: {hand_count} tiles")