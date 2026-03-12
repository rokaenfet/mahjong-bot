from helper import config
from helper.tile_util import MahjongMeld
from helper.player import MahjongPlayer
from helper.game_util import GameLogEntry, GamePhase
from helper.utility import MahjongConverter, Hand136

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.agari import Agari

from typing import List, Optional, Tuple, Union, Set, Dict, Any
from enum import Enum, auto

import random




class MahjongGame:
    """
    Main game engine for Riichi Mahjong simulation.
    
    Manages the complete game state machine from initial deal through
    final scoring, with support for custom AI agents and detailed logging.
    """
    
    # Game configuration constants
    BA_NAMES = ["East", "South"]  # Standard Ton-Nan; West/North optional
    STARTING_SCORE = 25000
    DEAD_WALL_SIZE = 14
    MIN_WALL_TILES = 14  # Game ends when live wall reaches this count
    
    def __init__(
        self,
        player_names: Optional[List[str]] = None,
        starting_score: int = STARTING_SCORE,
        total_ba: int = 2,  # 2 = Hanchan (East+South), 1 = Tonpuu (East only)
        enable_red_dora: bool = True,
        log_level: str = "full"  # "minimal", "actions", "full"
    ):
        """
        Initialize game engine.
        
        Args:
            player_names: List of player names (auto-generated if None)
            starting_score: Initial points per player (default: 25000)
            total_ba: Number of rounds (1=East only, 2=East+South)
            enable_red_dora: Whether to use red 5 tiles as dora
            log_level: Logging verbosity for _log_state()
        """
        # Player setup
        self.starting_score = starting_score
        self.total_ba = total_ba
        self.enable_red_dora = enable_red_dora
        self.log_level = log_level
        
        if player_names is None:
            player_names = [f"Player-{i+1}" for i in range(4)]
        self.player_names = player_names[:4]  # Ensure exactly 4 players
        
        # Initialize players with unique IDs
        self.players: List[MahjongPlayer] = [
            MahjongPlayer(player_id=i, name=name, initial_score=starting_score)
            for i, name in enumerate(self.player_names)
        ]
        
        # Table state
        self.wall: List[int] = []  # List of 136-format tile IDs
        self.dead_wall: List[int] = []  # Rinshan tiles
        self.dora_indicators: List[int] = []  # Revealed dora tiles (136-format)
        
        # Game progression counters
        self.current_ba: int = 0  # 0=East, 1=South
        self.current_kyoku: int = 0  # 1-4 (or more with renchan)
        self.honba_count: int = 0  # Counter sticks (300pt each)
        self.riichi_pot: int = 0  # Accumulated 1000pt sticks
        
        # Turn management
        self.dealer_index: int = 0  # Player index who is Oya
        self.current_turn: int = 0  # Player index whose turn it is
        self.last_discarder: Optional[int] = None  # For Chi validation
        
        # Win/settlement state
        self.kyoku_winner: Optional[int] = None  # Player who won this hand
        self.kyoku_win_type: Optional[str] = None  # "tsumo" or "ron"
        self.kyoku_ryukyoku: bool = False  # True if exhaustive draw
        
        # Logging
        self.game_log: List[GameLogEntry] = []
        self.hand_calculator = HandCalculator()
        
        # Initialize game state
        self._reset_game_state()
    
    # =========================================================================
    # GAME LIFECYCLE
    # =========================================================================
    
    def _reset_game_state(self) -> None:
        """Reset all game state to initial values."""
        self.current_ba = 0
        self.current_kyoku = 0
        self.honba_count = 0
        self.riichi_pot = 0
        self.dealer_index = 0
        self.game_log.clear()
        
        # Reset all players
        for player in self.players:
            player.reset_for_kyoku(seat_wind=0, is_dealer=False)
    
    def play_game(self) -> Dict[str, Any]:
        """
        Execute complete game from start to finish.
        
        Returns:
            Dictionary with final scores, rankings, and game metadata
        """
        self._log_state(GamePhase.SETUP.value, "GAME_START")
        while not self._is_game_over():
            self._play_ba()
            self._log_state(GamePhase.SETUP.value, f"BA_COMPLETE_{self.BA_NAMES[self.current_ba - 1]}")
        
        # Final scoring with Uma
        results = self._calculate_final_results()
        self._log_state(GamePhase.GAME_END.value, "GAME_OVER", metadata=results)
        
        return results
    
    def _is_game_over(self) -> bool:
        """Check if game has completed all rounds."""
        # Game ends after South 4 (All Last) is fully resolved
        if self.current_ba >= len(self.BA_NAMES):
            return True
        # Optional: End early if a player reaches target score
        # if any(p.score >= self.ending_score for p in self.players):
        #     return True
        return False
    
    def _play_ba(self) -> None:
        """Play through one complete round (East or South)."""
        ba_name = self.BA_NAMES[self.current_ba]
        self._log_state(GamePhase.SETUP.value, f"BA_START_{ba_name}")
        
        # Play kyoku until round completion condition
        while self._is_ba_active():
            self._setup_kyoku()
            self._play_kyoku()
            self._settle_kyoku()
            self._log_state(GamePhase.SETTLEMENT.value, f"KYOKU_COMPLETE_{ba_name}{self.current_kyoku}")
        
        # Advance to next ba
        self.current_ba += 1
        self.current_kyoku = 0
        self.honba_count = 0  # Reset honba between rounds
    
    def _is_ba_active(self) -> bool:
        """Check if current round should continue."""
        # Standard: Play at least 4 kyoku, continue if dealer tenpai/wins
        if self.current_kyoku < 4:
            return True
        # All Last (South 4) ends immediately after settlement
        if self.current_ba == 1 and self.current_kyoku == 4:
            return False
        # Renchan: continue if dealer was tenpai or won
        dealer = self.players[self.dealer_index]
        return dealer.is_tenpai_last or self.kyoku_winner == self.dealer_index
    
    # =========================================================================
    # KYOKU SETUP & TEARDOWN
    # =========================================================================
    
    def _setup_kyoku(self) -> None:
        """Initialize state for a new hand."""
        self.current_kyoku += 1
        self.kyoku_winner = None
        self.kyoku_win_type = None
        self.kyoku_ryukyoku = False
        self.last_discarder = None

        # Reset player state for new kyoku
        for i, player in enumerate(self.players):
            seat_wind = (i - self.dealer_index) % 4
            is_dealer = (i == self.dealer_index)
            player.reset_for_kyoku(seat_wind=seat_wind, is_dealer=is_dealer)

        # Build wall and dead wall
        self._build_wall()
        # Deal 13 tiles to everyone
        for player in self.players:
            hand = [self.wall.pop() for _ in range(13)]
            player.hand = Hand136(hand)

        # Dealer gets final tile (14th)
        self.players[self.dealer_index].hand.draw(self.wall.pop())
        
        # Reveal initial dora indicator
        self.dora_indicators = [self.dead_wall.pop()]
        
        # Set starting turn (dealer starts)
        self.current_turn = self.dealer_index
        
        self._log_state(
            GamePhase.SETUP.value, 
            "KYOKU_SETUP",
            metadata={
                "ba": self.BA_NAMES[self.current_ba],
                "kyoku": self.current_kyoku,
                "dealer": self.players[self.dealer_index].name,
                "dora": self._tile_ids_to_mspzd(self.dora_indicators)
            }
        )
    
    def _build_wall(self) -> None:
        """Shuffle and build live wall + dead wall."""
        # Create and shuffle all tiles
        all_tiles = list(range(136))
        if not self.enable_red_dora:
            # Remove red dora IDs if disabled
            all_tiles = [t for t in all_tiles if t not in config.RED_DORA_IDS]
        random.shuffle(all_tiles)
        
        # Reserve dead wall (last 14 tiles)
        self.dead_wall = all_tiles[-self.DEAD_WALL_SIZE:]
        self.wall = all_tiles[:-self.DEAD_WALL_SIZE]
    
    def _settle_kyoku(self) -> None:
        """Handle end-of-hand scoring and state progression."""
        if self.kyoku_winner is not None:
            # Hand was won: calculate and distribute points
            self._settle_agari()
        else:
            # Exhaustive draw: check tenpai and apply penalties
            self._settle_ryukyoku()
        
        # Update dealer rotation
        self._update_dealer_rotation()
    
    def _update_dealer_rotation(self) -> None:
        """Determine if dealer stays (renchan) or rotates."""
        dealer = self.players[self.dealer_index]
        
        # Renchan conditions: dealer won OR dealer is tenpai at ryukyoku
        if self.kyoku_winner == self.dealer_index or dealer.is_tenpai_last:
            self.honba_count += 1  # Add counter stick
            # Dealer stays, kyoku counter may increment for display
        else:
            # Rotate dealer clockwise
            self.dealer_index = (self.dealer_index + 1) % 4
            self.honba_count = 0  # Reset counter sticks
    
    # =========================================================================
    # KYOKU GAME LOOP
    # =========================================================================
    
    def _play_kyoku(self) -> None:
        """Execute main turn loop for a single hand."""
        while not self._is_kyoku_over():
            current_player = self.players[self.current_turn]
            
            # === DRAW PHASE ===
            if not (current_player.is_dealer and len(current_player.hand) == 14):
                # Dealer starts with 14 tiles, skips first draw
                if not self.wall:
                    break  # Wall exhausted
                drawn_tile = self.wall.pop()
                current_player.draw_tile(drawn_tile)
                self._log_state(
                    GamePhase.DRAW.value,
                    "DRAW",
                    tile=self._tile_id_to_mspzd(drawn_tile),
                    metadata={"player": current_player.name}
                )
                
                # Check for Ankan (closed kan) - must be before win check
                ankan_type = current_player.decide_ankan()
                if ankan_type is not None and self._can_declare_ankan(current_player, ankan_type):
                    self._execute_ankan(current_player, ankan_type)
                    continue  # Repeat decision cycle after kan draw
            
            # Check for Tsumo win
            if self._check_tsumo_win(current_player):
                break
            
            # === DISCARD PHASE ===
            discard_tile = current_player.decide_discard()
            if not current_player.discard_tile(discard_tile):
                raise RuntimeError(f"Failed to discard tile {discard_tile}")
            
            self._log_state(
                GamePhase.DISCARD.value,
                "DISCARD",
                tile=self._tile_id_to_mspzd(discard_tile),
                metadata={
                    "player": current_player.name,
                    "is_riichi": current_player.is_riichi
                }
            )
            
            # Check for Riichi declaration (on discard)
            if (current_player.is_menzen and current_player.is_tenpai() 
                and not current_player.is_riichi and current_player.score >= 1000):
                if current_player.declare_riichi(discard_tile):
                    self.riichi_pot += 1000
                    self._log_state(
                        GamePhase.DISCARD.value,
                        "RIICHI_DECLARED",
                        metadata={"player": current_player.name}
                    )
            
            # === CALL RESOLUTION PHASE ===
            call_result = self._resolve_calls(discard_tile, self.current_turn)
            
            if call_result["called"]:
                # A call was made: turn jumps to caller
                self.current_turn = call_result["caller_idx"]
                self.last_discarder = self.current_turn
                
                # If Kan was called, caller draws replacement immediately
                if call_result["call_type"] == "kan":
                    self._handle_kan_replacement_draw(
                        self.players[self.current_turn]
                    )
                # Pon/Chi: caller discards immediately without drawing
                self._execute_caller_discard()
            else:
                # No calls: normal turn rotation
                self.current_turn = (self.current_turn + 1) % 4
                self.last_discarder = self.current_turn
    
    def _is_kyoku_over(self) -> bool:
        """Check if current hand has ended."""
        return (self.kyoku_winner is not None or 
                self.kyoku_ryukyoku or 
                len(self.wall) < self.MIN_WALL_TILES)
    
    # =========================================================================
    # CALL RESOLUTION (Naki)
    # =========================================================================
    
    def _resolve_calls(self, discarded_tile: int, discarder_idx: int) -> Dict[str, Any]:
        """
        Check for calls on a discard in priority order: Ron > Kan > Pon > Chi.
        
        Returns dict with call details or {"called": False}.
        """
        discarded_34 = discarded_tile // 4
        result = {"called": False}
        
        # Check players in turn order starting after discarder
        for offset in range(1, 4):
            player_idx = (discarder_idx + offset) % 4
            player = self.players[player_idx]
            is_from_left = (offset == 1)  # Only left player can Chi
            
            # === RON CHECK (Highest Priority) ===
            if player.can_call_ron(discarded_34):
                # Validate yaku before confirming win
                if self._validate_yaku_for_ron(player, discarded_tile):
                    result.update({
                        "called": True,
                        "call_type": "ron",
                        "caller_idx": player_idx,
                        "winner": player_idx
                    })
                    self.kyoku_winner = player_idx
                    self.kyoku_win_type = "ron"
                    return result
                # If no yaku, cannot Ron (continue checking other calls)
            
            # === KAN CHECK (Daiminkan) ===
            if player.can_call_kan(discarded_34, is_closed=False):
                result.update({
                    "called": True,
                    "call_type": "kan",
                    "caller_idx": player_idx
                })
                self._execute_open_kan(player, discarded_tile, discarder_idx)
                return result
            
            # === PON CHECK ===
            if player.can_call_pon(discarded_34):
                result.update({
                    "called": True,
                    "call_type": "pon",
                    "caller_idx": player_idx
                })
                self._execute_pon(player, discarded_tile, discarder_idx)
                return result
            
            # === CHI CHECK (Only from left player) ===
            if is_from_left and player.can_call_chi(discarded_34, is_from_left=True):
                result.update({
                    "called": True,
                    "call_type": "chi",
                    "caller_idx": player_idx
                })
                self._execute_chi(player, discarded_tile, discarder_idx)
                return result
        
        return result
    
    def _validate_yaku_for_ron(self, player: MahjongPlayer, winning_tile: int) -> bool:
        """
        Validate that a winning hand has at least one yaku.
        Uses mahjong-python library for calculation.
        """
        tiles_34 = player.hand.to_34()
        melds_34 = [m.tile_34_types for m in player.melds]
        
        # Build config for hand calculator
        config = {
            "tiles": tiles_34,
            "melds": melds_34,
            "dora_indicators": [t // 4 for t in self.dora_indicators],
            "is_riichi": player.is_riichi,
            "is_tsumo": False,
            "is_ippatsu": player.riichi_ippatsu,
            "is_haitei": len(self.wall) == 0,  # Last tile of wall
            "is_houtei": False,  # Last discard of hand
            "is_rinshan": False,  # Kan replacement draw
            "is_chankan": False,  # Winning on kan discard
            "seat_wind": player.seat_wind,
            "prevalent_wind": self.current_ba,  # 0=East, 1=South
            "honba": self.honba_count
        }
        
        try:
            result = self.hand_calculator.estimate_hand_value(**config)
            return result["han"] > 0  # Must have at least 1 han
        except Exception:
            # Fallback: basic yaku check if library fails
            return self._basic_yaku_check(player, winning_tile)
    
    def _basic_yaku_check(self, player: MahjongPlayer, winning_tile: int) -> bool:
        """Simple yaku validation fallback."""
        # Common yaku that don't require complex pattern matching
        if player.is_riichi:
            return True
        if not player.is_menzen and any(m.meld_type != MahjongMeld.MELD_KAN_CLOSED for m in player.melds):
            # Check for simple yakuhai (seat/prevalent wind, dragons)
            for meld in player.melds:
                for tile_type in meld.tile_34_types:
                    if tile_type in [player.seat_wind + 27, self.current_ba + 27, config.HAKU, config.HATSU, config.CHUN]:
                        return True
        return False
    
    def _execute_pon(self, player: MahjongPlayer, discarded_tile: int, from_player: int) -> None:
        """Execute Pon call logic."""
        tile_34 = discarded_tile // 4
        # Find two matching tiles in hand
        matching = [t for t in player.hand.ids if t // 4 == tile_34][:2]
        
        meld = MahjongMeld(
            meld_type=MahjongMeld.MELD_PON,
            tiles=matching + [discarded_tile],
            from_player=from_player,
            called_tile=discarded_tile
        )
        player.add_exposed_meld(meld)
        
        # Remove consumed tiles from hand
        for t in matching:
            player.hand.discard(t)
        
        self._log_state(
            GamePhase.CALL_RESOLUTION.value,
            "PON",
            tile=self._tile_id_to_mspzd(discarded_tile),
            metadata={"caller": player.name, "from": self.players[from_player].name}
        )
    
    def _execute_chi(self, player: MahjongPlayer, discarded_tile: int, from_player: int) -> None:
        """Execute Chi call logic (only from left player)."""
        tile_34 = discarded_tile // 4
        hand_34 = player.hand.to_34()
        
        # Find valid sequence
        suit_start = (tile_34 // 9) * 9
        rank = tile_34 % 9
        
        sequences = [
            (tile_34 - 2, tile_34 - 1) if rank >= 2 else None,
            (tile_34 - 1, tile_34 + 1) if 1 <= rank <= 7 else None,
            (tile_34 + 1, tile_34 + 2) if rank <= 6 else None,
        ]
        
        for seq in sequences:
            if seq and hand_34[seq[0]] > 0 and hand_34[seq[1]] > 0:
                # Found valid sequence
                tile1 = next(t for t in player.hand.ids if t // 4 == seq[0])
                tile2 = next(t for t in player.hand.ids if t // 4 == seq[1])
                
                meld = MahjongMeld(
                    meld_type=MahjongMeld.MELD_CHI,
                    tiles=[tile1, tile2, discarded_tile],
                    from_player=from_player,
                    called_tile=discarded_tile
                )
                player.add_exposed_meld(meld)
                
                # Remove consumed tiles
                player.hand.discard(tile1)
                player.hand.discard(tile2)
                break
        
        self._log_state(
            GamePhase.CALL_RESOLUTION.value,
            "CHI",
            tile=self._tile_id_to_mspzd(discarded_tile),
            metadata={"caller": player.name, "from": self.players[from_player].name}
        )
    
    def _execute_open_kan(self, player: MahjongPlayer, discarded_tile: int, from_player: int) -> None:
        """Execute Daiminkan (open kan) call logic."""
        tile_34 = discarded_tile // 4
        matching = [t for t in player.hand.ids if t // 4 == tile_34][:3]
        
        meld = MahjongMeld(
            meld_type=MahjongMeld.MELD_KAN_OPEN,
            tiles=matching + [discarded_tile],
            from_player=from_player,
            called_tile=discarded_tile
        )
        player.add_exposed_meld(meld)
        
        # Remove consumed tiles
        for t in matching:
            player.hand.discard(t)
        
        self._log_state(
            GamePhase.CALL_RESOLUTION.value,
            "KAN_OPEN",
            tile=self._tile_id_to_mspzd(discarded_tile),
            metadata={"caller": player.name}
        )
    
    def _can_declare_ankan(self, player: MahjongPlayer, tile_34_type: int) -> bool:
        """Validate Ankan declaration conditions."""
        if not player.is_menzen:
            return False
        if player.hand.count_type(tile_34_type) != 4:
            return False
        # Additional rules: cannot ankan after riichi unless wait unchanged
        if player.is_riichi:
            # Simplified: allow only if ankan doesn't change waits
            return True  # Full implementation would check wait preservation
        return True
    
    def _execute_ankan(self, player: MahjongPlayer, tile_34_type: int) -> None:
        """Execute Ankan (closed kan) declaration."""
        # Get all four copies of the tile
        matching = [t for t in player.hand.ids if t // 4 == tile_34_type]
        
        meld = MahjongMeld(
            meld_type=MahjongMeld.MELD_KAN_CLOSED,
            tiles=matching,
            from_player=player.player_id,
            called_tile=matching[0]
        )
        player.add_exposed_meld(meld)
        
        # Remove from hand (already exposed)
        for t in matching:
            player.hand.discard(t)
        
        # Draw replacement from dead wall
        self._handle_kan_replacement_draw(player)
        
        self._log_state(
            GamePhase.DRAW.value,
            "ANKAN",
            tile=self._tile_34_to_mspzd(tile_34_type),
            metadata={"player": player.name}
        )
    
    def _handle_kan_replacement_draw(self, player: MahjongPlayer) -> None:
        """Handle rinshan draw after kan declaration."""
        if not self.dead_wall:
            return  # Should not happen in valid game
        
        # Draw replacement tile
        replacement = self.dead_wall.pop()
        player.hand.draw(replacement)
        
        # Reveal new dora indicator
        if self.dead_wall:
            new_indicator = self.dead_wall.pop()
            self.dora_indicators.append(new_indicator)
            # Replenish dead wall from live wall
            if self.wall:
                self.dead_wall.insert(0, self.wall.pop())
        
        self._log_state(
            GamePhase.DRAW.value,
            "RINSHAN_DRAW",
            tile=self._tile_id_to_mspzd(replacement),
            metadata={
                "player": player.name,
                "new_dora": self._tile_id_to_mspzd(new_indicator) if self.dead_wall else None
            }
        )
    
    def _execute_caller_discard(self) -> None:
        """Handle discard after Pon/Chi call (no draw phase)."""
        player = self.players[self.current_turn]
        
        # Riichi players must auto-discard
        if player.is_riichi and len(player.hand) == 14:
            discard = player.hand.ids[-1]  # Last drawn tile
        else:
            discard = player.decide_discard()
        
        if not player.discard_tile(discard):
            raise RuntimeError("Failed to discard after call")
        
        self._log_state(
            GamePhase.DISCARD.value,
            "DISCARD_AFTER_CALL",
            tile=self._tile_id_to_mspzd(discard),
            metadata={"player": player.name}
        )
        
        # Check calls on this new discard
        call_result = self._resolve_calls(discard, self.current_turn)
        if call_result["called"]:
            self.current_turn = call_result["caller_idx"]
            if call_result["call_type"] == "kan":
                self._handle_kan_replacement_draw(self.players[self.current_turn])
            self._execute_caller_discard()  # Recursive for multiple calls
        else:
            # Normal turn rotation after call sequence
            self.current_turn = (self.current_turn + 1) % 4
    
    # =========================================================================
    # WIN CONDITION CHECKS
    # =========================================================================
    
    def _check_tsumo_win(self, player: MahjongPlayer) -> bool:
        """Check if player can win by self-draw."""
        if not player.is_tenpai():
            return False
        
        # Get last drawn tile (most recently added to hand)
        last_tile = player.hand.ids[-1]
        tile_34 = last_tile // 4
        
        if not player.can_win_on_tile(tile_34, is_tsumo=True):
            return False
        
        # Validate yaku for tsumo
        if not self._validate_yaku_for_tsumo(player):
            return False
        
        # Win confirmed
        self.kyoku_winner = player.player_id
        self.kyoku_win_type = "tsumo"
        
        self._log_state(
            GamePhase.DRAW.value,
            "TSUMO_WIN",
            tile=self._tile_id_to_mspzd(last_tile),
            metadata={"player": player.name}
        )
        return True
    
    def _validate_yaku_for_tsumo(self, player: MahjongPlayer) -> bool:
        """Validate yaku for tsumo win."""
        tiles_34 = player.hand.to_34()
        melds_34 = [m.tile_34_types for m in player.melds]
        
        config = {
            "tiles": tiles_34,
            "melds": melds_34,
            "dora_indicators": [t // 4 for t in self.dora_indicators],
            "is_riichi": player.is_riichi,
            "is_tsumo": True,
            "is_ippatsu": player.riichi_ippatsu,
            "is_haitei": len(self.wall) == 0,
            "seat_wind": player.seat_wind,
            "prevalent_wind": self.current_ba,
            "honba": self.honba_count
        }
        
        try:
            result = self.hand_calculator.estimate_hand_value(**config)
            return result["han"] > 0
        except Exception:
            return self._basic_yaku_check(player, player.hand.ids[-1])
    
    # =========================================================================
    # KYOKU SETTLEMENT
    # =========================================================================
    
    def _settle_agari(self) -> None:
        """Calculate and distribute points for a winning hand."""
        winner = self.players[self.kyoku_winner]
        tiles_34 = winner.hand.to_34()
        melds_34 = [m.tile_34_types for m in winner.melds]
        
        config = {
            "tiles": tiles_34,
            "melds": melds_34,
            "dora_indicators": [t // 4 for t in self.dora_indicators],
            "is_riichi": winner.is_riichi,
            "is_tsumo": (self.kyoku_win_type == "tsumo"),
            "is_ippatsu": winner.riichi_ippatsu,
            "seat_wind": winner.seat_wind,
            "prevalent_wind": self.current_ba,
            "honba": self.honba_count
        }
        
        try:
            result = self.hand_calculator.estimate_hand_value(**config)
            points = result["costs"]
            
            if self.kyoku_win_type == "ron":
                # Ron: discarder pays all
                discarder = self.players[self.last_discarder]
                payment = points["ron"]
                winner.add_score(payment)
                discarder.add_score(-payment)
            else:
                # Tsumo: others pay share (dealer pays double)
                for i, player in enumerate(self.players):
                    if i == self.kyoku_winner:
                        continue
                    payment = points["tsumo"][1] if player.is_dealer else points["tsumo"][0]
                    winner.add_score(payment)
                    player.add_score(-payment)
            
            # Award riichi pot if applicable
            if self.riichi_pot > 0:
                winner.add_score(self.riichi_pot)
                self.riichi_pot = 0
            
            # Mark dealer tenpai status for renchan logic
            winner.is_tenpai_last = True
            
        except Exception as e:
            # Fallback scoring if library fails
            self._fallback_scoring(winner)
    
    def _settle_ryukyoku(self) -> None:
        """Handle exhaustive draw settlement."""
        self.kyoku_ryukyoku = True
        
        # Check tenpai for all players
        tenpai_players = [p for p in self.players if p.is_tenpai()]
        noten_players = [p for p in self.players if not p.is_tenpai()]
        
        # Store tenpai status for renchan logic
        for player in self.players:
            player.is_tenpai_last = player.is_tenpai()
        
        # Noten bappu: 3000 points transferred from noten to tenpai players
        if tenpai_players and noten_players:
            points_per_tenpai = 3000 // len(tenpai_players)
            for tenpai_p in tenpai_players:
                tenpai_p.add_score(points_per_tenpai)
            for noten_p in noten_players:
                noten_p.add_score(-3000 // len(noten_players))
        
        self._log_state(
            GamePhase.SETTLEMENT.value,
            "RYUKYOKU",
            metadata={
                "tenpai": [p.name for p in tenpai_players],
                "noten": [p.name for p in noten_players]
            }
        )
    
    def _fallback_scoring(self, winner: MahjongPlayer) -> None:
        """Simple fallback scoring if hand calculator fails."""
        base_points = 1000
        if winner.is_dealer:
            base_points *= 2
        if self.kyoku_win_type == "ron":
            self.players[self.last_discarder].add_score(-base_points)
            winner.add_score(base_points)
        else:
            for player in self.players:
                if player != winner:
                    payment = base_points // 2 if player.is_dealer else base_points // 3
                    player.add_score(-payment)
                    winner.add_score(payment)
    
    # =========================================================================
    # FINAL RESULTS & LOGGING
    # =========================================================================
    
    def _calculate_final_results(self) -> Dict[str, Any]:
        """Calculate final rankings with Uma."""
        # Apply Uma: +15/-5/+5/-15 for 1st/2nd/3rd/4th
        uma = [15, 5, -5, -15]
        
        # Sort players by score descending
        ranked = sorted(
            enumerate(self.players),
            key=lambda x: x[1].score,
            reverse=True
        )
        
        results = []
        for rank, (idx, player) in enumerate(ranked):
            final_score = player.score + uma[rank] * 1000  # Uma in thousands
            results.append({
                "rank": rank + 1,
                "player": player.name,
                "final_score": final_score,
                "starting_score": self.starting_score
            })

        return {
            "results": results,
            "ba_played": self.current_ba,
            "total_kyoku": sum(1 for _ in self.game_log if "KYOKU" in _.phase),
            "riichi_pot_remaining": self.riichi_pot
        }
    
    def _log_state(
        self,
        phase: str,
        action: str,
        tile: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record game state snapshot for visualization/replay.
        
        Args:
            phase: Current game phase (from GamePhase enum)
            action: Specific action taken
            tile: Tile involved in MSPZD notation
            metadata: Additional context dictionary
        """
        if self.log_level == "minimal":
            return
        
        # === FIX: Convert phase to string name ===
        # If phase is a GamePhase enum, get its name
        if isinstance(phase, GamePhase):
            phase_str = phase.name  # e.g., "SETUP", "DRAW", "DISCARD"
        else:
            phase_str = str(phase)  # Already a string
        # === END FIX ===
        
        # Prepare player state snapshots
        player_states = []
        for player in self.players:
            state = {
                "id": player.player_id,
                "name": player.name,
                "score": player.score,
                "seat_wind": player.seat_wind,
                "is_dealer": player.is_dealer,
                "is_riichi": player.is_riichi,
                "furiten": player.furiten,
                "hand_size": len(player.hand) if player.hand else 0,
                "meld_count": len(player.melds),
                "discard_count": len(player.discards)
            }
            
            # Include full hand only in "full" log level
            if self.log_level == "full" and player.hand:
                state["hand_tiles"] = [
                    self._tile_id_to_mspzd(t) for t in player.hand.ids
                ]
                state["discards"] = [
                    self._tile_id_to_mspzd(t) for t in player.discards
                ]
            
            player_states.append(state)
        
        # Create log entry with phase_str instead of phase
        entry = GameLogEntry(
            phase=phase_str,  # ✅ Store as string
            ba=self.BA_NAMES[self.current_ba] if self.current_ba < len(self.BA_NAMES) else "Complete",
            kyoku=self.current_kyoku,
            honba=self.honba_count,
            turn_player=self.current_turn,
            action=action,
            tile=tile,
            dora_indicators=self._tile_ids_to_mspzd(self.dora_indicators),
            players=player_states,
            metadata=metadata or {}
        )
        
        self.game_log.append(entry)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _tile_id_to_mspzd(self, tile_id: int) -> str:
        """Convert single 136-format tile ID to MSPZD notation."""
        return MahjongConverter.to_str([tile_id])
    
    def _tile_34_to_mspzd(self, tile_34: int) -> str:
        """Convert 34-format tile type to MSPZD notation."""
        # Create a dummy 136-ID for conversion
        dummy_id = tile_34 * 4
        return self._tile_id_to_mspzd(dummy_id)
    
    def _tile_ids_to_mspzd(self, tile_ids: List[int]) -> List[str]:
        """Convert list of tile IDs to MSPZD notation list."""
        return [self._tile_id_to_mspzd(t) for t in tile_ids]
    
    def get_game_log(self) -> List[Dict[str, Any]]:
        """Return game log as list of dictionaries for visualization."""
        return [entry.to_dict() for entry in self.game_log]
    
    def export_log_json(self, filepath: str) -> None:
        """Export game log to JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_game_log(), f, indent=2, ensure_ascii=False)