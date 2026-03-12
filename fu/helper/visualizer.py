from helper import config
from helper.game import GamePhase, GameLogEntry
from helper.tile_util import MahjongMeld
from helper.utility import MahjongConverter, Hand136, MSPZD

from typing import List, Optional, Tuple, Union, Set, Dict, Any
from enum import Enum, auto

import os
import sys
import json

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_DARK = "\033[48;5;236m"
    BG_BLUE = "\033[48;5;25m"
    BG_GREEN = "\033[48;5;22m"
    BG_RED = "\033[48;5;52m"
    
    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output."""
        for attr in dir(cls):
            if attr.isupper():
                setattr(cls, attr, "")

# OPTIONAL NOTEBOOK MODE
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    NOTEBOOK_AVAILABLE = True
except ImportError:
    NOTEBOOK_AVAILABLE = False

# OPTIONAL KEYBOARD INPUT
try:
    import keyboard  # Optional: for non-blocking key input
    USE_KEYBOARD_LIB = True
except ImportError:
    USE_KEYBOARD_LIB = False
    print(f"{Colors.YELLOW}Note: Install 'keyboard' library for better controls{Colors.RESET}")
    print(f"{Colors.DIM}Using fallback input() method{Colors.RESET}\n")


# =============================================================================
# VISUALIZATION MODES
# =============================================================================

class VizMode(Enum):
    """Visualization output mode."""
    TERMINAL = auto()
    NOTEBOOK = auto()
    HTML_FILE = auto()


# =============================================================================
# MAHJONG REPLAY VISUALIZER
# =============================================================================

class MahjongReplay:
    """Interactive visualization for MahjongGame replay logs."""
    
    # === NEW KEYBINDS ===
    KEY_NEXT_STEP = ['n', 'N', 'l', 'L', 'right', 'RIGHT']
    KEY_PREV_STEP = ['p', 'P', 'h', 'H', 'left', 'LEFT']
    KEY_NEXT_KYOKU = ['k', 'K', 'down', 'DOWN']
    KEY_PREV_KYOKU = ['j', 'J', 'up', 'UP']
    KEY_QUIT = ['q', 'Q', 'x', 'X']
    
    def __init__(
        self,
        game_log: List[Dict[str, Any]],
        mode: VizMode = VizMode.TERMINAL,
        show_hand_details: bool = True,
        show_meld_details: bool = True,
        tile_image_base_url: str = ""
    ):
        self.game_log = game_log
        self.current_idx = 0
        self.mode = mode
        self.show_hand_details = show_hand_details
        self.show_meld_details = show_meld_details
        self.tile_image_base_url = tile_image_base_url
        
        if mode == VizMode.TERMINAL and not sys.stdout.isatty():
            Colors.disable()
        
        self.kyoku_indices = self._find_kyoku_boundaries()
        self.ba_indices = self._find_ba_boundaries()
        
        # Cache for shanten/wait calculations
        self._enable_shanten_calc = False
        try:
            from mahjong.shanten import Shanten
            from mahjong.agari import Agari
            self._shanten_calc = Shanten()
            self._agari_calc = Agari()
            self._enable_shanten_calc = True
        except ImportError:
            pass
    
    # =========================================================================
    # NAVIGATION & BOUNDARIES
    # =========================================================================
    
    def _find_kyoku_boundaries(self) -> List[int]:
        """Find log indices where each kyoku starts (SETUP phase)."""
        indices = []
        for i, entry in enumerate(self.game_log):
            entry: GameLogEntry
            phase = entry.phase if entry.phase else ''
            action = entry.action if entry.action else ''
            if phase == 'SETUP' and 'KYOKU' in action:
                indices.append(i)
        return indices
    
    def _find_ba_boundaries(self) -> List[int]:
        """Find log indices where each ba (round) starts."""
        indices = [0]
        current_ba = None
        for i, entry in enumerate(self.game_log):
            entry: GameLogEntry
            ba = entry.ba if entry.ba else ''
            if ba != current_ba:
                indices.append(i)
                current_ba = ba
        return indices
    
    def _jump_to_kyoku(self, kyoku_offset: int) -> None:
        """Jump forward/backward by kyoku count."""
        if kyoku_offset > 0:
            for idx in self.kyoku_indices:
                if idx > self.current_idx:
                    kyoku_offset -= 1
                    if kyoku_offset == 0:
                        self.current_idx = idx
                        return
            self.current_idx = len(self.game_log) - 1
        else:
            for idx in reversed(self.kyoku_indices):
                if idx < self.current_idx:
                    kyoku_offset += 1
                    if kyoku_offset == 0:
                        self.current_idx = idx
                        return
            self.current_idx = 0
    
    def _jump_to_ba(self, ba_offset: int) -> None:
        """Jump forward/backward by ba (round) count."""
        if ba_offset > 0:
            for idx in self.ba_indices:
                if idx > self.current_idx:
                    ba_offset -= 1
                    if ba_offset == 0:
                        self.current_idx = idx
                        return
            self.current_idx = len(self.game_log) - 1
        else:
            for idx in reversed(self.ba_indices):
                if idx < self.current_idx:
                    ba_offset += 1
                    if ba_offset == 0:
                        self.current_idx = idx
                        return
            self.current_idx = 0
    
    # =========================================================================
    # STATE CALCULATION (Shanten, Waits)
    # =========================================================================
    
    def _calculate_player_state(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate shanten and wait tiles for a player."""
        result = {
            'shanten': '?',
            'waits': set(),
            'is_tenpai': False,
            'furiten': player_data.get('furiten', False)
        }
        
        if not self._enable_shanten_calc or not self.show_hand_details:
            return result
        
        try:
            hand_tiles = player_data.get('hand_tiles', [])
            if not hand_tiles:
                return result
            
            hand_str = ''.join(hand_tiles)
            tiles_136 = MahjongConverter.to_136(hand_str)
            tiles_34 = MahjongConverter.to_34_array(tiles_136)
            
            melds_34 = []
            
            shanten = self._shanten_calc.calculate_shanten(tiles_34, melds_34)
            result['shanten'] = shanten
            result['is_tenpai'] = (shanten == 0)
            
            if shanten == 0:
                waits = set()
                for tile_type in range(34):
                    if tiles_34[tile_type] >= 4:
                        continue
                    test_tiles = tiles_34.copy()
                    test_tiles[tile_type] += 1
                    if self._agari_calc.is_agari(test_tiles, melds_34):
                        waits.add(tile_type)
                result['waits'] = waits
            
        except Exception:
            pass
        
        return result
    
    # =========================================================================
    # TILE FORMATTING HELPERS
    # =========================================================================
    
    def _format_tile(self, tile_str: str, highlight: bool = False) -> str:
        """Format a single tile string for terminal display."""
        if highlight:
            return f"{Colors.BG_YELLOW}{Colors.BOLD}{Colors.BLACK} {tile_str} {Colors.RESET}"
        else:
            return f"{Colors.BG_DARK}{Colors.WHITE} {tile_str} {Colors.RESET}"
    
    def _format_drawn_tile(self, tile_str: str) -> str:
        """Format drawn tile with visual separation from hand."""
        # Green background + border effect for drawn tile
        return f"{Colors.BG_GREEN}{Colors.BOLD}{Colors.WHITE} [{tile_str}] {Colors.RESET}"
    
    def _format_meld_type(self, meld_type: str) -> str:
        """Format meld type with color coding."""
        type_colors = {
            'PON': Colors.MAGENTA,
            'CHI': Colors.CYAN,
            'KAN_OPEN': Colors.YELLOW,
            'KAN_CLOSED': Colors.BLUE,
            'PON*': Colors.MAGENTA,  # Daiminkan
        }
        color = type_colors.get(meld_type, Colors.WHITE)
        return f"{color}{meld_type}{Colors.RESET}"
    
    def _tile_type_to_mspzd(self, tile_type: int) -> str:
        """Convert 34-format tile type to MSPZD string."""
        if tile_type < 9:
            return f"{tile_type + 1}m"
        elif tile_type < 18:
            return f"{tile_type - 8}p"
        elif tile_type < 27:
            return f"{tile_type - 17}s"
        else:
            honor_names = ['E', 'S', 'W', 'N', 'Haku', 'Hatsu', 'Chun']
            idx = tile_type - 27
            return honor_names[idx] if 0 <= idx < len(honor_names) else '?'
    
    # =========================================================================
    # RENDERING - TERMINAL MODE
    # =========================================================================
    
    def _render_terminal(self) -> None:
        """Render current state to terminal with ANSI colors."""
        if not self.game_log:
            print(f"{Colors.RED}Error: Game log is empty{Colors.RESET}")
            return
        
        entry: GameLogEntry = self.game_log[self.current_idx]
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # === HEADER WITH WALL COUNT ===
        ba = entry.ba if entry.ba else 'Unknown'
        kyoku = entry.kyoku if entry.kyoku else 0
        honba = entry.honba if entry.honba else 0
        turn_player = entry.turn_player if entry.turn_player else 0
        action = entry.action if entry.action else 'N/A'
        tile = entry.tile if entry.tile else ''
        dora = entry.dora_indicators if entry.dora_indicators else []
        
        # === IMPROVEMENT 3: Wall tiles remaining ===
        metadata = entry.metadata
        tiles_remaining = metadata.get('tiles_remaining', '?')
        dead_wall = metadata.get('dead_wall_remaining', '?')
        
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.WHITE}  {ba} {kyoku} Kyoku{Colors.RESET}"
            f"{Colors.DIM} | Honba: {honba}{Colors.RESET}"
            f"{Colors.DIM} | Turn: P{turn_player}{Colors.RESET}"
            f"{Colors.DIM} | Wall: {Colors.GREEN}{tiles_remaining}{Colors.RESET}"
            f"{Colors.DIM} | Dead: {Colors.YELLOW}{dead_wall}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}  Action: {action}{Colors.RESET}"
            f"{Colors.WHITE} {tile}{Colors.RESET}"
            f"{Colors.DIM} | Step: {self.current_idx}/{len(self.game_log)-1}{Colors.RESET}")
        
        # Dora indicators
        if dora:
            dora_str = '  '.join(dora)
            print(f"{Colors.DIM}  Dora: {Colors.YELLOW}{dora_str}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        
        # === PLAYERS ===
        players = entry.players if entry.players else []
        for i, p in enumerate(players):
            is_turn = (i == turn_player)
            is_dealer = p.get('is_dealer', False)
            is_riichi = p.get('is_riichi', False)
            score = p.get('score', 0)
            name = p.get('name', f'P{i}')
            seat_wind = p.get('seat_wind', i)
            wind_names = ['East', 'South', 'West', 'North']
            wind = wind_names[seat_wind] if 0 <= seat_wind < 4 else '?'
            
            # Player header
            bg = Colors.BG_BLUE if is_turn else Colors.BG_DARK
            riichi_mark = f"{Colors.RED}[RIICHI]{Colors.RESET} " if is_riichi else ""
            dealer_mark = f"{Colors.YELLOW}[DEALER]{Colors.RESET} " if is_dealer else ""
            
            print(f"{bg}{Colors.WHITE}  [{wind}] {riichi_mark}{dealer_mark}{name}"
                  f"{Colors.DIM} | Score: {score:,}{Colors.RESET}")
            
            # === IMPROVEMENT 1 & 2: Hand tiles + Drawn tile separation + Meld details ===
            hand_tiles = p.get('hand_tiles', [])
            hand_size = p.get('hand_size', len(hand_tiles))
            melds = p.get('melds', [])
            meld_count = p.get('meld_count', len(melds) if melds else 0)
            
            # Display hand tiles
            if self.show_hand_details and hand_tiles:
                # Check if current action is DRAW for this player
                is_draw_action = (action == 'DRAW' and is_turn)
                drawn_tile = entry.tile if entry.tile else '' if is_draw_action else None
                
                # Split hand: all tiles except last are "hand", last might be "drawn"
                if is_draw_action and drawn_tile and len(hand_tiles) > 0:
                    # Show all tiles as hand, then drawn tile separately
                    hand_display = hand_tiles  # All tiles in hand
                    print(f"{Colors.DIM}    Hand ({len(hand_display)}): ", end='')
                    for t in hand_display:
                        print(self._format_tile(t), end=' ')
                    # Drawn tile shown separately with gap
                    print(f"{Colors.DIM}  →  Drawn: {self._format_drawn_tile(drawn_tile)}{Colors.RESET}")
                else:
                    tiles_str = ' '.join([self._format_tile(t) for t in hand_tiles])
                    print(f"{Colors.DIM}    Hand ({hand_size}): {tiles_str}{Colors.RESET}")
            else:
                print(f"{Colors.DIM}    Hand: {hand_size} tiles{Colors.RESET}")
            
            # === IMPROVEMENT 2: Display meld details ===
            if self.show_meld_details and melds and len(melds) > 0:
                print(f"{Colors.DIM}    Melds:{Colors.RESET}")
                for meld in melds:
                    meld_type = meld.get('type', 'UNKNOWN')
                    meld_tiles = meld.get('tiles', [])
                    from_player = meld.get('from_player', '?')
                    
                    # Format meld type with color
                    type_display = self._format_meld_type(meld_type.upper())
                    
                    # Format tiles
                    tile_display = ' '.join([self._format_tile(t) for t in meld_tiles])
                    
                    # Show who discarded (for Pon/Chi/Kan)
                    if from_player != '?':
                        from_display = f"{Colors.DIM}(from P{from_player}){Colors.RESET}"
                    else:
                        from_display = ""
                    
                    print(f"{Colors.DIM}      [{type_display}] {tile_display} {from_display}{Colors.RESET}")
            elif meld_count > 0:
                print(f"{Colors.DIM}    Melds: {Colors.MAGENTA}{meld_count} exposed{Colors.RESET}")
            
            # Shanten and waits
            player_state = self._calculate_player_state(p)
            shanten = player_state['shanten']
            is_tenpai = player_state['is_tenpai']
            furiten = player_state['furiten']
            
            if shanten != '?':
                shanten_color = Colors.GREEN if shanten == 0 else Colors.WHITE
                tenpai_mark = f"{Colors.RED} [TENPAI]{Colors.RESET}" if is_tenpai else ""
                furiten_mark = f"{Colors.RED} [FURITEN]{Colors.RESET}" if furiten else ""
                print(f"{Colors.DIM}    Shanten: {shanten_color}{shanten}{Colors.RESET}{tenpai_mark}{furiten_mark}")
                
                if is_tenpai and player_state['waits']:
                    wait_tiles = []
                    for wt in list(player_state['waits'])[:7]:  # Show first 7 waits
                        wait_tiles.append(self._tile_type_to_mspzd(wt))
                    if len(player_state['waits']) > 7:
                        wait_tiles.append(f"...+{len(player_state['waits'])-7}")
                    print(f"{Colors.DIM}    Waits: {Colors.CYAN}{', '.join(wait_tiles)}{Colors.RESET}")
            
            # Discards (last few)
            discards = p.get('discards', [])
            if discards:
                recent = discards[-8:] if len(discards) > 8 else discards
                discard_str = ' '.join([self._format_tile(d) for d in recent])
                if len(discards) > 8:
                    prefix = f"... ({len(discards)-8}) "
                else:
                    prefix = ""
                print(f"{Colors.DIM}    Discards: {prefix}{discard_str}{Colors.RESET}")
            
            print()  # Empty line between players
        
        # === NAVIGATION HELP ===
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.DIM}  Navigation: "
              f"{Colors.WHITE}[→/N] Next Step{Colors.DIM} | "
              f"{Colors.WHITE}[←/P] Prev Step{Colors.DIM} | "
              f"{Colors.WHITE}[↓/K] Next Kyoku{Colors.DIM} | "
              f"{Colors.WHITE}[↑/J] Prev Kyoku{Colors.DIM} | "
              f"{Colors.WHITE}[Q] Quit{Colors.DIM}{Colors.RESET}")
        print(f"{Colors.DIM}  Tip: Hold arrow keys for continuous navigation{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    
    # =========================================================================
    # DISPLAY & NAVIGATION
    # =========================================================================
    
    def render(self) -> None:
        """Render current state based on mode."""
        if self.mode == VizMode.TERMINAL:
            self._render_terminal()
        elif self.mode == VizMode.NOTEBOOK:
            self._render_notebook()
    
    def show(self) -> None:
        """Start interactive visualization session."""
        if self.mode == VizMode.TERMINAL:
            self._show_terminal()
        elif self.mode == VizMode.NOTEBOOK:
            self._show_notebook()
    
    def _show_terminal(self) -> None:
        """Terminal-based interactive navigation with improved keybinds."""
        print(f"{Colors.GREEN}Starting Mahjong Replay Viewer{Colors.RESET}")
        print(f"{Colors.DIM}Total states: {len(self.game_log)}{Colors.RESET}\n")
        print(f"{Colors.DIM}Controls: "
              f"{Colors.WHITE}→/N{Colors.DIM}=next step | "
              f"{Colors.WHITE}←/P{Colors.DIM}=prev step | "
              f"{Colors.WHITE}↓/K{Colors.DIM}=next kyoku | "
              f"{Colors.WHITE}↑/J{Colors.DIM}=prev kyoku | "
              f"{Colors.WHITE}Q{Colors.DIM}=quit{Colors.RESET}\n")
        
        self.render()
        
        if USE_KEYBOARD_LIB:
            import time
            
            last_key_time = 0
            debounce_delay = 0.15
            
            while True:
                current_time = time.time()
                
                # Always check quit first
                if keyboard.is_pressed('q') or keyboard.is_pressed('x'):
                    print(f"\n{Colors.GREEN}Replay ended{Colors.RESET}")
                    break
                
                # Debounce check
                if current_time - last_key_time < debounce_delay:
                    time.sleep(0.02)
                    continue
                
                # === IMPROVEMENT 4: New keybinds ===
                # Next/Prev STEP (individual state changes)
                if (keyboard.is_pressed('n') or keyboard.is_pressed('l') or 
                    keyboard.is_pressed('right')):
                    self.current_idx = min(len(self.game_log) - 1, self.current_idx + 1)
                    self.render()
                    last_key_time = current_time
                    time.sleep(debounce_delay)
                    
                elif (keyboard.is_pressed('p') or keyboard.is_pressed('h') or 
                      keyboard.is_pressed('left')):
                    self.current_idx = max(0, self.current_idx - 1)
                    self.render()
                    last_key_time = current_time
                    time.sleep(debounce_delay)
                
                # Next/Prev KYOKU (jump to kyoku boundaries)
                elif (keyboard.is_pressed('k') or keyboard.is_pressed('down')):
                    self._jump_to_kyoku(1)
                    self.render()
                    last_key_time = current_time
                    time.sleep(debounce_delay)
                    
                elif (keyboard.is_pressed('j') or keyboard.is_pressed('up')):
                    self._jump_to_kyoku(-1)
                    self.render()
                    last_key_time = current_time
                    time.sleep(debounce_delay)
                
                else:
                    time.sleep(0.03)
        
        else:
            # Fallback input() method
            while True:
                try:
                    cmd = input(f"\n{Colors.CYAN}Command (n/p/k/j/q): {Colors.RESET}").strip().lower()
                    
                    if cmd in ['n', 'l', 'right', '']:
                        self.current_idx = min(len(self.game_log) - 1, self.current_idx + 1)
                    elif cmd in ['p', 'h', 'left']:
                        self.current_idx = max(0, self.current_idx - 1)
                    elif cmd in ['k', 'down']:
                        self._jump_to_kyoku(1)
                    elif cmd in ['j', 'up']:
                        self._jump_to_kyoku(-1)
                    elif cmd in ['q', 'x']:
                        print(f"\n{Colors.GREEN}Replay ended{Colors.RESET}")
                        break
                    else:
                        print(f"{Colors.YELLOW}Unknown command{Colors.RESET}")
                        continue
                    
                    self.render()
                    
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{Colors.GREEN}Replay ended{Colors.RESET}")
                    break
    
    def _render_notebook(self) -> None:
        """Notebook rendering (simplified for now)."""
        if not NOTEBOOK_AVAILABLE:
            print("ipywidgets not available. Use TERMINAL mode.")
            return
        
        # Similar to terminal but with HTML widgets
        self._render_terminal()  # Fallback for now
    
    def _show_notebook(self) -> None:
        """Notebook-based navigation."""
        if not NOTEBOOK_AVAILABLE:
            print("ipywidgets not available. Use TERMINAL mode.")
            return
        
        self.output = widgets.Output()
        
        btn_prev = widgets.Button(description="◀ Step", layout=widgets.Layout(width='80px'))
        btn_next = widgets.Button(description="Step ▶", layout=widgets.Layout(width='80px'))
        btn_prev_kyoku = widgets.Button(description="⏪ Kyoku", layout=widgets.Layout(width='80px'))
        btn_next_kyoku = widgets.Button(description="Kyoku ⏩", layout=widgets.Layout(width='80px'))
        btn_quit = widgets.Button(description="Quit", layout=widgets.Layout(width='60px'))
        
        def on_prev(_):
            self.current_idx = max(0, self.current_idx - 1)
            self.render()
        
        def on_next(_):
            self.current_idx = min(len(self.game_log) - 1, self.current_idx + 1)
            self.render()
        
        def on_prev_kyoku(_):
            self._jump_to_kyoku(-1)
            self.render()
        
        def on_next_kyoku(_):
            self._jump_to_kyoku(1)
            self.render()
        
        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        btn_prev_kyoku.on_click(on_prev_kyoku)
        btn_next_kyoku.on_click(on_next_kyoku)
        
        ui = widgets.VBox([
            widgets.HBox([btn_prev, btn_next, btn_prev_kyoku, btn_next_kyoku, btn_quit]),
            self.output
        ])
        
        self.render()
        display(ui)
    
    # =========================================================================
    # EXPORT & UTILITIES
    # =========================================================================
    
    def get_current_state(self) -> Dict[str, Any]:
        """Return current log entry."""
        return self.game_log[self.current_idx] if self.game_log else {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate game statistics from log."""
        stats = {
            'total_states': len(self.game_log),
            'total_kyoku': len(self.kyoku_indices),
            'total_ba': len(self.ba_indices),
            'riichi_declarations': 0,
            'ron_wins': 0,
            'tsumo_wins': 0,
            'ryukyoku': 0
        }
        
        for entry in self.game_log:
            entry: GameLogEntry
            action = entry.action if entry.action else ''
            if 'RIICHI' in action:
                stats['riichi_declarations'] += 1
            elif 'RON' in action:
                stats['ron_wins'] += 1
            elif 'TSUMO' in action:
                stats['tsumo_wins'] += 1
            elif 'RYUKYOKU' in action:
                stats['ryukyoku'] += 1
        
        return stats