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
    """
    Interactive visualization for MahjongGame replay logs.
    
    Supports terminal-based navigation with keyboard controls
    or notebook-based with ipywidgets buttons.
    """
    
    # Navigation key mappings for terminal mode
    KEY_NEXT_TURN = ['n', 'N', 'l', 'L', 'RIGHT']
    KEY_PREV_TURN = ['p', 'P', 'h', 'H', 'LEFT']
    KEY_NEXT_KYOKU = ['k', 'K', 'DOWN']
    KEY_PREV_KYOKU = ['j', 'J', 'UP']
    KEY_QUIT = ['q', 'Q', 'x', 'X']
    
    def __init__(
        self,
        game_log: List[Dict[str, Any]],
        mode: VizMode = VizMode.TERMINAL,
        show_hand_details: bool = True,
        tile_image_base_url: str = ""
    ):
        """
        Initialize replay visualizer.
        
        Args:
            game_log: List of log entry dictionaries from MahjongGame
            mode: Output mode (TERMINAL, NOTEBOOK, or HTML_FILE)
            show_hand_details: If True, show full hand tiles (else show counts)
            tile_image_base_url: Base URL for tile images (HTML mode)
        """
        self.game_log = game_log
        self.current_idx = 0
        self.mode = mode
        self.show_hand_details = show_hand_details
        self.tile_image_base_url = tile_image_base_url
        
        # Disable colors if not in terminal
        if mode == VizMode.TERMINAL and not sys.stdout.isatty():
            Colors.disable()
        
        # Pre-calculate kyoku boundaries for navigation
        self.kyoku_indices = self._find_kyoku_boundaries()
        self.ba_indices = self._find_ba_boundaries()
        
        # Cache for shanten/wait calculations (optional, requires mahjong lib)
        self._enable_shanten_calc = False
        try:
            from mahjong.shanten import Shanten
            from mahjong.agari import Agari
            self._shanten_calc = Shanten()
            self._agari_calc = Agari()
            self._enable_shanten_calc = True
        except ImportError:
            print(f"{Colors.YELLOW}Warning: mahjong library not found. Shanten/waits disabled.{Colors.RESET}")
    
    # =========================================================================
    # NAVIGATION & BOUNDARIES
    # =========================================================================
    
    def _find_kyoku_boundaries(self) -> List[int]:
        """Find log indices where each kyoku starts (SETUP phase)."""
        indices = []
        for i, entry in enumerate(self.game_log):
            entry: GameLogEntry
            phase = entry.phase
            action = entry.action
            if phase == 'SETUP' and 'KYOKU' in action:
                indices.append(i)
        return indices
    
    def _find_ba_boundaries(self) -> List[int]:
        """Find log indices where each ba (round) starts."""
        indices = [0]  # Game start
        current_ba = None
        for i, entry in enumerate(self.game_log):
            ba = entry.ba
            if ba != current_ba:
                indices.append(i)
                current_ba = ba
        return indices
    
    def _jump_to_kyoku(self, kyoku_offset: int) -> None:
        """Jump forward/backward by kyoku count."""
        if kyoku_offset > 0:
            # Find next kyoku start
            for idx in self.kyoku_indices:
                if idx > self.current_idx:
                    kyoku_offset -= 1
                    if kyoku_offset == 0:
                        self.current_idx = idx
                        return
            # If not found, go to end
            self.current_idx = len(self.game_log) - 1
        else:
            # Find previous kyoku start
            for idx in reversed(self.kyoku_indices):
                if idx < self.current_idx:
                    kyoku_offset += 1
                    if kyoku_offset == 0:
                        self.current_idx = idx
                        return
            # If not found, go to start
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
        """
        Calculate shanten and wait tiles for a player.
        Requires mahjong library integration.
        """
        result = {
            'shanten': '?',
            'waits': set(),
            'is_tenpai': False,
            'furiten': player_data.get('furiten', False)
        }
        
        if not self._enable_shanten_calc or not self.show_hand_details:
            return result
        
        try:
            # Get hand tiles from log
            hand_tiles = player_data.get('hand_tiles', [])
            if not hand_tiles:
                return result
            
            # Convert MSPZD strings to 34-array
            hand_str = ''.join(hand_tiles)
            tiles_136 = MahjongConverter.to_136(hand_str)
            tiles_34 = MahjongConverter.to_34_array(tiles_136)
            
            # Get melds from player data (if available)
            melds_34 = []  # Would need meld data in log for full accuracy
            
            # Calculate shanten
            shanten = self._shanten_calc.calculate_shanten(tiles_34, melds_34)
            result['shanten'] = shanten
            result['is_tenpai'] = (shanten == 0)
            
            # Calculate waits if tenpai
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
            
        except Exception as e:
            # Silently fail shanten calculation
            pass
        
        return result
    
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
        
        # === HEADER ===
        ba = entry.ba
        kyoku = entry.kyoku
        honba = entry.honba
        turn_player = entry.turn_player
        action = entry.action
        tile = entry.tile
        dora = entry.dora_indicators
        
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.WHITE}  {ba} {kyoku} Kyoku{Colors.RESET}"
              f"{Colors.DIM} | Honba: {honba}{Colors.RESET}"
              f"{Colors.DIM} | Turn: P{turn_player}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}  Action: {action}{Colors.RESET}"
              f"{Colors.WHITE} {tile}{Colors.RESET}"
              f"{Colors.DIM} | Step: {self.current_idx}/{len(self.game_log)-1}{Colors.RESET}")
        
        # Dora indicators
        if dora:
            dora_str = ' '.join(dora)
            print(f"{Colors.DIM}  Dora: {Colors.YELLOW}{dora_str}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        # === PLAYERS ===
        players = entry.players
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
            
            # Hand tiles
            hand_tiles = p.get('hand_tiles', [])
            hand_size = p.get('hand_size', len(hand_tiles))
            
            if self.show_hand_details and hand_tiles:
                # Group tiles by suit for display
                tiles_str = ' '.join(hand_tiles)
                print(f"{Colors.DIM}    Hand ({hand_size}): {Colors.WHITE}{tiles_str}{Colors.RESET}")
            else:
                print(f"{Colors.DIM}    Hand: {hand_size} tiles{Colors.RESET}")
            
            # Melds (if available in log)
            meld_count = p.get('meld_count', 0)
            if meld_count > 0:
                print(f"{Colors.DIM}    Melds: {Colors.MAGENTA}{meld_count} exposed{Colors.RESET}")
            
            # Shanten and waits (if calculated)
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
                    # Convert wait tile types to MSPZD
                    wait_tiles = []
                    for wt in player_state['waits']:
                        if wt < 9:
                            wait_tiles.append(f"{wt+1}m")
                        elif wt < 18:
                            wait_tiles.append(f"{wt-8}p")
                        elif wt < 27:
                            wait_tiles.append(f"{wt-17}s")
                        else:
                            honor_names = ['E', 'S', 'W', 'N', 'Haku', 'Hatsu', 'Chun']
                            wait_tiles.append(honor_names[wt-27] if wt-27 < len(honor_names) else '?')
                    print(f"{Colors.DIM}    Waits: {Colors.CYAN}{', '.join(wait_tiles)}{Colors.RESET}")
            
            # Discards (last few)
            discards = p.get('discards', [])
            if discards:
                recent = discards[-6:] if len(discards) > 6 else discards
                discard_str = ' '.join(recent)
                print(f"{Colors.DIM}    Discards: {Colors.DIM}{discard_str}{Colors.RESET}")
            
            print()  # Empty line between players
        
        # === NAVIGATION HELP ===
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.DIM}  Navigation: "
              f"{Colors.WHITE}[N]ext{Colors.DIM} | "
              f"{Colors.WHITE}[P]rev{Colors.DIM} | "
              f"{Colors.WHITE}[K]yoku+{Colors.DIM} | "
              f"{Colors.WHITE}[J]yoku-{Colors.DIM} | "
              f"{Colors.WHITE}[Q]uit{Colors.DIM}{Colors.RESET}")
        print(f"{Colors.DIM}  Notebook: Use show_notebook() instead{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    
    # =========================================================================
    # RENDERING - NOTEBOOK MODE (ipywidgets)
    # =========================================================================
    
    def _render_notebook(self) -> None:
        """Render current state using ipywidgets (Jupyter/VS Code)."""
        if not NOTEBOOK_AVAILABLE:
            print("ipywidgets not available. Use TERMINAL mode instead.")
            return
        
        if not self.game_log:
            with self.output:
                print("Error: Game log is empty.")
            return
        
        entry: GameLogEntry = self.game_log[self.current_idx]
        
        with self.output:
            clear_output(wait=True)
            
            # Header
            ba = entry.ba
            kyoku = entry.kyoku
            honba = entry.honba
            action = entry.action
            tile = entry.tile
            dora = entry.dora_indicators
            
            dora_html = ''.join([
                f'<span style="background:#333; color:#ffd700; padding:2px 6px; '
                f'border-radius:3px; margin-right:4px;">{d}</span>'
                for d in dora
            ]) if dora else '<span style="color:#555">None</span>'
            
            html = f"""
            <div style="background:#1e1e1e; color:#d4d4d4; padding:15px; 
                        border-radius:8px; font-family:Consolas,monospace; 
                        border:1px solid #333; max-width:900px;">
                <div style="border-bottom:2px solid #4ec9b0; padding-bottom:8px; 
                            margin-bottom:15px; display:flex; justify-content:space-between;">
                    <div>
                        <span style="font-size:1.2em; font-weight:bold; color:#4ec9b0;">
                            {ba} {kyoku} Kyoku
                        </span>
                        <span style="margin-left:15px; color:#858585;">Honba: {honba}</span>
                    </div>
                    <div>
                        <span style="color:#ffd700; margin-right:5px;">DORA:</span>
                        {dora_html}
                    </div>
                </div>
                <div style="color:#ce9178; margin-bottom:10px;">
                    Action: <b>{action}</b> {tile} | Step: {self.current_idx}/{len(self.game_log)-1}
                </div>
            """
            
            # Players
            players = entry.players
            turn_player = entry.turn_player if entry.turn_player else 0
            
            for i, p in enumerate(players):
                is_turn = (i == turn_player)
                is_dealer = p.get('is_dealer', False)
                is_riichi = p.get('is_riichi', False)
                score = p.get('score', 0)
                name = p.get('name', f'P{i}')
                seat_wind = p.get('seat_wind', i)
                wind_names = ['🀀', '🀁', '🀂', '🀃']
                wind = wind_names[seat_wind] if 0 <= seat_wind < 4 else '?'
                
                bg = "#252526" if is_turn else "#1a1a1a"
                border = "#569cd6" if is_turn else "#333"
                riichi_badge = '<span style="color:#f44747; font-size:0.7em; ' \
                              'border:1px solid #f44747; padding:1px 4px; ' \
                              'border-radius:3px; margin-left:8px;">RIICHI</span>' if is_riichi else ''
                dealer_badge = '<span style="color:#ffd700; font-size:0.7em; ' \
                              'border:1px solid #ffd700; padding:1px 4px; ' \
                              'border-radius:3px; margin-left:8px;">DEALER</span>' if is_dealer else ''
                
                # Hand display
                hand_tiles = p.get('hand_tiles', [])
                hand_size = p.get('hand_size', len(hand_tiles))
                
                if self.show_hand_details and hand_tiles:
                    tile_html = ''.join([
                        f'<span style="display:inline-block; background:#333; '
                        f'color:#fff; padding:2px 5px; margin:1px; border-radius:3px; '
                        f'font-size:0.9em;">{t}</span>'
                        for t in hand_tiles
                    ])
                else:
                    tile_html = f'<span style="color:#666">{hand_size} tiles</span>'
                
                # Shanten/waits
                player_state = self._calculate_player_state(p)
                shanten = player_state['shanten']
                is_tenpai = player_state['is_tenpai']
                furiten = player_state['furiten']
                
                state_html = ""
                if shanten != '?':
                    shanten_color = "#4ec9b0" if shanten == 0 else "#d4d4d4"
                    tenpai_mark = '<span style="color:#f44747; margin-left:5px;">TENPAI</span>' if is_tenpai else ''
                    furiten_mark = '<span style="color:#f44747; margin-left:5px;">FURITEN</span>' if furiten else ''
                    state_html = f'<div style="color:#858585; font-size:0.85em; margin-top:4px;">' \
                                f'Shanten: <span style="color:{shanten_color};">{shanten}</span>' \
                                f'{tenpai_mark}{furiten_mark}'
                    if is_tenpai and player_state['waits']:
                        waits = list(player_state['waits'])[:5]  # Show first 5 waits
                        state_html += f' | Waits: {len(waits)} tiles'
                    state_html += '</div>'
                
                html += f"""
                <div style="background:{bg}; border:1px solid {border}; 
                            border-radius:6px; padding:10px; margin:8px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span>
                            <span style="color:#9cdcfe;">{wind}</span>
                            <b style="color:#d4d4d4; margin-left:8px;">{name}</b>
                            {dealer_badge}{riichi_badge}
                        </span>
                        <span style="color:#b5cea8; font-weight:bold;">{score:,}</span>
                    </div>
                    <div style="background:#111; padding:8px; border-radius:4px; 
                                min-height:40px; font-family:Consolas,monospace;">
                        {tile_html}
                    </div>
                    {state_html}
                </div>
                """
            
            html += "</div>"
            display(HTML(html))
    
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
        """
        Start interactive visualization session.
        
        Terminal mode: Keyboard navigation loop
        Notebook mode: Display with ipywidgets buttons
        """
        if self.mode == VizMode.TERMINAL:
            self._show_terminal()
        elif self.mode == VizMode.NOTEBOOK:
            self._show_notebook()
    
    def _show_terminal(self) -> None:
        """Terminal-based interactive navigation."""
        print(f"{Colors.GREEN}Starting Mahjong Replay Viewer{Colors.RESET}")
        print(f"{Colors.DIM}Total states: {len(self.game_log)}{Colors.RESET}\n")
        
        self.render()
        
        if USE_KEYBOARD_LIB:
            # Non-blocking keyboard input
            while True:
                if keyboard.is_pressed('n') or keyboard.is_pressed('l'):
                    self.current_idx = min(len(self.game_log) - 1, self.current_idx + 1)
                    self.render()
                    keyboard.wait('n', suppress=True)
                elif keyboard.is_pressed('p') or keyboard.is_pressed('h'):
                    self.current_idx = max(0, self.current_idx - 1)
                    self.render()
                    keyboard.wait('p', suppress=True)
                elif keyboard.is_pressed('k'):
                    self._jump_to_kyoku(1)
                    self.render()
                    keyboard.wait('k', suppress=True)
                elif keyboard.is_pressed('j'):
                    self._jump_to_kyoku(-1)
                    self.render()
                    keyboard.wait('j', suppress=True)
                elif keyboard.is_pressed('q') or keyboard.is_pressed('x'):
                    print(f"\n{Colors.GREEN}Replay ended{Colors.RESET}")
                    break
        else:
            # Fallback: blocking input()
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
                        print(f"{Colors.YELLOW}Unknown command. Use: n=next, p=prev, k=kyoku+, j=kyoku-, q=quit{Colors.RESET}")
                        continue
                    
                    self.render()
                    
                except KeyboardInterrupt:
                    print(f"\n{Colors.GREEN}Replay ended{Colors.RESET}")
                    break
                except EOFError:
                    break
    
    def _show_notebook(self) -> None:
        """Notebook-based interactive navigation with buttons."""
        if not NOTEBOOK_AVAILABLE:
            print("ipywidgets not available. Use TERMINAL mode.")
            return
        
        self.output = widgets.Output()
        
        # Create navigation buttons
        btn_prev_ba = widgets.Button(description="⏮ BA", layout=widgets.Layout(width='70px'))
        btn_prev_kyoku = widgets.Button(description="⏪ Kyoku", layout=widgets.Layout(width='80px'))
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width='70px'))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width='70px'))
        btn_next_kyoku = widgets.Button(description="Kyoku ⏩", layout=widgets.Layout(width='80px'))
        btn_next_ba = widgets.Button(description="BA ⏭", layout=widgets.Layout(width='70px'))
        
        # Button callbacks
        def on_prev_ba(_):
            self._jump_to_ba(-1)
            self.render()
        
        def on_prev_kyoku(_):
            self._jump_to_kyoku(-1)
            self.render()
        
        def on_prev(_):
            self.current_idx = max(0, self.current_idx - 1)
            self.render()
        
        def on_next(_):
            self.current_idx = min(len(self.game_log) - 1, self.current_idx + 1)
            self.render()
        
        def on_next_kyoku(_):
            self._jump_to_kyoku(1)
            self.render()
        
        def on_next_ba(_):
            self._jump_to_ba(1)
            self.render()
        
        btn_prev_ba.on_click(on_prev_ba)
        btn_prev_kyoku.on_click(on_prev_kyoku)
        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        btn_next_kyoku.on_click(on_next_kyoku)
        btn_next_ba.on_click(on_next_ba)
        
        # Progress slider
        slider = widgets.IntSlider(
            value=self.current_idx,
            min=0,
            max=len(self.game_log) - 1,
            step=1,
            description='Step:',
            layout=widgets.Layout(width='400px')
        )
        
        def on_slider_change(change):
            self.current_idx = change['new']
            self.render()
        
        slider.observe(on_slider_change, names='value')
        
        # Display UI
        nav_row1 = widgets.HBox([btn_prev_ba, btn_prev_kyoku, btn_prev])
        nav_row2 = widgets.HBox([btn_next, btn_next_kyoku, btn_next_ba])
        ui = widgets.VBox([
            nav_row1,
            nav_row2,
            slider,
            self.output
        ])
        
        self.render()
        display(ui)
    
    # =========================================================================
    # EXPORT & UTILITIES
    # =========================================================================
    
    def export_html(self, filepath: str) -> None:
        """Export full replay as interactive HTML file."""
        import json
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mahjong Replay - {self.game_log[0].get('ba', 'Game')}</title>
            <style>
                body {{ background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; padding: 20px; }}
                .state {{ display: none; }}
                .state.active {{ display: block; }}
                .player {{ background: #252526; border: 1px solid #333; border-radius: 6px; padding: 10px; margin: 8px 0; }}
                .player.active {{ border-color: #569cd6; }}
                .nav {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #333; padding: 10px; border-radius: 8px; }}
                button {{ background: #4ec9b0; color: #1e1e1e; border: none; padding: 8px 16px; margin: 0 5px; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background: #6adbc0; }}
            </style>
        </head>
        <body>
            <div id="states">
        """
        
        for i, entry in enumerate(self.game_log):
            entry: GameLogEntry
            active = 'active' if i == 0 else ''
            html_template += f'<div class="state {active}" data-idx="{i}">'
            html_template += f'<h2>{entry.ba if entry.ba else "?"} {entry.kyoku if entry.kyoku else 0} Kyoku</h2>'
            html_template += f'<p>Action: {entry.action if entry.action else "?"} {entry.tile if entry.tile else ""}</p>'
            # Add player states...
            html_template += '</div>'
        
        html_template += """
            </div>
            <div class="nav">
                <button onclick="prev()">◀ Prev</button>
                <button onclick="next()">Next ▶</button>
            </div>
            <script>
                let currentIdx = 0;
                const states = document.querySelectorAll('.state');
                function show(idx) {
                    states.forEach((s, i) => s.classList.toggle('active', i === idx));
                    currentIdx = idx;
                }
                function next() { show(Math.min(states.length - 1, currentIdx + 1)); }
                function prev() { show(Math.max(0, currentIdx - 1)); }
                document.addEventListener('keydown', (e) => {
                    if (e.key === 'ArrowRight' || e.key === 'n') next();
                    if (e.key === 'ArrowLeft' || e.key === 'p') prev();
                });
            </script>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"HTML replay exported to: {filepath}")
    
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