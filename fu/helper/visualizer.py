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
    
    def _render_html_file(self, output_path: str = "mahjong_replay.html") -> None:
        """Export replay viewer as an interactive HTML file using tile images."""

        states = []

        for entry in self.game_log:
            entry: GameLogEntry
            metadata = entry.metadata if entry.metadata else {}

            players = []
            for p in entry.players:
                player_state = self._calculate_player_state(p)

                players.append({
                    "name": p.get("name"),
                    "score": p.get("score"),
                    "seat_wind": p.get("seat_wind"),
                    "is_dealer": p.get("is_dealer"),
                    "is_riichi": p.get("is_riichi"),
                    "hand_tiles": p.get("hand_tiles"),
                    "hand_size": p.get("hand_size"),
                    "discards": p.get("discards"),
                    "melds": p.get("melds"),
                    "shanten": player_state["shanten"],
                    "is_tenpai": player_state["is_tenpai"],
                    "furiten": player_state["furiten"],
                    "waits": list(player_state["waits"])
                })

            states.append({
                "ba": entry.ba,
                "kyoku": entry.kyoku,
                "honba": entry.honba,
                "turn_player": entry.turn_player,
                "action": entry.action,
                "tile": entry.tile,
                "dora": entry.dora_indicators,
                "tiles_remaining": metadata.get("tiles_remaining"),
                "dead_wall": metadata.get("dead_wall_remaining"),
                "players": players
            })

        tile_base = config.TILE_IMAGE_DIR.replace("\\", "/")

        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Mahjong Replay Viewer</title>
            <style>
                :root {
                    --bg-primary: #1e1e1e;
                    --bg-secondary: #252526;
                    --bg-tertiary: #2d2d2d;
                    --bg-card: #1a1a1a;
                    --border-color: #333;
                    --border-active: #569cd6;
                    --text-primary: #d4d4d4;
                    --text-secondary: #858585;
                    --text-accent: #4ec9b0;
                    --text-gold: #ffd700;
                    --text-red: #f44747;
                    --text-green: #4ec9b0;
                    --text-blue: #9cdcfe;
                    --text-cyan: #6adbc0;
                    --text-magenta: #d7ba7d;
                    --shadow: 0 2px 8px rgba(0,0,0,0.3);
                    --radius: 8px;
                }

                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }

                body {
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    padding: 20px;
                    min-height: 100vh;
                }

                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }

                /* === CONTROLS === */
                .controls {
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: var(--radius);
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    flex-wrap: wrap;
                    box-shadow: var(--shadow);
                    position: sticky;
                    top: 0;
                    z-index: 100;
                }

                .controls button {
                    background: var(--text-accent);
                    color: var(--bg-primary);
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: bold;
                    font-family: inherit;
                    transition: all 0.2s;
                }

                .controls button:hover {
                    background: #6adbc0;
                    transform: translateY(-1px);
                }

                .controls button:active {
                    transform: translateY(0);
                }

                .controls input[type="range"] {
                    flex: 1;
                    min-width: 200px;
                    height: 6px;
                    -webkit-appearance: none;
                    background: var(--bg-tertiary);
                    border-radius: 3px;
                    outline: none;
                }

                .controls input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    width: 18px;
                    height: 18px;
                    background: var(--text-accent);
                    border-radius: 50%;
                    cursor: pointer;
                }

                .step-counter {
                    background: var(--bg-tertiary);
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 13px;
                    min-width: 100px;
                    text-align: center;
                }

                .keyboard-hint {
                    font-size: 12px;
                    color: var(--text-secondary);
                    margin-left: auto;
                }

                .keyboard-hint kbd {
                    background: var(--bg-tertiary);
                    padding: 2px 6px;
                    border-radius: 3px;
                    border: 1px solid var(--border-color);
                    font-size: 11px;
                }

                /* === HEADER === */
                .game-header {
                    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
                    border: 1px solid var(--border-color);
                    border-left: 4px solid var(--text-accent);
                    border-radius: var(--radius);
                    padding: 15px 20px;
                    margin-bottom: 20px;
                    box-shadow: var(--shadow);
                }

                .header-main {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                    flex-wrap: wrap;
                    gap: 10px;
                }

                .header-title {
                    font-size: 20px;
                    font-weight: bold;
                    color: var(--text-accent);
                }

                .header-info {
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                    font-size: 13px;
                    color: var(--text-secondary);
                }

                .header-info span {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }

                .header-info .highlight {
                    color: var(--text-gold);
                    font-weight: bold;
                }

                .header-info .wall-count {
                    color: var(--text-green);
                    font-weight: bold;
                }

                .dora-display {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 12px;
                    background: var(--bg-card);
                    border-radius: 4px;
                    margin-top: 10px;
                }

                .dora-label {
                    color: var(--text-gold);
                    font-weight: bold;
                    font-size: 13px;
                }

                .dora-tiles {
                    display: flex;
                    gap: 4px;
                }

                .dora-tiles img {
                    height: 32px;
                    width: 24px;
                    border: 1px solid var(--text-gold);
                    border-radius: 3px;
                }

                .action-display {
                    margin-top: 10px;
                    padding: 8px 12px;
                    background: var(--bg-card);
                    border-radius: 4px;
                    font-size: 13px;
                }

                .action-display b {
                    color: var(--text-cyan);
                }

                /* === PLAYERS GRID === */
                .players-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }

                .player-card {
                    background: var(--bg-card);
                    border: 1px solid var(--border-color);
                    border-radius: var(--radius);
                    padding: 15px;
                    box-shadow: var(--shadow);
                    transition: all 0.2s;
                }

                .player-card.turn {
                    border-color: var(--border-active);
                    background: linear-gradient(135deg, #1a1a2e 0%, var(--bg-card) 100%);
                    box-shadow: 0 0 15px rgba(86, 156, 214, 0.3);
                }

                .player-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 12px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid var(--border-color);
                }

                .player-name {
                    font-weight: bold;
                    font-size: 15px;
                    color: var(--text-blue);
                }

                .player-wind {
                    background: var(--bg-tertiary);
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                    margin-right: 8px;
                }

                .player-score {
                    color: var(--text-green);
                    font-weight: bold;
                    font-size: 15px;
                }

                .player-flags {
                    display: flex;
                    gap: 6px;
                    margin-top: 4px;
                }

                .flag {
                    font-size: 11px;
                    padding: 1px 6px;
                    border-radius: 3px;
                    font-weight: bold;
                }

                .flag.dealer {
                    background: rgba(255, 215, 0, 0.2);
                    color: var(--text-gold);
                    border: 1px solid var(--text-gold);
                }

                .flag.riichi {
                    background: rgba(244, 71, 71, 0.2);
                    color: var(--text-red);
                    border: 1px solid var(--text-red);
                }

                /* === HAND DISPLAY === */
                .hand-section {
                    margin-bottom: 12px;
                }

                .hand-label {
                    font-size: 12px;
                    color: var(--text-secondary);
                    margin-bottom: 6px;
                }

                .hand-tiles {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 2px;
                    align-items: center;
                    min-height: 44px;
                    padding: 8px;
                    background: var(--bg-tertiary);
                    border-radius: 4px;
                }

                .hand-tiles img {
                    height: 40px;
                    width: 30px;
                    border: 1px solid #000;
                    border-radius: 3px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                }

                .drawn-tile {
                    margin-left: 12px;
                    padding-left: 12px;
                    border-left: 2px dashed var(--text-secondary);
                }

                .drawn-tile img {
                    border: 2px solid var(--text-gold);
                    box-shadow: 0 0 8px rgba(255, 215, 0, 0.4);
                }

                .tile-separator {
                    color: var(--text-secondary);
                    font-size: 16px;
                    margin: 0 8px;
                }

                /* === MELDS === */
                .meld-section {
                    margin-bottom: 12px;
                }

                .meld-label {
                    font-size: 12px;
                    color: var(--text-secondary);
                    margin-bottom: 6px;
                }

                .meld-list {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }

                .meld-item {
                    background: var(--bg-tertiary);
                    padding: 6px 10px;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                }

                .meld-type {
                    font-size: 11px;
                    font-weight: bold;
                    padding: 2px 6px;
                    border-radius: 3px;
                    min-width: 50px;
                    text-align: center;
                }

                .meld-type.pon {
                    background: rgba(215, 186, 125, 0.2);
                    color: var(--text-magenta);
                    border: 1px solid var(--text-magenta);
                }

                .meld-type.chi {
                    background: rgba(106, 219, 192, 0.2);
                    color: var(--text-cyan);
                    border: 1px solid var(--text-cyan);
                }

                .meld-type.kan_open {
                    background: rgba(255, 215, 0, 0.2);
                    color: var(--text-gold);
                    border: 1px solid var(--text-gold);
                }

                .meld-type.kan_closed {
                    background: rgba(156, 220, 254, 0.2);
                    color: var(--text-blue);
                    border: 1px solid var(--text-blue);
                }

                .meld-tiles {
                    display: flex;
                    gap: 1px;
                }

                .meld-tiles img {
                    height: 28px;
                    width: 21px;
                    border: 1px solid #000;
                    border-radius: 2px;
                }

                .meld-source {
                    font-size: 11px;
                    color: var(--text-secondary);
                    margin-left: auto;
                }

                /* === SHANTEN & WAITS === */
                .status-section {
                    background: var(--bg-tertiary);
                    padding: 8px 10px;
                    border-radius: 4px;
                    margin-bottom: 12px;
                    font-size: 13px;
                }

                .shanten-display {
                    margin-bottom: 6px;
                }

                .shanten-value {
                    font-weight: bold;
                    color: var(--text-green);
                }

                .shanten-value.high {
                    color: var(--text-red);
                }

                .status-badge {
                    display: inline-block;
                    font-size: 11px;
                    padding: 1px 6px;
                    border-radius: 3px;
                    margin-left: 6px;
                    font-weight: bold;
                }

                .status-badge.tenpai {
                    background: rgba(78, 201, 176, 0.2);
                    color: var(--text-green);
                    border: 1px solid var(--text-green);
                }

                .status-badge.furiten {
                    background: rgba(244, 71, 71, 0.2);
                    color: var(--text-red);
                    border: 1px solid var(--text-red);
                }

                .waits-display {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    flex-wrap: wrap;
                }

                .waits-label {
                    color: var(--text-secondary);
                    font-size: 12px;
                }

                .waits-tiles {
                    display: flex;
                    gap: 2px;
                }

                .waits-tiles img {
                    height: 24px;
                    width: 18px;
                    border: 1px solid var(--text-blue);
                    border-radius: 2px;
                }

                /* === DISCARDS === */
                .discard-section {
                    margin-top: 12px;
                    padding-top: 12px;
                    border-top: 1px solid var(--border-color);
                }

                .discard-label {
                    font-size: 12px;
                    color: var(--text-secondary);
                    margin-bottom: 6px;
                }

                .discard-river {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 2px;
                    max-height: 80px;
                    overflow-y: auto;
                    padding: 6px;
                    background: var(--bg-tertiary);
                    border-radius: 4px;
                }

                .discard-river img {
                    height: 28px;
                    width: 21px;
                    border: 1px solid #000;
                    border-radius: 2px;
                    opacity: 0.8;
                }

                .discard-river img:last-child {
                    opacity: 1;
                    border-color: var(--text-secondary);
                }

                .discard-count {
                    font-size: 11px;
                    color: var(--text-secondary);
                    margin-left: 6px;
                }

                /* === SCROLLBAR === */
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }

                ::-webkit-scrollbar-track {
                    background: var(--bg-primary);
                }

                ::-webkit-scrollbar-thumb {
                    background: var(--border-color);
                    border-radius: 4px;
                }

                ::-webkit-scrollbar-thumb:hover {
                    background: var(--text-secondary);
                }

                /* === RESPONSIVE === */
                @media (max-width: 768px) {
                    .players-grid {
                        grid-template-columns: 1fr;
                    }

                    .controls {
                        flex-direction: column;
                        align-items: stretch;
                    }

                    .controls input[type="range"] {
                        width: 100%;
                    }

                    .keyboard-hint {
                        text-align: center;
                        margin-left: 0;
                        margin-top: 10px;
                    }

                    .header-main {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="controls">
                    <button onclick="prevStep()">◀ Prev</button>
                    <button onclick="nextStep()">Next ▶</button>
                    <button onclick="prevKyoku()">⏪ Kyoku</button>
                    <button onclick="nextKyoku()">Kyoku ⏩</button>
                    <input type="range" min="0" max="__MAX__" value="0" id="slider">
                    <span class="step-counter" id="step-counter">0 / __MAX__</span>
                    <div class="keyboard-hint">
                        <kbd>←</kbd> <kbd>→</kbd> Step &nbsp;
                        <kbd>↑</kbd> <kbd>↓</kbd> Kyoku &nbsp;
                        <kbd>Q</kbd> Quit
                    </div>
                </div>

                <div id="replay"></div>
            </div>

            <script>
                const TILE_BASE = "__TILE_BASE__";
                let states = __STATES__;
                let index = 0;
                let kyokuIndices = [];

                // Find kyoku boundaries
                states.forEach((s, i) => {
                    if (s.action && s.action.includes('KYOKU_SETUP')) {
                        kyokuIndices.push(i);
                    }
                });

                const slider = document.getElementById("slider");
                const stepCounter = document.getElementById("step-counter");

                slider.oninput = () => {
                    index = parseInt(slider.value);
                    render();
                };

                // Keyboard navigation
                document.addEventListener('keydown', (e) => {
                    if (e.target.tagName === 'INPUT') return;

                    switch(e.key) {
                        case 'ArrowRight':
                        case 'n':
                        case 'N':
                        case 'l':
                        case 'L':
                            nextStep();
                            break;
                        case 'ArrowLeft':
                        case 'p':
                        case 'P':
                        case 'h':
                        case 'H':
                            prevStep();
                            break;
                        case 'ArrowDown':
                        case 'k':
                        case 'K':
                            nextKyoku();
                            break;
                        case 'ArrowUp':
                        case 'j':
                        case 'J':
                            prevKyoku();
                            break;
                        case 'q':
                        case 'Q':
                            alert('Use browser back/close to exit');
                            break;
                    }
                });

                function nextStep() {
                    index = Math.min(states.length - 1, index + 1);
                    slider.value = index;
                    render();
                }

                function prevStep() {
                    index = Math.max(0, index - 1);
                    slider.value = index;
                    render();
                }

                function nextKyoku() {
                    for (let idx of kyokuIndices) {
                        if (idx > index) {
                            index = idx;
                            slider.value = index;
                            render();
                            return;
                        }
                    }
                    index = states.length - 1;
                    slider.value = index;
                    render();
                }

                function prevKyoku() {
                    for (let i = kyokuIndices.length - 1; i >= 0; i--) {
                        if (kyokuIndices[i] < index) {
                            index = kyokuIndices[i];
                            slider.value = index;
                            render();
                            return;
                        }
                    }
                    index = 0;
                    slider.value = index;
                    render();
                }

                function tilePath(tile) {
                    if (!tile) return "";

                    // Remove trailing 'z' if present (e.g., "Ez" → "E", "Hakuz" → "Haku")
                    let cleanTile = tile;
                    if (tile.endsWith("z") && tile.length > 1) {
                        cleanTile = tile.slice(0, -1);
                    }

                    // === WINDS (E, S, W, N) ===
                    if (cleanTile === "E") return TILE_BASE + "/z/z1.png";
                    if (cleanTile === "S") return TILE_BASE + "/z/z2.png";
                    if (cleanTile === "W") return TILE_BASE + "/z/z3.png";
                    if (cleanTile === "N") return TILE_BASE + "/z/z4.png";

                    // === DRAGONS (Haku, Hatsu, Chun) ===
                    if (cleanTile === "Haku" || cleanTile === "Wht") return TILE_BASE + "/d/d1.png";
                    if (cleanTile === "Hatsu" || cleanTile === "Grn") return TILE_BASE + "/d/d2.png";
                    if (cleanTile === "Chun" || cleanTile === "Red") return TILE_BASE + "/d/d3.png";

                    // === NUMBERED SUITS (1m-9m, 1p-9p, 1s-9s) ===
                    let suit = cleanTile.slice(-1);  // Last character is suit (m, p, s)
                    let num = cleanTile.slice(0, -1);  // Everything before is number

                    if (suit === "m" || suit === "p" || suit === "s") {
                        return TILE_BASE + "/" + suit + "/" + suit + num + ".png";
                    }

                    // === FALLBACK: Handle raw z-suit numbers (1z-7z) ===
                    if (suit === "z") {
                        let honorNum = parseInt(num);
                        if (honorNum >= 1 && honorNum <= 4) {
                            return TILE_BASE + "/z/z" + honorNum + ".png";
                        }
                        if (honorNum === 5) return TILE_BASE + "/d/d1.png";
                        if (honorNum === 6) return TILE_BASE + "/d/d2.png";
                        if (honorNum === 7) return TILE_BASE + "/d/d3.png";
                    }

                    // === DEBUG: Log unknown tile formats ===
                    console.warn("Unknown tile format:", tile);
                    return "";
                }

                function tileHTML(tile, size = "normal") {
                    if (!tile) return "";
                    const height = size === "small" ? 24 : (size === "large" ? 40 : 32);
                    const width = Math.round(height * 0.75);
                    return `<img src="${tilePath(tile)}" title="${tile}" style="height:${height}px;width:${width}px;">`;
                }

                function render() {
                    let s = states[index];
                    if (!s) return;

                    stepCounter.innerText = `${index} / ${states.length - 1}`;

                    // Dora display
                    let doraHTML = "";
                    if (s.dora && s.dora.length > 0) {
                        doraHTML = s.dora.map(t => tileHTML(t, "large")).join("");
                    }

                    // Wall info
                    const wallInfo = s.tiles_remaining !== undefined ? 
                        `<span class="wall-count">🀄 Wall: ${s.tiles_remaining}</span>` : "";
                    const deadInfo = s.dead_wall !== undefined ? 
                        `<span>🀆 Dead: ${s.dead_wall}</span>` : "";

                    // Action display
                    const actionTile = s.tile ? tileHTML(s.tile, "large") : "";
                    const actionDisplay = s.action ? 
                        `<div class="action-display">Action: <b>${s.action}</b> ${actionTile}</div>` : "";

                    let html = `
                    <div class="game-header">
                        <div class="header-main">
                            <div class="header-title">🀄 ${s.ba} ${s.kyoku} Kyoku</div>
                            <div class="header-info">
                                <span>🔄 Honba: <span class="highlight">${s.honba}</span></span>
                                ${wallInfo}
                                ${deadInfo}
                                <span>🎯 Turn: <span class="highlight">P${s.turn_player}</span></span>
                            </div>
                        </div>
                        <div class="dora-display">
                            <span class="dora-label">DORA:</span>
                            <div class="dora-tiles">${doraHTML || "<span style='color:#555'>None</span>"}</div>
                        </div>
                        ${actionDisplay}
                    </div>

                    <div class="players-grid">
                    `;

                    const winds = ["🀀 East", "🀁 South", "🀂 West", "🀃 North"];

                    s.players.forEach((p, i) => {
                        const isTurn = (i === s.turn_player);
                        const cardClass = `player-card${isTurn ? " turn" : ""}`;

                        const dealerFlag = p.is_dealer ? '<span class="flag dealer">DEALER</span>' : '';
                        const riichiFlag = p.is_riichi ? '<span class="flag riichi">RIICHI</span>' : '';

                        // Hand tiles
                        let handHTML = "";
                        if (p.hand_tiles && p.hand_tiles.length > 0) {
                            const isDrawAction = (s.action === "DRAW" && i === s.turn_player);
                            
                            if (isDrawAction && s.tile) {
                                // Show hand + separated drawn tile
                                const handTiles = p.hand_tiles.slice(0, -1).map(t => tileHTML(t)).join("");
                                const drawnTile = tileHTML(s.tile, "large");
                                handHTML = `
                                    <div class="hand-tiles">
                                        ${handTiles}
                                        <span class="tile-separator">→</span>
                                        <span class="drawn-tile">${drawnTile}</span>
                                    </div>
                                `;
                            } else {
                                const tiles = p.hand_tiles.map(t => tileHTML(t)).join("");
                                handHTML = `<div class="hand-tiles">${tiles}</div>`;
                            }
                        } else {
                            handHTML = `<div class="hand-tiles"><span style="color:#555">No tiles</span></div>`;
                        }

                        // Melds
                        let meldHTML = "";
                        if (p.melds && p.melds.length > 0) {
                            meldHTML = '<div class="meld-section"><div class="meld-label">MELDS</div><div class="meld-list">';
                            p.melds.forEach(m => {
                                const typeClass = m.type ? m.type.toLowerCase().replace(' ', '_') : 'unknown';
                                const tiles = m.tiles ? m.tiles.map(t => tileHTML(t, "small")).join("") : "";
                                const source = m.from_player !== undefined ? `(from P${m.from_player})` : "";
                                meldHTML += `
                                    <div class="meld-item">
                                        <span class="meld-type ${typeClass}">${m.type}</span>
                                        <div class="meld-tiles">${tiles}</div>
                                        <span class="meld-source">${source}</span>
                                    </div>
                                `;
                            });
                            meldHTML += '</div></div>';
                        }

                        // Shanten & Status
                        let statusHTML = "";
                        if (p.shanten !== "?" && p.shanten !== undefined) {
                            const shantenClass = p.shanten > 3 ? "high" : "";
                            const tenpaiBadge = p.is_tenpai ? '<span class="status-badge tenpai">TENPAI</span>' : '';
                            const furitenBadge = p.furiten ? '<span class="status-badge furiten">FURITEN</span>' : '';
                            
                            let waitsHTML = "";
                            if (p.waits && p.waits.length > 0) {
                                const waitTiles = p.waits.slice(0, 10).map(wt => {
                                    // Convert wait tile type to MSPZD
                                    let waitTile;
                                    if (wt < 9) waitTile = (wt + 1) + "m";
                                    else if (wt < 18) waitTile = (wt - 8) + "p";
                                    else if (wt < 27) waitTile = (wt - 17) + "s";
                                    else {
                                        const honors = ["E", "S", "W", "N", "Haku", "Hatsu", "Chun"];
                                        waitTile = honors[wt - 27] || "?";
                                    }
                                    return tileHTML(waitTile, "small");
                                }).join("");
                                const moreWaits = p.waits.length > 10 ? `+${p.waits.length - 10}` : "";
                                waitsHTML = `
                                    <div class="waits-display">
                                        <span class="waits-label">Waits:</span>
                                        <div class="waits-tiles">${waitTiles}</div>
                                        ${moreWaits ? `<span class="discard-count">${moreWaits}</span>` : ""}
                                    </div>
                                `;
                            }

                            statusHTML = `
                                <div class="status-section">
                                    <div class="shanten-display">
                                        Shanten: <span class="shanten-value ${shantenClass}">${p.shanten}</span>
                                        ${tenpaiBadge}${furitenBadge}
                                    </div>
                                    ${waitsHTML}
                                </div>
                            `;
                        }

                        // Discards
                        let discardHTML = "";
                        if (p.discards && p.discards.length > 0) {
                            const recent = p.discards.slice(-12);
                            const hidden = p.discards.length - 12;
                            const tiles = recent.map(t => tileHTML(t, "small")).join("");
                            discardHTML = `
                                <div class="discard-section">
                                    <div class="discard-label">
                                        DISCARDS ${hidden > 0 ? `<span class="discard-count">(${hidden} more)</span>` : ""}
                                    </div>
                                    <div class="discard-river">${tiles}</div>
                                </div>
                            `;
                        }

                        html += `
                        <div class="${cardClass}">
                            <div class="player-header">
                                <div>
                                    <span class="player-wind">${winds[p.seat_wind] || "?"}</span>
                                    <span class="player-name">${p.name || "P" + i}</span>
                                    <div class="player-flags">${dealerFlag}${riichiFlag}</div>
                                </div>
                                <div class="player-score">${p.score?.toLocaleString() || 0}</div>
                            </div>

                            <div class="hand-section">
                                <div class="hand-label">HAND (${p.hand_size || 0} tiles)</div>
                                ${handHTML}
                            </div>

                            ${meldHTML}
                            ${statusHTML}
                            ${discardHTML}
                        </div>
                        `;
                    });

                    html += "</div>";
                    document.getElementById("replay").innerHTML = html;

                    // Scroll to top on kyoku change
                    if (kyokuIndices.includes(index)) {
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    }
                }

                // Initial render
                render();
            </script>
        </body>
        </html>
        """

        html = html.replace("__STATES__", json.dumps(states, ensure_ascii=False))
        html = html.replace("__MAX__", str(len(states) - 1))
        html = html.replace("__TILE_BASE__", tile_base)

        with open(output_path, "w", encoding="utf8") as f:
            f.write(html)

        print(f"Replay exported to: {output_path}")
    # =========================================================================
    # DISPLAY & NAVIGATION
    # =========================================================================
    
    def render(self) -> None:
        """Render current state based on mode."""
        if self.mode == VizMode.TERMINAL:
            self._render_terminal()
        elif self.mode == VizMode.NOTEBOOK:
            self._render_notebook()
        elif self.mode == VizMode.HTML_FILE:
            self._render_html_file()
    
    def show(self) -> None:
        """Start interactive visualization session."""
        if self.mode == VizMode.TERMINAL:
            self._show_terminal()
        elif self.mode == VizMode.NOTEBOOK:
            self._show_notebook()
        elif self.mode == VizMode.HTML_FILE:
            self._render_html_file()
    
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