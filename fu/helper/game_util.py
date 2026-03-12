from helper import config

from typing import List, Optional, Tuple, Union, Set, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

class GamePhase(Enum):
    """Enumeration of game phases for state tracking."""
    SETUP = auto()
    DRAW = auto()
    DISCARD = auto()
    CALL_RESOLUTION = auto()
    SETTLEMENT = auto()
    GAME_END = auto()


@dataclass
class GameLogEntry:
    """Structured log entry for visualization and replay."""
    phase: str
    ba: str
    kyoku: int
    honba: int
    turn_player: int
    action: str
    tile: Optional[str]  # MSPZD notation
    dora_indicators: List[str]
    players: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
