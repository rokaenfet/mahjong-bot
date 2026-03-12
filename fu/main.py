# Mahjong CONSTS
from mahjong.constants import EAST, SOUTH, WEST, NORTH, HAKU, HATSU, CHUN #former 4 are .WINDS, and all 7 are .HONOR_INDICES
from mahjong.constants import FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU # Red Dora number cards > .AKA_DORA_LIST
from mahjong.constants import DISPLAY_WINDS # Str display of the winds

# Mahjong methods/classes
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.shanten import Shanten
from mahjong.meld import Meld
from mahjong.tile import TilesConverter
from mahjong.agari import Agari

# imports
import ipywidgets as widgets
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Optional, Tuple, Union, Set, Dict, Any
from IPython.display import display, HTML, clear_output

import random
import re
import copy
import os
from pathlib import Path
import base64

# Import custom mahjong classes
import helper.config as config
from helper.game import MahjongGame


game = MahjongGame(
    player_names = ["lynn", "byron", "fu", "yagata"],
    starting_score = 25000,
    total_ba = 2,
    enable_red_dora = True,
    log_level = "full"
)

results = game.play_game()
log = game.game_log

