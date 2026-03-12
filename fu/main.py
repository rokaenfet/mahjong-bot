# Import custom mahjong classes
from helper.game import MahjongGame
from helper.visualizer import MahjongReplay, VizMode

from helper import config

# --- SIMULATE GAME ---

game = MahjongGame(
    player_names = ["lynn", "byron", "fu", "yagata"],
    starting_score = 25000,
    total_ba = 4,
    enable_red_dora = True,
    log_level = "full"
)

results = game.play_game()


# --- VISUALIZER ---

replayer = MahjongReplay(
    game_log = game.game_log,
    mode = VizMode.HTML_FILE,
    show_hand_details = True
)

replayer.show()