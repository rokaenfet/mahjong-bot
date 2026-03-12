# CONST

TOTAL_CARD_NUM = 136 # 4 * [9pin + 9sozu + 9manzu + 3sanngennpai + 4jihai]
TILE_IMAGE_DIR = "./tile_imgs/"

# Honor tile types (34-format)
EAST = 27
SOUTH = 28
WEST = 29
NORTH = 30
HAKU = 31
HATSU = 32
CHUN = 33

HONOR_TYPES = {EAST, SOUTH, WEST, NORTH, HAKU, HATSU, CHUN}
HONOR_NAMES = {
    EAST: "E", SOUTH: "S", WEST: "W", NORTH: "N",
    HAKU: "Haku", HATSU: "Hatsu", CHUN: "Chun"
}

# Red dora tile IDs (136-format) - specific copies that are red
FIVE_RED_MAN = 16   # 5-man, copy 0
FIVE_RED_PIN = 52   # 5-pin, copy 0  
FIVE_RED_SOU = 88   # 5-sou, copy 0

RED_DORA_IDS = {FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU}

# Mapping: 34-type → red dora 136-ID (for quick lookup)
RED_DORA_BY_TYPE = {
    4: FIVE_RED_MAN,   # 5-man type
    13: FIVE_RED_PIN,  # 5-pin type
    22: FIVE_RED_SOU,  # 5-sou type
}