"""
Label-based tile recognition for Mahjong Soul.

Uses a self-calibrating approach:
  1. First run: user labels the tiles from a screenshot (--calibrate)
  2. The system extracts label templates from those known tiles
  3. Future runs match against the calibrated label templates

The label is the small character (1-9, E, S, W, N) in the top-right of each tile.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tiles import Tile, Suit, TILE_LOOKUP

LABEL_TEMPLATES_DIR = Path(__file__).parent / "label_templates"
CALIBRATION_FILE = LABEL_TEMPLATES_DIR / "calibration.json"

LABEL_X_START = 0.55
LABEL_Y_END = 0.40

SUIT_MAP = {"man": Suit.MAN, "pin": Suit.PIN, "sou": Suit.SOU}

SHORTHAND_TO_LABEL = {}
for v in range(1, 10):
    SHORTHAND_TO_LABEL[f"{v}m"] = str(v)
    SHORTHAND_TO_LABEL[f"{v}p"] = str(v)
    SHORTHAND_TO_LABEL[f"{v}s"] = str(v)
SHORTHAND_TO_LABEL.update({"east": "E", "south": "S", "west": "W", "north": "N",
                            "haku": "Hk", "hatsu": "Ht", "chun": "Ch"})


def _crop_label(tile_img: np.ndarray) -> np.ndarray:
    h, w = tile_img.shape[:2]
    return tile_img[0:int(h * LABEL_Y_END), int(w * LABEL_X_START):]


def _normalize_label(label_crop: np.ndarray, extract_char: bool = True) -> np.ndarray:
    """Convert label crop to a normalized binary character image for matching."""
    gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)

    if extract_char:
        # Threshold to find the dark text on light background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clean small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find the largest connected component (the label character)
        coords = cv2.findNonZero(binary)
        if coords is not None and len(coords) > 5:
            x, y, bw, bh = cv2.boundingRect(coords)
            pad = 2
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(binary.shape[1] - x, bw + 2 * pad)
            bh = min(binary.shape[0] - y, bh + 2 * pad)
            char_roi = binary[y:y + bh, x:x + bw]
            return cv2.resize(char_roi, (28, 28))

    return cv2.resize(gray, (28, 28))


def _detect_suit(tile_img: np.ndarray) -> str:
    hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    sat_mask = s_ch > 50
    total = max(1, sat_mask.size)

    green = (sat_mask & (h_ch >= 35) & (h_ch <= 85)).sum() / total
    red = (sat_mask & ((h_ch <= 12) | (h_ch >= 168))).sum() / total
    blue = (sat_mask & (h_ch >= 86) & (h_ch <= 140)).sum() / total

    dark_mask = (v_ch < 120) & (v_ch > 20)
    dark_pct = dark_mask.sum() / total

    if green > 0.04 and green > red * 1.5:
        return "sou"

    if red > 0.04 and red > green * 2:
        # Pin tiles have dark circle patterns giving a high dark-to-red ratio
        # Man tiles have red characters but relatively little dark area
        if dark_pct > red * 1.2:
            return "pin"
        return "man"

    # Low saturation with dark structured patterns → pin
    if dark_pct > 0.10:
        sat_pct = sat_mask.sum() / total
        if sat_pct < 0.10:
            return "pin"
        return "honor"

    return "honor"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(tile_imgs: list[np.ndarray], tile_names: list[str]):
    """Save label templates from user-labeled tiles (additive).

    tile_names: list of shorthand strings like ["3m", "6m", "west", ...]
    Running calibrate again adds to existing templates rather than replacing them.
    """
    LABEL_TEMPLATES_DIR.mkdir(exist_ok=True)

    # Load existing calibration data if present
    label_map: dict[str, list[str]] = {}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            label_map = json.load(f)

    # Find next available index
    existing_files = list(LABEL_TEMPLATES_DIR.glob("label_*.png"))
    next_idx = len(existing_files)

    for i, (img, name) in enumerate(zip(tile_imgs, tile_names)):
        name = name.strip().lower()
        if name not in TILE_LOOKUP and name not in SHORTHAND_TO_LABEL:
            print(f"  Warning: unknown tile '{name}', skipping")
            continue

        label_char = SHORTHAND_TO_LABEL.get(name, name)
        label_crop = _crop_label(img)
        normalized = _normalize_label(label_crop)

        fname = f"label_{label_char}_{next_idx + i:02d}.png"
        cv2.imwrite(str(LABEL_TEMPLATES_DIR / fname), normalized)

        if label_char not in label_map:
            label_map[label_char] = []
        label_map[label_char].append(fname)

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    n_chars = len(label_map)
    n_imgs = sum(len(v) for v in label_map.values())
    print(f"\nCalibration saved: {n_chars} unique labels from {n_imgs} total templates")
    print(f"Templates in {LABEL_TEMPLATES_DIR}/")


def load_label_templates() -> dict[str, list[np.ndarray]]:
    """Load calibrated label templates."""
    if not CALIBRATION_FILE.exists():
        return {}

    with open(CALIBRATION_FILE) as f:
        label_map = json.load(f)

    templates: dict[str, list[np.ndarray]] = {}
    for label_char, fnames in label_map.items():
        templates[label_char] = []
        for fname in fnames:
            img = cv2.imread(str(LABEL_TEMPLATES_DIR / fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[label_char].append(img)

    return templates


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
_MATCH_PAD = 4  # pixels of padding added to each stored template during matching

def _match_label(tile_img: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    """Match a tile's label against calibrated templates.

    Each stored 28x28 template is padded by _MATCH_PAD pixels on all sides
    before matching, giving matchTemplate a search window that absorbs small
    positional differences between screenshots without re-calibrating.
    """
    label_crop = _crop_label(tile_img)
    query = _normalize_label(label_crop)  # 28x28

    best_char = "?"
    best_score = -1.0

    for label_char, tmpls in templates.items():
        for tmpl in tmpls:
            padded = cv2.copyMakeBorder(
                tmpl, _MATCH_PAD, _MATCH_PAD, _MATCH_PAD, _MATCH_PAD,
                cv2.BORDER_CONSTANT, value=0,
            )
            result = cv2.matchTemplate(padded, query, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            if score > best_score:
                best_score = score
                best_char = label_char

    return best_char, best_score


LABEL_TO_HONOR = {
    "E": (Suit.WIND, 1), "S": (Suit.WIND, 2),
    "W": (Suit.WIND, 3), "N": (Suit.WIND, 4),
    "Hk": (Suit.DRAGON, 1), "Ht": (Suit.DRAGON, 2), "Ch": (Suit.DRAGON, 3),
}


def recognize_tile(tile_img: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[Tile | None, dict]:
    """Recognize a tile using calibrated label matching + color analysis."""
    if not templates:
        return None, {"label": "?", "confidence": 0.0, "suit_guess": _detect_suit(tile_img),
                       "error": "not calibrated"}

    label_char, confidence = _match_label(tile_img, templates)
    suit_guess = _detect_suit(tile_img)

    debug = {"label": label_char, "confidence": confidence, "suit_guess": suit_guess}

    # Honor tiles
    if label_char in LABEL_TO_HONOR:
        suit, value = LABEL_TO_HONOR[label_char]
        return Tile(suit, value), debug

    # Numbered tiles
    if label_char.isdigit() and 1 <= int(label_char) <= 9:
        value = int(label_char)
        if suit_guess in SUIT_MAP:
            return Tile(SUIT_MAP[suit_guess], value), debug
        return None, debug

    return None, debug
