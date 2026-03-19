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
# Extended to 0.50 so that landscape-oriented meld tiles (e.g. the face-strip
# visible on the top player's melds) whose label sits at ~40–50 % of height
# are still captured.  For normal portrait tiles the extra area is blank white
# and the largest-component selector in _normalize_label still finds the glyph.
LABEL_Y_END = 0.50

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

        # Find the best connected component (the label character).
        # The label indicator is always anchored to the TOP-RIGHT corner of the
        # label zone.  Tile-body graphics (bamboo sticks, 発 kanji) that leak
        # into the left side of the zone must be ignored.
        #
        # Scoring: reward components that are (a) far right and (b) near the top.
        #   score = x_right_norm * 0.6 + (1 - y_top_norm) * 0.4
        # where x_right_norm = (x + w) / zone_width  and
        #       y_top_norm   = y / zone_height.
        #
        # Candidate filtering (all three conditions required):
        #   • area  ≥ 5 % of zone area  (removes single-pixel noise)
        #   • width ≥ 20 % of zone_w    (removes thin edge-shadow artifacts)
        #   • height ≥ 15 % of zone_h   (removes hairline horizontal lines)
        # If no component passes the size filters we fall back to the largest.
        # This is backward-compatible: for normal tiles only one significant
        # component exists, so it wins trivially.
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n_comp > 1:
            zone_h, zone_w = binary.shape[:2]
            min_area  = max(4,   zone_h * zone_w * 0.05)
            min_width = max(3,   zone_w * 0.20)
            min_height= max(2,   zone_h * 0.15)

            cands = [i + 1 for i, s in enumerate(stats[1:])
                     if (s[cv2.CC_STAT_AREA]   >= min_area
                         and s[cv2.CC_STAT_WIDTH]  >= min_width
                         and s[cv2.CC_STAT_HEIGHT] >= min_height)]
            if not cands:           # everything is noise — fall back to largest
                cands = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]

            def _top_right_score(idx: int) -> float:
                s = stats[idx]
                x_right = (s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]) / max(1, zone_w)
                y_top   = s[cv2.CC_STAT_TOP] / max(1, zone_h)
                return x_right * 0.6 + (1.0 - y_top) * 0.4

            best = max(cands, key=_top_right_score)
            x  = stats[best, cv2.CC_STAT_LEFT]
            y  = stats[best, cv2.CC_STAT_TOP]
            bw = stats[best, cv2.CC_STAT_WIDTH]
            bh = stats[best, cv2.CC_STAT_HEIGHT]
            pad = 2
            x  = max(0, x - pad)
            y  = max(0, y - pad)
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

    # Compute label-zone colour signals early — the top-right corner is a reliable
    # per-suit indicator even when the full-tile stats are diluted or ambiguous.
    th, tw = tile_img.shape[:2]
    label_zone = tile_img[:int(th * 0.40), int(tw * 0.55):]
    red_l = green_l = blue_l = 0.0
    if label_zone.size > 0:
        hsv_l = cv2.cvtColor(label_zone, cv2.COLOR_BGR2HSV)
        h_l, s_l, _ = cv2.split(hsv_l)
        sm_l = s_l > 50
        lz_tot = max(1, sm_l.size)
        red_l   = (sm_l & ((h_l <= 12) | (h_l >= 168))).sum() / lz_tot
        green_l = (sm_l & (h_l >= 35) & (h_l <= 85)).sum() / lz_tot
        blue_l  = (sm_l & (h_l >= 86) & (h_l <= 140)).sum() / lz_tot

    # Very dominant green (little red present) → sou, regardless of dark content
    if green > 0.04 and green > red * 3.0:
        return "sou"
    # Moderate green with bamboo-like dark structure → sou
    # (handles muted bamboo tiles where green/red ratio is 1.2–3x)
    if green > 0.05 and green > red * 1.2 and dark_pct > 0.09:
        return "sou"
    # Sou via label zone: green dominates in the label area even when full-tile
    # stats are muddied by mixed-colour bamboo (green ≈ red in tile body)
    if green_l > 0.10 and green_l > red_l * 1.5:
        return "sou"

    # Multi-colour bamboo fallback (e.g. 9s, 8s): tile art has mixed green+red so
    # the 3× and 1.2× thresholds above are not met, but green still exceeds red
    # with significant dark structure.  Blue guard avoids catching pin tiles whose
    # circles can produce high dark_pct.
    if green > 0.08 and green > red and dark_pct > 0.08 and blue < 0.10:
        return "sou"

    # Dominant blue → pin (Mahjong Soul pin circles are rendered in deep blue/purple)
    if blue > 0.15 and blue > red * 2 and blue > green * 2:
        return "pin"
    if blue_l > 0.10 and blue_l > red_l * 2 and blue_l > green_l * 2:
        return "pin"

    if red > 0.04 and red > green * 2:
        # Pin tiles have dark circle patterns giving a high dark-to-red ratio
        # Man tiles have red characters but relatively little dark area
        #
        # High-pip pin tiles (e.g. 6p, 7p) have many red circles that push the
        # overall red fraction very high, so dark_pct can fall below red * 1.2
        # even though the dark ring outlines are significant.  Man character
        # strokes are thin and keep dark_pct < 0.10 in practice; circle tiles
        # with multiple large rings reliably produce dark_pct > 0.12.
        if dark_pct > 0.12:
            return "pin"
        if dark_pct > red * 1.2:
            return "pin"
        return "man"

    # Low saturation with dark structured patterns → pin.
    # This must come BEFORE the label-zone man check: a pin dora indicator tile
    # has dark_pct > 0.10 + sat < 0.10, but also some red in its label zone
    # from the golden dora frame — we must not misclassify it as man.
    if dark_pct > 0.10:
        sat_pct = sat_mask.sum() / total
        if sat_pct < 0.10:
            return "pin"
        return "honor"

    # Label-zone red → man.  Only reaches here when dark_pct ≤ 0.10, which
    # rules out pin tiles (their circles produce dark content).  This catches
    # man dora indicator tiles whose full-tile red is diluted by white background.
    if red_l > 0.05 and red_l > green_l:
        return "man"

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

    # Honor tiles — use body colour to disambiguate visually-similar label glyphs.
    #
    # Dragon tiles (reliable colour signal):
    #   Hatsu → distinctive green body (sou)
    #   Chun  → distinctive red body (man)
    #
    # Wind tiles (E/S/W/N): body colour is NOT reliable — _detect_suit often
    # returns "man"/"pin" for wind tiles due to coloured label text.  However,
    # some digit glyphs can be confused with letters (e.g. "5" ≈ "S" at 28×28).
    # Guard: if a wind letter wins the label match but a DIGIT is nearly as
    # strong (within _WIND_DIGIT_MARGIN) AND the tile body looks like a numbered
    # suit, prefer the digit.  This handles the "5p misidentified as South" case
    # without breaking genuine South/North/etc. tiles (where no digit scores
    # close to the letter and/or the suit is "honor").
    if label_char in LABEL_TO_HONOR:
        if label_char == "Ch" and suit_guess == "sou":
            label_char = "Ht"
            debug["label"] = label_char
        elif label_char == "Ht" and suit_guess == "man":
            label_char = "Ch"
            debug["label"] = label_char
        elif label_char in ("E", "S", "W", "N") and suit_guess in SUIT_MAP:
            # Guard: measure red content in the label zone.
            #
            # In Mahjong Soul, numbered-tile corner indicators are RED for man
            # tiles (and similarly coloured for pin/sou).  Wind-tile indicators
            # are coloured per-wind (East=red, South=teal, West=blue, North=dark)
            # but the body of the indicator character typically carries little to
            # no RED hue.  The one exception is East (東, red character) — handled
            # below via the confidence threshold on the dragon override.
            #
            # Rule: only apply the digit-override AND the dragon-colour-override
            # when the label zone has measurable red (≥ 0.03 fraction of saturated
            # pixels).  When red_lz is near zero the tile is almost certainly a
            # genuine wind tile (South/West/North) or a Hatsu-body fake wind — in
            # either case we trust the wind letter or the sou colour signal.
            th, tw = tile_img.shape[:2]
            lz = tile_img[:int(th * LABEL_Y_END), int(tw * LABEL_X_START):]
            red_lz = 0.0
            if lz.size > 0:
                hsv_lz = cv2.cvtColor(lz, cv2.COLOR_BGR2HSV)
                h_lz, s_lz, _ = cv2.split(hsv_lz)
                sm_lz = s_lz > 50
                lz_tot = max(1, sm_lz.size)
                red_lz = (sm_lz & ((h_lz <= 12) | (h_lz >= 168))).sum() / lz_tot
            _LABEL_ZONE_RED_THRESH = 0.03

            if red_lz > _LABEL_ZONE_RED_THRESH:
                # Label zone has red → could be a numbered tile whose label
                # glyph resembles a wind letter.  Apply digit override.
                best_digit_score = -1.0
                best_digit_char  = None
                for lc2, tmpls2 in templates.items():
                    if not (lc2.isdigit() and 1 <= int(lc2) <= 9):
                        continue
                    for tmpl2 in tmpls2:
                        padded2 = cv2.copyMakeBorder(
                            tmpl2, _MATCH_PAD, _MATCH_PAD, _MATCH_PAD, _MATCH_PAD,
                            cv2.BORDER_CONSTANT, value=0,
                        )
                        lc2_crop = _crop_label(tile_img)
                        query    = _normalize_label(lc2_crop)
                        sc2 = cv2.matchTemplate(padded2, query, cv2.TM_CCOEFF_NORMED).max()
                        if sc2 > best_digit_score:
                            best_digit_score = sc2
                            best_digit_char  = lc2
                # Strict override: digit beats wind outright → prefer digit.
                strict = best_digit_char and best_digit_score >= confidence
                # Lenient override: man-suit tile with moderate wind confidence
                # and a decent digit match.  Handles numbered tiles (e.g. 5m, 7m)
                # whose label glyph looks like a wind letter at discard scale or
                # after 90° rotation — but only when the wind letter did NOT win
                # with high confidence (≥ 0.80), which would indicate a real wind.
                lenient = (
                    best_digit_char
                    and suit_guess == "man"
                    and confidence < 0.80
                    and best_digit_score >= 0.70
                )
                if strict or lenient:
                    label_char = best_digit_char
                    debug["label"] = label_char
                    debug["confidence"] = best_digit_score
                    value = int(label_char)
                    return Tile(SUIT_MAP[suit_guess], value), debug
                # Dragon colour override when label-zone has red (e.g. East wind
                # tile with red 東 character that _detect_suit reads as "man").
                # Guard: only when confidence is LOW so genuine East tiles (strong
                # "E" match) are not overridden, AND no decent digit candidate
                # exists (a strong digit match means it's a numbered tile, not
                # a dragon — the Ht/Ch assignment would be wrong).
                _DRAGON_OVERRIDE_MAX_CONF = 0.78
                _DRAGON_MIN_DIGIT_CLEAR = 0.70
                if confidence < _DRAGON_OVERRIDE_MAX_CONF and best_digit_score < _DRAGON_MIN_DIGIT_CLEAR:
                    if suit_guess == "sou":
                        label_char = "Ht"
                        debug["label"] = label_char
                    elif suit_guess == "man":
                        label_char = "Ch"
                        debug["label"] = label_char
            else:
                # Label zone has no significant red → genuine wind tile
                # (South=teal, West=blue, North=dark).  Trust the wind letter,
                # but still apply the sou dragon override unconditionally:
                # no wind tile has a green body, so suit_guess="sou" uniquely
                # identifies a Hatsu tile whose label was confused with a wind
                # letter (common at the left-player 270° rotation scale).
                if suit_guess == "sou":
                    label_char = "Ht"
                    debug["label"] = label_char
        suit, value = LABEL_TO_HONOR[label_char]
        return Tile(suit, value), debug

    # Numbered tiles — require a minimum label confidence to avoid random matches
    # on clipped or otherwise unusable tile images.
    _MIN_LABEL_CONF = 0.10
    if label_char.isdigit() and 1 <= int(label_char) <= 9:
        if confidence < _MIN_LABEL_CONF:
            return None, debug
        value = int(label_char)
        if suit_guess in SUIT_MAP:
            return Tile(SUIT_MAP[suit_guess], value), debug
        return None, debug

    return None, debug
