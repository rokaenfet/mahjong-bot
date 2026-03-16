"""
game_state.py – Full board state recognition for Mahjong Soul.

Recognises (all four players: self / top / left / right):
  - Discard piles
  - Open melds (pon / chii / kan) with tile identities
  - Riichi status
  - Dora indicator tiles and computed actual dora
  - Aka dora (red five) presence
  - Round wind + round number  (e.g. "East 2")
  - Wall tile count remaining
  - Player scores (gold digits in the centre panel)
  - Seat wind per player (badge icons at diamond corners)

Calibration commands (run with --help for full usage):
  python game_state.py screen.png --calibrate-round  East1
  python game_state.py screen.png --calibrate-wall   67
  python game_state.py screen.png --calibrate-score  self  25000
  python game_state.py screen.png --calibrate-score  left  25000
  python game_state.py screen.png --calibrate-seat   left  N
  python game_state.py screen.png --show-regions
  python game_state.py screen.png --debug

Score calibration notes:
  - self / top  → saves to label_templates/score_digits/  (horizontal rendering)
  - left        → saves to label_templates/score_digits_side/  (vertical stack)
  - right       → also saves to score_digits_side/ (rotated 180° before matching)
  - Calibrate from multiple screenshots to cover digits 0-9.
  - First-write-wins within each directory — recalibrate by deleting the file.

Wall count calibration:
  - Saves to label_templates/wall/  (teal "x NN" text on centre panel)
  - Recalibrate by deleting label_templates/wall/d<digit>.png files.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tiles import Tile, Suit
from label_ocr import recognize_tile, load_label_templates
from recognize import extract_tile_images, TILE_BRIGHTNESS_THRESH

# ──────────────────────────────────────────────────────────────────────────────
# Screen region constants  (fractions of full-image width / height).
# Tuned for the default Mahjong Soul 4-player horizontal layout.
# Adjust these if your game window size / layout differs.
# ──────────────────────────────────────────────────────────────────────────────

# Discard pile per player.  Left/right crops are rotated before processing.
# self=0°  top=180°  left=270°(CCW)  right=90°(CW)
#
# Calibrated from pixel-level blob analysis of 5 screenshots:
#   SELF:  tiles at y=0.501–0.678, x=0.401–0.591  (aspect ~1.0, square)
#   TOP:   tiles at y=0.149–0.264, x=0.421–0.580  (aspect ~0.84, slightly landscape)
#   LEFT:  tiles at y=0.268–0.491, x=0.302–0.409  (aspect ~0.49, landscape; rotated)
#   RIGHT: tiles at y=0.268–0.491, x=0.589–0.714  (aspect ~0.50, landscape; rotated)
SELF_DISCARD  = dict(y_start=0.482, y_end=0.720, x_start=0.375, x_end=0.615)
TOP_DISCARD   = dict(y_start=0.128, y_end=0.315, x_start=0.415, x_end=0.608)
LEFT_DISCARD  = dict(y_start=0.240, y_end=0.515, x_start=0.230, x_end=0.425)
RIGHT_DISCARD = dict(y_start=0.240, y_end=0.515, x_start=0.572, x_end=0.762)

# Open meld (pon / chii / kan) areas per opponent.
#
# Left/right player melds are in a vertical column on the respective screen
# edge (their right-hand side from their own perspective).  The same 270°/90°
# rotation used for discards is applied so tiles face upright before
# segmentation.  Regions are wide enough to hold up to 4 meld sets (≤ 16
# tiles) stacked in one row after rotation.
#
# Top player melds appear to OUR LEFT of the top discard area.  Tiles are
# displayed upside-down; the 180° rotation used for discards is also applied.
#
# Calibrated from mahjongsoul7.png (left pon of Hatsu at y≈0.700–0.887).
# Self meld tiles appear to the RIGHT of the hand tiles at the bottom;
# calibrated from mahjongsoul6.png (self pon of Hatsu at x≈0.800–0.930).
SELF_MELD  = dict(y_start=0.855, y_end=0.955, x_start=0.790, x_end=0.940)
# Top player's meld tiles are rendered in strong 3-D perspective: only a thin
# horizontal face-strip is visible at the very top of the screen.  Calibrated
# from mahjongsoul4.png where 6 meld tiles appear at abs x≈0.210–0.405,
# y≈0.020–0.095.  The old region (x=0.005–0.215) missed this strip entirely.
TOP_MELD   = dict(y_start=0.018, y_end=0.098, x_start=0.200, x_end=0.415)
LEFT_MELD  = dict(y_start=0.420, y_end=0.955, x_start=0.045, x_end=0.150)
RIGHT_MELD = dict(y_start=0.045, y_end=0.580, x_start=0.848, x_end=0.955)

# Thin strips in front of each player where a riichi stick appears.
# Kept narrow AND requires an elongated bright blob (see detect_riichi).
SELF_RIICHI  = dict(y_start=0.822, y_end=0.856, x_start=0.390, x_end=0.610)
TOP_RIICHI   = dict(y_start=0.145, y_end=0.179, x_start=0.390, x_end=0.610)
LEFT_RIICHI  = dict(y_start=0.420, y_end=0.580, x_start=0.358, x_end=0.398)
RIGHT_RIICHI = dict(y_start=0.420, y_end=0.580, x_start=0.602, x_end=0.642)

# Dora indicator display (top-left HUD strip).
# x_start=0 so the leftmost indicator tile is never clipped.
DORA_REGION = dict(y_start=0.000, y_end=0.082, x_start=0.000, x_end=0.210)

# Center diamond panel containing round wind, wall count, etc.
# Starts at ~0.330 so TOP_DISCARD can end at ~0.325 without overlap.
CENTER_PANEL    = dict(y_start=0.330, y_end=0.650, x_start=0.370, x_end=0.630)
# Sub-regions inside the centre panel crop (fractions of the panel crop itself).
# "East 1" badge text is teal/blue and sits roughly in the top-third.
ROUND_TEXT_SUB  = dict(y_start=0.08, y_end=0.36, x_start=0.18, x_end=0.82)
# "x 67" wall count is teal text just below the "East 1" badge (also teal).
# x range 0.40-0.62 skips the large badge-frame teal blobs at x≈0 and x≈0.80.
# y_start=0.20 catches screenshots where the wall-count line sits a bit higher
# (slightly above y=0.27 that we used before).  y_end=0.34 stays above the gold
# self-score row that starts at y≈0.35.
WALL_COUNT_SUB  = dict(y_start=0.20, y_end=0.34, x_start=0.40, x_end=0.62)

# Score sub-regions (fractions of CENTER_PANEL crop).
# Layout (confirmed by pixel analysis):
#   Top  : horizontal row at y≈0.03, upside-down  (read after 180° rotate)
#   Left : vertical stack of digits at x≈0.34, y=0.12-0.27
#   Right: vertical stack of digits at x≈0.64, y=0.12-0.25
#   Self : horizontal row at y≈0.35 (just below "x NN" wall count)
SCORE_SELF_SUB  = dict(y_start=0.31, y_end=0.42, x_start=0.38, x_end=0.62)
SCORE_TOP_SUB   = dict(y_start=0.00, y_end=0.09, x_start=0.38, x_end=0.62)
SCORE_LEFT_SUB  = dict(y_start=0.10, y_end=0.30, x_start=0.30, x_end=0.42)
SCORE_RIGHT_SUB = dict(y_start=0.10, y_end=0.28, x_start=0.58, x_end=0.70)

# Seat-wind badge sub-regions (small coloured icons flanking the self-score row).
# Left/right badges are confirmed at y≈0.38-0.55 at the horizontal edges.
# Top/self badges are not reliably visible in the center panel.
SEAT_LEFT_SUB   = dict(y_start=0.38, y_end=0.58, x_start=0.00, x_end=0.16)
SEAT_RIGHT_SUB  = dict(y_start=0.38, y_end=0.58, x_start=0.84, x_end=1.00)
SEAT_TOP_SUB    = dict(y_start=0.00, y_end=0.10, x_start=0.00, x_end=0.16)
SEAT_SELF_SUB   = dict(y_start=0.00, y_end=0.10, x_start=0.84, x_end=1.00)

# Template directories for text elements
_HERE               = Path(__file__).parent
ROUND_TMPL_DIR      = _HERE / "label_templates" / "round"
WALL_TMPL_DIR       = _HERE / "label_templates" / "wall"
SCORE_DIGIT_DIR     = _HERE / "label_templates" / "score_digits"
# Separate dir for left/right side scores — the digits there are rendered in a
# vertical stack (each ~17px wide x 10px tall) and look different when resized
# compared with the horizontal self/top digits.
SCORE_SIDE_DIGIT_DIR = _HERE / "label_templates" / "score_digits_side"
SEAT_WIND_TMPL_DIR  = _HERE / "label_templates" / "seat_wind"

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Meld:
    """One open meld (pon / chii / kan).

    tiles          – the 3 (or 4 for kan) recognised tile objects in
                     left-to-right order as seen after player rotation.
    called_tile_idx – index into ``tiles`` of the called tile (the one that
                     was rotated 90° in the display), or None if uncertain.
    """
    tiles:           list[Tile | None]
    called_tile_idx: int | None = None

    def __str__(self) -> str:
        parts = []
        for i, t in enumerate(self.tiles):
            s = str(t) if t else "???"
            parts.append(f"[{s}]" if i == self.called_tile_idx else s)
        return "(" + " ".join(parts) + ")"


@dataclass
class PlayerState:
    discards:  list[Tile | None] = field(default_factory=list)
    melds:     list[Meld]        = field(default_factory=list)
    riichi:    bool              = False
    score:     int | None        = None
    seat_wind: str | None        = None   # "East" / "South" / "West" / "North"

    @property
    def meld_count(self) -> int:
        """Total number of meld tile slots (legacy helper; 3 per pon/chi, 4 for kan)."""
        return sum(len(m.tiles) for m in self.melds)


@dataclass
class GameState:
    self_state:  PlayerState = field(default_factory=PlayerState)
    top_state:   PlayerState = field(default_factory=PlayerState)
    left_state:  PlayerState = field(default_factory=PlayerState)
    right_state: PlayerState = field(default_factory=PlayerState)

    dora_indicators: list[Tile | None] = field(default_factory=list)
    doras:           list[Tile | None] = field(default_factory=list)

    round_wind:   str | None = None   # "East" / "South" / "West" / "North"
    round_number: int | None = None   # 1–4
    wall_count:   int | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _crop(img: np.ndarray, region: dict) -> np.ndarray:
    h, w = img.shape[:2]
    y1, y2 = int(h * region["y_start"]), int(h * region["y_end"])
    x1, x2 = int(w * region["x_start"]), int(w * region["x_end"])
    return img[y1:y2, x1:x2]


_ROTATIONS = {
    0:   None,
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Discard pile segmentation  (multi-row, unlike the single-row hand)
# ──────────────────────────────────────────────────────────────────────────────

_DISCARD_MIN_AREA    = 200   # smaller than hand tiles' MIN_TILE_AREA
_DISCARD_ASPECT_LO   = 0.50
_DISCARD_ASPECT_HI   = 2.5
_DISCARD_BRIGHTNESS  = 130   # raw mean brightness required to accept a blob
_ROW_MERGE_RATIO     = 0.55  # blob y-centres within this * tile_h -> same row


def segment_discard_tiles(
    discard_img: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Detect all tiles in a multi-row discard pile image.
    Returns (x, y, w, h) bounding boxes sorted row-by-row, left-to-right.
    """
    h, w = discard_img.shape[:2]

    # Small blur only — a large kernel (≥ 7px) bridges the 0-3px gaps between
    # adjacent tiles in the rotated left/right crops and merges them into one
    # blob.  k=3 smoothes sensor noise without bridging inter-tile gaps.
    blurred  = cv2.GaussianBlur(discard_img, (3, 3), 0)
    gray     = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_raw = cv2.cvtColor(discard_img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, TILE_BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)

    # Minimal horizontal erode/dilate — just enough to clean ragged edges on
    # isolated tiles without collapsing the narrow gaps between touching tiles.
    # (Tiles in rotated left/right crops can have as little as 2 px between them.)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mask   = cv2.erode(mask, kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / max(1, bw)
        if bw * bh >= _DISCARD_MIN_AREA and _DISCARD_ASPECT_LO <= aspect <= _DISCARD_ASPECT_HI:
            if cv2.mean(gray_raw[y:y + bh, x:x + bw])[0] >= _DISCARD_BRIGHTNESS:
                boxes.append((x, y, bw, bh))

    if not boxes:
        return []

    # Remove slivers (width much smaller than median) — catches thin false
    # positives from wall-tile edges or centre-panel UI elements.
    if len(boxes) >= 2:
        med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[2] >= med_w * 0.4]

    # Remove short false positives (height much smaller than median) — catches
    # centre-panel badges / buttons that leak into the top of the crop.
    if len(boxes) >= 2:
        med_h = sorted(b[3] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[3] >= med_h * 0.4]

    # Group boxes into rows by overlapping y-centres
    boxes.sort(key=lambda b: b[1])
    ref_h = max(b[3] for b in boxes) if boxes else 1

    rows: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        bx, by, bw, bh = box
        cy = by + bh // 2
        placed = False
        for row in rows:
            ry, rh = row[0][1], row[0][3]
            row_cy = ry + rh // 2
            if abs(cy - row_cy) < ref_h * _ROW_MERGE_RATIO:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    result: list[tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        result.extend(row)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Discard pile recognition
# ──────────────────────────────────────────────────────────────────────────────

def recognize_discards(
    img: np.ndarray,
    region: dict,
    rotation_cw: int,
    label_templates: dict,
    debug_prefix: str | None = None,
) -> list[Tile | None]:
    """
    Detect and identify tiles in a player's discard region.

    rotation_cw: clockwise degrees to rotate the crop so tiles read upright.
                 self=0, top=180, left=270, right=90
    """
    crop = _crop(img, region)
    rot  = _ROTATIONS.get(rotation_cw)
    if rot is not None:
        crop = cv2.rotate(crop, rot)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_discard_crop.png", crop)

    boxes     = segment_discard_tiles(crop)
    tile_imgs = extract_tile_images(crop, boxes)

    tiles: list[Tile | None] = []
    for t_img in tile_imgs:
        tile, _ = recognize_tile(t_img, label_templates)
        tiles.append(tile)

    if debug_prefix:
        vis = crop.copy()
        for (x, y, bw, bh), tile in zip(boxes, tiles):
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
            label = str(tile) if tile else "?"
            cv2.putText(vis, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(f"{debug_prefix}_discard_vis.png", vis)

    return tiles


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Dora indicators + actual dora computation
# ──────────────────────────────────────────────────────────────────────────────

_DORA_NEXT_NUMBER = {v: (v % 9) + 1 for v in range(1, 10)}   # 9 -> 1
_DORA_NEXT_WIND   = {1: 2, 2: 3, 3: 4, 4: 1}                  # North -> East
_DORA_NEXT_DRAGON = {1: 2, 2: 3, 3: 1}                        # Chun -> Haku


def indicator_to_dora(tile: Tile | None) -> Tile | None:
    """Return the actual dora tile for a given indicator tile."""
    if tile is None:
        return None
    if tile.suit in (Suit.MAN, Suit.PIN, Suit.SOU):
        return Tile(tile.suit, _DORA_NEXT_NUMBER[tile.value])
    if tile.suit == Suit.WIND:
        return Tile(Suit.WIND, _DORA_NEXT_WIND[tile.value])
    if tile.suit == Suit.DRAGON:
        return Tile(Suit.DRAGON, _DORA_NEXT_DRAGON[tile.value])
    return None


def recognize_dora(
    img: np.ndarray,
    label_templates: dict,
    debug_prefix: str | None = None,
) -> tuple[list[Tile | None], list[Tile | None]]:
    """
    Detect dora indicator tiles and compute the actual dora tiles.
    Returns (indicators, doras).

    The dora HUD shows exactly one face-up tile (the indicator) followed by
    several face-down orange tiles.  We pad the crop so the leftmost tile is
    never edge-clipped, then filter to only the bright (face-up) tiles.
    """
    crop = _crop(img, DORA_REGION)
    # Pad left/top with dark pixels so the indicator tile (which can be flush
    # against the left edge) still gets a closed contour.  Dark (50) stays well
    # below TILE_BRIGHTNESS_THRESH so it never merges with the bright tile face.
    PAD = 12
    crop = cv2.copyMakeBorder(crop, PAD, PAD, PAD, 0,
                              cv2.BORDER_CONSTANT, value=(50, 50, 50))

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_dora_crop.png", crop)

    boxes = segment_discard_tiles(crop)

    # Keep only face-up (bright) tiles — face-down ones are orange/dark
    face_up: list[tuple[int, int, int, int]] = []
    gray_raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        x, y, bw, bh = box
        mean_br = cv2.mean(gray_raw[y:y + bh, x:x + bw])[0]
        if mean_br >= 170:          # face-up tiles are noticeably brighter
            face_up.append(box)

    # Shift boxes back for extract_tile_images (undo left pad)
    tile_imgs = extract_tile_images(crop, face_up)

    indicators: list[Tile | None] = []
    doras:      list[Tile | None] = []
    for t_img in tile_imgs:
        tile, _ = recognize_tile(t_img, label_templates)
        indicators.append(tile)
        doras.append(indicator_to_dora(tile))

    return indicators, doras


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Aka dora (red five) detection
# ──────────────────────────────────────────────────────────────────────────────

_AKA_HUE_HI       = 12    # red wraps near 0 and 180
_AKA_HUE_HI2      = 168
_AKA_MIN_SAT      = 120
_AKA_MIN_VAL      = 80
_AKA_PCT_THRESH   = 0.03   # fraction of tile pixels that must be red


def is_aka_dora(tile_img: np.ndarray, tile: Tile | None) -> bool:
    """
    Return True if the tile image is a red five (aka dora).
    Only evaluates 5m / 5p / 5s; always False for any other tile.
    """
    if tile is None or tile.value != 5:
        return False
    if tile.suit not in (Suit.MAN, Suit.PIN, Suit.SOU):
        return False

    hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red_mask = (((h <= _AKA_HUE_HI) | (h >= _AKA_HUE_HI2))
                & (s >= _AKA_MIN_SAT) & (v >= _AKA_MIN_VAL))
    return bool(red_mask.sum() / max(1, red_mask.size) > _AKA_PCT_THRESH)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Open meld recognition (pon / chii / kan)
# ──────────────────────────────────────────────────────────────────────────────

def _trim_to_face(tile_img: np.ndarray, min_gap: int = 3) -> np.ndarray:
    """Strip the 3-D side-edge from a perspective-rendered meld tile.

    Meld tiles in Mahjong Soul are displayed at a slight angle so the side of
    the tile is visible on the right of each bounding box.  The label character
    lives on the face (left portion); the side strip confuses ``_crop_label``.

    Uses an adaptive brightness threshold derived from the face's own pixel
    statistics: columns that dip below 55 % of the mean face brightness are
    considered the edge/gap region, and the face is trimmed there.
    """
    gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    col_frac = (gray > 160).sum(axis=0) / H   # fraction of bright pixels per col

    # Derive threshold from the left half of the tile (definitely face)
    face_mean = float(col_frac[:W // 2].mean()) if W >= 4 else 0.8
    threshold = face_mean * 0.55   # anything below this is in the gap / side

    x = W - 1
    while x > W // 3:
        if col_frac[x] < threshold:
            gap_right = x
            while x > 0 and col_frac[x] < threshold:
                x -= 1
            gap_left = x + 1
            if gap_right - gap_left + 1 >= min_gap:
                face_w = max(gap_left, W // 3)
                return tile_img[:, :face_w]
        else:
            x -= 1

    return tile_img   # no clear gap → assume already face-only


# Minimum tile area for a meld blob (larger than discard false-positives)
# Minimum tile area for a meld blob.
# Heavily-decorated tiles (Hatsu, bamboo) fill much of the white face area
# with coloured glyphs, leaving only ~2900–3100 S<60 (near-white) pixels in
# the face mask.  The threshold must stay below that to avoid dropping them.
# False-positive risk is low: UI chrome elements in meld regions are much
# smaller (< 1000 px²) or clearly non-white.
_MELD_MIN_AREA = 2500


def segment_meld_tiles(meld_img: np.ndarray) -> list[tuple]:
    """Find tile bounding boxes in a meld strip (after player rotation).

    Returns boxes sorted left-to-right.  Uses a stricter area filter than
    ``segment_discard_tiles`` because meld tiles are larger and false-positive
    UI elements (Tools button, avatar ornaments) are smaller.

    Uses a white-face mask (low saturation, high value) instead of a raw
    brightness threshold so that the orange/golden 3-D side edges that are
    visible on meld tiles do not get found as separate blobs.  The side edges
    have OpenCV-HSV saturation ≈ 200+ while tile faces are near-white (S < 60).
    """
    h, w = meld_img.shape[:2]
    blurred = cv2.GaussianBlur(meld_img, (3, 3), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # White-face mask: low saturation (face) OR already-bleached white
    # Excludes the orange 3-D side edges (S ≈ 200) and coloured backgrounds.
    face_mask = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    # Also keep high-brightness pixels that aren't strongly saturated orange,
    # so that tile art (green bamboo, red circles) within the face is still
    # enclosed in the bounding box after dilation.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    bright_mask = ((gray > TILE_BRIGHTNESS_THRESH) &
                   ~((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42) &
                     (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 140))
                   ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(face_mask, bright_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mask   = cv2.erode(mask, kernel, iterations=1)
    mask   = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= _MELD_MIN_AREA:
            boxes.append((x, y, bw, bh))

    if not boxes:
        return []

    # Keep only blobs whose area is at least 35 % of the largest blob
    max_area = max(b[2] * b[3] for b in boxes)
    boxes = [b for b in boxes if b[2] * b[3] >= max_area * 0.35]

    # Median-height filter to remove residual small UI elements
    if len(boxes) >= 2:
        med_h = sorted(b[3] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[3] >= med_h * 0.45]

    boxes.sort(key=lambda b: b[0])
    return boxes


def _bleach_tile_frame(tile_img: np.ndarray) -> np.ndarray:
    """Replace the golden/orange tile-frame pixels with white.

    Meld tiles have a visible decorative frame around the face (golden/orange
    border).  When the frame bleeds into the label zone it creates a large
    bright blob in the BINARY_INV image that wins the largest-component contest
    over the actual label glyph.  Bleaching those pixels to white makes them
    disappear after BINARY_INV (white → 0), leaving only the label character.

    HSV range chosen for the Mahjong Soul tile frame: H≈15–42°, S>70, V>140.
    """
    hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
    frame = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42) &
             (hsv[:, :, 1] > 70)  & (hsv[:, :, 2] > 140))
    out = tile_img.copy()
    out[frame] = (255, 255, 255)
    return out


def _trim_gray_top(tile_img: np.ndarray) -> np.ndarray:
    """Trim the gray shadow band from the top of a face-strip tile.

    The top player's meld tiles are rendered in steep 3-D perspective: the face
    blob extracted by segment_meld_tiles includes a gray shadow at the top
    (the angled top edge of the tile) before the actual white face area starts.
    OTSU thresholding treats this gray band as a large dark blob that outcompetes
    the small label character.

    This function finds the first row where ≥ 25 % of pixels are truly bright
    (> 210), indicating the start of the white face area, and crops from there.
    Falls back to the whole image if no such row is found.
    """
    gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    row_frac = (gray > 210).sum(axis=1) / max(1, W)
    for y in range(H):
        if row_frac[y] >= 0.25:
            return tile_img[y:, :] if y > 0 else tile_img
    return tile_img


def _recognize_meld_tile(
    tile_img: np.ndarray,
    label_templates: dict,
) -> tuple[Tile | None, dict]:
    """Recognize a single tile extracted from a meld strip.

    Steps:
    1. Trim the 3-D side edge (right) so ``_crop_label`` sees the face only.
    2. For landscape tiles (face-strip geometry), also trim the gray shadow band
       from the top rows so the white face area is at the image top.
    3. Bleach the golden tile-frame pixels so they don't dominate the binary.
    4. Try the standard top-right label zone.
    5. If confidence is low, also try the top-left zone (called tile fallback).
    6. Return whichever orientation gave the higher confidence.
    """
    from label_ocr import recognize_tile

    face = _trim_to_face(tile_img)

    # The 3-D perspective rendering adds a dark shadow band at the TOP of the
    # face crop: visible on landscape tiles (face-strip, top player) but also
    # on some portrait tiles (left/right players viewed at an angle).
    # Trim it unconditionally so _crop_label only sees the clean white face.
    face = _trim_gray_top(face)

    face  = _bleach_tile_frame(face)
    tile, dbg = recognize_tile(face, label_templates)
    conf = dbg.get("confidence", 0.0)

    # Always try the horizontally-flipped face as well.  Called tiles (rotated
    # 90° within the meld) can end up with their label at the top-LEFT after
    # the player-crop rotation, and perspective shadows sometimes produce a
    # spuriously high confidence for the wrong tile on the unflipped face.
    # We always keep whichever orientation gives the higher confidence score.
    face_flipped = cv2.flip(face, 1)
    tile2, dbg2 = recognize_tile(face_flipped, label_templates)
    if dbg2.get("confidence", 0.0) > conf:
        tile, dbg = tile2, dbg2

    return tile, dbg


def _group_into_melds(
    boxes: list[tuple],
    crop: np.ndarray,
    label_templates: dict,
) -> list[Meld]:
    """Group sorted tile boxes into meld sets of 3–4 and recognise each tile."""
    if not boxes:
        return []

    # Estimate a typical tile width from the median of all box widths
    med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]

    # Split into groups wherever there is a gap > 60 % of a tile width
    groups: list[list[tuple]] = []
    current = [boxes[0]]
    for i in range(1, len(boxes)):
        prev = boxes[i - 1]
        gap = boxes[i][0] - (prev[0] + prev[2])
        if gap > med_w * 0.6:
            groups.append(current)
            current = [boxes[i]]
        else:
            current.append(boxes[i])
    groups.append(current)

    # ---- post-process: split groups that are too large to be a single meld ----
    # When two adjacent melds have no visible gap (especially the top player's
    # face-strip tiles), all tiles land in one over-sized group.  We break such
    # groups into sub-groups of 3 (pon/chii) – or 4 (kan) if that's what the
    # tile count implies – using the called-tile count as a cross-check.
    def _split_oversized(grp: list[tuple]) -> list[list[tuple]]:
        n = len(grp)
        if n <= 4:
            return [grp]

        # Count called tiles (wider/shorter aspect than the rest)
        asps  = [b[3] / max(1, b[2]) for b in grp]
        med_a = sorted(asps)[n // 2]
        n_called = sum(
            1 for a in asps
            if abs(a - med_a) / max(0.01, med_a) > 0.25
        )

        # Infer meld size: if we have clearly 2× the called tiles of a single
        # meld, use 3-tile chunks; otherwise fall back to simple thirds.
        chunk = 3
        if n_called >= 2 and n % n_called == 0:
            # e.g. 6 tiles × 2 called → each meld has 3 tiles
            chunk = n // n_called

        # Build chunks; if the last piece is undersized, absorb it into prev.
        chunks = [grp[i : i + chunk] for i in range(0, n, chunk)]
        while len(chunks) > 1 and len(chunks[-1]) < 3:
            last = chunks.pop()
            chunks[-1].extend(last)
        return [c for c in chunks if len(c) >= 3]

    expanded: list[list[tuple]] = []
    for g in groups:
        expanded.extend(_split_oversized(g))
    groups = expanded

    melds: list[Meld] = []
    for group in groups:
        if len(group) < 3:
            continue   # too few tiles to be a valid meld

        # Identify the called tile: the one whose aspect ratio (h/w) differs
        # most from the group median (it was rotated an extra 90°)
        aspects = [b[3] / max(1, b[2]) for b in group]
        med_asp = sorted(aspects)[len(aspects) // 2]
        called_idx = None
        max_diff = 0.0
        for i, asp in enumerate(aspects):
            diff = abs(asp - med_asp) / max(0.01, med_asp)
            if diff > 0.25 and diff > max_diff:
                max_diff = diff
                called_idx = i

        tiles_out: list[Tile | None] = []
        confidences: list[float] = []
        for x, y, bw, bh in group:
            t_img = crop[y:y + bh, x:x + bw]
            t, dbg = _recognize_meld_tile(t_img, label_templates)
            tiles_out.append(t)
            confidences.append(dbg.get("confidence", 0.0))

        # Pon / kan majority-vote correction.
        #
        # A pon is always 3 identical tiles; a kan is 4.  When one tile is
        # mis-recognised (e.g. the called tile's label is in an unusual
        # position), the two/three correct tiles form a clear majority.
        # Replace any outlier with the majority tile.
        #
        # Guard against chii mis-application: if an outlier tile is sequential
        # to the majority tile (same suit, value ±1) that indicates a chii, not
        # a pon with a misread.  In that case we skip the override so the chii
        # tiles are preserved as-is.
        from collections import Counter
        valid = [(t, i) for i, t in enumerate(tiles_out) if t is not None]
        if valid:
            counts = Counter(t for t, _ in valid)
            majority_tile, majority_count = counts.most_common(1)[0]
            if majority_count >= 2:
                outliers = [t for t in tiles_out
                            if t is not None and t != majority_tile]
                # Chii is only possible with numbered suits (man/pin/sou).
                # Honour tiles (winds, dragons) can never form a chii, so the
                # sequential guard must not fire for them.
                _numbered = {Suit.MAN, Suit.PIN, Suit.SOU}
                is_chii_pattern = (
                    majority_tile.suit in _numbered
                    and any(
                        o.suit == majority_tile.suit
                        and abs(o.value - majority_tile.value) <= 2
                        for o in outliers
                    )
                )
                if not is_chii_pattern:
                    for i, t in enumerate(tiles_out):
                        if t != majority_tile:
                            tiles_out[i] = majority_tile

        melds.append(Meld(tiles=tiles_out, called_tile_idx=called_idx))

    return melds


def recognize_melds(
    img: np.ndarray,
    region: dict,
    rotation_cw: int,
    label_templates: dict,
    debug_prefix: str | None = None,
) -> list[Meld]:
    """Detect and identify all open melds in a player's meld region.

    rotation_cw: same convention as ``recognize_discards``
                 self=0, top=180, left=270, right=90
    """
    crop = _crop(img, region)
    rot  = _ROTATIONS.get(rotation_cw)
    if rot is not None:
        crop = cv2.rotate(crop, rot)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_meld_crop.png", crop)

    boxes = segment_meld_tiles(crop)

    if debug_prefix:
        vis = crop.copy()
        for x, y, bw, bh in boxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 128), 1)
        cv2.imwrite(f"{debug_prefix}_meld_vis.png", vis)

    return _group_into_melds(boxes, crop, label_templates)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Riichi stick detection
# ──────────────────────────────────────────────────────────────────────────────

_RIICHI_BRIGHT_THRESH = 210   # sticks are near-white/cream
_RIICHI_MIN_ASPECT    = 3.5   # stick is much longer than it is wide
_RIICHI_MIN_AREA_FRAC = 0.06  # blob must cover at least 6 % of the zone


def detect_riichi(img: np.ndarray, zone: dict) -> bool:
    """
    Return True if a riichi stick is visible in the given zone.

    A riichi stick is a thin, very bright (near-white) elongated rectangle.
    We require both high brightness AND a high aspect ratio so that ambient
    wall-tile edges don't trigger false positives.
    """
    crop = _crop(img, zone)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    zh, zw = gray.shape
    zone_area = max(1, zh * zw)

    bright_mask = (gray >= _RIICHI_BRIGHT_THRESH).astype(np.uint8)

    # Quick bail-out: if not enough bright pixels at all, skip
    if bright_mask.sum() / zone_area < 0.04:
        return False

    # Find connected bright blobs and test each for stick-like shape
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < zone_area * _RIICHI_MIN_AREA_FRAC:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect >= _RIICHI_MIN_ASPECT:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Round wind + round number  ("East 2"  ->  wind="East", number=2)
# ──────────────────────────────────────────────────────────────────────────────

_WIND_NAMES = ["East", "South", "West", "North"]


def _round_text_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of teal/cyan pixels (the round-indicator badge text colour)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # "East 1" badge text is bright teal/cyan: H≈80-115 in OpenCV 0-179 scale
    return cv2.inRange(hsv, (80, 80, 100), (115, 255, 255))


def _gold_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of gold/yellow pixels (score digits, wall-count digits)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (18, 100, 150), (38, 255, 255))


def _load_round_templates() -> dict[str, np.ndarray]:
    tmpls: dict[str, np.ndarray] = {}
    if not ROUND_TMPL_DIR.exists():
        return tmpls
    for p in ROUND_TMPL_DIR.glob("*.png"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tmpls[p.stem] = img   # stem = "East1", "East2", …
    return tmpls


def calibrate_round(img: np.ndarray, label: str):
    """
    Save a template for the given round indicator string (e.g. "East1", "East2").
    Run once per unique round you want to recognise.
    """
    panel = _crop(img, CENTER_PANEL)
    sub   = _crop(panel, ROUND_TEXT_SUB)
    mask  = _round_text_mask(sub)

    coords = cv2.findNonZero(mask)
    if coords is None:
        print("No orange text found in centre panel — check ROUND_TEXT_SUB region.")
        print("Saving debug image: debug_round_sub.png")
        cv2.imwrite("debug_round_sub.png", sub)
        return

    x, y, bw, bh = cv2.boundingRect(coords)
    normalized = cv2.resize(mask[y:y + bh, x:x + bw], (64, 24))

    ROUND_TMPL_DIR.mkdir(parents=True, exist_ok=True)
    out = ROUND_TMPL_DIR / f"{label}.png"
    cv2.imwrite(str(out), normalized)
    print(f"Saved round template -> {out}")


def recognize_round_info(
    img: np.ndarray,
    round_templates: dict[str, np.ndarray] | None = None,
    debug_prefix: str | None = None,
) -> tuple[str | None, int | None]:
    """
    Return (round_wind, round_number) from the centre panel.
    Returns (None, None) when uncalibrated or unrecognised.
    """
    panel = _crop(img, CENTER_PANEL)
    sub   = _crop(panel, ROUND_TEXT_SUB)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_round_sub.png", sub)
        cv2.imwrite(f"{debug_prefix}_round_mask.png", _round_text_mask(sub))

    if round_templates is None:
        round_templates = _load_round_templates()

    if not round_templates:
        return None, None

    mask   = _round_text_mask(sub)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None

    x, y, bw, bh = cv2.boundingRect(coords)
    query = cv2.resize(mask[y:y + bh, x:x + bw], (64, 24)).astype(np.float32)

    best_label, best_score = None, -1.0
    for label, tmpl in round_templates.items():
        padded = cv2.copyMakeBorder(
            tmpl.astype(np.float32), 4, 4, 4, 4, cv2.BORDER_CONSTANT
        )
        score = float(cv2.matchTemplate(padded, query, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_score, best_label = score, label

    if best_label is None or best_score < 0.60:
        return None, None

    for wind in _WIND_NAMES:
        if best_label.startswith(wind):
            try:
                return wind, int(best_label[len(wind):])
            except ValueError:
                return wind, None
    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Wall tile count
# ──────────────────────────────────────────────────────────────────────────────

def _bright_binary(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _segment_digits(
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """
    Return (x, y, w, h) boxes for digit-like blobs in a binary image,
    sorted reading-order (top-to-bottom, left-to-right within each row).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_h, img_w = binary.shape[:2]
    # Use a small fixed minimum so vertically-stacked score digits and small
    # wall-count digits (which can be as short as 4-5px) are not rejected.
    min_bh = max(4, img_h * 0.04)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 15 or bh < min_bh or bw > img_w * 0.6:
            continue
        aspect = bh / max(1, bw)
        if 0.18 <= aspect <= 5.0:
            boxes.append((x, y, bw, bh))
    if not boxes:
        return []
    # Bucket into rows: blobs whose y-centres are within one median height of
    # each other belong to the same row.
    med_h = sorted(b[3] for b in boxes)[len(boxes) // 2]
    boxes.sort(key=lambda b: b[1])  # by y first
    rows: list[list] = []
    for box in boxes:
        cy = box[1] + box[3] // 2
        placed = False
        for row in rows:
            row_cy = row[0][1] + row[0][3] // 2
            if abs(cy - row_cy) < med_h * 0.6:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    result = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        result.extend(row)
    return result


def _match_boxes(
    binary: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    digit_templates: dict[str, np.ndarray],
    min_score: float = 0.45,
) -> int | None:
    """Read a number from pre-computed boxes using digit template matching."""
    if not digit_templates or not boxes:
        return None
    digits = []
    for x, y, bw, bh in boxes:
        patch = cv2.resize(binary[y:y + bh, x:x + bw], (16, 24)).astype(np.float32)
        best_d, best_s = None, -1.0
        for d, tmpl in digit_templates.items():
            padded = cv2.copyMakeBorder(
                tmpl.astype(np.float32), 2, 2, 2, 2, cv2.BORDER_CONSTANT
            )
            s = float(cv2.matchTemplate(padded, patch, cv2.TM_CCOEFF_NORMED).max())
            if s > best_s:
                best_s, best_d = s, d
        if best_d is not None and best_s >= min_score:
            digits.append(best_d)
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


def _match_number(
    binary: np.ndarray,
    digit_templates: dict[str, np.ndarray],
    min_score: float = 0.45,
) -> int | None:
    """Read a number from a binary mask using digit template matching."""
    if not digit_templates:
        return None
    boxes = _segment_digits(binary)
    if not boxes:
        return None
    # Remove slivers (width << median)
    if len(boxes) >= 2:
        med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[2] >= med_w * 0.3]
    return _match_boxes(binary, boxes, digit_templates, min_score)


def _separate_teal_digits(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Erode horizontally to split merged teal characters (e.g. 'x67' -> x, 6, 7),
    then return boxes sorted in reading order.  The first box (leftmost) is the
    'x' prefix and is skipped by the caller.
    """
    # Erode with a horizontal kernel to break connections between adjacent chars
    ew = max(2, int(mask.shape[1] * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ew, 1))
    sep = cv2.erode(mask, kernel, iterations=2)
    return _segment_digits(sep)


def _load_wall_templates() -> dict[str, np.ndarray]:
    """Load per-digit wall-count templates (d0..d9)."""
    tmpls: dict[str, np.ndarray] = {}
    if not WALL_TMPL_DIR.exists():
        return tmpls
    for p in WALL_TMPL_DIR.glob("d*.png"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tmpls[p.stem[1:]] = img   # "d6" -> "6"
    return tmpls


def _load_score_digit_templates(side: bool = False) -> dict[str, np.ndarray]:
    """Load per-digit score templates (0..9).

    side=True loads the left/right "side" templates from SCORE_SIDE_DIGIT_DIR.
    side=False loads the self/top templates from SCORE_DIGIT_DIR.
    If the side dir is empty or missing, falls back to the main dir.
    """
    d = SCORE_SIDE_DIGIT_DIR if side else SCORE_DIGIT_DIR
    tmpls: dict[str, np.ndarray] = {}
    if d.exists():
        for p in d.glob("*.png"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tmpls[p.stem] = img
    # Fall back to main dir if side dir has no templates yet
    if side and not tmpls:
        return _load_score_digit_templates(side=False)
    return tmpls


def calibrate_wall_count(img: np.ndarray, count: int):
    """
    Save wall-count digit templates for the given tile count.
    The display reads "x NN"; we skip the "x" and save each digit individually.
    Run with a screenshot where the wall count equals `count`.
    """
    panel  = _crop(img, CENTER_PANEL)
    sub    = _crop(panel, WALL_COUNT_SUB)
    cv2.imwrite("debug_wall_sub.png", sub)

    # Wall count "x NN" text is teal, same colour as "East 1"
    mask = _round_text_mask(sub)
    cv2.imwrite("debug_wall_mask.png", mask)

    # x range is already trimmed to skip the large frame blobs and the "x" char;
    # digit blobs are the remaining small teal shapes.
    boxes = _segment_digits(mask)
    count_str = str(count)

    # If extra noise blobs are present, keep only the rightmost N (= digit count)
    if len(boxes) > len(count_str):
        boxes = boxes[len(boxes) - len(count_str):]

    if len(boxes) != len(count_str):
        print(f"WARNING: found {len(boxes)} blobs, expected {len(count_str)} digits.")
        print("Check debug_wall_sub.png / debug_wall_mask.png")
        if not boxes:
            return

    WALL_TMPL_DIR.mkdir(parents=True, exist_ok=True)
    saved = set()
    for i, (x, y, bw, bh) in enumerate(boxes):
        if i >= len(count_str):
            break
        d = count_str[i]
        if d in saved:
            continue
        patch = cv2.resize(mask[y:y + bh, x:x + bw], (16, 24))
        out = WALL_TMPL_DIR / f"d{d}.png"
        cv2.imwrite(str(out), patch)
        print(f"  Saved wall digit '{d}' -> {out}")
        saved.add(d)
    remaining = set(count_str) - saved
    if remaining:
        print(f"NOTE: digits {sorted(remaining)} not yet saved — calibrate from another screenshot.")


def recognize_wall_count(
    img: np.ndarray,
    wall_templates: dict[str, np.ndarray] | None = None,
    debug_prefix: str | None = None,
) -> int | None:
    """Return the number of tiles remaining in the wall, or None."""
    panel = _crop(img, CENTER_PANEL)
    sub   = _crop(panel, WALL_COUNT_SUB)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_wall_sub.png", sub)

    if wall_templates is None:
        wall_templates = _load_wall_templates()
    if not wall_templates:
        return None

    # Wall count "x NN" text is teal — use the round-text (teal) mask.
    mask = _round_text_mask(sub)
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_wall_mask.png", mask)

    # x range is trimmed (x=0.40-0.62 of panel) so badge-frame blobs are excluded.
    # The sub may still contain the "x" prefix of "x NN":  "x" is noticeably
    # shorter than the digit characters.  We keep only blobs at or above the
    # median blob height (with a 0.65 tolerance) to discard it.
    boxes = _segment_digits(mask)
    if len(boxes) >= 2:
        # Drop blobs that are much narrower than the median (noise)
        med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[2] >= med_w * 0.5]
    if len(boxes) >= 2:
        # Drop the "x" character: it is shorter than the digit blobs
        med_h = sorted(b[3] for b in boxes)[len(boxes) // 2]
        boxes = [b for b in boxes if b[3] >= med_h * 0.65]

    # Use a lower threshold for wall digits: the teal mask is already a strong
    # colour filter so false positives are rare, but digit rendering varies
    # slightly between game states.
    return _match_boxes(mask, boxes, wall_templates, min_score=0.30)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Player scores (centre-panel corners, gold text)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_score_digits(
    img: np.ndarray,
    score_str: str,
    seat: str = "self",
    debug: bool = False,
):
    """
    Extract digit templates from a known score value visible in the given seat's
    score sub-region of the centre panel.

      score_str : the score as a string, e.g. "25000"
      seat      : "self" / "top" / "left" / "right"
    """
    sub_map = {
        "self":  SCORE_SELF_SUB,
        "top":   SCORE_TOP_SUB,
        "left":  SCORE_LEFT_SUB,
        "right": SCORE_RIGHT_SUB,
    }
    sub_region = sub_map.get(seat.lower())
    if sub_region is None:
        print(f"Unknown seat {seat!r}. Use self / top / left / right.")
        return

    # Rotations must match what recognize_scores applies so templates align.
    _SCORE_ROTATIONS = {"top": cv2.ROTATE_180, "right": cv2.ROTATE_180}

    panel = _crop(img, CENTER_PANEL)
    crop  = _crop(panel, sub_region)
    rot   = _SCORE_ROTATIONS.get(seat.lower())
    if rot is not None:
        crop = cv2.rotate(crop, rot)
    mask  = _gold_mask(crop)

    if debug:
        cv2.imwrite(f"debug_score_{seat}.png", crop)
        cv2.imwrite(f"debug_score_{seat}_mask.png", mask)

    # Dilate slightly to merge split strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    boxes  = _segment_digits(mask_d)

    if len(boxes) != len(score_str):
        print(f"WARNING: found {len(boxes)} digit blobs, score '{score_str}' has "
              f"{len(score_str)} digits.")
        print("Check debug images. Continuing with what was found.")

    # Left/right scores are rendered in a vertical stack (each digit is a short
    # wide blob) — save their templates to a separate directory so they don't
    # corrupt the main self/top horizontal-rendering templates.
    is_side = seat.lower() in ("left", "right")
    tmpl_dir = SCORE_SIDE_DIGIT_DIR if is_side else SCORE_DIGIT_DIR
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    saved = set()
    for i, (x, y, bw, bh) in enumerate(boxes):
        if i >= len(score_str):
            break
        d = score_str[i]
        if d in saved:
            continue          # don't overwrite with a later occurrence of same digit
        out = tmpl_dir / f"{d}.png"
        if out.exists():
            print(f"  Skipped digit '{d}' (template already exists at {out})")
            continue          # keep first-calibrated template
        patch = cv2.resize(mask[y:y + bh, x:x + bw], (16, 24))
        cv2.imwrite(str(out), patch)
        print(f"  Saved score digit '{d}' -> {out}")
        saved.add(d)

    remaining = set(score_str) - saved
    if remaining:
        print(f"NOTE: digits {sorted(remaining)} not yet saved — calibrate from "
              "another screenshot that shows these digits in a score.")


def recognize_scores(
    img: np.ndarray,
    digit_templates: dict[str, np.ndarray] | None = None,
    side_templates: dict[str, np.ndarray] | None = None,
    debug_prefix: str | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """
    Read all four player scores from the centre panel corners.
    Returns (self_score, top_score, left_score, right_score).
    Any score that cannot be read is returned as None.

    digit_templates : templates for self/top (horizontal rendering)
    side_templates  : templates for left/right (vertical-stack rendering);
                      if None, falls back to digit_templates
    """
    if digit_templates is None:
        digit_templates = _load_score_digit_templates(side=False)
    if side_templates is None:
        side_templates = _load_score_digit_templates(side=True)

    panel = _crop(img, CENTER_PANEL)
    results: list[int | None] = []
    for sub, name, rotation in [
        (SCORE_SELF_SUB,  "self",  None),
        (SCORE_TOP_SUB,   "top",   cv2.ROTATE_180),   # top score is upside-down
        (SCORE_LEFT_SUB,  "left",  None),              # vertical stack, reads top-to-bottom
        (SCORE_RIGHT_SUB, "right", cv2.ROTATE_180),   # right side: upside-down top-to-bottom,
                                                       # rotating 180 makes digits upright and
                                                       # reverses order so string reads correctly
    ]:
        is_side = name in ("left", "right")
        tmpls = side_templates if is_side else digit_templates

        crop = _crop(panel, sub)
        if rotation is not None:
            crop = cv2.rotate(crop, rotation)
        mask = _gold_mask(crop)
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_score_{name}.png", crop)
            cv2.imwrite(f"{debug_prefix}_score_{name}_mask.png", mask)

        # Detect and optionally print boxes for debugging
        boxes = _segment_digits(mask)
        if len(boxes) >= 2:
            med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]
            boxes = [b for b in boxes if b[2] >= med_w * 0.3]

        if debug_prefix and boxes:
            print(f"  score_{name}: {len(boxes)} boxes: {boxes}")
            for bx, by, bw, bh in boxes:
                patch = cv2.resize(mask[by:by+bh, bx:bx+bw], (16, 24)).astype(np.float32)
                scores = {}
                for d, tmpl in tmpls.items():
                    padded = cv2.copyMakeBorder(tmpl.astype(np.float32), 2,2,2,2, cv2.BORDER_CONSTANT)
                    scores[d] = round(float(cv2.matchTemplate(padded, patch, cv2.TM_CCOEFF_NORMED).max()), 3)
                best_d = max(scores, key=scores.__getitem__) if scores else None
                print(f"    ({bx},{by},{bw},{bh}) -> {best_d}({scores.get(best_d, 0):.3f}) | {scores}")

        # Side scores use a lower threshold: their blobs are short (9-12px) and
        # resize to 16x24 with more stretch, reducing cross-correlation scores.
        min_s = 0.35 if is_side else 0.45
        results.append(_match_boxes(mask, boxes, tmpls, min_score=min_s))

    return tuple(results)   # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# 10. Seat wind detection (small badge icons at diamond corners)
# ──────────────────────────────────────────────────────────────────────────────

# Wind letter colours:  East=red, South=yellow/white, West=blue, North=green
# We use template matching on the letter silhouette (teal-ish) inside the badge.
_SEAT_WIND_LETTERS = {"E": "East", "S": "South", "W": "West", "N": "North"}


def _load_seat_wind_templates() -> dict[str, np.ndarray]:
    tmpls: dict[str, np.ndarray] = {}
    if not SEAT_WIND_TMPL_DIR.exists():
        return tmpls
    for p in SEAT_WIND_TMPL_DIR.glob("*.png"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tmpls[p.stem] = img   # "E", "S", "W", "N"
    return tmpls


def _badge_letter_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask isolating the white/bright letter inside a seat-wind badge."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return mask


def calibrate_seat_wind(img: np.ndarray, seat: str, wind_letter: str):
    """
    Save a seat-wind badge template for the given seat.

      seat        : "self" / "top" / "left" / "right"
      wind_letter : "E" / "S" / "W" / "N"
    """
    sub_map = {
        "self":  SEAT_SELF_SUB,
        "top":   SEAT_TOP_SUB,
        "left":  SEAT_LEFT_SUB,
        "right": SEAT_RIGHT_SUB,
    }
    sub = sub_map.get(seat.lower())
    if sub is None:
        print(f"Unknown seat {seat!r}.")
        return

    panel = _crop(img, CENTER_PANEL)
    crop  = _crop(panel, sub)
    mask  = _badge_letter_mask(crop)

    coords = cv2.findNonZero(mask)
    if coords is None:
        print(f"No bright content found in {seat} seat-wind region.")
        cv2.imwrite(f"debug_seat_{seat}.png", crop)
        return

    x, y, bw, bh = cv2.boundingRect(coords)
    normalized = cv2.resize(mask[y:y + bh, x:x + bw], (24, 24))
    SEAT_WIND_TMPL_DIR.mkdir(parents=True, exist_ok=True)
    out = SEAT_WIND_TMPL_DIR / f"{wind_letter.upper()}.png"
    cv2.imwrite(str(out), normalized)
    print(f"Saved seat-wind template '{wind_letter}' -> {out}")


def recognize_seat_winds(
    img: np.ndarray,
    seat_templates: dict[str, np.ndarray] | None = None,
    debug_prefix: str | None = None,
) -> dict[str, str | None]:
    """
    Detect seat winds for all four players from the centre panel badges.
    Returns {"self": "North"|None, "top": ..., "left": ..., "right": ...}.
    """
    if seat_templates is None:
        seat_templates = _load_seat_wind_templates()

    panel = _crop(img, CENTER_PANEL)
    winds: dict[str, str | None] = {}

    for seat, sub in [
        ("self",  SEAT_SELF_SUB),
        ("top",   SEAT_TOP_SUB),
        ("left",  SEAT_LEFT_SUB),
        ("right", SEAT_RIGHT_SUB),
    ]:
        crop = _crop(panel, sub)
        mask = _badge_letter_mask(crop)

        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_seat_{seat}.png", crop)

        if not seat_templates:
            winds[seat] = None
            continue

        coords = cv2.findNonZero(mask)
        if coords is None:
            winds[seat] = None
            continue

        x, y, bw, bh = cv2.boundingRect(coords)
        query = cv2.resize(mask[y:y + bh, x:x + bw], (24, 24)).astype(np.float32)

        best_letter, best_score = None, -1.0
        for letter, tmpl in seat_templates.items():
            padded = cv2.copyMakeBorder(
                tmpl.astype(np.float32), 2, 2, 2, 2, cv2.BORDER_CONSTANT
            )
            score = float(cv2.matchTemplate(padded, query, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_score, best_letter = score, letter

        if best_letter and best_score >= 0.55:
            winds[seat] = _SEAT_WIND_LETTERS.get(best_letter)
        else:
            winds[seat] = None

    return winds


# ──────────────────────────────────────────────────────────────────────────────
# 11. Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def recognize_game_state(
    screenshot_path: str,
    debug: bool = False,
) -> GameState:
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"Error: could not load '{screenshot_path}'")
        sys.exit(1)

    dbg = str(Path(screenshot_path).stem) if debug else None

    label_templates = load_label_templates()
    round_templates = _load_round_templates()
    wall_templates  = _load_wall_templates()
    score_templates      = _load_score_digit_templates(side=False)
    score_side_templates = _load_score_digit_templates(side=True)
    seat_templates       = _load_seat_wind_templates()

    state = GameState()

    # --- Discard piles ---
    # self=0°, top=180°, left=270° (CCW), right=90° (CW)
    state.self_state.discards  = recognize_discards(
        img, SELF_DISCARD,  0,   label_templates, dbg and f"{dbg}_self")
    state.top_state.discards   = recognize_discards(
        img, TOP_DISCARD,   180, label_templates, dbg and f"{dbg}_top")
    state.left_state.discards  = recognize_discards(
        img, LEFT_DISCARD,  270, label_templates, dbg and f"{dbg}_left")
    state.right_state.discards = recognize_discards(
        img, RIGHT_DISCARD, 90,  label_templates, dbg and f"{dbg}_right")

    # --- All player melds ---
    state.self_state.melds  = recognize_melds(
        img, SELF_MELD,  0,   label_templates, dbg and f"{dbg}_self")
    state.top_state.melds   = recognize_melds(
        img, TOP_MELD,   180, label_templates, dbg and f"{dbg}_top")
    state.left_state.melds  = recognize_melds(
        img, LEFT_MELD,  270, label_templates, dbg and f"{dbg}_left")
    state.right_state.melds = recognize_melds(
        img, RIGHT_MELD, 90,  label_templates, dbg and f"{dbg}_right")

    # --- Riichi ---
    state.self_state.riichi  = detect_riichi(img, SELF_RIICHI)
    state.top_state.riichi   = detect_riichi(img, TOP_RIICHI)
    state.left_state.riichi  = detect_riichi(img, LEFT_RIICHI)
    state.right_state.riichi = detect_riichi(img, RIGHT_RIICHI)

    # --- Dora ---
    state.dora_indicators, state.doras = recognize_dora(
        img, label_templates, dbg)

    # --- Round info ---
    state.round_wind, state.round_number = recognize_round_info(
        img, round_templates, dbg)

    # --- Wall count ---
    state.wall_count = recognize_wall_count(img, wall_templates, dbg)

    # --- Player scores ---
    (state.self_state.score,
     state.top_state.score,
     state.left_state.score,
     state.right_state.score) = recognize_scores(
        img, score_templates, score_side_templates, dbg)

    # --- Seat winds ---
    seat_winds = recognize_seat_winds(img, seat_templates, dbg)
    state.self_state.seat_wind  = seat_winds.get("self")
    state.top_state.seat_wind   = seat_winds.get("top")
    state.left_state.seat_wind  = seat_winds.get("left")
    state.right_state.seat_wind = seat_winds.get("right")

    return state


def print_state(gs: GameState):
    def _fmt(tiles):
        return "[" + ", ".join(str(t) if t else "???" for t in tiles) + "]"

    print(f"\n{'='*62}")
    print(f"  Round : {gs.round_wind or '?'} {gs.round_number or '?'}")
    print(f"  Wall  : {gs.wall_count if gs.wall_count is not None else '?'} tiles")
    print(f"  Dora  : {_fmt(gs.doras)}  (indicators: {_fmt(gs.dora_indicators)})")
    print(f"{'='*62}")
    for seat, ps in [
        ("Self",  gs.self_state),
        ("Top",   gs.top_state),
        ("Left",  gs.left_state),
        ("Right", gs.right_state),
    ]:
        riichi    = " [RIICHI]" if ps.riichi else ""
        wind_str  = f"  seat={ps.seat_wind}" if ps.seat_wind else ""
        score_str = f"  score={ps.score}" if ps.score is not None else ""
        print(f"  {seat:5s}{riichi}{wind_str}{score_str}")
        if ps.melds:
            melds_str = "  ".join(str(m) for m in ps.melds)
            print(f"         melds:    {melds_str}")
        print(f"         discards: {_fmt(ps.discards)}")
    print(f"{'='*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Region overlay (diagnostic)
# ──────────────────────────────────────────────────────────────────────────────

def show_regions(img: np.ndarray, out_path: str = "debug_regions.png"):
    """
    Draw every tracked region onto the image so you can visually verify
    the coordinate constants before running full recognition.
    Saves the annotated image to out_path.
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    def draw(region, colour, label):
        y1 = int(h * region["y_start"]); y2 = int(h * region["y_end"])
        x1 = int(w * region["x_start"]); x2 = int(w * region["x_end"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(vis, label, (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    draw(SELF_DISCARD,  (0, 255,   0), "self disc")
    draw(TOP_DISCARD,   (0, 200, 255), "top disc")
    draw(LEFT_DISCARD,  (255, 180,  0), "left disc")
    draw(RIGHT_DISCARD, (200,   0, 255), "right disc")

    draw(TOP_MELD,   (0, 200, 255), "top meld")
    draw(LEFT_MELD,  (255, 180,  0), "left meld")
    draw(RIGHT_MELD, (200,   0, 255), "right meld")

    draw(SELF_RIICHI,  (0, 255,   0), "self riichi")
    draw(TOP_RIICHI,   (0, 200, 255), "top riichi")
    draw(LEFT_RIICHI,  (255, 180,  0), "left riichi")
    draw(RIGHT_RIICHI, (200,   0, 255), "right riichi")

    draw(DORA_REGION,  (0, 255, 255), "dora")
    draw(CENTER_PANEL, (255, 255,  0), "center")

    # Also show sub-regions inside the centre panel
    cp_y1 = int(h * CENTER_PANEL["y_start"]); cp_x1 = int(w * CENTER_PANEL["x_start"])
    cp_h  = int(h * (CENTER_PANEL["y_end"] - CENTER_PANEL["y_start"]))
    cp_w  = int(w * (CENTER_PANEL["x_end"] - CENTER_PANEL["x_start"]))
    for sub, colour, label in [
        (ROUND_TEXT_SUB,  (255, 255,   0), "round"),
        (WALL_COUNT_SUB,  (255, 200,   0), "wall"),
        (SCORE_SELF_SUB,  (180, 255, 180), "sc self"),
        (SCORE_TOP_SUB,   (180, 255, 255), "sc top"),
        (SCORE_LEFT_SUB,  (255, 200, 180), "sc left"),
        (SCORE_RIGHT_SUB, (200, 180, 255), "sc right"),
        (SEAT_SELF_SUB,   (100, 255, 100), "seat self"),
        (SEAT_TOP_SUB,    (100, 255, 255), "seat top"),
        (SEAT_LEFT_SUB,   (255, 180, 100), "seat left"),
        (SEAT_RIGHT_SUB,  (180, 100, 255), "seat right"),
    ]:
        sy1 = cp_y1 + int(cp_h * sub["y_start"]); sy2 = cp_y1 + int(cp_h * sub["y_end"])
        sx1 = cp_x1 + int(cp_w * sub["x_start"]); sx2 = cp_x1 + int(cp_w * sub["x_end"])
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), colour, 1)
        cv2.putText(vis, label, (sx1 + 2, sy1 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"Region overlay saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mahjong Soul full board state recogniser."
    )
    parser.add_argument("screenshot", help="Path to screenshot image")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug crops for every region")
    parser.add_argument("--show-regions", action="store_true",
                        help="Draw all region boxes on the image and save debug_regions.png")
    parser.add_argument("--calibrate-round", metavar="LABEL",
                        help="Save round-wind template (e.g. 'East1', 'South2')")
    parser.add_argument("--calibrate-wall", metavar="COUNT", type=int,
                        help="Save wall-count digit templates for the given tile count")
    parser.add_argument("--calibrate-score", nargs=2, metavar=("SEAT", "VALUE"),
                        help="Save score digit templates. SEAT=self/top/left/right, "
                             "VALUE=score string e.g. '25000'")
    parser.add_argument("--calibrate-seat", nargs=2, metavar=("SEAT", "LETTER"),
                        help="Save seat-wind badge template. SEAT=self/top/left/right, "
                             "LETTER=E/S/W/N")
    args = parser.parse_args()

    src = cv2.imread(args.screenshot)
    if src is None:
        print(f"Cannot load {args.screenshot!r}")
        sys.exit(1)

    if args.show_regions:
        show_regions(src)
    elif args.calibrate_round:
        calibrate_round(src, args.calibrate_round)
    elif args.calibrate_wall is not None:
        calibrate_wall_count(src, args.calibrate_wall)
    elif args.calibrate_score:
        seat, value = args.calibrate_score
        calibrate_score_digits(src, value, seat, debug=args.debug)
    elif args.calibrate_seat:
        seat, letter = args.calibrate_seat
        calibrate_seat_wind(src, seat, letter)
    else:
        gs = recognize_game_state(args.screenshot, debug=args.debug)
        print_state(gs)
