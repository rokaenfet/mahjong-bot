"""
Mahjong Soul tile recognizer — MVP.

Workflow:
  1. Run with --build-templates on a screenshot to segment hand tiles
     and save them as numbered images for manual labeling.
  2. Move/copy each saved tile image into templates/<shorthand>.png
     (e.g. templates/1m.png, templates/5p.png, templates/chun.png).
  3. Run normally to recognize tiles via template matching.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tiles import Tile, Suit, ALL_TILES, TILE_LOOKUP
from label_ocr import recognize_tile, calibrate, load_label_templates

# ---------------------------------------------------------------------------
# Configuration — tweak these for your resolution / game window
# ---------------------------------------------------------------------------
# Percentages of the full screenshot that bound the player's hand area.
# x_end stops before the meld area so called tiles don't bleed in.
HAND_REGION = {
    "y_start": 0.855,
    "y_end": 0.965,
    "x_start": 0.10,
    "x_end": 0.82,
}

# Region where the player's called sets (pon/chii/kan) are displayed.
MELD_REGION = {
    "y_start": 0.855,
    "y_end": 0.985,
    "x_start": 0.82,
    "x_end": 0.97,
}

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Minimum match score to accept a template hit (0-1, higher = stricter)
MATCH_THRESHOLD = 0.75

# Tile segmentation parameters
TILE_BRIGHTNESS_THRESH = 160   # grayscale threshold for tile face
MIN_TILE_AREA = 400            # ignore contours smaller than this
TILE_ASPECT_LO = 1.0           # min height/width ratio
TILE_ASPECT_HI = 2.2           # max height/width ratio


# ---------------------------------------------------------------------------
# Hand region extraction
# ---------------------------------------------------------------------------
def crop_hand(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y1 = int(h * HAND_REGION["y_start"])
    y2 = int(h * HAND_REGION["y_end"])
    x1 = int(w * HAND_REGION["x_start"])
    x2 = int(w * HAND_REGION["x_end"])
    return img[y1:y2, x1:x2]


def count_meld_tiles(img: np.ndarray) -> int:
    """Count tiles in the player's exposed meld area (right of the hand).

    Returns the total number of meld tiles (always a multiple of 3 or 4).
    0 means no calls have been made.
    """
    h, w = img.shape[:2]
    y1 = int(h * MELD_REGION["y_start"])
    y2 = int(h * MELD_REGION["y_end"])
    x1 = int(w * MELD_REGION["x_start"])
    x2 = int(w * MELD_REGION["x_end"])
    meld_crop = img[y1:y2, x1:x2]

    mh, mw = meld_crop.shape[:2]
    blur_k = max(5, int(mh * 0.10)) | 1
    blurred = cv2.GaussianBlur(meld_crop, (blur_k, blur_k), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray_raw = cv2.cvtColor(meld_crop, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, TILE_BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)
    erode_w = max(3, int(mw * 0.004))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_w, 1))
    mask = cv2.erode(mask, h_kernel, iterations=2)
    mask = cv2.dilate(mask, h_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        # Meld tiles may be rotated so accept a wider aspect range
        aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 0
        if area >= MIN_TILE_AREA and 1.0 <= aspect <= 2.5:
            mean_brightness = cv2.mean(gray_raw[y:y + bh, x:x + bw])[0]
            if mean_brightness >= 140:
                tiles.append((x, y, bw, bh))

    count = _filter_slivers(tiles)
    print(f"Meld area: {len(count)} tile(s) detected")
    return len(count)


def split_hand_and_drawn(
    boxes: list[tuple[int, int, int, int]], expected_hand: int
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int] | None]:
    """Split detected boxes into main hand tiles and the drawn tile.

    Uses the expected hand size to find where the main hand ends, then
    looks for a single tile separated by a gap as the drawn tile.
    Returns (hand_boxes, drawn_box_or_None).
    """
    if not boxes:
        return [], None

    # Cap total candidates at expected_hand + 1 drawn to discard any meld bleed-in
    boxes = boxes[:expected_hand + 1]

    hand_boxes = boxes[:expected_hand]
    remainder = boxes[expected_hand:]

    if not remainder:
        return hand_boxes, None

    # The drawn tile should be separated from the main hand by a visible gap.
    if len(hand_boxes) > 0:
        median_w = sorted(b[2] for b in hand_boxes)[len(hand_boxes) // 2]
        last_hand_right = hand_boxes[-1][0] + hand_boxes[-1][2]
        gap = remainder[0][0] - last_hand_right
        if gap >= median_w * 0.3:
            return hand_boxes, remainder[0]

    return hand_boxes, None


# ---------------------------------------------------------------------------
# Tile segmentation
# ---------------------------------------------------------------------------
def segment_tiles(hand_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, w, h) for each tile, sorted left-to-right.

    Strategy: heavy blur + fixed threshold to find cleanly-separated tiles via
    contour detection, then fill in any wide gaps where tiles were missed.
    Falls back to uniform splitting when contour detection fails.
    """
    h, w = hand_img.shape[:2]

    # --- 1. Blur to merge artwork into bright blobs (scaled to crop size) ---
    blur_k = max(5, int(h * 0.10)) | 1
    blurred = cv2.GaussianBlur(hand_img, (blur_k, blur_k), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # --- 2. Threshold to isolate bright tile regions ---
    _, mask = cv2.threshold(gray, TILE_BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)

    # --- 3. Erode horizontally to break connections between tiles ---
    erode_w = max(3, int(w * 0.004))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_w, 1))
    mask = cv2.erode(mask, h_kernel, iterations=2)
    mask = cv2.dilate(mask, h_kernel, iterations=1)

    # Unblurred grayscale for brightness checks on raw pixel data
    gray_raw = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

    # --- 4. Find tile-shaped contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bh / bw if bw > 0 else 0
        if area >= MIN_TILE_AREA and TILE_ASPECT_LO <= aspect <= TILE_ASPECT_HI:
            mean_brightness = cv2.mean(gray_raw[y:y + bh, x:x + bw])[0]
            if mean_brightness >= 140:
                tiles.append((x, y, bw, bh))

    tiles.sort(key=lambda b: b[0])
    tiles = _split_wide_tiles(tiles)
    tiles = _normalize_heights(tiles, h)
    tiles = _deduplicate_tiles(tiles)
    tiles = _filter_slivers(tiles)

    # --- 5. If enough tiles found, fill gaps (brightness-checked) ---
    if len(tiles) >= 8:
        return _fill_gaps(tiles, gray_raw)

    # --- 6. Retry with lower threshold for darker themes ---
    _, mask2 = cv2.threshold(gray, TILE_BRIGHTNESS_THRESH - 30, 255, cv2.THRESH_BINARY)
    mask2 = cv2.erode(mask2, h_kernel, iterations=2)
    mask2 = cv2.dilate(mask2, h_kernel, iterations=1)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles2 = []
    for cnt in contours2:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / bw if bw > 0 else 0
        if bw * bh >= MIN_TILE_AREA and TILE_ASPECT_LO <= aspect <= TILE_ASPECT_HI:
            mean_brightness = cv2.mean(gray_raw[y:y + bh, x:x + bw])[0]
            if mean_brightness >= 140:
                tiles2.append((x, y, bw, bh))
    tiles2.sort(key=lambda b: b[0])
    tiles2 = _split_wide_tiles(tiles2)
    tiles2 = _normalize_heights(tiles2, h)
    tiles2 = _deduplicate_tiles(tiles2)
    tiles2 = _filter_slivers(tiles2)

    if len(tiles2) >= 8:
        return _fill_gaps(tiles2, gray_raw)

    # --- 7. Last resort: uniform split ---
    return _uniform_split(hand_img)


def _deduplicate_tiles(tiles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Remove tiles that substantially overlap an already-accepted tile.

    When a single tile produces multiple bright blobs (e.g. upper and lower
    circle clusters), each blob gets normalized to full height and they end
    up at the same x position. Keep only the widest one.
    """
    result: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in tiles:
        keep = True
        for i, (rx, ry, rbw, rbh) in enumerate(result):
            overlap = max(0, min(x + bw, rx + rbw) - max(x, rx))
            if overlap > min(bw, rbw) * 0.5:
                # Overlapping — keep the wider tile
                if bw > rbw:
                    result[i] = (x, y, bw, bh)
                keep = False
                break
        if keep:
            result.append((x, y, bw, bh))
    return result


def _normalize_heights(
    tiles: list[tuple[int, int, int, int]], crop_h: int
) -> list[tuple[int, int, int, int]]:
    """Extend blobs that are shorter than the reference tile height to the full
    crop height, so the label region (top-right corner) is always included.

    Tiles like 1p have a large dark circle that makes the bright blob cover
    only the lower portion of the face; without this fix the label is cropped off.
    """
    if not tiles:
        return tiles
    ref_h = max(t[3] for t in tiles)
    result = []
    for x, y, bw, bh in tiles:
        if bh < ref_h * 0.8:
            result.append((x, 0, bw, crop_h))
        else:
            result.append((x, y, bw, bh))
    return result


def _split_wide_tiles(tiles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Split any box that is much wider than the median into equal sub-tiles.

    Handles cases like 1p (large dark circle) merging with an adjacent tile
    into one wide contour.
    """
    if len(tiles) < 2:
        return tiles
    median_w = sorted(t[2] for t in tiles)[len(tiles) // 2]
    result = []
    for x, y, bw, bh in tiles:
        n = round(bw / median_w)
        if n >= 2:
            sub_w = bw // n
            for i in range(n):
                result.append((x + i * sub_w, y, sub_w, bh))
        else:
            result.append((x, y, bw, bh))
    return result


def _filter_slivers(tiles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Drop candidates whose width is much smaller than the median tile width."""
    if len(tiles) < 3:
        return tiles
    median_w = sorted(t[2] for t in tiles)[len(tiles) // 2]
    tiles = [t for t in tiles if t[2] >= median_w * 0.4]
    # A hand is at most 14 tiles; if more survive, keep the 14 widest
    if len(tiles) > 14:
        tiles = sorted(tiles, key=lambda t: t[2], reverse=True)[:14]
        tiles.sort(key=lambda t: t[0])
    return tiles


def _fill_gaps(
    tiles: list[tuple[int, int, int, int]],
    gray_raw: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """Fill wide gaps between detected tiles with evenly-spaced boxes.

    Only fills a gap if the gap region is bright (mean >= 140), indicating
    a real tile is present but was missed by the contour detector (e.g. 1p
    whose dark circles prevent a bright blob from forming). Dark gaps are
    skipped so phantom tiles are not inserted at the end of the hand.
    """
    if len(tiles) < 2:
        return tiles

    median_w = sorted(t[2] for t in tiles)[len(tiles) // 2]
    filled = list(tiles)

    i = 0
    while i < len(filled) - 1:
        right_of_curr = filled[i][0] + filled[i][2]
        left_of_next = filled[i + 1][0]
        gap = left_of_next - right_of_curr

        if gap > median_w * 1.1:
            # Check if the gap region contains a bright tile face.
            # Use only the top 35% of the crop (where the label lives) to avoid
            # false rejection from tiles like 1p that have large dark circles.
            if gray_raw is not None:
                gap_col_start = max(0, right_of_curr)
                gap_col_end = min(gray_raw.shape[1], left_of_next)
                if gap_col_end > gap_col_start:
                    top_rows = max(1, gray_raw.shape[0] * 35 // 100)
                    gap_region = gray_raw[:top_rows, gap_col_start:gap_col_end]
                    if gap_region.mean() < 140:
                        i += 1
                        continue

            n_missing = max(1, round(gap / median_w))
            sub_w = gap // n_missing
            ref_y = filled[i][1]
            ref_h = filled[i][3]
            for j in range(n_missing):
                x = right_of_curr + j * sub_w
                filled.insert(i + 1 + j, (x, ref_y, sub_w, ref_h))
            i += n_missing + 1
        else:
            i += 1

    return filled


def _uniform_split(hand_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback: split the hand into ~13 tiles based on the bright strip."""
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_mean = gray.mean(axis=1)
    bright_rows = np.where(row_mean > TILE_BRIGHTNESS_THRESH * 0.55)[0]
    if len(bright_rows) < 5:
        return []
    tile_top = int(bright_rows[0])
    tile_bot = int(bright_rows[-1])
    tile_h = tile_bot - tile_top

    col_mean = gray[tile_top:tile_bot, :].mean(axis=0)
    bright_cols = np.where(col_mean > TILE_BRIGHTNESS_THRESH * 0.55)[0]
    if len(bright_cols) < 10:
        return []
    strip_left = int(bright_cols[0])
    strip_right = int(bright_cols[-1])
    strip_w = strip_right - strip_left

    est_tile_w = tile_h / 1.25  # includes border/gap between tiles
    n_tiles = max(1, round(strip_w / est_tile_w))
    n_tiles = min(n_tiles, 14)  # a standard hand is 13 tiles + 1 drawn
    tile_w = strip_w // n_tiles

    return [(strip_left + i * tile_w, tile_top, tile_w, tile_h) for i in range(n_tiles)]


def extract_tile_images(
    hand_img: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
    return [hand_img[y : y + h, x : x + w] for x, y, w, h in boxes]


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------
def load_templates() -> dict[str, np.ndarray]:
    """Load all template images from the templates/ directory."""
    templates: dict[str, np.ndarray] = {}
    if not TEMPLATE_DIR.exists():
        return templates
    for file in TEMPLATE_DIR.iterdir():
        if file.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img = cv2.imread(str(file))
            if img is not None:
                name = file.stem  # e.g. "1m", "5p", "chun"
                templates[name] = img
    return templates


def match_tile(
    tile_img: np.ndarray, templates: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    """Match a single tile image against all templates. Returns (name, score)."""
    best_name = None
    best_score = -1.0

    for name, tmpl in templates.items():
        resized = cv2.resize(tmpl, (tile_img.shape[1], tile_img.shape[0]))
        result = cv2.matchTemplate(tile_img, resized, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Color-based suit heuristic (fallback when no templates exist)
# ---------------------------------------------------------------------------
def guess_suit(tile_img: np.ndarray) -> str:
    """Rough color-based guess at the tile's suit."""
    hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sat_mask = s > 60

    green_mask = sat_mask & (h >= 35) & (h <= 85)
    red_mask = sat_mask & ((h <= 10) | (h >= 170))
    blue_mask = sat_mask & (h >= 90) & (h <= 130)

    green_pct = green_mask.sum() / sat_mask.size
    red_pct = red_mask.sum() / sat_mask.size
    blue_pct = blue_mask.sum() / sat_mask.size

    scores = {"sou": green_pct, "man": red_pct, "pin": blue_pct}
    best = max(scores, key=scores.get)

    if scores[best] < 0.02:
        return "honor"
    return best


# ---------------------------------------------------------------------------
# Sort-order filter
# ---------------------------------------------------------------------------
_SUIT_ORDER = {Suit.MAN: 0, Suit.PIN: 1, Suit.SOU: 2, Suit.WIND: 3, Suit.DRAGON: 4}

def _tile_sort_key(tile: Tile) -> int:
    return _SUIT_ORDER[tile.suit] * 10 + tile.value

# Minimum confidence required to keep a tile that breaks the expected sort order
SORT_CONF_THRESHOLD = 0.70

def filter_by_sort_order(
    boxes: list, labels: list[str], recognized: list, infos: list[dict]
) -> tuple[list, list[str], list, list[dict]]:
    """Drop tiles that break the hand's ascending sort order AND have low confidence."""
    out_boxes, out_labels, out_recognized, out_infos = [], [], [], []
    prev_key = -1
    for box, label, tile, info in zip(boxes, labels, recognized, infos):
        if tile is not None:
            key = _tile_sort_key(tile)
            conf = info.get("confidence", 0)
            if key < prev_key and conf < SORT_CONF_THRESHOLD:
                print(f"  [sort-filter] dropping {label} (key {key} < {prev_key}, conf={conf:.2f})")
                continue
            prev_key = key
        out_boxes.append(box)
        out_labels.append(label)
        out_recognized.append(tile)
        out_infos.append(info)
    return out_boxes, out_labels, out_recognized, out_infos


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------
def draw_debug(
    hand_img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
) -> np.ndarray:
    vis = hand_img.copy()
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(screenshot_path: str, build_templates: bool = False, debug: bool = False,
        calibrate_labels: str = None):
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"Error: could not load '{screenshot_path}'")
        sys.exit(1)

    meld_count = count_meld_tiles(img)
    expected_hand = 13 - meld_count

    hand = crop_hand(img)
    boxes = segment_tiles(hand)
    hand_boxes, drawn_box = split_hand_and_drawn(boxes, expected_hand)

    all_boxes = hand_boxes + ([drawn_box] if drawn_box else [])
    tile_imgs = extract_tile_images(hand, all_boxes)

    print(f"Meld tiles: {meld_count}  |  Expected hand: {expected_hand}  |  "
          f"Detected: {len(hand_boxes)} hand + {1 if drawn_box else 0} drawn\n")

    if calibrate_labels:
        names = [n.strip() for n in calibrate_labels.split(",")]
        if len(names) != len(tile_imgs):
            print(f"Error: got {len(names)} labels but found {len(tile_imgs)} tiles.")
            print("Provide a comma-separated list matching tile count.")
            sys.exit(1)
        calibrate(tile_imgs, names)
        return

    if build_templates:
        out_dir = Path(__file__).parent / "segmented"
        out_dir.mkdir(exist_ok=True)
        for i, t_img in enumerate(tile_imgs):
            out_path = out_dir / f"tile_{i:02d}.png"
            cv2.imwrite(str(out_path), t_img)
            print(f"  Saved {out_path}")
        print(f"\nSegmented tiles saved to {out_dir}/")
        print("Label them by copying into templates/<shorthand>.png")
        print("  e.g.  templates/1m.png, templates/5p.png, templates/chun.png")
        if debug:
            index_labels = [str(i) for i in range(len(all_boxes))]
            vis = draw_debug(hand, all_boxes, index_labels)
            debug_path = Path(__file__).parent / "debug_hand.png"
            cv2.imwrite(str(debug_path), vis)
            print(f"Debug image saved to {debug_path}")
    else:
        label_tmpls = load_label_templates()
        img_templates = load_templates()
        labels: list[str] = []
        recognized: list[Tile] = []

        if label_tmpls:
            print("Using calibrated label recognition:\n")
        else:
            print("Not calibrated yet. Run with --calibrate to set up label recognition.")
            print("Falling back to color heuristic.\n")

        all_tiles: list[Tile | None] = []
        all_infos: list[dict] = []

        for t_img in tile_imgs:
            tile = None
            info = {}

            # Primary: calibrated label matching
            if label_tmpls:
                tile, info = recognize_tile(t_img, label_tmpls)

            # Fallback: image template matching
            if tile is None and img_templates:
                name, score = match_tile(t_img, img_templates)
                if name and name in TILE_LOOKUP:
                    tile = TILE_LOOKUP[name]
                    info = {"label": "tmpl", "confidence": score, "suit_guess": ""}

            all_tiles.append(tile)
            all_infos.append(info)

        # Filter out tiles that break sort order with low confidence
        boxes, labels, all_tiles, all_infos = filter_by_sort_order(
            boxes, ["???" if t is None else str(t) for t in all_tiles], all_tiles, all_infos
        )

        # Neighbor-context suit correction: if a numbered tile's suit doesn't
        # match both adjacent tiles' suit and the neighbors agree, override.
        for i in range(len(all_tiles)):
            tile = all_tiles[i]
            if tile is None or not tile.suit in (Suit.MAN, Suit.PIN, Suit.SOU):
                continue
            neighbors = [all_tiles[j] for j in (i - 1, i + 1)
                         if 0 <= j < len(all_tiles) and all_tiles[j] is not None
                         and all_tiles[j].suit in (Suit.MAN, Suit.PIN, Suit.SOU)]
            if len(neighbors) == 2 and neighbors[0].suit == neighbors[1].suit:
                neighbor_suit = neighbors[0].suit
                if tile.suit != neighbor_suit:
                    corrected = Tile(neighbor_suit, tile.value)
                    print(f"  [suit-correct] tile {i} {tile} -> {corrected} (neighbors agree)")
                    all_tiles[i] = corrected
                    labels[i] = str(corrected)

        # Recover drawn tile by sort order if gap-detection didn't find one.
        # The drawn tile is placed at the far right regardless of hand sort, so
        # it often has a lower sort key than the tile before it.
        drawn_label_recovered = None
        if drawn_box is None and len(all_tiles) >= 2:
            last_tile = all_tiles[-1]
            last_info = all_infos[-1]
            prev_tile = next((t for t in reversed(all_tiles[:-1]) if t is not None), None)

            should_recover = False
            if prev_tile is not None:
                if last_tile is not None:
                    if _tile_sort_key(last_tile) < _tile_sort_key(prev_tile):
                        should_recover = True
                elif last_info.get("label", "").isdigit() and last_info.get("suit_guess") == "honor":
                    # Suit unknown for digit-labeled last tile: assign the suit that
                    # creates the largest sort-key drop (most definitively a drawn tile).
                    value = int(last_info["label"])
                    prev_key = _tile_sort_key(prev_tile)
                    best_suit, best_drop = Suit.MAN, -1
                    for suit in (Suit.MAN, Suit.PIN, Suit.SOU):
                        drop = prev_key - (_SUIT_ORDER[suit] * 10 + value)
                        if drop > best_drop:
                            best_drop, best_suit = drop, suit
                    if best_drop > 0:
                        last_tile = Tile(best_suit, value)
                        all_tiles[-1] = last_tile
                        labels[-1] = str(last_tile)
                        print(f"  [drawn-suit] assigned {last_tile} (max sort-drop from {prev_tile})")
                        should_recover = True

            if should_recover:
                drawn_label_recovered = labels[-1]
                labels = labels[:-1]
                all_tiles = all_tiles[:-1]
                all_infos = all_infos[:-1]
                print(f"  [drawn-recover] {drawn_label_recovered} -> drawn tile\n")

        # Separate hand tiles from the gap-detected drawn tile for display.
        # When drawn_box was found via gap, the last item in all_tiles/labels is drawn.
        if drawn_box is not None and not drawn_label_recovered:
            hand_tiles_display = list(zip(all_tiles[:-1], all_infos[:-1]))
            drawn_display_label = labels[-1] if labels else None
            display_labels = labels[:-1]
        else:
            hand_tiles_display = list(zip(all_tiles, all_infos))
            drawn_display_label = drawn_label_recovered
            display_labels = labels

        for i, (tile, info) in enumerate(hand_tiles_display):
            if tile:
                recognized.append(tile)
                print(f"  Tile {i:2d}: {tile!s:>6}  "
                      f"(label='{info.get('label','')}' conf={info.get('confidence',0):.2f}"
                      f" suit={info.get('suit_guess','')})")
            else:
                print(f"  Tile {i:2d}: ???    ({info})")

        hand_str = f"Hand: [{', '.join(display_labels)}]"
        if drawn_display_label:
            hand_str += f"  +  Drawn: {drawn_display_label}"
        print(f"\n{hand_str}")

        if debug:
            vis = draw_debug(hand, boxes, labels)
            debug_path = Path(__file__).parent / "debug_hand.png"
            cv2.imwrite(str(debug_path), vis)
            print(f"\nDebug image saved to {debug_path}")

    if debug:
        hand_path = Path(__file__).parent / "debug_crop.png"
        cv2.imwrite(str(hand_path), hand)
        print(f"Cropped hand saved to {hand_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize mahjong tiles from a Mahjong Soul screenshot.")
    parser.add_argument("screenshot", help="Path to the screenshot image")
    parser.add_argument("--build-templates", action="store_true",
                        help="Segment tiles and save them for labeling instead of recognizing")
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Comma-separated tile names to calibrate label recognition "
                             "(e.g. '3m,6m,6m,8m,3p,8p,1s,4s,5s,5s,6s,west,north')")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug images showing detected regions")
    args = parser.parse_args()
    run(args.screenshot, build_templates=args.build_templates, debug=args.debug,
        calibrate_labels=args.calibrate)
