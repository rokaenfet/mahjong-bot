# Lynn — Mahjong Soul Tile Recognizer

Recognizes tiles from Mahjong Soul screenshots using OpenCV.
Reads the shorthand labels (1-9, E, S, W, N) in the top-right corner of each tile,
then combines with color analysis to identify the suit.

## Setup

```powershell
# Create venv (skip if already exists)
python -m venv lynn/venv

# Activate
.\lynn\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r lynn/requirements.txt
```

On macOS/Linux, activate with `source lynn/venv/bin/activate` instead.

### Download reference tile images

```bash
python lynn/download_tiles.py
```

Downloads clean tile PNGs from [FluffyStuff/riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles) into `lynn/reference/`.
These serve as a visual reference for all 37 tile types (including red dora).

## Usage

All commands are run from the **project root** with the venv activated.

### Step 1: Segment and view tiles

```bash
python lynn/recognize.py lynn/screenshot.png --build-templates --debug
```

This crops the hand region, isolates individual tiles, and saves them to `lynn/segmented/`.
Open the `debug_hand.png` to verify the segmentation looks correct.

### Step 2: Calibrate (one-time per theme)

Look at your segmented tiles and identify each one left-to-right, then run:

```bash
python lynn/recognize.py lynn/screenshot.png --calibrate "3m,6m,6m,8m,3p,8p,1s,4s,5s,5s,6s,west,north"
```

This extracts the label region from each tile and saves it as a template.
Calibration is **additive** — run it on multiple screenshots to cover more tile values.

| Shorthand | Tile                  |
|-----------|-----------------------|
| `1m`–`9m` | Man / Characters (萬) |
| `1p`–`9p` | Pin / Circles (筒)    |
| `1s`–`9s` | Sou / Bamboo (索)     |
| `east`    | East wind (東)        |
| `south`   | South wind (南)       |
| `west`    | West wind (西)        |
| `north`   | North wind (北)       |
| `haku`    | White dragon (白)     |
| `hatsu`   | Green dragon (發)     |
| `chun`    | Red dragon (中)       |

### Step 3: Recognize tiles

```bash
python lynn/recognize.py lynn/screenshot.png
```

The system matches each tile's label against your calibrated templates and uses
color analysis to determine the suit. Output:

```
Hand: [3m, 6m, 6m, 8m, 3p, 8p, 1s, 4s, 5s, 5s, 6s, West, North]
```

## How It Works

### Segmentation
1. **Blur + threshold** — merges tile artwork into solid bright blobs
2. **Erosion** — horizontal erosion breaks connections between adjacent tiles
3. **Contour detection** — tile-shaped contours are filtered by aspect ratio
4. **Gap filling** — missed tiles (e.g. darker bamboo) are inferred from gaps
5. **Fallback** — uniform split into ~13-14 tiles if contour detection fails

### Recognition
1. **Label matching** — the top-right corner of each tile contains a shorthand
   label (1-9 for numbered tiles, E/S/W/N for winds). These are matched against
   templates extracted during calibration.
2. **Suit detection** — color analysis of the tile body determines the suit:
   - Green dominant → Sou (bamboo)
   - Red dominant with low dark ratio → Man (characters)
   - High dark ratio → Pin (circles)
   - Neither → Honor (winds/dragons)
3. **Fallback** — image template matching if label matching fails

## File Overview

| File | Purpose |
|------|---------|
| `tiles.py` | Tile data model — `Suit` enum, `Tile` dataclass, shorthand parsing |
| `recognize.py` | Screenshot → segmentation → recognition pipeline |
| `label_ocr.py` | Label-based recognition: calibration + matching + suit detection |
| `download_tiles.py` | Downloads reference tile images into `reference/` |
| `reference/` | Clean tile PNGs for visual reference (37 tiles) |
| `templates/` | (Optional) full-tile reference images for template matching fallback |
| `label_templates/` | Calibrated label templates extracted from your screenshots |
| `segmented/` | Auto-generated segmented tile images |

## Tile Data Model

```python
from tiles import Tile, Suit, TILE_LOOKUP

t = Tile.from_shorthand("5p")   # Tile(suit=Suit.PIN, value=5)
print(t)                         # "5p"

west = Tile.from_shorthand("west")
print(west)                      # "West"

# All 34 unique tile types
from tiles import ALL_TILES
```

## Configuration

Edit the constants at the top of `recognize.py` to tune for your screen resolution:

- `HAND_REGION` — percentage bounds of the hand area in the screenshot
- `TILE_BRIGHTNESS_THRESH` — grayscale threshold for detecting tile faces (default 160)
- `MATCH_THRESHOLD` — minimum score (0-1) to accept a template match
