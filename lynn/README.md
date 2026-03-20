# Lynn — Mahjong Soul Image Recognizer

Reads Mahjong Soul screenshots using OpenCV template matching. Identifies tile labels from the top-right corner of each tile, combines with color analysis to determine suit, and extracts full board state including scores, winds, and discard piles.

## Setup

```powershell
python -m venv lynn/venv
.\lynn\venv\Scripts\Activate.ps1
pip install -r lynn/requirements.txt
```

On macOS/Linux: `source lynn/venv/bin/activate`

---

## Usage

### Hand recognition

```bash
python lynn/recognize.py screenshot.png
```

Output:
```
Hand: [1m, 1m, 2m, 3m, 4m, 5m, 7m, 1p, 6s, 8s, 9s, East, West]
Drawn: 5m (red)
```

### Full board state

```bash
python lynn/game_state.py screenshot.png
```

Output:
```
Round : East 1
Wall  : 52 tiles
Dora  : [6m]  (indicators: [5m])

West(self)   score=25000
     discards: [5p, South, Hatsu, 2s]
South        score=25000
     discards: [8s, Chun, 2m, 7m]
North        score=25000
     melds:    (Hatsu Hatsu Hatsu)
     discards: [South, North, South, 4p, 3p]
East         score=25000
     discards: [5m, North, West, 7p]
```

---

## Calibration

Both scripts use template matching and accumulate templates across sessions. Run calibration once whenever you have a screenshot with values not yet covered.

### Hand tiles

```bash
python lynn/recognize.py screenshot.png --calibrate "1m,2m,3m,east,west,hatsu"
```

List tiles left-to-right as they appear. If a drawn tile is present (rightmost, separated by a gap), include it as the 14th entry.

### Round wind

```bash
python lynn/game_state.py screenshot.png --calibrate-round East1
python lynn/game_state.py screenshot.png --calibrate-round South2
```

### Wall count

```bash
python lynn/game_state.py screenshot.png --calibrate-wall 52
```

Replace `52` with the count shown on screen.

### Player scores

```bash
python lynn/game_state.py screenshot.png --calibrate-score self  31000
python lynn/game_state.py screenshot.png --calibrate-score top   31000
python lynn/game_state.py screenshot.png --calibrate-score left  31000
python lynn/game_state.py screenshot.png --calibrate-score right 31000
```

Self and top scores render horizontally. Left and right scores render vertically — they use separate template sets.

### Seat winds

```bash
python lynn/game_state.py screenshot.png --calibrate-seat self  W
python lynn/game_state.py screenshot.png --calibrate-seat top   S
python lynn/game_state.py screenshot.png --calibrate-seat left  N
python lynn/game_state.py screenshot.png --calibrate-seat right E
```

Seat winds rotate each hand, so calibrate from multiple screenshots to cover all 4 letters per seat.

---

## Calibration gaps

| Feature | Calibrated | Missing |
|---------|-----------|---------|
| Wall count digits | 1 2 5 6 7 8 | **0 3 4 9** |
| Score digits (self/top) | 0 2 4 5 7 | **1 3 6 8 9** |
| Score digits (left/right) | 0 2 4 5 | **1 3 6 7 8 9** |
| Round winds | East 1 | **East 2–4, South 1–4, West 1–4** |
| Seat winds | E S W N | more screenshots for robustness |

Score digits 1, 3, 6, 8, 9 appear in scores like 31,000 / 16,800 / 38,500. One screenshot typically covers several digits at once.

---

## Tile shorthands

| Shorthand | Tile |
|-----------|------|
| `1m`–`9m` | Man / Characters (萬) |
| `1p`–`9p` | Pin / Circles (筒) |
| `1s`–`9s` | Sou / Bamboo (索) |
| `east` | East wind (東) |
| `south` | South wind (南) |
| `west` | West wind (西) |
| `north` | North wind (北) |
| `haku` | White dragon (白) |
| `hatsu` | Green dragon (發) |
| `chun` | Red dragon (中) |

---

## File overview

| File | Purpose |
|------|---------|
| `tiles.py` | `Suit` enum, `Tile` dataclass, `ALL_TILES`, `TILE_LOOKUP` |
| `recognize.py` | Hand segmentation and recognition pipeline |
| `label_ocr.py` | Label template matching, suit detection, calibration |
| `game_state.py` | Full board state: discards, scores, dora, winds, wall, seat winds |

### Template directories

| Directory | Contents |
|-----------|----------|
| `label_templates/` | Tile label crops (1–9, E, S, W, N, Ht, Hk, Ch) |
| `label_templates/round/` | Round wind badge templates |
| `label_templates/wall/` | Wall count digit templates (d0–d9) |
| `label_templates/score_digits/` | Score digit templates, horizontal (self/top) |
| `label_templates/score_digits_side/` | Score digit templates, vertical (left/right) |
| `label_templates/seat_wind/` | Seat wind badge templates (E, S, W, N) |

---

## How it works

### Hand segmentation (`recognize.py`)

1. Threshold + horizontal erosion separates adjacent tiles into blobs
2. Contour detection filters by aspect ratio; gaps between detected tiles fill in missed ones
3. A gap between the last two tiles triggers drawn-tile detection
4. Fallback: uniform split into 13–14 tiles if contour detection fails

### Tile recognition (`label_ocr.py`)

1. Crop the top-right corner of each tile and match against calibrated label templates
2. Run color analysis on the tile body to determine suit: green = sou, red = man, high-dark-ratio = pin, neither = honor
3. Red five detection: HSV red-channel check on any tile labeled 5m/5p/5s

### Board state (`game_state.py`)

Each board region is cropped using screen-fraction constants, then processed independently.

- **Discards**: multi-row grid segmentation, same label+suit pipeline as hand tiles
- **Dora**: same pipeline; `indicator_to_dora()` maps each indicator to its dora
- **Scores**: gold-pixel (HSV) mask, blob detection, digit template matching
- **Round/wall**: teal-pixel mask, blob detection, template matching
- **Seat winds**: badge crops at the 4 corners of the centre octagonal indicator, non-blue mask, template matching

Rotations before processing: top player 180°, left player 270° CCW, right player 90° CW.
