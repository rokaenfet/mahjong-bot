"""
Wipe corrupted label templates and rebuild from known debug discard tiles.

The old templates were generated with a buggy _normalize_label that used
findNonZero+boundingRect, capturing tile-body artifacts alongside the actual
label character.  The fixed code uses connectedComponentsWithStats to isolate
only the largest component, but that means the OLD stored templates no longer
match the NEW queries.  This script:

  1. Backs up the existing label_templates/ directory
  2. Deletes all label_*.png templates and calibration.json
  3. Re-calibrates from the known ground-truth debug discard tile images

After running this script you should also run the normal calibration workflow
(python recognize.py <screenshot> --calibrate "tiles...") to add templates
from a wider variety of tiles (especially numbers 2-8, and E/S/W winds).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Make sure we can import from the lynn/ directory regardless of cwd
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from label_ocr import calibrate, LABEL_TEMPLATES_DIR, CALIBRATION_FILE

# ---------------------------------------------------------------------------
# Ground-truth mapping:  filename  →  tile shorthand consumed by calibrate()
# ---------------------------------------------------------------------------
KNOWN_TILES: list[tuple[str, str]] = [
    ("dbg_disc_tile_self_0.png",  "hatsu"),
    ("dbg_disc_tile_left_0.png",  "hatsu"),   # second Hatsu example
    ("dbg_disc_tile_right_1.png", "chun"),
    ("dbg_disc_tile_left_1.png",  "chun"),    # second Chun example
    ("dbg_disc_tile_right_0.png", "north"),
    ("dbg_disc_tile_left_2.png",  "north"),   # second North example
    ("dbg_disc_tile_right_2.png", "1s"),
    ("dbg_disc_tile_self_1.png",  "9m"),
    ("dbg_disc_tile_self_2.png",  "9m"),      # second 9 example
]


def backup_old_templates() -> Path | None:
    """Copy the existing label_templates/ to label_templates_backup/."""
    backup = LABEL_TEMPLATES_DIR.parent / "label_templates_backup"
    if LABEL_TEMPLATES_DIR.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(LABEL_TEMPLATES_DIR, backup)
        print(f"Backed up old templates -> {backup}/")
        return backup
    return None


def wipe_label_templates() -> None:
    """Delete all label_*.png files and reset calibration.json.

    Sub-directories (round/, wall/, score_digits/, etc.) are left untouched.
    """
    if not LABEL_TEMPLATES_DIR.exists():
        return

    removed = 0
    for f in LABEL_TEMPLATES_DIR.glob("label_*.png"):
        f.unlink()
        removed += 1
    print(f"Removed {removed} old label template PNG files.")

    if CALIBRATION_FILE.exists():
        CALIBRATION_FILE.unlink()
        print("Removed old calibration.json.")


def main() -> None:
    print("=== Recalibrate label templates from ground-truth debug tiles ===\n")

    # 1. Backup
    backup_old_templates()

    # 2. Wipe
    wipe_label_templates()

    # 3. Collect tile images
    tile_imgs = []
    tile_names = []
    missing = []

    for fname, name in KNOWN_TILES:
        path = HERE / fname
        img = cv2.imread(str(path))
        if img is None:
            missing.append(fname)
            print(f"  WARNING: {fname} not found — skipping")
            continue
        tile_imgs.append(img)
        tile_names.append(name)
        print(f"  Loaded {fname!s:45s} -> {name}")

    if not tile_imgs:
        print("\nERROR: No debug tiles found.  Run game_state.py debug extraction first.")
        sys.exit(1)

    print()

    # 4. Calibrate with the new normalization code
    calibrate(tile_imgs, tile_names)

    if missing:
        print(f"\n  (Skipped {len(missing)} missing files: {missing})")

    print("\nDone!  Next steps:")
    print("  - Run game_state.py on mahjongsoul3.png and check discard recognition.")
    print("  - Add more templates (digits 2-8, E/S/W winds) via:")
    print("      python recognize.py <screenshot.png> --calibrate \"2m,3m,...\"")


if __name__ == "__main__":
    main()
