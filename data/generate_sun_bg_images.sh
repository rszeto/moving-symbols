#!/bin/sh

OLD_DIR=`pwd`
SCRIPT_DIR=`cd "$(dirname "$0")"; pwd`

CATEGORIES="l\living_room b\bedroom k\kitchen b\beach d\dining_room c\castle a\airport_terminal"

# Generate small SUN background images
cd "$SCRIPT_DIR"
python download_sun_bg.py --bg_categories $CATEGORIES
cd "$OLD_DIR"