#!/bin/sh

OLD_DIR=`pwd`
SCRIPT_DIR=`cd "$(dirname "$0")"; pwd`

cd "$SCRIPT_DIR"
echo "Downloading data..."
# TODO: Download icons8 data. Store in folder "icons8_raw"
echo "Just kidding. You should already have the icons8_raw folder in here!"

# Generate small icon8 images
echo "Processing images"
python icon8_raw_to_images.py

# Remove raw files
rm -r "icons8_raw"
echo "Done."
cd "$OLD_DIR"