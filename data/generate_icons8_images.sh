#!/bin/sh

OLD_DIR=`pwd`
SCRIPT_DIR=`cd "$(dirname "$0")"; pwd`

cd "$SCRIPT_DIR"
echo "Downloading data..."
wget http://web.eecs.umich.edu/~szetor/media/icons8.zip
unzip icons8.zip -d icons8_raw

# Generate small icon8 images
echo "Processing images"
python icons8_raw_to_images.py

# Remove raw files
rm -r "icons8_raw" "icons8.zip"
echo "Done."
cd "$OLD_DIR"