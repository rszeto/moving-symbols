#!/bin/sh

OLD_DIR=`pwd`
SCRIPT_DIR=`cd "$(dirname "$0")"; pwd`

cd "$SCRIPT_DIR"
wget https://github.com/brendenlake/omniglot/raw/84a36b46aa4e6020925a6e7a7054cf50a0d19d52/matlab/data_background.mat
wget https://github.com/brendenlake/omniglot/raw/84a36b46aa4e6020925a6e7a7054cf50a0d19d52/matlab/data_evaluation.mat

# Save letters to images
echo "Saving image files..."
python omniglot_raw_to_images.py

# Remove raw files
rm *.mat
echo "Done"
cd "$OLD_DIR"