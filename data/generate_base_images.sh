#!/bin/sh

OLD_DIR=`pwd`
SCRIPT_DIR=`cd "$(dirname "$0")"; pwd`

cd "$SCRIPT_DIR"
echo "Downloading data..."
# Download train info
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# Download test info
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# Gunzip
gunzip *.gz

# Save digits to images
echo "Saving image files..."
python mnist_raw_to_images.py

# Remove raw files
rm *-ubyte
echo "Done"
cd "$OLD_DIR"