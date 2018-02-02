import os
import struct
import numpy as np
from scipy.misc import imsave
import cv2
from PIL import Image

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def mnist_read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


def main():
    splits = ['training', 'testing']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_data_dir = os.path.join(script_dir, 'mnist')

    for split in splits:
        digit_counts = np.zeros(10)
        split_dir = os.path.join(mnist_data_dir, split)
        mnist_iter = mnist_read(split)
        for label, alpha_mask in mnist_iter:
            digit_dir = os.path.join(split_dir, str(label))
            if not os.path.isdir(digit_dir):
                os.makedirs(digit_dir)
            # Convert to RGBA
            _, color_mask = cv2.threshold(alpha_mask, 0, 255, cv2.THRESH_BINARY)
            image = np.stack((color_mask, color_mask, color_mask, alpha_mask), axis=-1)
            image_path = os.path.join(digit_dir, '%04d.png' % digit_counts[label])
            with open(image_path, 'w') as f:
                Image.fromarray(image).save(f)
            digit_counts[label] += 1


if __name__ == '__main__':
    main()