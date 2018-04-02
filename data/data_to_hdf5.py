"""Save a data folder as an HDF5 file.

@author Ryan Szeto
"""

import argparse
import os

import h5py
import numpy as np
from PIL import Image


def main(data_dir, h5_dest):
    """Store the images in `data_dir` to `h5_dest`, preserving folder hierarchy.

    @param data_dir: The root folder for the data
    @param h5_dest: The h5 path to save all data to
    """

    # Get splits
    splits = os.listdir(data_dir)
    # Get image classes
    image_classes = sorted(os.listdir(os.path.join(data_dir, splits[0])))
    # Get image size (in RGBA mode)
    image_dir = os.path.join(data_dir, splits[0], image_classes[0])
    first_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
    first_image = Image.open(first_image_path).convert('RGBA')
    first_image_np = np.array(first_image)
    image_size = first_image_np.shape

    with h5py.File(h5_dest, 'w') as f:
        # Store image classes as an attribute
        f.attrs['image_classes'] = image_classes

        for split in splits:
            print('Processing %s split' % split)
            split_group = f.create_group(split)
            for image_class in image_classes:
                print('Processing %s image class...' % image_class),
                # Get names of images in this class
                image_names = sorted(os.listdir(os.path.join(data_dir, split, image_class)))
                # Create dataset
                dataset_size = tuple([(len(image_names))] + list(image_size))
                image_class_dataset = split_group.create_dataset(image_class, dataset_size,
                                                                 dtype=np.uint8)
                for i, image_name in enumerate(image_names):
                    # Save image to h5_dest
                    image = Image.open(os.path.join(data_dir, split, image_class,
                                                    image_name)).convert('RGBA')
                    image_class_dataset[i] = np.array(image)
                print('done.')
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('h5_dest', type=str)

    args = parser.parse_args()
    main(**vars(args))