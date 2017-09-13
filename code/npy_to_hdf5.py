'''
Utility script to dump arrays stored in NumPy .npy files into HDF5 .h5 files
'''

import numpy as np
import h5py
import argparse
import os


def main(input_path, output_path):
    if output_path is None or len(output_path) == 0:
        input_path_noext, _ = os.path.splitext(input_path)
        output_path = input_path_noext + '.h5'
    print('Input .npy file: %s' % input_path)
    print('Output .h5 file: %s' % output_path)

    # Read the input file
    arr = np.load(input_path, mmap_mode='r')

    # Write to h5
    f = h5py.File(output_path, 'w')
    dset = f.create_dataset('data', arr.shape, dtype=arr.dtype)
    print('Now writing to .h5 file')
    dset[...] = arr
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to the .npy file to convert')
    parser.add_argument('--output_path', type=str, help='Path to the .h5 file to save to. By default, save to the same path as the input, but replacing ".npy" with ".h5"')
    args = parser.parse_args()

    main(**vars(args))