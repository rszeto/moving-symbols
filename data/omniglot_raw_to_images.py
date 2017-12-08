import os
from pprint import pprint

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from PIL import Image

######### Importing .mat files ###############################################
######### Reference: http://stackoverflow.com/a/8832212 ######################

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        # Handle case where elem is an array of mat_structs
        elif isinstance(elem, np.ndarray) and len(elem) > 0 and \
                isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
            dict[strg] = np.array([_todict(subelem) for subelem in elem])
        else:
            dict[strg] = elem
    return dict

######### End importing .mat files ############################################

def main():
    final_size = (28, 28)
    resample_strategy = Image.NEAREST

    training_mat = loadmat('data_background.mat')
    testing_mat = loadmat('data_evaluation.mat')
    complete_alphabet_set = np.concatenate((training_mat['images'], testing_mat['images']))

    for alphabet_index, alphabet_set in enumerate(complete_alphabet_set):
        for letter_index, letter_image_set in enumerate(alphabet_set):
            for image_index, image_np in enumerate(letter_image_set):
                split = 'training' if image_index % 2 == 0 else 'testing'
                save_image_index = image_index / 2 if (image_index % 2 == 0) \
                    else (image_index - 1) / 2
                image_np = 255 * (1 - np.array(image_np, dtype=np.uint8))
                rgba_image_np = np.stack((image_np, image_np, image_np, image_np), axis=-1)
                rgba_image = Image.fromarray(rgba_image_np, mode='RGBA')
                rgba_image = rgba_image.resize(final_size, resample_strategy)
                image_path = os.path.join('omniglot', split,
                                          '%02d_%02d' % (alphabet_index, letter_index),
                                          '%02d.png' % save_image_index)
                if not os.path.isdir(os.path.dirname(image_path)):
                    os.makedirs(os.path.dirname(image_path))
                rgba_image.save(image_path)

if __name__ == '__main__':
    main()