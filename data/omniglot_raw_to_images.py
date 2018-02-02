import os
from pprint import pprint
import shutil

import numpy as np
import scipy.io as spio
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
    # Clear mats for responsive debugging :)
    del training_mat
    del testing_mat

    for alphabet_index, alphabet_set in enumerate(complete_alphabet_set):
        for letter_index, letter_image_set in enumerate(alphabet_set):
            # Make folders
            if not os.path.isdir('omniglot/training/%02d_%02d' % (alphabet_index, letter_index)):
                os.makedirs('omniglot/training/%02d_%02d' % (alphabet_index, letter_index))
            if not os.path.isdir('omniglot/testing/%02d_%02d' % (alphabet_index, letter_index)):
                os.makedirs('omniglot/testing/%02d_%02d' % (alphabet_index, letter_index))
            # Keep track of how many images were generated
            num_training_images = 0
            num_testing_images = 0
            for image_index, image_np in enumerate(letter_image_set):
                split, save_image_index = ('training', num_training_images) \
                    if image_index % 2 == 0 else ('testing', num_testing_images)
                image_np = 255 * (1 - np.array(image_np, dtype=np.uint8))
                rgba_image_np = np.stack((image_np, image_np, image_np, image_np), axis=-1)
                rgba_image = Image.fromarray(rgba_image_np, mode='RGBA')
                rgba_image = rgba_image.resize(final_size, resample_strategy)
                # Skip saving if number of pixels in final image is too small
                if len(np.nonzero(np.array(rgba_image)[:, :, 3])[0]) < 28:
                    print('Alphabet %d, letter %d, image %d has too few white pixels. Skipping'
                          % (alphabet_index, letter_index, image_index))
                    continue
                image_path = os.path.join('omniglot', split,
                                          '%02d_%02d' % (alphabet_index, letter_index),
                                          '%02d.png' % save_image_index)
                rgba_image.save(image_path)
                # Update image counts
                if split == 'training':
                    num_training_images += 1
                else:
                    num_testing_images += 1

    # Some letters will have no images remaining, so clear those folders
    folders_to_remove = []
    for folder in os.listdir(os.path.join('omniglot', 'training')):
        if len(os.listdir(os.path.join('omniglot', 'training', folder))) == 0:
            folders_to_remove.append(folder)
    for folder in os.listdir(os.path.join('omniglot', 'testing')):
        if len(os.listdir(os.path.join('omniglot', 'testing', folder))) == 0:
            folders_to_remove.append(folder)

    for folder in folders_to_remove:
        print('Removing empty class %s' % folder)
        if os.path.isdir(os.path.join('omniglot', 'training', folder)):
            shutil.rmtree(os.path.join('omniglot', 'training', folder))
        if os.path.isdir(os.path.join('omniglot', 'testing', folder)):
            shutil.rmtree(os.path.join('omniglot', 'testing', folder))


if __name__ == '__main__':
    main()