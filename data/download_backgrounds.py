import numpy as np
import urllib2
import os
import scipy.io as spio
from functools import partial
from multiprocessing import Pool
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_URL = 'http://vision.cs.princeton.edu/projects/2010/SUN/urls/SUN397_urls.mat'
DEFAULT_BACKGROUND_CATEGORIES = [
    'c\\crosswalk',
    'g\\gas_station',
    'h\\highway',
    'p\\parking_lot',
    'r\\rainforest',
    't\\toll_plaza',
]
NUM_DOWNLOAD_THREADS = 8

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


def download_numbered_file(t, dir, filename_length=4, ext='jpg'):
    '''
    Download a file
    :param t: Tuple (i, url). Download image from URL to file named after i
    :param dir: The directory to save to
    :param filename_length: The length of the file name before the extension
    :param ext: The extension of the files to save. Does not include the '.'
    :return:
    '''
    i, url = t
    response = urllib2.urlopen(url)
    content = response.read()
    fmt_str = '%%0%dd.%s' % (filename_length, ext)
    with open(os.path.join(dir, fmt_str % i), 'w') as f:
        f.write(content)


def main(bg_categories, num_threads):
    os.chdir(SCRIPT_DIR)
    print('Downloading data...')

    # Download URL file
    mat_save_path = os.path.join(SCRIPT_DIR, 'SUN397_urls.mat')
    if not os.path.exists(mat_save_path):
        response = urllib2.urlopen(MAT_URL)
        content = response.read()
        with open(mat_save_path, 'w') as f:
            f.write(content)
    # Set background directory
    background_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'backgrounds'))

    # Parse URL file
    data = loadmat(mat_save_path)['SUN']
    # Filter to specified background categories
    data = [x for x in data if x.category in bg_categories]
    print('Found %d/%d categories' % (len(data), len(bg_categories)))

    # Start pool
    pool = Pool(NUM_DOWNLOAD_THREADS)

    for category_data in data:
        # Convert backslashes to underscores
        category_name = category_data.category.replace('\\', '_')
        # Make category directory
        cat_dir = os.path.join(background_dir, category_name)
        if not os.path.isdir(cat_dir):
            os.makedirs(cat_dir)
        # Save images
        print('Downloading %d images from category %s' % (len(category_data.images), category_name))
        fn = partial(download_numbered_file, dir=cat_dir)
        pool.map(fn, enumerate(category_data.images))

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bg_categories', type=str, nargs='+', default=DEFAULT_BACKGROUND_CATEGORIES,
                        help='SUN 397 categories to download. Backslashes must be escaped')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of download threads')

    args = parser.parse_args()
    main(**vars(args))