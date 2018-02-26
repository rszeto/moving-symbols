import numpy as np
import urllib2
import os
import scipy.io as spio
from functools import partial
import multiprocessing
import argparse
from PIL import Image
from StringIO import StringIO
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_URL = 'http://vision.cs.princeton.edu/projects/2010/SUN/urls/SUN397_urls.mat'

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


def download_numbered_file((url, category_name, is_train), dataset_root):
    '''
    Download a file
    '''
    # Get file byte string
    try:
        response = urllib2.urlopen(url)
        content = response.read()
        # Convert to Image via string buffer
        buff = StringIO()
        buff.write(content)
        buff.seek(0)
        image = Image.open(buff)
        # Resize image
        image = image.resize((64, 64), Image.BICUBIC)
        # Convert to RGB
        image = image.convert('RGB')
        # Save resized image
        with open(os.path.join(dataset_root, 'training' if is_train else 'testing', category_name,
                               os.path.basename(url)), 'w') as f:
            image.save(f)
    except:
        print('Failed to save %s, see traceback' % ((url, category_name, is_train),))
        traceback.print_exc()


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
    background_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'sun_bg'))

    # Parse URL file
    data = loadmat(mat_save_path)['SUN']
    # Filter to specified background categories
    if bg_categories is not None:
        data = [x for x in data if x.category in bg_categories]
    print('Found %d categories' % len(data))

    # Start pool
    pool = multiprocessing.Pool(num_threads)

    all_save_info = []
    for category_data in data:
        # Generate random training and testing split for this category
        num_images = len(category_data.images)
        split = np.zeros(num_images, dtype=np.bool)
        split[:num_images/2] = True
        np.random.shuffle(split)

        # Convert backslashes in category name to underscores
        processed_category_name = category_data.category.replace('\\', '_')

        # Make category directories
        train_dir = os.path.join(background_dir, 'training', processed_category_name)
        test_dir = os.path.join(background_dir, 'testing', processed_category_name)
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        save_info = [(url, processed_category_name, split[i])
                     for i, url in enumerate(category_data.images)]
        all_save_info += save_info

        # Print category info
        print('Found %d images for category %s (%s)' % (num_images, category_data.category,
                                                        processed_category_name))

    # Save images
    print('Downloading a total of %d images...' % len(all_save_info))
    fn = partial(download_numbered_file, dataset_root=background_dir)
    iter = pool.imap(fn, all_save_info)
    # iter = map(fn, all_save_info)
    for i, _ in enumerate(iter):
        if i % 200 == 0:
            print('Finished %d/%d images' % (i, len(all_save_info)))

    # Delete URL file
    os.remove(mat_save_path)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bg_categories', type=str, nargs='+',
                        help='SUN 397 categories to download (e.g. a\\abbey or '
                             'a\\apartment_building\\outdoor')
    parser.add_argument('--num_threads', type=int, default=multiprocessing.cpu_count(),
                        help='Number of download threads')

    args = parser.parse_args()
    main(**vars(args))