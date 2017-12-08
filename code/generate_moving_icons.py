import math
import multiprocessing
import os

import numpy as np

from moving_icons import MovingIconEnvironment
from moving_icons_utils import merge_dicts

def get_param_dicts():

    mnist_training_param_dicts = {
        'dataset=mnist+translation=on+split=training': {
            'data_dir': '../data/mnist',
            'split': 'training',
            'color_output': False,
            'icon_labels': range(10),
            'position_speed_limits': [1, 9]
        },
        'dataset=mnist+rotation=on+split=training': {
            'data_dir': '../data/mnist',
            'split': 'training',
            'color_output': False,
            'icon_labels': range(10),
            'rotation_speed_limits': [math.radians(3), math.radians(26)]
        },
        'dataset=mnist+scale=on+split=training': {
            'data_dir': '../data/mnist',
            'split': 'training',
            'color_output': False,
            'icon_labels': range(10),
            'scale_limits': [0.5, 1.5],
            'scale_period_limits': [13, 50],
            'scale_function_type': 'triangle'
        }
    }

    # Construct MNIST testing params
    mnist_testing_param_dicts = {}
    for training_video_set_name, training_params in mnist_training_param_dicts.iteritems():
        testing_video_set_name = training_video_set_name.replace('split=training', 'split=testing')
        test_params = dict(training_params)
        test_params['split'] = 'testing'
        mnist_testing_param_dicts[testing_video_set_name] = test_params

    # Construct Omniglot training params
    omniglot_training_param_dicts = {}
    for mnist_video_set_name, mnist_params in mnist_training_param_dicts.iteritems():
        omniglot_video_set_name = mnist_video_set_name.replace('dataset=mnist', 'dataset=omniglot')
        omniglot_params = dict(mnist_params)
        omniglot_params['data_dir'] = '../data/omniglot'
        omniglot_params['icon_labels'] = os.listdir('../data/omniglot/training')
        omniglot_training_param_dicts[omniglot_video_set_name] = omniglot_params

    # Construct Omniglot testing params
    omniglot_testing_param_dicts = {}
    for training_video_set_name, training_params in omniglot_training_param_dicts.iteritems():
        testing_video_set_name = training_video_set_name.replace('split=training', 'split=testing')
        test_params = dict(training_params)
        test_params['split'] = 'testing'
        omniglot_testing_param_dicts[testing_video_set_name] = test_params

    # Construct icons8 training params
    icons8_training_param_dicts = {}
    for mnist_video_set_name, mnist_params in mnist_training_param_dicts.iteritems():
        icons8_video_set_name = mnist_video_set_name.replace('dataset=mnist', 'dataset=icons8')
        icons8_params = dict(mnist_params)
        icons8_params['data_dir'] = '../data/icons8'
        icons8_params['icon_labels'] = os.listdir('../data/icons8/training')
        icons8_training_param_dicts[icons8_video_set_name] = icons8_params

    # Construct icons8 testing params
    icons8_testing_param_dicts = {}
    for training_video_set_name, training_params in icons8_training_param_dicts.iteritems():
        testing_video_set_name = training_video_set_name.replace('split=training', 'split=testing')
        test_params = dict(training_params)
        test_params['split'] = 'testing'
        icons8_testing_param_dicts[testing_video_set_name] = test_params

    # Merge the training and testing dictionaries
    training_dicts =  merge_dicts(
        mnist_training_param_dicts,
        omniglot_training_param_dicts,
        icons8_training_param_dicts,
    )
    testing_dicts =  merge_dicts(
        mnist_testing_param_dicts,
        omniglot_testing_param_dicts,
        icons8_testing_param_dicts,
    )

    return training_dicts, testing_dicts


def generate_moving_icons_video((seed, num_frames, params)):
    """Create the T x H x W (x C) NumPy array for one video."""
    env = MovingIconEnvironment(params, seed)

    all_frames = []
    for i in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))

    return np.array(all_frames, dtype=np.uint8)


def generate_all_moving_icon_videos(pool, pool_seed, num_videos, num_frames, params,
                                    dataset_name):
    output_dir = os.path.join('..', 'output')
    arg_tups = [(seed, num_frames, params) for seed in xrange(pool_seed, pool_seed+num_videos)]
    # Get list of V TxHxW(xC) videos
    videos = pool.map(generate_moving_icons_video, arg_tups)
    # videos = map(generate_moving_icons_video, arg_tups)
    videos = np.array(videos, dtype=np.uint8)
    # Swap to bizarro Toronto dimensions (T x V x H x W (x C))
    videos = videos.swapaxes(0, 1)
    np.save(os.path.join(output_dir, '%s_videos.npy' % dataset_name), videos)


def main():
    pool_seed = 123
    num_training_videos = 100
    num_training_frames = 20
    num_testing_videos = 10
    num_testing_frames = 30

    pool = multiprocessing.Pool()
    training_params, testing_params = get_param_dicts()
    for dataset_name, params in training_params.iteritems():
        generate_all_moving_icon_videos(pool, pool_seed, num_training_videos, num_training_frames,
                                        params, dataset_name)
    for dataset_name, params in testing_params.iteritems():
        generate_all_moving_icon_videos(pool, pool_seed, num_testing_videos, num_testing_frames,
                                        params, dataset_name)

if __name__ == '__main__':
    main()