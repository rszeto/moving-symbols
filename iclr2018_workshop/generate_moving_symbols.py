"""
Generates the training and testing splits used in our ICLR 2018 Workshop Track submission
"A Dataset To Evaluate The Representations Learned By Video Prediction Models".

Author: Ryan Szeto
"""

import multiprocessing
import os
import sys

import numpy as np

PROJ_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT_DIR)

from moving_symbols import MovingSymbolsEnvironment


class MovingSymbolsClassTrajectoryTracker:
    """Object that gets the symbol classes and trajectories of the generated video"""

    def __init__(self):
        self.symbol_classes = {}
        self.trajectories = {}


    def process_message(self, message):
        """Store the message."""
        meta = message['meta']
        if message['type'] == 'symbol_init':
            self.symbol_classes[meta['symbol_id']] = meta['label']
        elif message['type'] == 'symbol_state':
            if meta['symbol_id'] not in self.trajectories:
                self.trajectories[meta['symbol_id']] = []
            self.trajectories[meta['symbol_id']].append(meta['position'])

    def get_info(self):
        """Return the trajectories and symbol classes

        :return: num_symbols np.array, num_symbols x T x 2 np.array
        """
        sorted_keys = sorted(self.symbol_classes.keys())
        symbol_classes_np = np.array([self.symbol_classes[k] for k in sorted_keys])
        for k in sorted_keys:
            self.trajectories[k] = np.stack(self.trajectories[k], axis=0)
        trajectories_np = np.stack([self.trajectories[k] for k in sorted_keys], axis=0)
        return symbol_classes_np, trajectories_np


def get_param_dicts():
    # Generalizing rate, slow -> fast and fast -> slow
    mnist_training_slow_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'training',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (1, 5)
    }
    mnist_training_fast_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'training',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (6, 9)
    }
    mnist_testing_fast_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'testing',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (6, 9)
    }
    mnist_testing_slow_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'testing',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (1, 5)
    }

    # Generalizing rate, slow & fast -> medium
    mnist_training_slow_fast_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'training',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': [(1, 3), (7, 9)]
    }
    mnist_testing_medium_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'testing',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (4, 6)
    }

    # Generalizing appearance, MNIST -> Icons8
    mnist_training_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'training',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (1, 9)
    }
    icons8_testing_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'icons8'),
        'split': 'testing',
        'color_output': False,
        'symbol_labels': os.listdir(os.path.join(PROJ_ROOT_DIR, 'data', 'icons8', 'training')),
        'position_speed_limits': (1, 9)
    }

    # Generalizing appearance, Icons8 -> MNIST
    icons8_training_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'icons8'),
        'split': 'training',
        'color_output': False,
        'symbol_labels': os.listdir(os.path.join(PROJ_ROOT_DIR, 'data', 'icons8', 'training')),
        'position_speed_limits': (1, 9)
    }
    mnist_testing_params = {
        'data_dir': os.path.join(PROJ_ROOT_DIR, 'data', 'mnist'),
        'split': 'testing',
        'color_output': False,
        'symbol_labels': range(10),
        'position_speed_limits': (1, 9)
    }

    training_dicts = {
        'mnist_training_slow': mnist_training_slow_params,
        'mnist_training_fast': mnist_training_fast_params,
        'mnist_training_slow_fast': mnist_training_slow_fast_params,
        'mnist_training': mnist_training_params,
        'icons8_training': icons8_training_params
    }

    testing_dicts = {
        'mnist_testing_fast': mnist_testing_fast_params,
        'mnist_testing_slow': mnist_testing_slow_params,
        'mnist_testing_medium': mnist_testing_medium_params,
        'icons8_testing': icons8_testing_params,
        'mnist_testing': mnist_testing_params
    }

    return training_dicts, testing_dicts


def generate_moving_symbols_video((seed, num_frames, params)):
    """Create the T x H x W (x C) NumPy array for one video."""
    sub = MovingSymbolsClassTrajectoryTracker()
    env = MovingSymbolsEnvironment(params, seed)
    env.add_subscriber(sub)

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
    video_tensor = np.array(all_frames, dtype=np.uint8)
    symbol_classes, trajectories = sub.get_info()

    return video_tensor, symbol_classes, trajectories


def generate_all_moving_symbol_videos(pool, pool_seed, num_videos, num_frames, params, dataset_name):
    print('Working on %s...' % dataset_name)
    output_dir = os.path.join(PROJ_ROOT_DIR, 'output')
    arg_tups = [(seed, num_frames, params) for seed in xrange(pool_seed, pool_seed+num_videos)]
    # Get list of V TxHxW(xC) videos
    video_data = pool.map(generate_moving_symbols_video, arg_tups)
    videos, symbol_classes, trajectories = zip(*video_data)
    videos = np.stack(videos, axis=0)  # V x T x H x W (x C)
    symbol_classes = np.stack(symbol_classes, axis=0)  # V x D
    trajectories = np.stack(trajectories, axis=0)  # V x D x T x 2
    # Swap to bizarro Toronto dimensions (T x V x H x W (x C))
    videos = videos.swapaxes(0, 1)
    np.save(os.path.join(output_dir, '%s_videos.npy' % dataset_name), videos)
    np.save(os.path.join(output_dir, '%s_symbol_classes.npy' % dataset_name), symbol_classes)
    np.save(os.path.join(output_dir, '%s_trajectories.npy' % dataset_name), trajectories)


def main():
    pool_seed = 123
    num_training_videos = 10000
    num_training_frames = 20
    num_testing_videos = 1000
    num_testing_frames = 30

    # Make output directory
    output_dir = os.path.join(PROJ_ROOT_DIR, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    pool = multiprocessing.Pool()
    training_params, testing_params = get_param_dicts()
    for dataset_name, params in training_params.iteritems():
        generate_all_moving_symbol_videos(pool, pool_seed, num_training_videos, num_training_frames,
                                        params, dataset_name)
    for dataset_name, params in testing_params.iteritems():
        generate_all_moving_symbol_videos(pool, pool_seed, num_testing_videos, num_testing_frames,
                                        params, dataset_name)

if __name__ == '__main__':
    main()