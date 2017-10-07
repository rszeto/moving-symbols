import json
import numpy as np
import os
from multiprocessing import Pool
import sys
from collections import OrderedDict
import argparse

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'code'))
import main

base = {
    "num_images": 1,
    "num_timesteps": 20,
    "video_size": [64, 64],
    "digit_labels": [4],
    "angle_lim": [0, 0],
    "angle_init_lim": [0, 0]
}

def merge_dicts(a, b):
    '''
    Merge two dictionaries by adding b's key-value pairs to a, or
    replacing if a also has some of b's keys
    :param a: The first (base) dictionary
    :param b: The second (overwritten) dictionary
    :return:
    '''
    ret = a.copy()
    for key, value in b.iteritems():
        # If key starts with '-', delete the key instead of adding it
        if key[0] == '-':
            real_key = key[1:]
            if real_key in ret.keys():
                del ret[key[1:]]
        else:
            ret[key] = value
    return ret

def combine_extension_dicts(keys):
    '''
    Merge the dictionaries specified by the given keys
    :param keys: A list of valid extension_dicts keys
    :return:
    '''
    ret = base.copy()
    for key in keys:
        ret = merge_dicts(ret, extension_dicts[key])
    return ret


extension_dicts = OrderedDict([
    ('translation=on', {
        "x_speed_lim": [-3, 3],
        "y_speed_lim": [-3, 3]
    }),
    ('translation=on_fast', {
        "x_speed_lim": [4, 7],
        "y_speed_lim": [4, 7]
    }),
    ('rotation=limit_a', {
        "angle_lim": [-45, 45],
        "angle_init_lim": [0, 0],
        "angle_speed_lim": [-4, 4]
    }),
    ('rotation=limit_b', {
        "angle_lim": [45, 135],
        "angle_init_lim": [90, 90],
        "angle_speed_lim": [-4, 4]
    }),
    ('rotation=no_limit', {
        "-angle_lim": None,
        "angle_speed_lim": [-4, 4]
    }),
    ('rotation=cw', {
        "-angle_lim": None,
        "angle_speed_lim": [-4, 0]
    }),
    ('rotation=cw_fast', {
        "-angle_lim": None,
        "angle_speed_lim": [-10, -6]
    }),
    ('rotation=ccw', {
        "-angle_lim": None,
        "angle_speed_lim": [0, 4]
    }),
    ('scale=limit_a', {
        "scale_lim": [0.5, 1],
        "scale_init_lim": [0.75, 0.75],
        "scale_speed_lim": [-0.05, 0.05]
    }),
    ('scale=limit_b', {
        "scale_lim": [0.75, 1.25],
        "scale_init_lim": [1.0, 1.0],
        "scale_speed_lim": [-0.05, 0.05]
    }),
    ('scale=limit_a_fast', {
        "scale_lim": [0.5, 1],
        "scale_init_lim": [0.75, 0.75],
        "scale_speed_lim": [0.075, 0.135]
    }),
    ('flashing=sync_a', {
        "blink_rates": [2, 2, 2]
    }),
    ('flashing=sync_b', {
        "blink_rates": [4, 4, 4]
    }),
    ('flashing=async_a', {
        "blink_rates": [2, 5, 0]
    }),
    ('flashing=async_b', {
        "blink_rates": [3, 4, 5]
    }),
    ('num_digits=2', {
        "num_images": 2
    }),
    ('num_digits=3', {
        "num_images": 3
    }),
    ('image=label_subset_a', {
        "digit_labels": range(7)
    }),
    ('image=label_subset_b', {
        "digit_labels": range(7, 10)
    }),
    ('image=any', {
        "-digit_labels": None
    }),
    ('num_timesteps=100', {
        "num_timesteps": 100
    })
    # ('background=single', {
    #     "use_background": True,
    #     "background_file_cats": ["c_crosswalk"],
    #     "background_file_id": 0
    # }),
    # ('background=subset_a', {
    #     "use_background": True,
    #     "background_file_cats": ["c_crosswalk"]
    # }),
    # ('background=subset_b', {
    #     "use_background": True,
    #     "background_file_cats": ["r_rainforest"]
    # }),
    # ('background=any', {
    #     "use_background": True,
    #     "background_file_cats": ["c_crosswalk", "r_rainforest"]
    # })
])

dynamic_params = ['translation', 'rotation', 'scale', 'flashing']


from itertools import chain, combinations

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    Source: https://stackoverflow.com/a/16915734
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))


def is_valid_param_combo(keys):
    '''
    Determine if the given set of parameter keys are compatible, which is
    True iff they do not have set the same experimental setting
    :param keys: list of key strings from extension_dicts
    :return:
    '''
    setting_names = sorted([key[:key.index('=')] for key in keys])
    for i in range(len(setting_names) - 1):
        if setting_names[i] == setting_names[i+1]:
            return False
    return True


def generate_params(params_root, max_num_settings=len(extension_dicts.keys())):
    keys_arr = np.array(extension_dicts.keys())
    combo_tuples = powerset(range(len(keys_arr)))
    ret = []
    for t in combo_tuples:
        if len(t) > max_num_settings:
            continue

        # Get keys to each parameter setting
        key_list = list(keys_arr[list(t)])
        # Do not allow the same parameter to be set multiple times
        if not is_valid_param_combo(key_list):
            continue

        # Get names of parameters being set
        param_names = [x.split('=')[0] for x in key_list]
        # Do not allow videos that have no dynamics
        if not set(param_names) & set(dynamic_params):
            continue

        # Combine keys into experiment name
        exp_name = '+'.join(key_list) if len(key_list) > 0 else 'base'
        print(exp_name)
        ret.append(exp_name)

        # Save parameter file
        json_path = os.path.join(params_root, '%s.json' % exp_name)
        combined_dict = combine_extension_dicts(key_list)
        with open(json_path, 'w') as f:
            json.dump(combined_dict, f, sort_keys=True, indent=2)

        # Save parameter file for long-term predictions
        json_path = os.path.join(params_root, '%s_long.json' % exp_name)
        combined_dict = combine_extension_dicts(key_list)
        combined_dict['num_timesteps'] = 60
        with open(json_path, 'w') as f:
            json.dump(combined_dict, f, sort_keys=True, indent=2)

    return ret


def generate_videos(exp_names,
                    num_train_videos,
                    num_val_videos,
                    num_test_videos,
                    num_long_videos,
                    params_root,
                    output_root,
                    verbosity_params_path=os.path.join(
                        SCRIPT_DIR, '..', 'verbosity_params', 'verbosity_short.json'),
                    num_procs=None,
                    generate_overlap=False):
    pool = Pool(num_procs) if num_procs is not None else Pool()
    for exp_name in exp_names:
        # Generate videos
        json_path = os.path.join(params_root, '%s.json' % exp_name)
        # Train set
        train_save_prefix = os.path.join(output_root, exp_name)
        main.main([json_path], [num_train_videos], train_save_prefix, verbosity_params_path, num_procs, 0, pool=pool)
        if generate_overlap and 'num_digits=2' in exp_name:
            train_save_prefix = os.path.join(output_root, '%s+occlusion=on' % exp_name)
            main.main([json_path], [num_train_videos], train_save_prefix, verbosity_params_path, num_procs, 0, pool=pool, keep_overlap_only=True)

        # Validation set
        val_save_prefix = os.path.join(output_root, '%s_val' % exp_name)
        main.main([json_path], [num_val_videos], val_save_prefix, verbosity_params_path, num_procs, 10, pool=pool)
        if generate_overlap and 'num_digits=2' in exp_name:
            val_save_prefix = os.path.join(output_root, '%s+occlusion=on_val' % exp_name)
            main.main([json_path], [num_val_videos], val_save_prefix, verbosity_params_path, num_procs, 10, pool=pool, keep_overlap_only=True)

        # Test set
        test_save_prefix = os.path.join(output_root, '%s_test' % exp_name)
        main.main([json_path], [num_test_videos], test_save_prefix, verbosity_params_path, num_procs, 20, pool=pool)
        if generate_overlap and 'num_digits=2' in exp_name:
            test_save_prefix = os.path.join(output_root, '%s+occlusion=on_test' % exp_name)
            main.main([json_path], [num_test_videos], test_save_prefix, verbosity_params_path, num_procs, 20, pool=pool, keep_overlap_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_root', type=str, default=os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'params')),
                        help='Where the generated parameter JSON files will be saved')
    parser.add_argument('--output_root', type=str, default=os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'output')),
                         help='Where the generated dataset files will be saved')
    args = parser.parse_args()

    if not os.path.isdir(args.params_root):
        os.makedirs(args.params_root)
    if not os.path.isdir(args.output_root):
        os.makedirs(args.output_root)

    print('Generating parameters')
    exp_names = generate_params(args.params_root)

    print('Retrieving datasets to generate from file')
    mnist_slices_file = os.path.join(SCRIPT_DIR, 'mnist_slices.txt')
    with open(mnist_slices_file, 'r') as f:
        exp_names = [line.strip() for line in f.readlines()]
    # Filter empty or commented lines
    exp_names = filter(lambda x: len(x) > 0 and not x.startswith('#'), exp_names)

    print('Generating videos')
    generate_videos(exp_names, 10000, 1000, 1000, 10000, args.params_root, args.output_root, generate_overlap=False)