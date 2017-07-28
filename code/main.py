import numpy as np
from generate_moving_mnist import MovingMNISTGenerator, SimpleMovingMNISTLogger
from multiprocessing import Pool
import json
import os
from functools import partial
import matplotlib.pyplot as plt
import argparse
import time
from text_description import create_description_from_logger


def generate_video_data(index, gen_params, seed, verbosity_params):
    '''
    Generate a video tensor with the given parameters.
    :param index: Which job this is
    :param gen_params: The dictionary of parameters
    :return:
    '''
    logger = SimpleMovingMNISTLogger()
    gen_seed = seed + index
    gen = MovingMNISTGenerator(seed=gen_seed, observers=[logger], **gen_params)
    gen.run_dynamics()

    video_tensor = gen.get_video_tensor_copy()
    desc = create_description_from_logger(logger, **verbosity_params)
    text_desc = str(desc)
    return video_tensor, logger.messages, text_desc


def main(param_file_paths, stratum_sizes, save_prefix, verbosity_params_path,
         num_procs, seed, pool=None):
    if len(param_file_paths) != len(stratum_sizes):
        print('Number of param file paths must equal number of stratum sizes')
        return

    num_strata = len(param_file_paths)
    if pool is None:
        pool = Pool(processes=num_procs)

    video_tensors_list = []
    messages_list = []
    text_descs_list = []

    # Load verbosity settings
    verbosity_params = {}
    if verbosity_params_path:
        with open(verbosity_params_path, 'r') as f:
            verbosity_params = json.load(f)

    for i in range(num_strata):
        param_file_path = param_file_paths[i]
        num_videos = stratum_sizes[i]

        # Load the generation parameters
        with open(param_file_path, 'r') as f:
            gen_params = json.load(f)
        # Generate video frames with a multiprocessing pool
        fn = partial(generate_video_data, gen_params=gen_params, seed=seed,
                     verbosity_params=verbosity_params)
        # data = pool.map(fn, range(num_videos))
        data = map(fn, range(num_videos))
        cur_video_tensors_list, cur_messages_list, cur_text_descs_list = zip(*data)
        # Convert from tuples to actual lists
        video_tensors_list += list(cur_video_tensors_list)
        messages_list += list(cur_messages_list)
        text_descs_list += list(cur_text_descs_list)

    # Combine the frames into one tensor
    video_tensors = np.stack(video_tensors_list, axis=0)
    # Swap to bizarro Toronto dims
    if video_tensors.ndim == 4:
        video_tensors = video_tensors.transpose((3, 0, 1, 2))
    else:
        video_tensors = video_tensors.transpose((4, 0, 1, 2, 3))

    # JSONify each set of messages for storing in NumPy file
    json_messages_list = [json.dumps(messages) for messages in messages_list]

    # Save the file
    save_prefix = os.path.abspath(save_prefix)
    if not os.path.exists(os.path.dirname(save_prefix)):
        os.makedirs(os.path.dirname(save_prefix))
    np.save('%s_videos.npy' % save_prefix, video_tensors)
    np.save('%s_messages.npy' % save_prefix, np.array(json_messages_list))
    np.save('%s_text_descs.npy' % save_prefix, np.array(text_descs_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--param_file_paths', type=str, nargs='+', required=True,
                        help='The path to the parameter JSON file for each stratum')
    parser.add_argument('--stratum_sizes', type=int, nargs='+', required=True,
                        help='How many videos to generate for each stratum')
    parser.add_argument('--save_prefix', type=str, required=True,
                        help='The path prefix for the saved files')
    parser.add_argument('--verbosity_params_path', type=str,
                        help='Path to the verbosity settings to pass to the description generator')
    parser.add_argument('--num_procs', type=int, default=1, help='How many processors to use')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='Seed for RNG')

    args = parser.parse_args()
    main(**vars(args))