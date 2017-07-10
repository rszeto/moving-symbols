import numpy as np
from generate_moving_mnist import MovingMNISTGenerator
from multiprocessing import Pool
import json
import os
from functools import partial
import matplotlib.pyplot as plt
import argparse


def get_video_tensor(index, gen_params):
    '''
    Generate a video tensor with the given parameters.
    :param index: Which job this is
    :param gen_params: The dictionary of parameters
    :return:
    '''
    np.random.seed(index)
    gen = MovingMNISTGenerator(**gen_params)
    return gen.get_video_tensor_copy()


def main(param_file_path, save_path, num_videos, num_procs):
    # Load the generation parameters
    with open(param_file_path, 'r') as f:
        gen_params = json.load(f)

    # Generate video frames with a multiprocessing pool
    fn = partial(get_video_tensor, gen_params=gen_params)
    pool = Pool(processes=num_procs)
    video_tensors_list = pool.map(fn, range(num_videos))

    # Combine the frames into one tensor
    video_tensors = np.stack(video_tensors_list, axis=0)
    # Swap to bizarro Toronto dims
    video_tensors = video_tensors.transpose((3, 0, 1, 2))

    # Save the file
    save_path = os.path.abspath(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, video_tensors)

    # TODO: Extract save_video fn from generator object


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file_path', type=str, help='The path to the parameter JSON file')
    parser.add_argument('save_path', type=str, help='The path to the file to store the data')
    parser.add_argument('num_videos', type=int, help='How many videos to generate')
    parser.add_argument('--num_procs', type=int, default=1, help='How many processors to use')

    args = parser.parse_args()
    main(**vars(args))