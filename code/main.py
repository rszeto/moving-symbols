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
import h5py


def generate_video_data(index, gen_params, seed):
    '''
    Generate a video tensor with the given parameters.
    :param index: Which job this is
    :param gen_params: The dictionary of parameters
    :return:
    '''
    logger = SimpleMovingMNISTLogger()
    gen_seed = seed + index
    # print(gen_seed)
    gen = MovingMNISTGenerator(seed=gen_seed, observers=[logger], **gen_params)

    video_tensor = gen.get_video_tensor_copy()
    desc = create_description_from_logger(logger)
    text_desc = str(desc)
    return video_tensor, logger.messages, text_desc


def main(param_file_path, save_prefix, num_videos, num_procs, seed):
    # Load the generation parameters
    with open(param_file_path, 'r') as f:
        gen_params = json.load(f)

    # Generate video frames with a multiprocessing pool
    fn = partial(generate_video_data, gen_params=gen_params, seed=seed)
    pool = Pool(processes=num_procs)
    # data = pool.map(fn, range(num_videos))
    data = map(fn, range(num_videos))
    video_tensors_list, messages_list, text_descs_list = zip(*data)
    # Convert from tuples to actual lists
    video_tensors_list = list(video_tensors_list)
    messages_list = list(messages_list)
    text_descs_list = list(text_descs_list)

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

    # TODO: Extract save_video fn from generator object


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file_path', type=str, help='The path to the parameter JSON file')
    parser.add_argument('save_prefix', type=str, help='The path prefix for the saved files')
    parser.add_argument('num_videos', type=int, help='How many videos to generate')
    parser.add_argument('--num_procs', type=int, default=1, help='How many processors to use')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='Seed for RNG')

    args = parser.parse_args()
    main(**vars(args))