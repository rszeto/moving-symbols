import numpy as np
from generate_moving_mnist import MovingMNISTGenerator
from multiprocessing import Pool
import json
import os
from functools import partial
import matplotlib.pyplot as plt

def generate_one_video(index, gen_params):
    '''
    Generate a video tensor with the given parameters.
    :param gen_params: The dictionary of parameters
    :return:
    '''
    np.random.seed(index)
    gen = MovingMNISTGenerator(**gen_params)
    gen.save_video('videos/output_%04d.avi' % index)

def main():
    param_file_path = '../params/test.json'

    pool = Pool(processes=4)
    with open(param_file_path, 'r') as f:
        gen_params = json.load(f)
    if not os.path.exists('videos'):
        os.makedirs('videos')

    fn = partial(generate_one_video, gen_params=gen_params)
    video_tensors = pool.map(fn, range(100))
    # video_tensors = map(fn, range(10))

    # TODO: Save videos as npy
    # TODO: Extract save_video fn from generator object


if __name__ == '__main__':
    main()