import cPickle as pickle
import math
import os
from pprint import pprint
import time

import cv2
import numpy as np

from moving_icons import MovingIconEnvironment, AbstractMovingIconSubscriber

class MovingIconCaptionGenerator(AbstractMovingIconSubscriber):
    """Object that collects all Moving Icon event messages and turns them into a paragraph"""

    def __init__(self):
        self.messages = []


    def process_message(self, message):
        """Store the message."""
        self.messages.append(message)


if __name__ == '__main__':

    seed = int(time.time())
    seed = 1512518328

    debug_options = dict(
        show_bounding_poly=True,
        show_frame_number=True
    )
    debug_options = None

    params = dict(
        data_dir='../data/mnist',
        split='training',
        num_icons=1,
        video_size=(100, 100),
        color_output=False,
        icon_labels=range(10),
        scale_limits = [0.5, 1.5],
        scale_period_limits = [40, 60],
        rotation_speed_limits = [math.radians(5), math.radians(15)],
        position_speed_limits = [1, 5],
        # interacting_icons = True,
        scale_function_type = 'triangle'
    )

    sub = MovingIconCaptionGenerator()
    env = MovingIconEnvironment(params, seed, debug_options=debug_options,
                                initial_subscribers=[sub])
    print(env.cur_rng_seed)


    # Display loop
    images = []
    for _ in xrange(150):
        cv_image = env.next()
        images.append(cv_image)

    with open('test.pkl', 'w') as f:
        pickle.dump(sub.messages, f)
    with open('test.pkl', 'r') as f:
        dumped = pickle.load(f)
    pprint(dumped)
    os.remove('test.pkl')

    for image in images:
        cv2.imshow(None, np.array(image.convert('RGB'))[:, :, ::-1])
        cv2.waitKey(1000/60)