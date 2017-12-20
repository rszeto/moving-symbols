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
    # seed = 1513280009

    debug_options = dict(
        show_bounding_poly=True,
        show_frame_number=True
    )
    debug_options = None

    params = dict(
        data_dir='../data/mnist',
        split='training',
        num_icons=4,
        video_size=(100, 100),
        color_output=False,
        icon_labels=range(10),
        scale_limits = (0.5, 1.5),
        scale_period_limits = (40, 40),
        rotation_speed_limits = (0, 0),
        position_speed_limits = [(1, 5), (20, 20)],
        scale_function_type = 'sine'
    )

    sub = MovingIconCaptionGenerator()
    env = MovingIconEnvironment(params, seed, debug_options=debug_options,
                                initial_subscribers=[sub])
    print(env.cur_rng_seed)


    # # Display loop
    # for _ in xrange(50):
    #     image = env.next()
    #     cv2.imshow(None, np.array(image.convert('RGB'))[:, :, ::-1])
    #     cv2.waitKey(1000/10)
    #     # cv2.waitKey()

    # Print messages and show video
    images = []
    for _ in xrange(50):
        image = env.next()
        images.append(image)
    # messages = filter(lambda x: x['type'] != 'icon_state', sub.messages)
    messages = sub.messages
    pprint(messages)

    # Show frames sequentially
    for i, image in enumerate(images):
        cv_image = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        cv2.putText(cv_image, '%d' % i, (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0))
        cv2.imshow(None, cv_image)
        cv2.waitKey(1000/10)

    # # Show all frames in grid
    # grid_w = 6
    # grid_h = int(math.ceil(len(images)/float(grid_w)))
    # margin = 5
    # im_w, im_h = images[0].size
    # full_image_w = grid_w * im_w + (grid_w-1) * margin
    # full_image_h = grid_h * im_h + (grid_h-1) * margin
    # full_image = np.zeros((full_image_h, full_image_w, 3), dtype=np.uint8)
    # full_image[:, :, :] = np.array([255, 0, 0])
    #
    # for i, image in enumerate(images):
    #     grid_x = i % grid_w
    #     grid_y = (i - grid_x) / grid_w
    #     cv_image = np.array(image.convert('RGB'))[:, :, ::-1].copy()
    #     cv2.putText(cv_image, '%d' % i, (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0))
    #     full_image[
    #         grid_y * (im_h+margin):grid_y*(im_h+margin)+im_h,
    #         grid_x * (im_w+margin):grid_x*(im_w+margin)+im_w,
    #         :
    #     ] = cv_image
    #
    # cv2.imshow(None, full_image)
    # cv2.waitKey()
