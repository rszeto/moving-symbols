"""
Generate Moving MNIST with captions as described in "Attentive Semantic Video Generation using
Captions" by Marwah et al., ICCV 2017. https://arxiv.org/abs/1708.05980
"""

import time

import cv2
import numpy as np

from moving_icons import MovingIconEnvironment, AbstractMovingIconSubscriber


class MarwahMovingMnistCaptionGenerator(AbstractMovingIconSubscriber):
    """
    Collects all messages and describes the motion as 'digit X is going A and digit Y is going B
    and ...', where X and Y are digit labels and A and B can be 'left and right' or 'up and down'.
    """

    def __init__(self):
        AbstractMovingIconSubscriber.__init__(self)
        self.icon_init_messages = []
        self.first_icon_state_messages = []

    def process_message(self, message):
        """Process the identity and the motion of each digit."""

        # Only process messages with step equal to -1 or 0 (i.e. init or first state messages)
        if message['step'] not in [-1, 0]:
            return

        if message['type'] == 'icon_init':
            self.icon_init_messages.append(message['meta'])
        elif message['type'] == 'icon_state':
            self.first_icon_state_messages.append(message['meta'])


    def generate_caption(self):
        # Define error tolerance for arctangent computation
        eps = 1e-5
        # Group the messages for each icon together
        self.icon_init_messages.sort(key=lambda x: x['icon_id'])
        self.first_icon_state_messages.sort(key=lambda x: x['icon_id'])
        icon_message_tuples = zip(self.icon_init_messages, self.first_icon_state_messages)

        # Describe each icon, then stitch the descriptions together
        icon_captions = []
        for icon_init_message, first_icon_state_message in icon_message_tuples:
            icon_label = icon_init_message['label']
            icon_velocity = first_icon_state_message['velocity']
            # Get vertical or horizontal motion description from velocity
            vel_arctan = np.arctan2(icon_velocity[1], icon_velocity[0])
            if np.abs(vel_arctan - np.pi/2) < eps or np.abs(vel_arctan + np.pi/2) < eps:
                motion_fragment = 'up and down'
            else:
                motion_fragment = 'left and right'
            # Add icon caption
            icon_caption = 'digit %d is going %s' % (icon_label, motion_fragment)
            icon_captions.append(icon_caption)

        return ' and '.join(icon_captions)


def generate_single_video_caption_pair((seed, num_frames, params)):
    sub = MarwahMovingMnistCaptionGenerator()
    env = MovingIconEnvironment(params, seed, initial_subscribers=[sub])

    # Generate video sequence
    cv_images = []
    for _ in xrange(num_frames):
        image = env.next()
        cv_image = np.array(image)
        cv_images.append(cv_image)

    video_tensor = np.array(cv_images)
    caption = sub.generate_caption()

    return video_tensor, caption



def main():
    seed = int(time.time())
    num_frames = 10
    train_params = dict(
        data_dir='../data/mnist',
        split='training',
        num_icons=1,
        video_size=(64, 64),
        color_output=False,
        icon_labels=range(10),
        position_speed_limits=[8, 8],
        lateral_motion_at_start=True
    )

    print(seed)
    video_tensor, caption = generate_single_video_caption_pair((seed, num_frames, train_params))

    print(caption)
    for i, cv_image in enumerate(video_tensor):
        cv2.imshow(None, cv_image)
        cv2.waitKey(1000/5)

if __name__ == '__main__':
    main()