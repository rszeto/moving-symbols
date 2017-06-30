import numpy as np
import os
import sys
import cv2
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
import json

def tight_crop(image):
    '''
    Tightly crop around the non-zero values of the given image
    :param image: w x h x <2 or 3> image matrix
    :return:
    '''
    if image.ndim not in [2, 3]:
        raise ValueError('Obtained image has %d dimensions, but must have 2 or 3' % image.ndim)

    if image.ndim == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    nonzero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(nonzero_points)

    # Get crop depending on number of channels
    if image.ndim == 2:
        crop = image[y:y + h, x:x + w]
    else:
        crop = image[y:y + h, x:x + w, :]

    return crop


def get_rotated_scaled_image_tight_crop(image, deg, scale):
    '''
    Transform the image and return the tight crop around it
    :param image: The image to modify and crop
    :param def: The counterclockwise angle to rotate the image, in degrees
    :param scale: The scale of the image
    :return:
    '''
    # Rotate the image
    rot = get_rotation_matrix(image.shape, deg)
    # Scale the rotated image
    s = get_scale_matrix(scale)
    # Translate the result to the center of the workspace
    work_size = (int(3 * scale * np.max(image.shape)),) * 2
    trans = get_translation_matrix(work_size[0]/4, work_size[1]/4)
    M = np.dot(trans, np.dot(s, rot))

    # Generate image on work space
    if image.ndim == 2:
        work_image = cv2.warpPerspective(image, M, tuple(work_size))
    elif image.ndim == 3:
        # Transform each channel individually
        work_channels = []
        for c in range(3):
            work_channel = cv2.warpPerspective(image[:, :, c], M, tuple(work_size[:2]))
            work_channels.append(work_channel)
        work_image = np.stack(work_channels, axis=-1)
    else:
        raise ValueError('Image matrix must have either two or three dimensions')

    crop = tight_crop(work_image)
    return crop


def get_translation_matrix(x, y):
    '''
    Get the 3x3 matrix corresponding to a translation from the top-left corner
    :param x: The horizontal displacement (coming from the left)
    :param y: The vertical displacement (coming from the top)
    :return:
    '''
    trans = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
    return np.concatenate((trans, [[0, 0, 1]]))


def get_center_translation_matrix(image, x, y):
    '''
    Get the 3x3 matrix corresponding to moving the image's center to (x, y)
    :param image: The image to be translated
    :param x: The horizontal displacement (coming from the left)
    :param y: The vertical displacement (coming from the top)
    :return:
    '''
    return get_translation_matrix(x - image.shape[1]/2, y - image.shape[0]/2)


def get_rotation_matrix(image_size, deg):
    '''
    Get the 3x3 matrix corresponding to a counter-clockwise rotation from the center of the image
    :param image_size: The size of the image to rotate (height x width)
    :param deg: The angle to rotate by in degrees
    :return:
    '''

    rot = cv2.getRotationMatrix2D((image_size[1]/2, image_size[0]/2), deg, 1)
    return np.concatenate((rot, [[0, 0, 1]]))


def get_scale_matrix(scale):
    '''
    Get the 3x3 matrix corresponding to a scaling.
    :param scale: The ratio of the result to the original image
    :return:
    '''
    s = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
    return np.concatenate((s, [[0, 0, 1]]))


def overlay_image(fg, bg):
    '''
    Overlay the non-zero pixels in the foreground over the pixels in the background
    :param fg: The foreground image
    :param bg: The background image
    :return:
    '''
    if fg.shape != bg.shape:
        raise ValueError('Foreground and background have different sizes (fg: %s, bg: %s)' % (fg.shape, bg.shape))
    if fg.ndim not in [2, 3]:
        raise ValueError('Foreground image has %d dimensions, but must have 2 or 3' % fg.ndim)
    if bg.ndim not in [2, 3]:
        raise ValueError('Background image has %d dimensions, but must have 2 or 3' % bg.ndim)

    return np.minimum(255, fg.astype(np.int) + bg.astype(np.int)).astype(np.uint8)


def save_tensor_as_mjpg(filename, video_tensor):
    '''
    Saves the given video tensor as a video
    :param filename: The location of the video
    :param video_tensor: The H x W (x C) x T tensor of frames
    :return:
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 15.0, tuple(video_tensor.shape[:2][::-1]))
    for i in range(video_tensor.shape[-1]):
        if video_tensor.ndim == 3:
            video_frame = cv2.cvtColor(video_tensor[:, :, i], cv2.COLOR_GRAY2BGR)
        else:
            video_frame = video_tensor[:, :, :, i]
        out.write(video_frame[:, :, ::-1])
    out.release()


class MovingMNISTGenerator:

    __SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))

    def __init__(self,
                 data_dir=os.path.abspath(os.path.join(__SCRIPT_DIR__, '..', 'data')), split='training',
                 num_images=1, max_image_size=28, video_size=(64, 64), num_timesteps=30,
                 binary_output=False,
                 angle_lim=[0, 0], scale_lim=[1, 1],
                 x_speed_lim=[0, 0], y_speed_lim=[0, 0], scale_speed_lim=[0, 0], angle_speed_lim=[0, 0]):
        self.data_dir = data_dir
        self.split = split
        self.num_images = num_images
        self.max_image_size = max_image_size
        self.video_size = tuple(video_size)
        self.num_timesteps = num_timesteps
        self.binary_output = binary_output
        self.angle_lim = [x % 360 for x in angle_lim]
        self.scale_lim = scale_lim
        self.x_speed_lim = x_speed_lim
        self.y_speed_lim = y_speed_lim
        self.scale_speed_lim = scale_speed_lim
        self.angle_speed_lim = angle_speed_lim

        self.x_lim = [0, video_size[0]]
        self.y_lim = [0, video_size[1]]

        # Get digits and split them into image and label lists
        digit_infos = [self.__get_digit__() for _ in range(num_images)]
        self.images, self.labels = zip(*digit_infos)

        # Set image channel count
        self.is_src_grayscale = (self.images[0].ndim == 2)

        # Sample starting states and update params
        self.states = [self.__sample_start_states__() for _ in range(num_images)]
        self.update_params = [self.__sample_update_params__() for _ in range(num_images)]

        # Set empty video cache
        self.video_tensor = None


    def __get_digit__(self):
        '''
        Get random digit images (cropped) and labels
        :return:
        '''
        label = np.random.randint(10)
        digit_dir = os.path.join(self.data_dir, self.split, str(label))
        image = imread(os.path.join(digit_dir, '%04d.png' % np.random.randint(len(os.listdir(digit_dir)))))
        image = np.stack([image, np.zeros(image.shape), np.zeros(image.shape)], axis=-1).astype(np.uint8)
        crop = tight_crop(image)
        return crop, label


    def __sample_start_states__(self):
        '''
        Choose starting parameters based on possible position, scale, etc.
        :return:
        '''
        scale_start = np.random.uniform(self.scale_lim[0], self.scale_lim[1])
        # if self.angle_lim[0] == self.angle_lim[1]:
        #     angle_start = self.angle_lim[0]
        # else:
        #     angle_start = np.random.randint(self.angle_lim[0], self.angle_lim[1])
        angle_start = 0
        # Start far from the video frame border to avoid image clipping
        pad = (self.max_image_size / 2) * np.sqrt(2) * scale_start
        x_start = np.random.randint(np.ceil(self.x_lim[0] + pad), np.floor(self.x_lim[1] - pad))
        y_start = np.random.randint(np.ceil(self.y_lim[0] + pad), np.floor(self.y_lim[1] - pad))

        return dict(
            scale=scale_start,
            x=x_start,
            y=y_start,
            angle=angle_start
        )


    def __sample_update_params__(self):
        '''
        Choose update parameters based on possible position, scale updates, etc.
        :return:
        '''
        scale_speed = np.random.uniform(self.scale_speed_lim[0],
                                        self.scale_speed_lim[1])
        x_speed = np.random.uniform(self.x_speed_lim[0],
                                    self.x_speed_lim[1])
        y_speed = np.random.uniform(self.y_speed_lim[0],
                                    self.y_speed_lim[1])
        if self.angle_speed_lim[0] == self.angle_speed_lim[1]:
            angle_speed = self.angle_speed_lim[0]
        else:
            angle_speed = np.random.randint(self.angle_speed_lim[0],
                                            self.angle_speed_lim[1])

        return dict(
            scale_speed=scale_speed,
            x_speed=x_speed,
            y_speed=y_speed,
            angle_speed=angle_speed
        )


    def render_current_state(self):
        '''
        Render all the digits onto an image
        :return:
        '''
        # Get frame for each digit
        digit_frames = []
        for j in range(self.num_images):
            trans = get_center_translation_matrix(self.rotated_images[j], self.states[j]['x'], self.states[j]['y'])
            if self.is_src_grayscale:
                digit_frame = cv2.warpPerspective(self.rotated_images[j], trans, self.video_size)
            else:
                digit_frame = np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.uint8)
                for c in range(3):
                    digit_frame[:, :, c] = cv2.warpPerspective(np.squeeze(self.rotated_images[j][:, :, c]), trans, self.video_size)
            digit_frames.append(digit_frame)

        # Overlay frames
        stitched_frame = np.zeros(digit_frames[0].shape, dtype=np.uint8)
        for j in range(self.num_images):
            stitched_frame = overlay_image(digit_frames[j], stitched_frame)

        # Binarize frame if specified
        if self.binary_output:
            _, stitched_frame = cv2.threshold(stitched_frame, 1, 255, cv2.THRESH_BINARY)
        return stitched_frame


    def step(self):
        '''
        Step through the dynamics for each digit
        :return:
        '''

        self.rotated_images = []

        for j in range(self.num_images):
            image_state = self.states[j]
            update_params = self.update_params[j]

            # Update state
            image_state['x'] += update_params['x_speed']
            image_state['y'] += update_params['y_speed']
            image_state['angle'] = (image_state['angle'] + update_params['angle_speed'])
            image_state['scale'] += update_params['scale_speed']

            # Update parameters that don't require the image size
            # Scale
            if image_state['scale'] > self.scale_lim[1]:
                image_state['scale'] = self.scale_lim[1] - (image_state['scale'] - self.scale_lim[1])
                update_params['scale_speed'] *= -1
            elif image_state['scale'] < self.scale_lim[0]:
                image_state['scale'] = self.scale_lim[0] - (image_state['scale'] - self.scale_lim[0])
                update_params['scale_speed'] *= -1

            # Angle
            new_angle = image_state['angle']
            if self.angle_lim[0] <= self.angle_lim[1] and (new_angle < self.angle_lim[0] or new_angle > self.angle_lim[1]):
                # We crossed an angle border
                if update_params['angle_speed'] < 0:
                    # Crossed first border
                    diff = (self.angle_lim[0] - new_angle) % 360
                    image_state['angle'] = (self.angle_lim[0] + diff)
                    update_params['angle_speed'] *= -1
                else:
                    # Crossed second border
                    diff = (new_angle - self.angle_lim[1]) % 360
                    image_state['angle'] = (self.angle_lim[1] - diff)
                    update_params['angle_speed'] *= -1
            elif self.angle_lim[0] > self.angle_lim[1] and (new_angle > self.angle_lim[1] and new_angle < self.angle_lim[0]):
                # We crossed an angle border
                if update_params['angle_speed'] < 0:
                    # Crossed first border
                    diff = (self.angle_lim[0] - new_angle) % 360
                    image_state['angle'] = (self.angle_lim[0] + diff)
                    update_params['angle_speed'] *= -1
                else:
                    # Crossed second border
                    diff = (new_angle - self.angle_lim[1]) % 360
                    image_state['angle'] = (self.angle_lim[1] - diff)
                    update_params['angle_speed'] *= -1
            image_state['angle'] %= 360

            # Generate the cropped image
            cropped_digit = get_rotated_scaled_image_tight_crop(self.images[j], image_state['angle'], image_state['scale'])
            self.rotated_images.append(cropped_digit)

            # Bounce image off the walls (requires image size)
            x_right = image_state['x'] + cropped_digit.shape[1] / 2
            x_left = image_state['x'] - cropped_digit.shape[1] / 2
            if x_right > self.x_lim[1] or x_left < self.x_lim[0]:
                if x_right > self.x_lim[1]:
                    # x_right = x_max - (x_right - x_max)
                    x_right = self.x_lim[1]
                    image_state['x'] = x_right - cropped_digit.shape[1] / 2
                else:
                    # x_left = x_min - (x_left - x_min)
                    x_left = self.x_lim[0]
                    image_state['x'] = x_left + cropped_digit.shape[1] / 2
                # Flip x speed
                update_params['x_speed'] *= -1

            y_bottom = image_state['y'] + cropped_digit.shape[0] / 2
            y_top = image_state['y'] - cropped_digit.shape[0] / 2
            if y_bottom > self.y_lim[1] or y_top < self.y_lim[0]:
                if y_bottom > self.y_lim[1]:
                    # y_bottom = y_max - (y_bottom - y_max)
                    y_bottom = self.y_lim[1]
                    image_state['y'] = y_bottom - cropped_digit.shape[0] / 2
                else:
                    # y_top = y_min - (y_top - y_min)
                    y_top = self.y_lim[0]
                    image_state['y'] = y_top + cropped_digit.shape[0] / 2
                # Flip y speed
                update_params['y_speed'] *= -1


    def populate_video_tensor(self):
        '''
        Generate the video frame cache and store it as video_tensor
        :return:
        '''
        if not self.video_tensor:
            # Generate frames
            frames = []
            for i in range(self.num_timesteps):
                # Iterate digit dynamics
                self.step()
                # Render and store the current image
                stitched_frame = self.render_current_state()
                frames.append(stitched_frame)
                print('Finished frame %d' % i)

            # Convert frames to tensor
            self.video_tensor = np.stack(frames, axis=-1)


    def save_video(self, file_path):
        '''
        Save the frames for this generator instance to either a NumPy or video file
        :param file_path: Path to save to. Must end in ".npy" or ".avi"
        :return:
        '''
        _, ext = os.path.splitext(file_path)
        if ext not in ['.npy', '.avi']:
            raise ValueError('File extension must be either ".npy" or ".avi"')

        # Generate frames
        if self.video_tensor is None:
            self.populate_video_tensor()

        # Save
        if ext == '.npy':
            np.save(file_path, self.video_tensor)
        else:
            save_tensor_as_mjpg(file_path, self.video_tensor)


    def get_video_tensor_copy(self):
        '''
        Get a copy of the video tensor
        :return:
        '''
        if self.video_tensor is None:
            self.populate_video_tensor()
        return self.video_tensor.copy()


def create_moving_mnist_generator(param_file):
    '''
    Create the MovingMNISTGenerator object with parameters defined by the given JSON file
    :param param_file: Path to the JSON file with keyword args for MovingMNISTGenerator
    :return:
    '''
    with open(param_file, 'r') as f:
        params = json.load(f)
    return MovingMNISTGenerator(**params)


def main():
    # Create the generator
    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_dir = os.path.join(script_dir, '..', 'params')
    gen = create_moving_mnist_generator(os.path.join(param_dir, 'test.json'))

    # Save tensor
    gen.save_video('output.npy')
    # Save preview video
    gen.save_video('output.avi')


if __name__ == '__main__':
    # np.random.seed(123)
    start = time()
    main()
    end = time()
    print('%d s' % (end-start))
