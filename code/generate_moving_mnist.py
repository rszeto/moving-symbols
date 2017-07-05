import os
from scipy.misc import imread
from time import time
import json

from utils import *

class MovingMNISTGenerator:

    __SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))

    def __init__(self,
                 data_dir=os.path.abspath(os.path.join(__SCRIPT_DIR__, '..', 'data')), split='training',
                 num_images=1, max_image_size=28, video_size=(64, 64), num_timesteps=30,
                 binary_output=False,
                 x_lim=None, y_lim=None,
                 x_init_lim=None, y_init_lim=None,
                 angle_lim=[0, 0], scale_lim=[1, 1],
                 x_speed_lim=[0, 0], y_speed_lim=[0, 0], scale_speed_lim=[0, 0], angle_speed_lim=[0, 0],
                 background_file_path=None):
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

        self.x_lim = [0, video_size[0]] if x_lim is None else x_lim
        self.y_lim = [0, video_size[1]] if y_lim is None else y_lim
        self.x_init_lim = self.x_lim if x_init_lim is None else x_init_lim
        self.y_init_lim = self.y_lim  if y_init_lim is None else y_init_lim

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

        # Set background image
        if self.is_src_grayscale:
            self.background = np.zeros((video_size[1], video_size[0]), dtype=np.uint8)
            if background_file_path:
                bg = imread(background_file_path)
                if bg.ndim == 2:
                    self.background[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0])] = \
                        bg[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0])]
                else:
                    raise ValueError('Only grayscale backgrounds are supported for grayscale digits')
        else:
            self.background = np.zeros((video_size[1], video_size[0], 4), dtype=np.uint8)
            self.background[:, :, 3] = 255
            if background_file_path:
                bg = imread(background_file_path)
                if bg.ndim != 3:
                    raise ValueError('Only color backgrounds are supported for colored digits')
                else:
                    if bg.shape[2] == 4:
                        raise NotImplementedError('Backgrounds with alpha channels are not supported')
                    else:
                        self.background[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0]), :3] = \
                            bg[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0]), :3]


    def __get_digit__(self):
        '''
        Get random digit images (cropped) and labels
        :return:
        '''
        label = np.random.randint(10)
        digit_dir = os.path.join(self.data_dir, self.split, str(label))
        image = imread(os.path.join(digit_dir, '%04d.png' % np.random.randint(len(os.listdir(digit_dir)))))
        # if label % 2 == 0:
        #     image = np.stack([image, np.zeros(image.shape), np.zeros(image.shape), image], axis=-1).astype(np.uint8)
        # else:
        #     image = np.stack([np.zeros(image.shape), image, np.zeros(image.shape), image], axis=-1).astype(np.uint8)
        crop = tight_crop(image)
        return crop, label


    def __sample_start_states__(self):
        '''
        Choose starting parameters based on possible position, scale, etc.
        :return:
        '''
        # Choose scale
        scale_start = np.random.uniform(self.scale_lim[0], self.scale_lim[1])
        # Choose angle
        if self.angle_lim[0] == self.angle_lim[1]:
            # Only one choice
            angle_start = self.angle_lim[0]
        elif self.angle_lim[0] < self.angle_lim[1]:
            # Going CCW ends up at larger angle
            angle_start = np.random.randint(self.angle_lim[0], self.angle_lim[1])
        else:
            # Going CCW ends up at smaller angle
            diff = self.angle_lim[1] + (360 - self.angle_lim[0])
            # Choose between 0 and diff, inclusive
            offset = np.random.randint(diff+1)
            # Move diff degrees past first interval endpoint
            angle_start = (self.angle_lim[0] + offset) % 360
        # Start far from the video frame border to avoid image clipping
        pad = (self.max_image_size / 2) * np.sqrt(2) * scale_start
        x_start = np.random.randint(np.ceil(self.x_init_lim[0] + pad), np.floor(self.x_init_lim[1] - pad))
        y_start = np.random.randint(np.ceil(self.y_init_lim[0] + pad), np.floor(self.y_init_lim[1] - pad))

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
                digit_frame = np.zeros((self.video_size[1], self.video_size[0], self.rotated_images[j].shape[2]), dtype=np.uint8)
                for c in range(digit_frame.shape[2]):
                    digit_frame[:, :, c] = cv2.warpPerspective(np.squeeze(self.rotated_images[j][:, :, c]), trans, self.video_size)
            digit_frames.append(digit_frame)

        # Overlay frames
        # stitched_frame = np.zeros(digit_frames[0].shape, dtype=np.uint8)
        stitched_frame = self.background
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

            # Convert frames to tensor
            self.video_tensor = np.stack(frames, axis=-1)
            # Discard alpha channel
            if self.video_tensor.ndim == 4 and self.video_tensor.shape[2] == 4:
                # Make sure all values from alpha channel are maximized
                alpha = self.video_tensor[:, :, 3, :]
                if np.max(alpha) == np.min(alpha) and np.min(alpha) == 255:
                    self.video_tensor = self.video_tensor[:, :, :3, :]
                else:
                    raise RuntimeError('Video tensor has alpha channel, but not all alpha values are maximized')


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
    gen = create_moving_mnist_generator(os.path.join(param_dir, 'toronto.json'))

    # Save tensor
    gen.save_video('output.npy')
    # Save preview video
    gen.save_video('output.avi')


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    print('%d s' % (end-start))
