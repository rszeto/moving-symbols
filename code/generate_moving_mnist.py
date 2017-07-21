import os
from scipy.misc import imread
from time import time
import json
from abc import ABCMeta, abstractmethod
from types import UnicodeType

from utils import *


class MovingMNISTEventObserver:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def receive_message(self, message):
        pass


class SimpleMovingMNISTLogger(MovingMNISTEventObserver):

    def __init__(self):
        self.messages = []

    def receive_message(self, message):
        self.messages.append(message)

    def print_messages(self):
        for message in self.messages:
            print(message)


class MovingMNISTGenerator:

    __SCRIPT_DIR__ = os.path.dirname(os.path.abspath(__file__))

    def __init__(self,
                 seed=0,
                 observers=[],
                 data_dir=os.path.abspath(os.path.join(__SCRIPT_DIR__, '..', 'data')), split='training',
                 num_images=1, max_image_size=28, video_size=(64, 64), num_timesteps=30,
                 x_lim=None, y_lim=None,
                 x_init_lim=None, y_init_lim=None,
                 angle_lim=[0, 0], scale_lim=[1, 1],
                 x_speed_lim=[0, 0], y_speed_lim=[0, 0], scale_speed_lim=[0, 0], angle_speed_lim=[0, 0],
                 use_background=False,
                 background_file_dir=os.path.abspath(os.path.join(__SCRIPT_DIR__, '..', 'backgrounds')),
                 background_file_cats=None,
                 background_file_id=None,
                 enable_image_interaction=False,
                 visual_debug=False,
                 use_color=False, image_colors=None,
                 digit_labels=None,
                 digit_image_id=None,
                 blink_rate=0):

        # Store copy of initial settings
        settings = locals().copy()
        # Remove variables not related to starting conditions
        del settings['self']
        del settings['observers']

        # Initialize RNG stuff
        self.__seed_counter__ = seed
        # Initialize step count
        self.step_count = 0

        # Initialize observer list
        self.observers = observers

        self.data_dir = data_dir
        self.split = split
        self.num_images = num_images
        self.max_image_size = max_image_size
        self.video_size = tuple(video_size)
        self.num_timesteps = num_timesteps
        self.angle_lim = None if angle_lim is None else [x % 360 for x in angle_lim]
        self.scale_lim = scale_lim
        self.x_speed_lim = x_speed_lim
        self.y_speed_lim = y_speed_lim
        self.scale_speed_lim = scale_speed_lim
        self.angle_speed_lim = angle_speed_lim
        self.enable_image_interaction = enable_image_interaction
        self.visual_debug = visual_debug
        self.blink_rate = blink_rate

        self.x_lim = [0, video_size[0]] if x_lim is None else x_lim
        self.y_lim = [0, video_size[1]] if y_lim is None else y_lim
        # If unspecified, set init limits far from the border to avoid image clipping
        pad = (self.max_image_size / 2) * scale_lim[1]
        self.x_init_lim = [
            int(np.ceil(self.x_lim[0] + pad)),
            int(np.floor(self.x_lim[1] - pad))
        ] if x_init_lim is None else x_init_lim
        self.y_init_lim = [
            int(np.ceil(self.y_lim[0] + pad)),
            int(np.floor(self.y_lim[1] - pad))
        ] if y_init_lim is None else y_init_lim

        # Set image channel count
        self.use_color = use_color
        self.image_colors = image_colors

        # Emit settings message
        message = dict(
            type='settings',
            step=-1,
            meta=settings.copy()
        )
        self.publish_message(message)

        # Check that digit_image_id and digit_labels are compatible. If digit_image_id is
        # defined, digit_labels must have length 1.
        if digit_image_id is not None and (digit_labels is None or len(digit_labels) != 1):
            raise ValueError('If digit_image_id is defined, digit_labels must be given and '
                             'have length 1')

        # Get digits and split them into image and label lists
        self.digit_image_id = digit_image_id
        self.digit_labels = range(10) if digit_labels is None else digit_labels
        digit_infos = [self.__get_digit__(i) for i in range(num_images)]
        self.images, self.labels = zip(*digit_infos)
        self.images = list(self.images)
        self.labels = list(self.labels)
        if self.use_color:
            self.__colorize_digits__()

        # Sample starting states and update params, and initialize rotated images and BBs
        self.states = [self.__sample_start_states__(i) for i in range(num_images)]
        self.update_params = [self.__sample_update_params__(i) for i in range(num_images)]
        self.init_rotated_images_and_bounding_boxes()

        # Set empty video cache
        self.video_tensor = None
        self.ran_dynamics = False

        # Setting background
        # Check consistency of parameters
        if background_file_id is not None:
            if not use_background:
                raise ValueError('If background_file_id is given, then use_background must be True')
            if not background_file_cats or len(background_file_cats) != 1:
                raise ValueError('If background_file_id is given, then background_file_cats must be'
                                 ' given and have length 1')
        # Set background parameters and background
        self.use_background = use_background
        self.background_file_dir = background_file_dir
        self.background_file_cats = background_file_cats if background_file_cats \
            else os.listdir(self.background_file_dir)
        self.background_file_id = background_file_id
        self.set_background()


    def set_background(self):
        '''
        Sample and set the background image
        :return:
        '''

        # Sample background image directory
        self.__reseed_rng__()
        bg_folder = os.path.join(self.background_file_dir, self.background_file_cats[
            np.random.randint(len(self.background_file_cats))
        ])
        # Sample background from folder
        self.__reseed_rng__()
        background_file_path = os.path.join(bg_folder, '%04d.jpg'
                                            % np.random.randint(len(os.listdir(bg_folder))))
        # Override if the background ID was defined
        if self.background_file_id:
            background_file_path = os.path.join(bg_folder, '%04d.jpg' % self.background_file_id)

        # Override background if choosing not to use one. This is needed here so all random seeds
        # are called
        if not self.use_background:
            background_file_path = None

        video_size = self.video_size
        if not self.use_color:
            self.background = np.zeros((video_size[1], video_size[0]), dtype=np.uint8)
            if background_file_path:
                bg = imread(background_file_path)
                if bg.ndim != 2:
                    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
                self.background[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0])] = \
                    bg[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0])]
        else:
            self.background = np.zeros((video_size[1], video_size[0], 4), dtype=np.uint8)
            self.background[:, :, 3] = 255
            if background_file_path:
                bg = imread(background_file_path)
                if bg.ndim != 3:
                    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
                self.background[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0]), :3] = \
                    bg[:min(bg.shape[0], video_size[1]), :min(bg.shape[1], video_size[0]), :3]

        # Publish background message
        message = dict(
            type='background',
            step=-1,
            meta=dict(
                image_path=background_file_path
            )
        )
        self.publish_message(message)


    def publish_message(self, message):
        self.verify_message(message)
        for observer in self.observers:
            observer.receive_message(message)


    def verify_message(self, message):
        '''
        Makes sure a given message is properly defined
        :param message:
        :return:
        '''
        assert(set_equal(message.keys(), ['type', 'step', 'meta']))

        message_type = message['type']
        step = message['step']
        meta = message['meta']

        assert(isinstance(step, int) and step >= -1)
        assert(isinstance(meta, dict))

        if message_type == 'settings':
            assert(step == -1)
        elif message_type == 'digit':
            assert(step == -1)
            assert(set_equal(meta.keys(), ['label', 'image_path', 'id']))
            assert(isinstance(meta['label'], int))
            assert(isinstance(meta['image_path'], str))
            assert(isinstance(meta['id'], int))
        elif message_type == 'digit_color':
            assert(step == -1)
            assert(set_equal(meta.keys(), ['digit_id', 'color']))
            assert(isinstance(meta['digit_id'], int))
            assert(isinstance(meta['digit_id'], int))
        elif message_type == 'background':
            assert(step == -1)
            assert(set_equal(meta.keys(), ['image_path']))
            assert(meta['image_path'] is None
                   or isinstance(meta['image_path'], UnicodeType)
                   or isinstance(meta['image_path'], str))
        elif message_type == 'start_state':
            assert(step == -1)
            assert(set_equal(meta.keys(), ['digit_id', 'scale', 'x', 'y', 'angle']))
            for key, value in meta.iteritems():
                if key == 'scale':
                    assert(isinstance(value, float))
                else:
                    assert(isinstance(value, int))
        elif message_type == 'start_update_params':
            assert(step == -1)
            assert(set_equal(meta.keys(), ['digit_id', 'scale_speed', 'x_speed', 'y_speed', 'angle_speed']))
            for key, value in meta.iteritems():
                if key == 'scale_speed':
                    assert (isinstance(value, float))
                else:
                    assert (isinstance(value, int))
        elif message_type == 'reverse_scale_speed':
            assert(step > -1)
            assert(set_equal(meta.keys(), ['digit_id', 'new_direction']))
            assert(isinstance(meta['digit_id'], int))
            assert(isinstance(meta['new_direction'], int))
            assert(abs(meta['new_direction']) == 1)
        elif message_type == 'reverse_angle_speed':
            assert(step > -1)
            assert(set_equal(meta.keys(), ['digit_id', 'new_direction']))
            assert(isinstance(meta['digit_id'], int))
            assert(isinstance(meta['new_direction'], int))
            assert(abs(meta['new_direction']) == 1)
        elif message_type == 'overlap':
            assert(step > -1)
            assert(set_equal(meta.keys(), ['digit_id_a', 'digit_id_b']))
            assert(isinstance(meta['digit_id_a'], int))
            assert(isinstance(meta['digit_id_b'], int))
        elif message_type == 'bounce_off_digit':
            assert(step > -1)
            assert(set_equal(meta.keys(), ['digit_id_a', 'digit_id_b']))
            assert(isinstance(meta['digit_id_a'], int))
            assert(isinstance(meta['digit_id_b'], int))
        elif message_type == 'bounce_off_wall':
            assert(step > -1)
            assert(set_equal(meta.keys(), ['digit_id', 'wall_label']))
            assert(isinstance(meta['digit_id'], int))
            assert(meta['wall_label'] in ['north', 'south', 'east', 'west'])
        else:
            raise NotImplementedError('Messages of type %s are not supported' % message_type)


    def __reseed_rng__(self):
        '''
        Reseed the RNG. This must be called before each np.random.* call
        :return: 
        '''
        self.__seed_counter__ += 1
        np.random.seed(self.__seed_counter__)


    def __get_digit__(self, i):
        '''
        Get random digit images (cropped) and labels
        :param i: The digit ID
        :return:
        '''
        self.__reseed_rng__()
        label = self.digit_labels[np.random.randint(len(self.digit_labels))]
        digit_dir = os.path.join(self.data_dir, self.split, str(label))
        self.__reseed_rng__()
        image_path = os.path.join(digit_dir, '%04d.png' % np.random.randint(len(os.listdir(digit_dir))))
        # Override if a specific image was specified
        if self.digit_image_id is not None:
            image_path = os.path.join(digit_dir, '%04d.png' % self.digit_image_id)
        image = imread(image_path)
        crop = tight_crop(image)

        # Publish digit message
        message = dict(
            type='digit',
            step=-1,
            meta=dict(
                label=label,
                image_path=image_path,
                id=i
            )
        )
        self.publish_message(message)

        return crop, label


    def __colorize_digits__(self):
        '''
        Convert the stored digits into RGBA images
        :return:
        '''
        if not self.use_color:
            raise ValueError('Attempted to colorize digits for grayscale video')
        if self.images[0].ndim != 2:
            raise RuntimeError('Attempted to colorize digits more than once')

        # Create digit image with alpha channel
        for i in range(self.num_images):
            crop = self.images[i]
            color = [255, 255, 255] if self.image_colors is None else self.image_colors[i % len(self.image_colors)]
            channels = [color[c] * np.ones(crop.shape) for c in range(3)]
            channels.append(crop)
            crop = np.stack(channels, axis=-1).astype(np.uint8)
            self.images[i] = crop

            # Publish color message
            message = dict(
                type='digit_color',
                step=-1,
                meta=dict(
                    digit_id=i,
                    color=color
                )
            )
            self.publish_message(message)


    def __sample_start_states__(self, i):
        '''
        Choose starting parameters based on possible position, scale, etc.
        :param i: The digit ID
        :return:
        '''
        # Choose scale
        self.__reseed_rng__()
        scale_start = np.random.uniform(self.scale_lim[0], self.scale_lim[1])
        # Choose angle. Start by choosing position in range as percentile
        self.__reseed_rng__()
        angle_percent = np.random.uniform()
        if self.angle_lim[0] <= self.angle_lim[1]:
            # Choose point in range
            angle_diff = self.angle_lim[1] - self.angle_lim[0]
            angle_start = self.angle_lim[0] + angle_percent * angle_diff
            # Discretize
            angle_start = int(np.round(angle_start))
        else:
            # Going CCW ends up at smaller angle
            angle_diff = self.angle_lim[1] + (360 - self.angle_lim[0])
            angle_start = self.angle_lim[0] + angle_percent * angle_diff
            # Discretize and place within [0, 360)
            angle_start = int(np.round(angle_start)) % 360
        # Start far from the video frame border to avoid image clipping
        self.__reseed_rng__()
        x_start = np.random.randint(self.x_init_lim[0], self.x_init_lim[1] + 1)
        self.__reseed_rng__()
        y_start = np.random.randint(self.y_init_lim[0], self.y_init_lim[1] + 1)

        # Set start state
        start_state = dict(
            digit_id=i,
            scale=scale_start,
            x=x_start,
            y=y_start,
            angle=angle_start
        )

        # Publish start state
        message = dict(
            type='start_state',
            step=-1,
            meta=start_state.copy()
        )
        self.publish_message(message)

        return start_state


    def __sample_update_params__(self, i):
        '''
        Choose update parameters based on possible position, scale updates, etc.
        :return:
        '''
        self.__reseed_rng__()
        x_speed = int(np.floor(np.random.uniform(self.x_speed_lim[0],
                                    self.x_speed_lim[1])))
        self.__reseed_rng__()
        y_speed = int(np.floor(np.random.uniform(self.y_speed_lim[0],
                                    self.y_speed_lim[1])))
        self.__reseed_rng__()
        scale_speed = np.random.uniform(self.scale_speed_lim[0],
                                        self.scale_speed_lim[1])
        self.__reseed_rng__()
        angle_speed = int(np.floor(np.random.uniform(self.angle_speed_lim[0],
                                                     self.angle_speed_lim[1])))

        update_params = dict(
            digit_id=i,
            scale_speed=scale_speed,
            x_speed=x_speed,
            y_speed=y_speed,
            angle_speed=angle_speed
        )

        # Publish update params
        message = dict(
            type='start_update_params',
            step=-1,
            meta=update_params.copy()
        )
        self.publish_message(message)

        return update_params


    def render_current_state(self):
        '''
        Render all the digits onto an image
        :return:
        '''
        # Get frame for each digit
        digit_frames = []
        for j in range(self.num_images):
            trans = get_center_translation_matrix(self.rotated_images[j], self.states[j]['x'], self.states[j]['y'])
            if not self.use_color:
                digit_frame = cv2.warpPerspective(self.rotated_images[j], trans, self.video_size)
            else:
                digit_frame = np.zeros((self.video_size[1], self.video_size[0], self.rotated_images[j].shape[2]), dtype=np.uint8)
                for c in range(digit_frame.shape[2]):
                    digit_frame[:, :, c] = cv2.warpPerspective(np.squeeze(self.rotated_images[j][:, :, c]), trans, self.video_size)
            digit_frames.append(digit_frame)

        # Overlay frames
        stitched_frame = self.background
        # Draw digits if this is not a blink frame
        if self.blink_rate <= 1 or (self.blink_rate > 1 and (
                        (self.step_count + 1) % self.blink_rate != 0)):
            for j in range(self.num_images):
                stitched_frame = overlay_image(digit_frames[j], stitched_frame)

            # Draw bounding boxes
            if self.visual_debug:
                for j in range(self.num_images):
                    a = (int(self.bounding_boxes[j][0][0]), int(self.bounding_boxes[j][0][1]))
                    b = (int(self.bounding_boxes[j][1][0]), int(self.bounding_boxes[j][1][1]))
                    if self.use_color:
                        stitched_frame[:, :, :3] = cv2.rectangle(stitched_frame[:, :, :3].copy(), a, b, (128, 128, 128))
                    else:
                        stitched_frame = cv2.rectangle(stitched_frame, a, b, 128)

        return stitched_frame


    def init_rotated_images_and_bounding_boxes(self):
        '''
        Set the self.rotated_images and self.bounding_boxes fields based on the computed images
        :return:
        '''
        self.rotated_images = []
        self.bounding_boxes = []

        for j in range(self.num_images):
            image_state = self.states[j]
            # Generate the cropped image
            cropped_digit = get_rotated_scaled_image_tight_crop(self.images[j], image_state['angle'], image_state['scale'])
            self.rotated_images.append(cropped_digit)

            # Compute the coordinates of each box's corners
            x_right = image_state['x'] + cropped_digit.shape[1] / 2
            x_left = image_state['x'] - cropped_digit.shape[1] / 2
            y_bottom = image_state['y'] + cropped_digit.shape[0] / 2
            y_top = image_state['y'] - cropped_digit.shape[0] / 2
            self.bounding_boxes.append([(x_left, y_top), (x_right, y_bottom)])



    def step(self):
        '''
        Step through the dynamics for each digit
        :return:
        '''
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

                # Publish scale speed reversal message
                message = dict(
                    type='reverse_scale_speed',
                    step=self.step_count,
                    meta=dict(
                        digit_id=j,
                        new_direction=-1
                    )
                )
                self.publish_message(message)

            elif image_state['scale'] < self.scale_lim[0]:
                image_state['scale'] = self.scale_lim[0] - (image_state['scale'] - self.scale_lim[0])
                update_params['scale_speed'] *= -1

                # Publish scale speed reversal message
                message = dict(
                    type='reverse_scale_speed',
                    step=self.step_count,
                    meta=dict(
                        digit_id=j,
                        new_direction=1
                    )
                )
                self.publish_message(message)

            # Angle
            new_angle = image_state['angle']
            if self.angle_lim is not None:
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

                    # Publish reverse angle speed message
                    message = dict(
                        type='reverse_angle_speed',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            new_direction=int(np.sign(update_params['angle_speed']))
                        )
                    )
                    self.publish_message(message)

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

                    # Publish reverse angle speed message
                    message = dict(
                        type='reverse_angle_speed',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            new_direction=int(np.sign(update_params['angle_speed']))
                        )
                    )
                    self.publish_message(message)

            image_state['angle'] %= 360

            # Generate the cropped image
            cropped_digit = get_rotated_scaled_image_tight_crop(self.images[j], image_state['angle'], image_state['scale'])
            self.rotated_images[j] = cropped_digit

            # Compute the coordinates of each box's corners
            x_right = image_state['x'] + cropped_digit.shape[1] / 2
            x_left = image_state['x'] - cropped_digit.shape[1] / 2
            y_bottom = image_state['y'] + cropped_digit.shape[0] / 2
            y_top = image_state['y'] - cropped_digit.shape[0] / 2
            self.bounding_boxes[j] = [(x_left, y_top), (x_right, y_bottom)]

            # Bounce image off other images
            for k in range(self.num_images):
                if j == k: continue
                if is_overlapping(self.bounding_boxes[k], self.bounding_boxes[j]):
                    if self.enable_image_interaction:
                        other_box = self.bounding_boxes[k]
                        # Find the angle (w/ y pointing up) of current image's center relative to other image
                        angle_rad = np.arctan2(self.states[k]['y']-image_state['y'], image_state['x']-self.states[k]['x'])
                        angle_deg = (angle_rad * 180 / np.pi + 360) % 360
                        if (angle_deg >= 0 and angle_deg < 45) or (angle_deg >= 315 and angle_deg <= 359):
                            # Align current image's left side with other image's right
                            x_left = other_box[1][0] + 1
                            image_state['x'] = x_left + cropped_digit.shape[1] / 2
                            x_right = image_state['x'] + cropped_digit.shape[1] / 2
                        elif (angle_deg >= 45 and angle_deg < 135):
                            # Align current image's bottom side with other image's top
                            y_bottom = other_box[0][1] - 1
                            image_state['y'] = y_bottom - cropped_digit.shape[0] / 2
                            y_top = image_state['y'] - cropped_digit.shape[0] / 2
                        elif (angle_deg >= 135 and angle_deg < 225):
                            # Align current image's right side with other image's left side
                            x_right = other_box[0][0] - 1
                            image_state['x'] = x_right - cropped_digit.shape[1] / 2
                            x_left = image_state['x'] - cropped_digit.shape[1] / 2
                        else:
                            # Align current image's top side with other image's bottom
                            y_top = other_box[1][1] + 1
                            image_state['y'] = y_top + cropped_digit.shape[0] / 2
                            y_bottom = image_state['y'] + cropped_digit.shape[0] / 2

                        v1 = np.array((update_params['x_speed'], update_params['y_speed']))
                        v2 = np.array((self.update_params[k]['x_speed'], self.update_params[k]['y_speed']))
                        x1 = np.array((image_state['x'], image_state['y']))
                        x2 = np.array((self.states[k]['x'], self.states[k]['y']))
                        v1_new = v1 - np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)
                        v2_new = v2 - np.dot(v2 - v1, x2 - x1) / np.linalg.norm(x2 - x1) ** 2 * (x2 - x1)
                        update_params['x_speed'] = v1_new[0]
                        update_params['y_speed'] = v1_new[1]
                        self.update_params[k]['x_speed'] = v2_new[0]
                        self.update_params[k]['y_speed'] = v2_new[1]

                        # Publish bounce message
                        message = dict(
                            type='bounce_off_digit',
                            step=self.step_count,
                            meta=dict(
                                digit_id_a=j,
                                digit_id_b=k
                            )
                        )
                        self.publish_message(message)
                    else:
                        # Publish overlap message if current digit overlaps the other
                        if j > k:
                            message = dict(
                                type='overlap',
                                step=self.step_count,
                                meta=dict(
                                    digit_id_a=j,
                                    digit_id_b=k
                                )
                            )
                            self.publish_message(message)

            # Bounce image off the walls (requires image size)
            if x_right > self.x_lim[1] or x_left < self.x_lim[0]:
                if x_right > self.x_lim[1]:
                    # Hit right (east) wall
                    x_right = self.x_lim[1]
                    image_state['x'] = x_right - cropped_digit.shape[1] / 2
                    x_left = image_state['x'] - cropped_digit.shape[1] / 2

                    # Publish wall message
                    message = dict(
                        type='bounce_off_wall',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            wall_label='east'
                        )
                    )
                    self.publish_message(message)
                else:
                    # Hit left wall
                    x_left = self.x_lim[0]
                    image_state['x'] = x_left + cropped_digit.shape[1] / 2
                    x_right = image_state['x'] + cropped_digit.shape[1] / 2

                    # Publish wall message
                    message = dict(
                        type='bounce_off_wall',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            wall_label='west'
                        )
                    )
                    self.publish_message(message)
                # Flip x speed
                update_params['x_speed'] *= -1

            if y_bottom > self.y_lim[1] or y_top < self.y_lim[0]:
                if y_bottom > self.y_lim[1]:
                    # Hit bottom (south) wall
                    y_bottom = self.y_lim[1]
                    image_state['y'] = y_bottom - cropped_digit.shape[0] / 2
                    y_top = image_state['y'] - cropped_digit.shape[0] / 2

                    # Publish wall message
                    message = dict(
                        type='bounce_off_wall',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            wall_label='south'
                        )
                    )
                    self.publish_message(message)
                else:
                    # Hit top (north) wall
                    y_top = self.y_lim[0]
                    image_state['y'] = y_top + cropped_digit.shape[0] / 2
                    y_bottom = image_state['y'] + cropped_digit.shape[0] / 2

                    # Publish wall message
                    message = dict(
                        type='bounce_off_wall',
                        step=self.step_count,
                        meta=dict(
                            digit_id=j,
                            wall_label='north'
                        )
                    )
                    self.publish_message(message)
                # Flip y speed
                update_params['y_speed'] *= -1

            self.bounding_boxes[j] = [(x_left, y_top), (x_right, y_bottom)]

        # Increment step
        self.step_count += 1


    def get_video_tensor_copy(self):
        '''
        Get a copy of the video tensor
        :return:
        '''
        if not self.ran_dynamics:
            raise RuntimeError('You must call run_dynamics() before obtaining a video tensor copy')
        return self.video_tensor.copy()


    def run_dynamics(self):
        '''
        Run through the dynamics for the whole time period, populating the video tensor
        and event log
        :return:
        '''
        if not self.ran_dynamics:
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

            # Update run flag
            self.ran_dynamics = True