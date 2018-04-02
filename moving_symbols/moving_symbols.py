"""@mainpage Moving Symbols API

# Entry Point

The `moving_symbols` package provides the `MovingSymbolsEnvironment` class, which is the one you
should be using to generate Moving Symbols videos.

# Tiny Example

The following code snippet puts the frames of one Moving Symbols video into a list:

```python
    from moving_symbols import MovingSymbolsEnvironment

    env = MovingSymbolsEnvironment(params, seed)

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
```

# %MovingSymbolsEnvironment as a Publisher

A MovingSymbolsEnvironment instance publishes messages corresponding to the initialization and
state of each symbol at all time steps. The following code snippet shows an example where a
subscriber collects all the published messages:

```python
    from moving_symbols import MovingSymbolsEnvironment

    class Subscriber:
        def process_message(self, message):
            print(message)

    env = MovingSymbolsEnvironment(params, seed)
    sub = Subscriber()
    env.add_subscriber(sub)

    all_frames = []
    for _ in xrange(num_frames):
        frame = env.next()
        all_frames.append(np.array(frame))
```

Messages start getting published as soon as `env.next()` is first called.

"""

import math
import os
import sys
from warnings import warn

import cv2
import h5py
import numpy as np
import pygame as pg
import pymunk as pm
import pymunk.pygame_util as pmu
from PIL import Image

from moving_symbols_utils import merge_dicts, tight_crop, compute_pm_hull_vertices, \
    create_sine_fn, create_triangle_fn

_COLLISION_TYPES = dict(
    symbol=0,
    wall=1
)


class FileImageLoader:
    """File system-based image loader.

    This class samples images (with their corresponding file path and label) from a dataset stored
    on a file system.
    """

    def __init__(self, root, split, labels, mode=None):
        """Constructor

        @param root: The path to the root directory containing all images.
        @param split: The name of the split within the root directory to sample images from.
        @param labels: list of possible labels to sample from
        @param mode: String that indicates how to transform the image for rendering. Options
        include "tight_crop", which crops based on the alpha channel, or None for no processing.
        """

        self.root = root
        self.split = split
        self.labels = labels
        self.mode = mode


    def get_image(self):
        """Return a random, pre-processed image, along with its label and source path.

        The image is pre-processed based on the ImageLoader's mode.

        @retval image: A PIL Image from the given label set
        @retval image_path: The path to the selected image
        """

        label = self.labels[np.random.randint(len(self.labels))]
        class_path = os.path.join(self.root, self.split, str(label))
        class_image_names = os.listdir(class_path)
        image_idx = np.random.randint(len(class_image_names))
        image_path = os.path.join(class_path, class_image_names[image_idx])
        image = Image.open(image_path, 'r')
        if self.mode == 'tight_crop':
            image = tight_crop(image)
        return image, label, image_path


    def __str__(self):
        return 'fudge'


class HDF5ImageLoader:
    """HDF5 file-based image loader.

    This class samples images (with their corresponding file path and label) from a dataset stored
    in an HDF5 file.
    """

    def __init__(self, h5_file_path, split, labels, mode=None):
        """Constructor

        @param h5_file_path: The path to an HDF5 file containing all images.
        @param split: The name of the split within the root directory to sample images from.
        @param labels: list of possible labels to sample from
        @param mode: String that indicates how to transform the image for rendering. Options
        include "tight_crop", which crops based on the alpha channel, or None for no processing.
        """

        self.h5_file_path = h5_file_path
        self.mode = mode
        self.labels = labels
        self.split = split

        self.h5_file = h5py.File(h5_file_path)


    def get_image(self):
        """Return a random, pre-processed image, along with its label and source path.

        The image is pre-processed based on the ImageLoader's mode.

        @retval image: A PIL Image from the given label set
        @retval image_path: The path to the selected image, formatted as:
                            <h5_file_path>:/<split>/<image_class>[<image_num>]
        """

        label = self.labels[np.random.randint(len(self.labels))]
        # Get the given image class dataset
        image_class_dataset = self.h5_file['/%s/%s' % (self.split, label)]
        # Get random image
        image_idx = np.random.randint(len(image_class_dataset))
        image_np = image_class_dataset[image_idx]
        image = Image.fromarray(image_np, 'RGBA')
        # Process image
        if self.mode == 'tight_crop':
            image = tight_crop(image)

        # Get image path
        image_path = '%s:/%s/%s[%d]' % (self.h5_file_path, self.split, label, image_idx)

        return image, label, image_path



class Symbol:

    def __init__(self, id, label, image, image_path, scale_fn):
        """Constructor

        @param id: A numerical ID for this symbol
        @param label: The class label of this symbol
        @param image: The base PIL image for this symbol
        @param image_path: The path to the (unprocessed) image file
        @param scale_fn: A function that returns the scale of the symbol at any given time t
        """
        self.id = id
        self.label = label
        self.image = image
        self.image_path = image_path
        self.pg_image = pg.image.fromstring(image.tobytes(), image.size, image.mode)

        # Get vertices
        hull = compute_pm_hull_vertices(self.image)

        self.body = pm.Body(1, pm.inf)
        self._base_vertices = [pt - [dim/2. for dim in self.image.size] for pt in hull]
        self.shape = pm.Poly(self.body, self._base_vertices)
        self.scale_fn = scale_fn
        self.scale = None

        # Store the angular velocity
        self.angular_velocity = None

        # Set random things so objects interact properly
        self.shape.elasticity = 1.0
        self.shape.collision_type = _COLLISION_TYPES['symbol']


    def get_render_image_and_position(self, screen_size):
        """Get the PyGame Surface and center coordinate of the scaled, rotated symbol.

        @param screen_size: (width, height) of the PyGame screen
        @retval rotated_image: pygame.Surface of the scaled, rotated image
        @retval pg_image_pos: PyGame coordinates of the scaled, rotated image
        """
        # Get scaled image
        scaled_image = pg.transform.smoothscale(
            self.pg_image, tuple([int(x * self.scale) for x in self.image.size])
        )
        # Get body position in PyGame coordinates
        pg_body_pos = pm.Vec2d(self.body.position.x, -self.body.position.y + screen_size[1])
        # Rotate image
        angle_deg = math.degrees(self.body.angle)
        rotated_image = pg.transform.rotate(scaled_image, angle_deg)
        # Offset image coordinates from body due to rotation
        offset = pm.Vec2d(rotated_image.get_size()) / 2.
        pg_image_pos = pg_body_pos - offset

        return rotated_image, pg_image_pos


    def set_scale(self, step):
        """Set the scale of the symbol to the scaling function at the given time step.

        @param step: The current time step of the environment
        """
        self.scale = self.scale_fn(step)
        self.shape.unsafe_set_vertices(self._base_vertices,
                                       transform=pm.Transform(self.scale, 0, 0, self.scale, 0, 0))


    def get_state_message(self, step):
        """Produce a message about the state of the symbol at the given time step.

        Return a message containing information about the current pose and motion of the symbol.
        Everything is a float except for id (int), position (np.float array with shape (2,)),
        and velocity (np.float array with shape (2,)).

        @param step: The time step of the MovingSymbolEnvironment
        @retval message: A dict describing the symbol's state. Its key-value pairs are:
                         - step: Current time step
                         - type: 'symbol_state'
                         - meta: dict with following key-value pairs:
                            - symbol_id: The ID for this symbol
                            - position: The symbol's PyGame coordinates as an np.array
                            - angle: The symbol's PyGame angle
                            - scale: The symbol's scale
                            - velocity: The symbol's PyGame velocity as an np.array
                            - angular_velocity: The symbol's angular velocity
                            - scale_velocity: The symbol's scale velocity
        """
        dt = 0.001
        scale_velocity = (self.scale_fn(step + dt) - self.scale_fn(step)) / dt
        return dict(
            step=step,
            type='symbol_state',
            meta=dict(
                symbol_id=self.id,
                position=np.array(self.body.position),
                angle=self.body.angle,
                scale=self.scale,
                velocity=np.array(self.body.velocity),
                angular_velocity=self.angular_velocity,
                scale_velocity=scale_velocity
            )
        )

    def get_init_message(self):
        """Produce a message about fixed properties of the symbol, i.e.\ image data.

        Return a message containing information required to reconstruct the appearance and shape
        of the symbol. The returned meta information includes the symbol ID, the image as a HxWx4
        np.uint8 array, the path to the source image, and the vertices of the symbol shape,
        in PyMunk coordinates, as a Vx2 np.float array.

        @retval message: A dict describing the symbol's initial state. Its key-value pairs are:
                         - step: Current time step
                         - type: 'symbol_init'
                         - meta: dict with following key-value pairs:
                            - symbol_id: The ID for this symbol
                            - label: The class label of this symbol
                            - image: An np.array of the full image (dimensions H x W x 4)
                            - image_path: Path to the source image (uncropped)
                            - vertices: The PyMunk coordinates defining the symbol's hitbox as a
                              V x 2 np.float array
        """
        ret = dict(
            step=-1,
            type='symbol_init',
            meta=dict(
                symbol_id=self.id,
                label=self.label,
                image=np.array(self.image),
                image_path=self.image_path,
                vertices=np.array(self._base_vertices)
            )
        )
        return ret


class MovingSymbolsEnvironment:
    """Generator that produces Moving %Symbol video frames.

    This class manages a physical environment in which symbols move around. It also handles
    rendering of the current physical state. Renders are returned as PIL images, either in RGB or
    L (8-bit grayscale) mode. It implements the native Python generator interface.

    The physical state is initialized based on the parameters given to the constructor (default
    values are supplied by DEFAULT_PARAMS). Below are the key-value pairs that can be specified:

    - **symbol_image_loader, FileImageLoader or HDF5ImageLoader**: The image dataset sampler from
      which to sample symbols
    - **num_symbols, int**: How many symbols should appear in the video
    - **video_size, (int, int)**: The resolution of the video as (width, height)
    - **color_output, bool**: Whether to produce "RGB" color images or "L" grayscale images
    - **symbol_labels, Sequence**: The labels for the symbol classes. These must be strings or
      ints (or any object with __str__ implemented) that match the names of the folders in each
      split directory
    - **scale_limits, (float, float)**: The minimum and maximum scale of an object relative to its
      original size
    - **scale_period_limits, (float, float) or list of (float, float)**: The minimum and maximum
      duration of a full scale cycle in number of frames
    - **rotation_speed_limits, (float, float) or list of (float, float)**: The minimum and maximum
      angular speed, in radians per frame
    - **position_speed_limits, (float, float) or list of (float, float)**: The minimum and maximum
      translational speed, in pixels per frame
    - **interacting_symbols, bool**: Whether symbols will bounce off each other
    - **scale_function_type, str**: The class of function used to define the scale of each symbol at
      each time step. Supported options are:
      - "sine": Scale is determined by a sine wave
      - "triangle": Scale is determined by a triangle wave (i.e. scaling speed is constant,
        but switches directions if the digit gets too big or small)
      - "constant": The symbols do not change scale. Initial scale is randomly sampled from
        within scale_limits
    - **rotate_at_start, bool**: Whether symbols can start at a rotated angle
    - **rescale_at_start, bool**: Whether symbols can start at any scale in the specified range. If
      not. the scale of all symbols is initialized to the middle of the allowed scale range.
    - **lateral_motion_at_start, bool**: Whether symbols can only translate left/right/up/down to
      start. If this is True, symbols can only move non-laterally if they bounce off of other
      symbols.
    - **background_image_loader, FileImageLoader or HDF5ImageLoader**: The image dataset sampler
      from which to sample backgrounds

    A MovingSymbolsEnvironment object also publishes messages that describe each symbol's
    initialization and state; subscribers can be added with add_subscriber().
    A subscriber must implement the `process_message` method that takes exactly one argument, a
    `dict` containing the published message. All messages have the following key-value pairs:

    - **step, int**: The time step that the message describes. For initialization messages,
    this is -1. The first rendered frame corresponds to step 0.
    - **type, str**: An identifier for what kind of message this is. This can be used to filter
      out messages and determine the structure of the meta-information without probing it.
    - **meta, dict**: The actual contents of the message.

    Below is a list of key-value pairs associated with the `meta` dict for each message type:

    * **params**: This is a copy of the `params` argument passed in to the MovingSymbolsEnvironment
      object's constructor.
    * **debug_options**: This is a copy of the `debug_options` argument passed in to the
    MovingSymbolsEnvironment object's constructor.
    * **background**
        - label: The name of the background class
        - image: An np.array containing the full background image of the video
        - image_path: Path to the source image
    * **symbol_init**
      - symbol_id: The `id` field of the relevant Symbol
      - label: The `label` field of the relevant Symbol
      - image: An np.array of the Symbol's full image (dimensions H x W x 4)
      - image_path: Path to the Symbol's source image (uncropped)
      - vertices: The PyMunk coordinates defining the Symbol's hitbox as a V x 2 np.float array
    * **symbol_state**
      - symbol_id: The `id` for the relevant Symbol
      - position: The Symbol's PyGame coordinates as an np.array
      - angle: The Symbol's PyGame angle
      - scale: The Symbol's `scale`
      - velocity: The Symbol's PyGame velocity as an np.array
      - angular_velocity: The Symbol's `angular_velocity`
      - scale_velocity: The Symbol's scale velocity
    * **start_overlap**
        - symbol_ids: The `id`s of the Symbols that have started overlapping at this time step
        (as a list)
    * **end_overlap**
        - symbol_ids: The `id`s of the Symbols that have stopped overlapping at this time step
        (as a list)
    * **hit_wall**
        - symbol_id: The `id` of the Symbol that hit a wall at this time step
        - wall_label: A label for the wall that was hit at this time step. Can be "left",
        "right", "top", or "bottom".

    """

    DEFAULT_PARAMS = dict(
        num_symbols=1,
        video_size=(64, 64),
        color_output=True,
        scale_limits=(1.0, 1.0),
        scale_period_limits=(1, 1),
        rotation_speed_limits=(0, 0),
        position_speed_limits=(0, 0),
        interacting_symbols=False,
        scale_function_type='constant',
        rotate_at_start=False,
        rescale_at_start=True,
        lateral_motion_at_start=False,
        symbol_image_loader=None,
        background_image_loader=None
    )

    DEFAULT_DEBUG_OPTIONS = dict(
        frame_number_font_size=30,
        show_pymunk_debug=False,
        show_bounding_poly=False,
        show_frame_number=False,
        frame_rate=sys.maxint
    )


    def __init__(self, params, seed, fidelity=10, debug_options=None):
        """Constructor

        @param params: dict of parameters that define how symbols behave and are rendered. See the
        detailed description for this class for supported parameters.
        @param seed: Seed for the RNG (int)
        @param fidelity: How many iterations to run in the physics simulator per step (int)
        @param debug_options: dict with options for visual debugging, or None if visual debugging
                              should be turned off. The following key-value pairs are supported:
                              - show_pymunk_debug, bool: Whether to use PyMunk's default drawing
                                function
                              - show_bounding_poly, bool: Whether to render PyMunk surface outlines
                              - show_frame_number, bool: Whether to show the index of the frame
                              - frame_number_font_size, int: Size of the frame index font
                              - frame_rate, int: Frame rate of the debug visualization

        """

        # Make sure symbol loader is defined
        if not params.get('symbol_image_loader', False):
            raise ValueError('No symbol image loader was provided')

        self.params = merge_dicts(MovingSymbolsEnvironment.DEFAULT_PARAMS, params)
        self.fidelity = fidelity
        self.debug_options = None if debug_options is None \
            else merge_dicts(MovingSymbolsEnvironment.DEFAULT_DEBUG_OPTIONS, debug_options)
        self.video_size = self.params['video_size']

        self._subscribers = []
        self._init_messages = []
        self._step_called = False

        self._add_init_message(dict(
            step=-1,
            type='params',
            meta=dict(self.params)
        ))

        # Convert translation/rotation/scale period/speed limits to lists
        if isinstance(self.params['scale_period_limits'], tuple):
            self.params['scale_period_limits'] = [self.params['scale_period_limits']]
        if isinstance(self.params['rotation_speed_limits'], tuple):
            self.params['rotation_speed_limits'] = [self.params['rotation_speed_limits']]
        if isinstance(self.params['position_speed_limits'], tuple):
            self.params['position_speed_limits'] = [self.params['position_speed_limits']]

        self.cur_rng_seed = seed
        np.random.seed(self.cur_rng_seed)

        if self.debug_options is not None:
            self._pg_screen = pg.display.set_mode(self.video_size)
            self._pg_draw_options = pmu.DrawOptions(self._pg_screen)
            pg.font.init()
            font_size = self.debug_options['frame_number_font_size']
            self._pg_font = pg.font.SysFont(pg.font.get_default_font(), font_size)
            self._pg_clock = pg.time.Clock()
            self._add_init_message(dict(
                step=-1,
                type='debug_options',
                meta=dict(debug_options)
            ))

        self._space = pm.Space()
        self.symbols = []

        image_loader = self.params['symbol_image_loader']
        bg_image_loader = self.params['background_image_loader']

        for id in xrange(self.params['num_symbols']):
            image, label, image_path = image_loader.get_image()

            # Define the scale function
            period_limits_index = np.random.choice(len(self.params['scale_period_limits']))
            period = np.random.uniform(
                *tuple(self.params['scale_period_limits'][period_limits_index])
            )
            amplitude = (self.params['scale_limits'][1] - self.params['scale_limits'][0]) / 2.
            x_offset = np.random.uniform(period)
            # Override offset if digits should not start at random scale
            if not self.params['rescale_at_start']:
                x_offset = 0
            # Randomly shift offset (i.e. symbol can either grow or shrink at start)
            x_offset += (period/2 if np.random.choice([True, False]) else 0)
            y_offset = (self.params['scale_limits'][1] + self.params['scale_limits'][0]) / 2.
            if self.params['scale_function_type'] == 'sine':
                scale_fn = create_sine_fn(period, amplitude, x_offset, y_offset)
            elif self.params['scale_function_type'] == 'triangle':
                scale_fn = create_triangle_fn(period, amplitude, x_offset, y_offset)
            elif self.params['scale_function_type'] == 'constant':
                scale = np.random.uniform(*self.params['scale_limits'])
                scale_fn = lambda x: scale
            else:
                raise ValueError('scale_function_type "%s" is unsupported'
                                 % self.params['scale_function_type'])

            symbol = Symbol(id, label, image, image_path, scale_fn)

            # Set the symbol's initial rotation and scale
            symbol.set_scale(0)
            start_angle = np.random.uniform(2 * math.pi)
            if self.params['rotate_at_start']:
                symbol.body.angle = start_angle

            # Compute the minimum possible margin between the symbol's center and the wall
            w_half = image.size[0] / 2.
            h_half = image.size[1] / 2.
            margin = math.sqrt(w_half ** 2 + h_half ** 2) * self.params['scale_limits'][1]
            # Set the symbol position at least one margin's distance from any wall
            x_limits = (margin+1, self.video_size[0] - margin - 1)
            y_limits = (margin+1, self.video_size[1] - margin - 1)
            symbol.body.position = (np.random.uniform(*x_limits), np.random.uniform(*y_limits))
            # If symbols will interact with each other, make sure they don't overlap initially
            while self.params['interacting_symbols'] and len(self._space.shape_query(symbol.shape)) > 0:
                symbol.body.position = (np.random.uniform(*x_limits), np.random.uniform(*y_limits))

            # Set angular velocity
            rotation_speed_limit_index = np.random.choice(len(self.params['rotation_speed_limits']))
            symbol.body.angular_velocity = np.random.uniform(
                *tuple(self.params['rotation_speed_limits'][rotation_speed_limit_index])
            )
            symbol.body.angular_velocity *= 1 if np.random.binomial(1, .5) else -1
            symbol.angular_velocity = symbol.body.angular_velocity

            # Set translational velocity
            sampled_velocity = np.random.uniform(-1, 1, 2)
            # If only lateral motion is allowed, map velocity to the nearest lateral one
            if self.params['lateral_motion_at_start']:
                v_angle = math.degrees(np.arctan2(sampled_velocity[1], sampled_velocity[0]))
                if v_angle >= -135 and v_angle < -45:
                    sampled_velocity = np.array([0, -1])
                elif v_angle >= -45 and v_angle < 45:
                    sampled_velocity = np.array([1, 0])
                elif v_angle >= 45 and v_angle < 135:
                    sampled_velocity = np.array([0, 1])
                else:
                    sampled_velocity = np.array([-1, 0])
            symbol.body.velocity = sampled_velocity / np.linalg.norm(sampled_velocity)
            position_speed_limit_index = np.random.choice(len(self.params['position_speed_limits']))
            symbol.body.velocity *= np.random.uniform(
                *tuple(self.params['position_speed_limits'][position_speed_limit_index])
            )

            # Add symbol to the space and environment
            self._space.add(symbol.body, symbol.shape)
            self.symbols.append(symbol)

            # Publish message about the symbol
            self._add_init_message(symbol.get_init_message())

        # Add walls
        self._add_walls()
        # Add collision handlers
        self._add_collision_handlers(
            interacting_symbols=self.params['interacting_symbols']
        )
        # Init step count
        self._step_count = 0

        # Set background image
        self.background = Image.fromarray(np.zeros((self.video_size[0], self.video_size[1], 3),
                                                   dtype=np.uint8))
        if bg_image_loader is not None:
            # Choose an image
            bg_image, category_name, full_image_path = bg_image_loader.get_image()
            self.background.paste(bg_image)

            # Publish information about the chosen background
            self._add_init_message(dict(
                step=-1,
                type='background',
                meta=dict(
                    label=category_name,
                    image=np.array(self.background),
                    image_path=full_image_path
                )
            ))


    def _add_walls(self):
        space = self._space
        vs = self.video_size
        walls = [
            # Bottom
            pm.Segment(space.static_body, (0, 0), (vs[0], 0), 0.0),
            # Left
            pm.Segment(space.static_body, (-1, 0), (-1, vs[1]), 0.0),
            # Right
            pm.Segment(space.static_body, vs, (vs[0], 0), 0.0),
            # Top
            pm.Segment(space.static_body, (vs[0], vs[1]+1), (0, vs[1]+1), 0.0)
        ]
        for wall in walls:
            wall.elasticity = 1.0
            wall.collision_type = _COLLISION_TYPES['wall']
        space.add(walls)


    @staticmethod
    def _symbol_wall_pre_handler(arbiter, space, data):
        """Remove angular velocity of the symbol.

        This handler sets the angular velocity of the symbol to zero, which prevents the physics
        simulation from adding kinetic energy due to rotation.

        @param arbiter:
        @param space:
        @param data:
        @retval:
        """
        set_ = arbiter.contact_point_set
        if len(arbiter.contact_point_set.points) > 0:
            body = arbiter.shapes[0].body
            body.angular_velocity = 0
            set_.points[0].distance = 0
        arbiter.contact_point_set = set_
        return True

    @staticmethod
    def _symbol_wall_post_handler(arbiter, space, data):
        """Restore angular velocity of the symbol.

        This handler restores the angular velocity after the collision has been solved. It looks
        up the fixed angular velocity from the Symbol instance associated with the body in the
        collision.

        @param arbiter:
        @param space:
        @param data:
        @retval:
        """
        if len(arbiter.contact_point_set.points) > 0:
            body = arbiter.shapes[0].body
            body.angular_velocity = data['body_symbol_map'][body].angular_velocity
        return True

    @staticmethod
    def _symbol_symbol_pre_handler(arbiter, space, data):
        """Remove angular velocity of both symbols.

        This handler sets the angular velocity of each symbol to zero, which prevents the physics
        simulation from adding kinetic energy due to rotation.

        @param arbiter:
        @param space:
        @param data:
        @retval:
        """
        set_ = arbiter.contact_point_set
        if len(arbiter.contact_point_set.points) > 0:
            for shape in arbiter.shapes:
                shape.body.angular_velocity = 0
            set_.points[0].distance = 0
        arbiter.contact_point_set = set_
        return True

    @staticmethod
    def _symbol_symbol_post_handler(arbiter, space, data):
        """Restore angular velocity of both symbols.

        This handler restores the angular velocity after the collision has been solved. It looks
        up the fixed angular velocity from the Symbol instances associated with each body in the
        collision.

        @param arbiter:
        @param space:
        @param data:
        @retval:
        """
        if len(arbiter.contact_point_set.points) > 0:
            for shape in arbiter.shapes:
                shape.body.angular_velocity = data['body_symbol_map'][shape.body].angular_velocity
        return True


    @staticmethod
    def _symbol_wall_begin_handler(arbiter, space, data):
        """Log the point where an symbol first touches a wall.

        @param arbiter:
        @param space:
        @param data:
        @retval:
        """
        symbol_id = data['body_symbol_map'][arbiter.shapes[0].body].id

        # Identify the wall that was hit based on the wall shape's normal
        wall_normal = arbiter.shapes[1].normal
        if wall_normal == pm.Vec2d(1, 0):
            wall_label = 'left'
        elif wall_normal == pm.Vec2d(-1, 0):
            wall_label = 'right'
        elif wall_normal == pm.Vec2d(0, -1):
            wall_label = 'top'
        elif wall_normal == pm.Vec2d(0, 1):
            wall_label = 'bottom'

        data['mie']._publish_message(dict(
            step=data['mie']._step_count,
            type='hit_wall',
            meta=dict(
                symbol_id=symbol_id,
                wall_label=wall_label
            )
        ))
        return True


    @staticmethod
    def _symbol_symbol_overlap_begin_handler(arbiter, space, data):
        overlapping_symbol_ids = (data['body_symbol_map'][arbiter.shapes[0].body].id,
                                data['body_symbol_map'][arbiter.shapes[1].body].id)
        data['mie']._publish_message(dict(
            step=data['mie']._step_count,
            type='start_overlap',
            meta=dict(
                symbol_ids=overlapping_symbol_ids
            )
        ))
        # Don't call pre_solve or post_solve handlers (separate handler will still be called)
        return False


    @staticmethod
    def _symbol_symbol_overlap_separate_handler(arbiter, space, data):
        overlapping_symbol_ids = (data['body_symbol_map'][arbiter.shapes[0].body].id,
                                data['body_symbol_map'][arbiter.shapes[1].body].id)
        data['mie']._publish_message(dict(
            step=data['mie']._step_count,
            type='end_overlap',
            meta=dict(
                symbol_ids=overlapping_symbol_ids
            )
        ))
        return True


    def _add_collision_handlers(self, interacting_symbols=False):
        """Add custom collision handlers in the PyMunk space.

        This method adds collision handlers to ensure that angular velocities do not affect
        translational speeds.

        @param interacting_symbols: Boolean for whether symbols should bounce off each other
        @retval:
        """
        # Define the symbol-wall handler
        body_symbol_map = {symbol.body: symbol for symbol in self.symbols}
        h = self._space.add_collision_handler(_COLLISION_TYPES['symbol'], _COLLISION_TYPES['wall'])
        h.begin = MovingSymbolsEnvironment._symbol_wall_begin_handler
        h.pre_solve = MovingSymbolsEnvironment._symbol_wall_pre_handler
        h.post_solve = MovingSymbolsEnvironment._symbol_wall_post_handler
        h.data['body_symbol_map'] = body_symbol_map
        h.data['mie'] = self

        # Define the symbol-symbol handler
        h = self._space.add_collision_handler(
            _COLLISION_TYPES['symbol'], _COLLISION_TYPES['symbol']
        )
        h.data['body_symbol_map'] = body_symbol_map
        h.data['mie'] = self
        if interacting_symbols:
            h.pre_solve = MovingSymbolsEnvironment._symbol_symbol_pre_handler
            h.post_solve = MovingSymbolsEnvironment._symbol_symbol_post_handler
        else:
            h.begin = MovingSymbolsEnvironment._symbol_symbol_overlap_begin_handler
            h.separate = MovingSymbolsEnvironment._symbol_symbol_overlap_separate_handler

        # TODO: Also add handlers that log events


    def _step(self):
        """Update the positions, scales, and rotations of each symbol."""

        # Publish all init messages, and update _step_called flag
        if not self._step_called:
            self._step_called = True
            for message in self._init_messages:
                self._publish_message(message)

        # First, publish messages about current symbol states
        for symbol in self.symbols:
            self._publish_message(symbol.get_state_message(self._step_count))
        # Now walk through simulation
        self._step_count += 1
        # Update scale of each symbol
        for symbol in self.symbols:
            symbol.set_scale(self._step_count)
        # Take several partial steps in the simulator to stop symbols from phasing through objects
        for _ in xrange(self.fidelity):
            self._space.step(1 / float(self.fidelity))


    def _render_pg(self):
        """Create a debugging visualization of the scene with PyGame.

        Generate an Image containing the visualized scene as rendered by PyGame. This renders the
        collision meshes and draws images at integer locations, so it should NOT be used to
        obtain the final render.

        @retval: Image (RGB format)
        """
        if self.debug_options is None:
            raise RuntimeError('_render_pg cannot be called since no debug options were given.')

        # Draw background image
        bg_sprite = pg.image.fromstring(self.background.tobytes(), self.background.size,
                                        self.background.mode)
        self._pg_screen.blit(bg_sprite, (0, 0))

        # Use PyMunk's default drawing function (guaranteed correctness)
        if self.debug_options['show_pymunk_debug']:
            self._space.debug_draw(self._pg_draw_options)

        # Draw each symbol
        for symbol in self.symbols:
            rotated_image, pg_image_pos = symbol.get_render_image_and_position(self.video_size)
            self._pg_screen.blit(rotated_image, pg_image_pos)

        # Draw polygon outline on top
        if self.debug_options['show_bounding_poly']:
            for symbol in self.symbols:
                ps = [p.rotated(symbol.body.angle) + symbol.body.position
                      for p in symbol.shape.get_vertices()]
                ps = [(p.x, self.video_size[1] - p.y) for p in ps]
                ps += [ps[0]]
                pg.draw.lines(self._pg_screen, pg.color.THECOLORS['red'], False, ps, 1)

        # Print step number
        text = self._pg_font.render(str(self._step_count), False, pg.color.THECOLORS['green'])
        if self.debug_options['show_frame_number']:
            self._pg_screen.blit(text, (0, 0))

        # Refresh PyGame screen
        pg.display.flip()
        self._pg_clock.tick(self.debug_options['frame_rate'])
        # Return an Image of the current PyGame (debug) screen
        pg_screen_bytes = pg.image.tostring(self._pg_screen, 'RGB')
        return Image.frombytes('RGB', self.video_size, pg_screen_bytes)


    def _render_cv(self):
        """Generate a PIL Image of the scene at the current state.

        Generate an Image containing the visualized scene as rendered by OpenCV. This renders the
        scene using floating-point position, scale, and rotation, so it should be used for the
        final render. It outputs a PIL Image in either "RGB" or "L" mode depending on the value
        of the "color_output" flag specified in the constructor parameters.

        @retval: Image (RGB or L format)
        """
        ret = np.array(self.background, dtype=np.float32) / 255.

        for x, symbol in enumerate(self.symbols):
            angle = symbol.body.angle
            scale = symbol.scale
            position = (symbol.body.position[0], self.video_size[1] - symbol.body.position[1])
            width, height = symbol.image.size

            M = cv2.getRotationMatrix2D((width/2., height/2.), math.degrees(angle), scale)
            M[0, 2] += position[0] - width/2.
            M[1, 2] += position[1] - height/2.

            overlay = cv2.warpAffine(np.array(symbol.image), M, self.video_size) / 255.
            alpha = np.stack([overlay[:, :, 3] for _ in xrange(3)], axis=2)
            ret = (1 - alpha) * ret + alpha * overlay[:, :, :3]

        # Get image to RGB 0-255 format
        ret = np.multiply(ret, 255).astype(np.uint8)
        # Convert image to grayscale if specified
        if not self.params['color_output']:
            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(ret)


    def _publish_message(self, message):
        """Publish a message to any subscribers

        @param message: Dict of information to publish
        """
        assert(isinstance(message, dict))
        if not self._step_called:
            raise RuntimeError('_publish_message() was called before _step(), which is not allowed')
        for subscriber in self._subscribers:
            subscriber.process_message(dict(message))


    def add_subscriber(self, subscriber):
        """Add a subscriber of published messages.\ The subscriber must have a callable
        "process_message" method.

        @param subscriber: An object with a callable "process_message" method
        """
        process_message_fn = getattr(subscriber, 'process_message', None)
        if not callable(process_message_fn):
            raise ValueError('The given subscriber does not have a "process_message" method')

        self._subscribers.append(subscriber)


    def _add_init_message(self, message):
        """Add a message that will be published when _step is called for the first time.

        @param message: Dict of information to publish
        """
        if self._step_called:
            raise RuntimeError('_add_init_message() was called after _step(), which is not allowed')
        self._init_messages.append(message)


    """GENERATOR METHODS"""
    def send(self, _):
        if self.debug_options is not None:
            self._render_pg()
        ret = self._render_cv()
        self._step()
        return ret

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __iter__(self):
        return self

    def next(self):
        return self.send(None)

    def close(self):
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError('Generator ignored GeneratorExit')
    """END GENERATOR METHODS"""
