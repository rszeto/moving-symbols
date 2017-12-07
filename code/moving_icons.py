import pymunk as pm
import pymunk.pygame_util as pmu
import pygame as pg
import pygame.locals as pgl
import time
import numpy as np
import math
import os
from PIL import Image
import cv2
import sys

import matplotlib.pyplot as plt


_COLLISION_TYPES = dict(
    icon=0,
    wall=1
)


def merge_dicts(*dicts):
    """Merge the given dictionaries. Key-value pairs in later dictionaries will replace pairs in
    earlier ones.
    """
    ret = {}
    for d in dicts:
        for k, v in d.iteritems():
            ret[k] = v
    return ret


def pil_grid(images, grid_size, margin=0):
    """Create a PIL Image grid of the given images

    :param images: A sequence of Image objects to tile
    :param grid_size: Grid size (w x h)
    :param margin: How many blank pixels to place between each image
    :return:
    """
    # Get max image size
    max_dims = [-1, -1]
    for image in images:
        max_dims[0] = max(image.size[0], max_dims[0])
        max_dims[1] = max(image.size[1], max_dims[1])
    grid_w, grid_h = grid_size
    ret_size = (max_dims[0] * grid_w + margin * (grid_w-1),
                max_dims[1] * grid_h + margin * (grid_h-1))
    ret = Image.new('RGB', ret_size)
    for i, image in enumerate(images):
        grid_x = i % grid_w
        grid_y = (i - grid_x) / grid_w
        ret.paste(image, (grid_x * (margin + max_dims[0]), grid_y * (margin + max_dims[1])))
    return ret


def tight_crop_2(image):
    """Produce a tightly-cropped version of the image, and add alpha channel if needed

    :param image: PIL image
    :return: PIL image
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    alpha = np.array(image)[:,:,3]
    nonzero_points = cv2.findNonZero(alpha)
    x, y, w, h = cv2.boundingRect(nonzero_points)
    cropped_image = image.crop((x, y, x+w, y+h))
    return cropped_image


def compute_pm_hull_vertices(image):
    """Get PyMunk vertices that enclose the alpha channel of a given RGBA image."""
    alpha = np.array(image)[:,:,3]
    nonzero_points = cv2.findNonZero(alpha)
    hull = cv2.convexHull(nonzero_points).squeeze()
    # Flip the y-axis since it points up in PyMunk
    hull[:, 1] = image.size[1] - hull[:, 1]
    return hull


def create_sine_fn(period, amplitude, x_offset, y_offset):
    period = float(period)
    amplitude = float(amplitude)
    x_offset = float(x_offset)
    y_offset = float(y_offset)
    return lambda x: amplitude * math.sin((x - x_offset) * (2 * math.pi / period)) + y_offset


def create_triangle_fn(period, amplitude, x_offset, y_offset):
    p = float(period)
    a = float(amplitude)
    x_offset = float(x_offset)
    y_offset = float(y_offset)
    def ret(x):
        in_ = math.fmod(x - x_offset, p) + 7*p/4
        return 4*a/p * (math.fabs(math.fmod(in_, p) - p/2) - p/4) + y_offset
    return ret


class ImageLoader:

    def __init__(self, root, mode):
        """Constructor

        :param root: The path to the root directory containing all images.
        :param mode: String that indicates how to transform the image for rendering. Options
        include "tight_crop".
        """

        self.root = root
        self.mode = mode


    def get_image(self, label):
        """Return a random, pre-processed image in the given label set and its source path.

        The image is pre-processed based on the ImageLoader's mode.

        :param label: The label of the class to sample from
        :return:
        """

        class_path = os.path.join(self.root, str(label))
        class_image_names = os.listdir(class_path)
        image_idx = np.random.randint(len(class_image_names))
        image_path = os.path.join(class_path, class_image_names[image_idx])
        image = Image.open(image_path, 'r')
        if self.mode == 'tight_crop':
            image = tight_crop_2(image)
        return image, image_path


class Icon:

    def __init__(self, id, image, image_path, scale_fn):
        self.id = id
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
        self.shape.collision_type = _COLLISION_TYPES['icon']


    def get_render_image_and_position(self, screen_size):
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
        self.scale = self.scale_fn(step)
        self.shape.unsafe_set_vertices(self._base_vertices,
                                       transform=pm.Transform(self.scale, 0, 0, self.scale, 0, 0))


class MovingIconEnvironment:
    """Generator that produces Moving Icon video frames.

    This class manages a physical environment in which icons move around. It also handles
    rendering of the current physical state. Renders are returned as PIL images, either in RGB or
    L (8-bit grayscale) mode.

    The physical state is initialized based on the parameters given to the constructor. Below are
    the key-value pairs that can be specified:

    - data_dir, str: Path to the dataset directory. The dataset directory should contain one
      folder for each split, e.g. "training" and "testing", and each split directory should
      contain folders for each class (e.g. 0, ..., 9 for MNIST digits).
    - split, str: Name of the data split to sample from
    - num_icons, int: How many icons should appear in the video
    - video_size, (int, int): The resolution of the video as (width, height)
    - color_output, bool: Whether to produce "RGB" color images or "L" grayscale images
    - icon_labels, Sequence: The labels for the icon classes. These must be strings or ints (or
      any object with __str__ implemented) that match the names of the folders in each split
      directory
    - scale_limits, [float, float]: The minimum and maximum scale of an object relative to its
      original size
    - scale_period_limits, [float, float]: The minimum and maximum duration of a full scale cycle
      in number of frames
    - rotation_speed_limits, [float, float]: The minimum and maximum angular speed, in radians
      per frame
    - position_speed_limits, [float, float]: The minimum and maximum translational speed,
      in pixels per frame
    - interacting_icons, bool: Whether icons will bounce off each other
    - scale_function_type, str: The class of function used to define the scale of each icon at
      each time step. Supported options are:
      - "sine": Scale is determined by a sine wave
      - "triangle": Scale is determined by a triangle wave (i.e. scaling speed is constant,
        but switches directions if the digit gets too big or small)
      - "constant": The icons do not change scale. Initial scale is randomly sampled from within
        scale_limits

    Additionally, debugging options can be given to the constructor. Below are the key-value
    pairs that can be specified:

    - frame_number_font_size: How large to make the frame count text
    - show_pymunk_debug: Draw objects with PyMunk's built-in rendering function
    - show_bounding_poly: Draw a polygon
    """

    DEFAULT_PARAMS = dict(
        data_dir='../data/mnist',
        split='training',
        num_icons=1,
        video_size=(64, 64),
        color_output=True,
        icon_labels=[0],
        scale_limits=[1.0, 1.0],
        scale_period_limits=[1, 1],
        rotation_speed_limits=[0, 0],
        position_speed_limits=[0, 0],
        interacting_icons=False,
        scale_function_type='constant'
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

        :param params: Parameters that define how icons behave and are rendered. See method
        description for supported commands.
        :param seed: Seed for the RNG (int)
        :param fidelity: How many iterations to run in the physics simulator per step (int)
        :param debug_options: dict with options for visual debugging. The following key-value
                              pairs are supported:
                              - show_pymunk_debug, bool: Whether to use PyMunk's default drawing
                                function
                              - show_bounding_poly, bool: Whether to render PyMunk surface outlines
                              - show_frame_number, bool: Whether to show the index of the frame
                              - frame_number_font_size, int: Size of the frame index font
                              - frame_rate, int: Frame rate of the debug visualization
        """

        self.params = merge_dicts(MovingIconEnvironment.DEFAULT_PARAMS, params)
        self.fidelity = fidelity
        self.debug_options = None if debug_options is None \
            else merge_dicts(MovingIconEnvironment.DEFAULT_DEBUG_OPTIONS, debug_options)
        self.video_size = self.params['video_size']

        self.cur_rng_seed = seed
        np.random.seed(self.cur_rng_seed)

        if self.debug_options is not None:
            self._pg_screen = pg.display.set_mode(self.video_size)
            self._pg_draw_options = pmu.DrawOptions(self._pg_screen)
            pg.font.init()
            font_size = self.debug_options['frame_number_font_size']
            self._pg_font = pg.font.SysFont(pg.font.get_default_font(), font_size)
            self._pg_clock = pg.time.Clock()

        self._space = pm.Space()
        self.icons = []
        image_loader = ImageLoader(os.path.join(self.params['data_dir'], self.params['split']),
                                   'tight_crop')

        for id in xrange(self.params['num_icons']):
            label = self.params['icon_labels'][np.random.randint(len(self.params['icon_labels']))]
            image, image_path = image_loader.get_image(label)

            # Define the scale function
            period = np.random.uniform(*tuple(self.params['scale_period_limits']))
            amplitude = (self.params['scale_limits'][1] - self.params['scale_limits'][0]) / 2.
            x_offset = np.random.uniform(period)
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

            icon = Icon(id, image, image_path, scale_fn)

            # Set the icon's initial rotation and scale
            icon.set_scale(0)
            icon.body.angle = np.random.uniform(2 * math.pi)

            # Compute the minimum possible margin between the icon's center and the wall
            w_half = image.size[0] / 2.
            h_half = image.size[1] / 2.
            margin = math.sqrt(w_half ** 2 + h_half ** 2) * self.params['scale_limits'][1]
            # Set the icon position at least one margin's distance from any wall
            x_limits = (margin+1, self.video_size[0] - margin - 1)
            y_limits = (margin+1, self.video_size[1] - margin - 1)
            icon.body.position = (np.random.uniform(*x_limits), np.random.uniform(*y_limits))
            # If icons will interact with each other, make sure they don't overlap initially
            while self.params['interacting_icons'] and len(self._space.shape_query(icon.shape)) > 0:
                icon.body.position = (np.random.uniform(*x_limits), np.random.uniform(*y_limits))

            # Finally, set speeds
            icon.body.angular_velocity = np.random.uniform(
                *tuple(self.params['rotation_speed_limits'])
            )
            icon.body.angular_velocity *= 1 if np.random.binomial(1, .5) else -1
            icon.angular_velocity = icon.body.angular_velocity
            icon.body.velocity = np.random.uniform(-1, 1, 2)
            icon.body.velocity = icon.body.velocity.normalized()
            icon.body.velocity *= np.random.uniform(*tuple(self.params['position_speed_limits']))

            # Add icon to the space and environment
            self._space.add(icon.body, icon.shape)
            self.icons.append(icon)

        # Add walls
        self._add_walls()
        # Add collision handlers
        self._add_collision_handlers(
            interacting_icons=self.params['interacting_icons']
        )
        # Init step count
        self._step_count = 0


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
    def _icon_wall_pre_handler(arbiter, space, data):
        """Remove angular velocity of the icon.

        This handler sets the angular velocity of the icon to zero, which prevents the physics
        simulation from adding kinetic energy due to rotation.

        :param arbiter:
        :param space:
        :param data:
        :return:
        """
        set_ = arbiter.contact_point_set
        if len(arbiter.contact_point_set.points) > 0:
            body = arbiter.shapes[0].body
            body.angular_velocity = 0
            set_.points[0].distance = 0
        arbiter.contact_point_set = set_
        return True

    @staticmethod
    def _icon_wall_post_handler(arbiter, space, data):
        """Restore angular velocity of the icon.

        This handler restores the angular velocity after the collision has been solved. It looks
        up the fixed angular velocity from the Icon instance associated with the body in the
        collision.

        :param arbiter:
        :param space:
        :param data:
        :return:
        """
        if len(arbiter.contact_point_set.points) > 0:
            body = arbiter.shapes[0].body
            body.angular_velocity = data['body_icon_map'][body].angular_velocity
        return True

    @staticmethod
    def _icon_icon_pre_handler(arbiter, space, data):
        """Remove angular velocity of both icons.

        This handler sets the angular velocity of each icon to zero, which prevents the physics
        simulation from adding kinetic energy due to rotation.

        :param arbiter:
        :param space:
        :param data:
        :return:
        """
        set_ = arbiter.contact_point_set
        if len(arbiter.contact_point_set.points) > 0:
            for shape in arbiter.shapes:
                shape.body.angular_velocity = 0
            set_.points[0].distance = 0
        arbiter.contact_point_set = set_
        return True

    @staticmethod
    def _icon_icon_post_handler(arbiter, space, data):
        """Restore angular velocity of both icons.

        This handler restores the angular velocity after the collision has been solved. It looks
        up the fixed angular velocity from the Icon instances associated with each body in the
        collision.

        :param arbiter:
        :param space:
        :param data:
        :return:
        """
        if len(arbiter.contact_point_set.points) > 0:
            for shape in arbiter.shapes:
                shape.body.angular_velocity = data['body_icon_map'][shape.body].angular_velocity
        return True


    def _add_collision_handlers(self, interacting_icons=False):
        # Define the icon-wall handler
        body_icon_map = {icon.body: icon for icon in self.icons}
        h = self._space.add_collision_handler(_COLLISION_TYPES['icon'], _COLLISION_TYPES['wall'])
        h.pre_solve = MovingIconEnvironment._icon_wall_pre_handler
        h.post_solve = MovingIconEnvironment._icon_wall_post_handler
        h.data['body_icon_map'] = body_icon_map

        # Define the icon-icon handler
        h = self._space.add_collision_handler(_COLLISION_TYPES['icon'], _COLLISION_TYPES['icon'])
        if interacting_icons:
            h.pre_solve = MovingIconEnvironment._icon_icon_pre_handler
            h.post_solve = MovingIconEnvironment._icon_icon_post_handler
            h.data['body_icon_map'] = body_icon_map
        else:
            h.begin = lambda arbiter, space, data: False

        # TODO: Also add handlers that log events


    def _step(self):
        """Update the positions, scales, and rotations of each icon."""
        self._step_count += 1
        # Update scale of each icon
        for icon in self.icons:
            icon.set_scale(self._step_count)
        # Take several partial steps in the simulator to stop icons from phasing through objects
        for _ in xrange(self.fidelity):
            self._space.step(1 / float(self.fidelity))


    def _render_pg(self):
        """Create a debugging visualization of the scene with PyGame.

        Generate an Image containing the visualized scene as rendered by PyGame. This renders the
        collision meshes and draws images at integer locations, so it should NOT be used to
        obtain the final render.

        :return: Image (RGB format)
        """
        if self.debug_options is None:
            raise RuntimeError('_render_pg cannot be called since no debug options were given.')

        # Use black background
        self._pg_screen.fill(pg.color.THECOLORS['black'])

        # Use PyMunk's default drawing function (guaranteed correctness)
        if self.debug_options['show_pymunk_debug']:
            self._space.debug_draw(self._pg_draw_options)

        # Draw each icon
        for icon in self.icons:
            rotated_image, pg_image_pos = icon.get_render_image_and_position(self.video_size)
            self._pg_screen.blit(rotated_image, pg_image_pos)

        # Draw polygon outline on top
        if self.debug_options['show_bounding_poly']:
            for icon in self.icons:
                ps = [p.rotated(icon.body.angle) + icon.body.position
                      for p in icon.shape.get_vertices()]
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

        :return: Image (RGB or L format)
        """
        ret = np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.float32)

        for x, icon in enumerate(self.icons):
            angle = icon.body.angle
            scale = icon.scale
            position = (icon.body.position[0], self.video_size[1] - icon.body.position[1])
            width, height = icon.image.size

            M = cv2.getRotationMatrix2D((width/2., height/2.), math.degrees(angle), scale)
            M[0, 2] += position[0] - width/2.
            M[1, 2] += position[1] - height/2.

            overlay = cv2.warpAffine(np.array(icon.image), M, self.video_size) / 255.
            alpha = np.stack([overlay[:, :, 3] for _ in xrange(3)], axis=2)
            ret = (1 - alpha) * ret + alpha * overlay[:, :, :3]

        # Get image to RGB 0-255 format
        ret = np.multiply(ret, 255).astype(np.uint8)
        # Convert image to grayscale if specified
        if not self.params['color_output']:
            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(ret)


    ### GENERATOR METHODS ###
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
    ### END GENERATOR METHODS ###



if __name__ == '__main__':

    seed = int(time.time())
    # seed = 1512518328

    debug_options = dict(
        show_bounding_poly=True,
        show_frame_number=True
    )
    debug_options = None

    data_dir = '../data/icons8'

    params = dict(
        data_dir=data_dir,
        split='training',
        num_icons=10,
        video_size=(200, 200),
        color_output=True,
        icon_labels=os.listdir(os.path.join(data_dir, 'training')),
        scale_limits = [0.5, 1.5],
        scale_period_limits = [40, 60],
        rotation_speed_limits = [math.radians(5), math.radians(15)],
        position_speed_limits = [1, 5],
        interacting_icons = False,
        scale_function_type = 'sine'
    )

    env = MovingIconEnvironment(params, seed, debug_options=debug_options)
    print(env.cur_rng_seed)

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    writer = cv2.VideoWriter('demo.avi', fourcc, 30, (200, 200))

    for i in xrange(150):
        cv_image = env.next()
        writer.write(np.array(cv_image.convert('RGB'))[:, :, ::-1])
    writer.release()