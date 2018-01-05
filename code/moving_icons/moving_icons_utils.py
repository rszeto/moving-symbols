import cv2
import math

import pymunk as pm
import numpy as np
from PIL import Image


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


def tight_crop(image):
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


def get_closest_axis_vector(v):
    """Map v to whichever of (0, 1), (0, -1), (1, 0), (-1, 0) is closest to v by angle.
    :param v: A PyMunk Vec2d object
    :return: A PyMunk Vec2d object
    """
    # Get angle from (0, 1)
    angle = math.degrees(np.arctan2(v[1], v[0]))
    if -135 <= angle < -45:
        return pm.Vec2d(0, -1)
    elif -45 <= angle < 45:
        return pm.Vec2d(1, 0)
    elif 45 <= angle < 135:
        return pm.Vec2d(0, 1)
    else:
        return pm.Vec2d(-1, 0)