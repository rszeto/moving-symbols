import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        for c in range(image.shape[2]):
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
    Overlay pixels in the foreground over the pixels in the background. For grayscale images, the alpha channel is
    computed from the brightness value. Images without alpha channels are not supported.
    :param fg: The foreground image. Can be H x W grayscale for H x W x C+A
    :param bg: The background image with same dimensions as foreground
    :return:
    '''
    if fg.shape != bg.shape:
        raise ValueError('Foreground and background have different sizes (fg: %s, bg: %s)' % (fg.shape, bg.shape))
    if fg.ndim not in [2, 3]:
        raise ValueError('Given images have %d dimensions, but must have 2 or 3' % fg.ndim)
    if fg.ndim == 3 and not fg.shape[2] == 4:
        raise NotImplementedError('Color images without alpha channels are not supported')

    if fg.ndim == 2:
        # Set the foreground brightness as alpha mask
        alpha = fg / 255.0
        ret = alpha * 255.0 + (1.0 - alpha) * bg
    else:
        alpha = fg[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        ret_no_alpha = alpha * fg[:, :, :3] + (1.0 - alpha) * bg[:, :, :3]
        ret = 255 * np.ones(fg.shape)
        ret[:, :, :3] = ret_no_alpha

    return ret.astype(np.uint8)


def save_tensor_as_mjpg(filename, video_tensor):
    '''
    Saves the given video tensor as a video
    :param filename: The location of the video
    :param video_tensor: The H x W (x C) x T tensor of frames
    :return:
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, 10.0, tuple(video_tensor.shape[:2][::-1]))
    for i in range(video_tensor.shape[-1]):
        if video_tensor.ndim == 3:
            video_frame = cv2.cvtColor(video_tensor[:, :, i], cv2.COLOR_GRAY2BGR)
        else:
            video_frame = video_tensor[:, :, :, i]
        out.write(video_frame[:, :, ::-1])
    out.release()


def is_overlapping(a, b):
    '''
    Determine if two boxes are overlapping
    :param a: The first bounding box given as [(x1, y1), (x2, y2)]
    :param b: The second bounding box given as [(x1, y1), (x2, y2)]
    :return:
    '''
    x_coords = [a[0][0], a[1][0], b[0][0], b[1][0]]
    x_coords_alt = [b[0][0], b[1][0], a[0][0], a[1][0]]
    is_horiz_overlap = not (np.array_equal(x_coords, sorted(x_coords)) or np.array_equal(x_coords_alt, sorted(x_coords)))

    y_coords = [a[0][1], a[1][1], b[0][1], b[1][1]]
    y_coords_alt = [b[0][1], b[1][1], a[0][1], a[1][1]]
    is_vert_overlap = not (np.array_equal(y_coords, sorted(y_coords)) or np.array_equal(y_coords_alt, sorted(y_coords)))

    # if is_horiz_overlap and is_vert_overlap:
    #     # Compute angle of the second box's center relative to the first
    #     a_center = [(a[0][0] + a[1][0])/2, (a[0][1] + a[1][1])/2]
    #     b_center = [(b[0][0] + b[1][0])/2, (b[0][1] + b[1][1])/2]
    #     angle_rad = np.arctan2((a_center[1]-b_center[1]), (b_center[0]-a_center[0]))
    #     return (angle_rad * 180 / np.pi + 360) % 360
    # else:
    #     return None

    return (is_horiz_overlap and is_vert_overlap)