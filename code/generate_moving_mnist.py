import numpy as np
import os
import sys
import cv2
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from pprint import pprint
from time import time

def tight_crop(image):
    '''
    Produce a tight crop around the given image
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

def get_random_image(split):
    '''
    Get a random image and label from the given split
    :param split: The name of the split, either 'training' or 'testing'
    :return:
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    label = np.random.randint(10)
    digit_dir = os.path.join(data_dir, split, str(label))
    image = imread(os.path.join(digit_dir, '%04d.png' % np.random.randint(len(os.listdir(digit_dir)))))
    # image = imread('/home/szetor/Pictures/profile.jpg')
    # image = imread(os.path.join(data_dir, split, '0', '0000.png'))
    crop = tight_crop(image)

    return crop, label


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
    # return work_image

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

    if fg.ndim == 2:
        fg_gray = fg
    else:
        fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(fg_gray, 1, 255, cv2.THRESH_BINARY)

    ret = bg.copy().astype(np.int)
    for y in range(fg.shape[0]):
        for x in range(fg.shape[1]):
            if mask[y, x]:
                if fg.ndim == 2:
                    ret[y, x] = min(255, ret[y, x] + fg[y, x])
                else:
                    for c in range(3):
                        ret[y, x, c] = min(255, ret[y, x, c] + fg[y, x, c])
    return ret.astype(np.uint8)


def sample_start_params(limit_params):
    '''
    Choose starting parameters from the limit params
    :param limit_params: A dict with the following fields:
        x_lim (list)
        y_lim (list)
        scale_lim (list)
    :return:
    '''
    MAX_IMAGE_SIZE = 28.0
    scale_start = np.random.uniform(limit_params['scale_lim'][0], limit_params['scale_lim'][1])
    angle_start = np.random.randint(360)
    # Start far from the video frame border to avoid image clipping
    pad = (MAX_IMAGE_SIZE / 2) * np.sqrt(2) * scale_start
    x_start = np.random.randint(np.ceil(limit_params['x_lim'][0] + pad), np.floor(limit_params['x_lim'][1] - pad))
    y_start = np.random.randint(np.ceil(limit_params['y_lim'][0] + pad), np.floor(limit_params['y_lim'][1] - pad))
    
    return dict(
        scale=scale_start,
        x=x_start,
        y=y_start,
        angle=angle_start
    )


def sample_update_params(update_limit_params):
    '''
    Choose update parameters from the limit params
    :param update_limit_params: A dict with the following fields:
        x_speed_lim (list)
        y_speed_lim (list)
        scale_speed_lim (list)
        angle_speed_lim (list)
    :return: 
    '''
    scale_speed = np.random.uniform(update_limit_params['scale_speed_lim'][0], 
                                    update_limit_params['scale_speed_lim'][1])
    x_speed = np.random.uniform(update_limit_params['x_speed_lim'][0],
                                update_limit_params['x_speed_lim'][1])
    y_speed = np.random.uniform(update_limit_params['y_speed_lim'][0],
                                update_limit_params['y_speed_lim'][1])
    if update_limit_params['angle_speed_lim'][0] == update_limit_params['angle_speed_lim'][1]:
        angle_speed = update_limit_params['angle_speed_lim'][0]
    else:
        angle_speed = np.random.randint(update_limit_params['angle_speed_lim'][0],
                                        update_limit_params['angle_speed_lim'][1])
    
    return dict(
        scale_speed=scale_speed,
        x_speed=x_speed,
        y_speed=y_speed,
        angle_speed=angle_speed
    )


def main():
    # Set overall video parameters
    num_digits = 3
    video_size = (64, 64)  # width x height
    num_timesteps = 200

    # Set parameters defining the digit dynamics
    limit_params = dict(
        x_lim=[0, video_size[0]],
        y_lim=[0, video_size[1]],
        scale_lim=[.5, 1.5]
    )
    update_limit_params = dict(
        x_speed_lim=[-7, 7],
        y_speed_lim=[-7, 7],
        scale_speed_lim=[0, .2],
        angle_speed_lim=[-30, 30]
    )

    # num_digits = 50
    # video_size = (1280, 720)  # width x height
    # num_timesteps = 100
    #
    # limit_params = dict(
    #     x_lim=[0, video_size[0]],
    #     y_lim=[0, video_size[1]],
    #     scale_lim=[1.5, 3.5]
    # )
    # update_limit_params = dict(
    #     x_speed_lim=[-20, 20],
    #     y_speed_lim=[-20, 20],
    #     scale_speed_lim=[0, .4],
    #     angle_speed_lim=[-30, 30]
    # )

    # Sample digits and dynamics
    digit_infos = [get_random_image('training') for _ in range(num_digits)]
    state_params = [sample_start_params(limit_params) for _ in range(num_digits)]
    update_params = [sample_update_params(update_limit_params) for _ in range(num_digits)]

    # Set up tensor to store video frames
    num_image_channels = 1 if digit_infos[0][0].ndim == 2 else 3
    video_tensor = np.zeros((video_size[1], video_size[0], num_image_channels, num_timesteps), dtype=np.uint8)
    
    for i in range(num_timesteps):
        digit_frames = []

        for j in range(num_digits):
            # Update image parameters
            state_params[j]['x'] += update_params[j]['x_speed']
            state_params[j]['y'] += update_params[j]['y_speed']
            state_params[j]['angle'] = (state_params[j]['angle'] + update_params[j]['angle_speed']) % 360
            state_params[j]['scale'] += update_params[j]['scale_speed']
            # Adjust scale parameter
            if state_params[j]['scale'] > limit_params['scale_lim'][1]:
                state_params[j]['scale'] = limit_params['scale_lim'][1] - (state_params[j]['scale'] - limit_params['scale_lim'][1])
                update_params[j]['scale_speed'] *= -1
            elif state_params[j]['scale'] < limit_params['scale_lim'][0]:
                state_params[j]['scale'] = limit_params['scale_lim'][0] - (state_params[j]['scale'] - limit_params['scale_lim'][0])
                update_params[j]['scale_speed'] *= -1
            # Generate the cropped image
            cropped_digit = get_rotated_scaled_image_tight_crop(digit_infos[j][0], state_params[j]['angle'], state_params[j]['scale'])

            # Bounce image off the walls
            x_right = state_params[j]['x'] + cropped_digit.shape[1]/2
            x_left = state_params[j]['x'] - cropped_digit.shape[1]/2
            if x_right > limit_params['x_lim'][1] or x_left < limit_params['x_lim'][0]:
                if x_right > limit_params['x_lim'][1]:
                    # x_right = x_max - (x_right - x_max)
                    x_right = limit_params['x_lim'][1]
                    state_params[j]['x'] = x_right - cropped_digit.shape[1] / 2
                else:
                    # x_left = x_min - (x_left - x_min)
                    x_left = limit_params['x_lim'][0]
                    state_params[j]['x'] = x_left + cropped_digit.shape[1] / 2
                # Flip x speed
                update_params[j]['x_speed'] *= -1

            y_bottom = state_params[j]['y'] + cropped_digit.shape[0]/2
            y_top = state_params[j]['y'] - cropped_digit.shape[0]/2
            if y_bottom > limit_params['y_lim'][1] or y_top < limit_params['y_lim'][0]:
                if y_bottom > limit_params['y_lim'][1]:
                    # y_bottom = y_max - (y_bottom - y_max)
                    y_bottom = limit_params['y_lim'][1]
                    state_params[j]['y'] = y_bottom - cropped_digit.shape[0] / 2
                else:
                    # y_top = y_min - (y_top - y_min)
                    y_top = limit_params['y_lim'][0]
                    state_params[j]['y'] = y_top + cropped_digit.shape[0] / 2
                # Flip y speed
                update_params[j]['y_speed'] *= -1

            trans = get_center_translation_matrix(cropped_digit, state_params[j]['x'], state_params[j]['y'])
            digit_frame = cv2.warpPerspective(cropped_digit, trans, video_size)
            digit_frames.append(digit_frame)

        stitched_frame = np.zeros(video_size[::-1])
        for j in range(num_digits):
            stitched_frame = overlay_image(digit_frames[j], stitched_frame)

        plt.clf()
        plt.imshow(stitched_frame, cmap='gray')
        plt.draw()
        plt.pause(.01)

        video_tensor[:, :, :, i] = stitched_frame[:, :, np.newaxis]
        print('Finished frame %d' % i)

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 15.0, video_size)
    for i in range(num_timesteps):
        frame = np.stack((np.squeeze(video_tensor[:, :, :, i]),) * 3, axis=-1)
        out.write(frame)
    out.release()


if __name__ == '__main__':
    # np.random.seed(123)
    start = time()
    main()
    end = time()
    print('%d s' % (end-start))
