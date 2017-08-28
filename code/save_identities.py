import numpy as np
import argparse
import json
from scipy.misc import imread
import cv2

def main(desc_path):
    json_str_messages = np.load(desc_path)
    json_messages = [json.loads(message) for message in json_str_messages]
    images = []
    digit_classes = []
    for video_message in json_messages:
        digit_messages = filter(lambda x: x['type'] == 'digit', video_message)
        digit_messages = sorted(digit_messages, key=lambda x: x['meta']['id'])
        if len(digit_messages) > 1 and len(digit_messages) < 5:
            image = np.zeros((64, 64), dtype=np.uint8)
            cur_digit_classes = []
            for i in range(len(digit_messages)):
                digit_image = imread(digit_messages[i]['meta']['image_path'])
                if i == 0:
                    x_offset, y_offset = 0, 0
                elif i == 1:
                    x_offset, y_offset = 36, 0
                elif i == 2:
                    x_offset, y_offset = 0, 36
                elif i == 3:
                    x_offset, y_offset = 36, 36
                image[y_offset:y_offset+digit_image.shape[0], x_offset:x_offset+digit_image.shape[1]] = digit_image
                digit_class = digit_messages[i]['meta']['label']
                cur_digit_classes.append(digit_class)
            digit_classes.append(cur_digit_classes)
        else:
            # Put image in center
            digit_image = imread(digit_messages[0]['meta']['image_path'])
            # Pad the image to make it 64x64
            image = cv2.copyMakeBorder(digit_image, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=0)
            digit_class = digit_messages[0]['meta']['label']
            digit_classes.append(digit_class)
        images.append(image)
    images = np.stack(images, axis=0)
    digit_classes = np.array(digit_classes, dtype=np.uint8)
    out_path = desc_path.replace('_messages.npy', '_identities')
    np.savez(out_path, images=images, digit_classes=digit_classes)
    print('Saved identities to %s' % desc_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc_path', type=str, help='Path to the *_messages.npy file')
    args = parser.parse_args()

    main(**vars(args))