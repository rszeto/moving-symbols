import numpy as np
import matplotlib.pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from textwrap import wrap
import re
import os
from pprint import pprint
import json
import cv2

def view_toronto_mnist_tensor(toronto_tensor,
                              vid_ids=None, delay=None, prompt_keypress=False,
                              titles=None, gen_seeds=None):
    '''
    View a video stored with wacky Toronto dimensions
    :param toronto_tensor: A tensor with dims T x V x H x W or T x V x H x W x C
    :param vid_id: Index of video to play. If None, choose random video
    :param delay: How long to wait between frames. If None, advance on click
    :return:
    '''
    vid_id_list = range(toronto_tensor.shape[1]) if vid_ids is None else vid_ids
    while True:
        for vid_id in vid_id_list:
            video = toronto_tensor[:, vid_id]

            # Show info
            for i in range(video.shape[0]):
                large_frame = cv2.resize(video[i], (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(None, large_frame)
                if delay:
                    # plt.pause(delay)
                    cv2.waitKey(delay)
                    if i == video.shape[0]-1 and prompt_keypress:
                        print('Ended video. Press any key to continue.')
                        # plt.waitforbuttonpress()
                        cv2.waitKey()
                else:
                    # plt.waitforbuttonpress()
                    cv2.waitKey()
                # plt.clf()

def main():
    # Select file path (https://stackoverflow.com/a/3579625)
    Tk().withdraw()
    filename = askopenfilename(initialdir='../output', title='Select one of the Moving Icons files')
    if not filename: exit()
    # Check the file is one of the valid files
    m = re.search('(.+?)(_videos.npy$)', filename)
    if m is None:
        print('The selected file is not a video file.')
        return

    # Load arrays
    video_tensor_path = filename
    video_tensor = np.load(video_tensor_path, mmap_mode='r')

    # Show info
    print('Video tensor shape: %s ' % str(video_tensor.shape))
    view_toronto_mnist_tensor(video_tensor, delay=1000/30, vid_ids=None)


if __name__ == '__main__':
    main()