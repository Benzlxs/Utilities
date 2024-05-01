###
import numpy as np
import os
import sys
import glob
import argparse
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2 as cv
from moviepy.editor import ImageSequenceClip


def making_video(args, file_list, extra_name=""):
    file_list.sort()
    clip = ImageSequenceClip(file_list, fps=args.fps)
    clip.write_videofile( args.image_path +  extra_name + str(args.fps) + ".mp4",fps=args.fps)

def main(args):
    file_list = glob.glob(os.path.join(args.image_path, '*'))
    file_list = sorted(file_list)
    making_video(args, file_list, extra_name='full')


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='./imagee', type=str, help='path to images folder')
    parser.add_argument("--fps", type=int, default=5, help="the fps")
    args = parser.parse_args()

    main(args)

