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
    clip.write_videofile( extra_name + str(args.fps) + ".mp4",fps=args.fps)

def main(args):
    if args.resize == True:
        print("resize image")
        original_dir_images = os.path.join(args.image_path, f"image")
        original_images = glob.glob(original_dir_images + '/*')
        dir_images = os.path.join(args.image_path, f"image_re")
        os.makedirs(dir_images, exist_ok=True)
        for o_img_dir in original_images:
            img_rgb  = cv.imread(o_img_dir)
            wid = args.imsize[0]
            hei = args.imsize[1]
            img_name = o_img_dir.split('/')[-1]
            img_resize = cv.resize(img_rgb, (wid, hei), interpolation = cv.INTER_AREA)
            cv.imwrite(os.path.join(dir_images, img_name), img_resize)

        file_list = glob.glob(os.path.join(dir_images,'*'))
    else:
        file_list = glob.glob(os.path.join(args.image_path, "image_re", '*'))

    making_video(args, file_list, extra_name='full')

    if args.crop:
        print("cropping image")
        original_dir_images = os.path.join(args.image_path, f"image_re")
        original_images = glob.glob(original_dir_images + '/*')
        dir_images = os.path.join(args.image_path, f"image_crop")
        os.makedirs(dir_images, exist_ok=True)
        s_p = args.start_point
        e_p = args.end_point
        for o_img_dir in original_images:
            img_rgb  = cv.imread(o_img_dir)
            img_name = o_img_dir.split('/')[-1]
            img_crop = img_rgb[s_p[0]:e_p[0], s_p[1]:e_p[1]]
            img_crop = cv.resize(img_crop, (3000, 1400), interpolation=cv.INTER_AREA)
            cv.imwrite(os.path.join(dir_images, img_name), img_crop)
        file_list = glob.glob(os.path.join(dir_images, '*'))
    else:
        file_list = glob.glob(os.path.join(args.image_path, "image_crop", '*'))
    making_video(args, file_list, extra_name='crop')

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='./imagee', type=str, help='path to images folder')
    parser.add_argument("--resize", action='store_true', help="resize the image")
    parser.add_argument("--imsize", default=(10000, 3000), nargs="+")
    parser.add_argument("--fps", type=int, default=15, help="the fps")
    parser.add_argument("--crop", action='store_true', help="cropping the image")
    parser.add_argument("--start_point", default=(2350,3350), nargs="+")
    parser.add_argument("--end_point", default=(2950,4950), nargs="+")
    args = parser.parse_args()

    main(args)

