"""
Evaluate the performance of different similarity metrics
"""
import sys
import os
import struct
import collections
import glob
import time
from os import path
import cv2 as cv
DIR_PATH = path.dirname(os.path.realpath(__file__))
sys.path.append(path.join(DIR_PATH, ".."))
import numpy as np
import math
from argparse import ArgumentParser


def main(args):
    img_0 = cv.imread(args.img0_dir)
    print("img 0 size:", img_0.shape)
    h_0, w_0, _ = img_0.shape
    img_1 = cv.imread(args.img1_dir)
    print("img 1 size:", img_1.shape)
    img_1 = cv.resize(img_1, [w_0, h_0])

    img_0_gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
    img_1_gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)

    if args.similarity_metric == "NCC":
        similarity = function_ncc(img_0_gray, img_1_gray)
    elif args.similarity_metric == "SSIM":
        similarity = function_ssim(img_0_gray, img_1_gray)
    else:
        raise NotImplementedError("Other methods have been implimented!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img0_dir", type=str, default="000026_a_window.png", help="image 0 directory")
    parser.add_argument("--img1_dir", type=str, default="000001_a_window.png", help="image 1 directory")
    parser.add_argument("--similarity_metric", type=str, default="NCC")
    args = parser.parse_args()
    main(args)


