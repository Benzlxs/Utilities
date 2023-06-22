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



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="dataset root")
    parser.add_argument(
        "--min_dist_threshold", type=float, default=0.001, help="min distance for valid point"
    )
    parser.add_argument("--column_gap", type=int, default=200, help="the starting column entry")
    args = parser.parse_args()
    main(args)


