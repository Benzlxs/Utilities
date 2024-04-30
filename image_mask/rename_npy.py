import cv2
import os
import numpy as np
from pathlib import Path

# Example usage
# root_dir = 'exp_data/skating'
root_dir = '/home/li325/dataset/dynamic_scene/data/HyperNeRF/hand1-dense-v2'
root_dir = Path(root_dir)

ratio = "2x"

image_dir = root_dir.joinpath("rgb/" + ratio)
mask_source = root_dir.joinpath("mask_npy_bk/" + ratio)
output_dir = root_dir.joinpath("mask_npy/" + ratio)
output_dir.mkdir(parents=True, exist_ok=True)

# all_dir_list = list(image_dir.glob("*.jpg"))
all_dir_list = sorted(list(image_dir.glob("*.png")))
start_idx = 0
for one_img_dir in all_dir_list:
    image_name = one_img_dir.name
    # print(one_img_dir)
    #os.system(f"cp ")
    img_no = int(image_name.split(".")[0])-1
    mask_path = str(output_dir.joinpath("%05d"%(img_no) + ".npy"))
    # one_image_output = output_dir.joinpath(f"{start_idx:05}.png")
    mask_source_path = str(mask_source.joinpath("%05d"%(start_idx) + ".npy"))
    cmd = f"cp {str(mask_source_path)} {str(mask_path)}"
    print(cmd)
    os.system(cmd)
    start_idx += 1
