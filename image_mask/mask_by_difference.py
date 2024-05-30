import cv2
import numpy as np
from pathlib import Path


def apply_mask_to_image_opencv(image_path, mask_path, output_path):
    # Read the original image
    image_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    h, w, _ = image_raw.shape
    # Convert image to RGBA if it is not already
    # if image.shape[2] < 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = mask[:,:,[2, 0, 1]]
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask_diff = image - mask
    # mask_diff = mask_diff.mean(axis=2)
    # Ensure mask is the same size as image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # mask = cv2.erode(mask, np.ones((8,8), np.uint8), iterations=1)

    # Convert the mask to 4 channels
    # alpha_channel = cv2.bitwise_not(mask)
    alpha_channel = mask

    # Split the original image into its component channels
    b, g, r = cv2.split(image)
    mask_sign =  (mask/255).astype(bool)
    b *= mask_sign
    g *= mask_sign
    r *= mask_sign

    # Merge the channels and apply the inverted mask as the new alpha channel
    final_image = cv2.merge((b, g, r, alpha_channel))
    # final_image = cv2.merge((r, g, b, alpha_channel))
    # final_image = cv2.merge((b, g, r))

    # Save the resulting image
    cv2.imwrite(output_path, final_image)

# Example usage
# root_dir = 'exp_data/skating'
root_dir = '/home/li325/dataset/dynamic_scene/data/HyperNeRF/keyboard'
root_dir = Path(root_dir)

ratio = "2x"

image_dir = root_dir.joinpath("rgb/" + ratio)
mask_dir = root_dir.joinpath("mask_png/" + ratio)
output_dir = root_dir.joinpath("mask/" + ratio)
output_dir.mkdir(parents=True, exist_ok=True)

# all_dir_list = list(image_dir.glob("*.jpg"))
all_dir_list = list(image_dir.glob("*.png"))
for one_img_dir in all_dir_list:
    image_name = one_img_dir.name
    print("Process:", image_name)
    image_path = str(one_img_dir)
    # mask_path = str(mask_dir.joinpath(image_name.replace(".jpg", ".png")))
    # mask_path = str(mask_dir.joinpath(str(int(image_name.split(".")[0])) + ".png"))
    img_no = int(image_name.split(".")[0])
    mask_path = str(mask_dir.joinpath("%05d"%(img_no) + ".png"))
    output_path = str(output_dir.joinpath(image_name))
    apply_mask_to_image_opencv(image_path, mask_path, output_path)
