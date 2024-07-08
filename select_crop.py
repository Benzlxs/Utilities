###
import numpy as np
import os
import sys
import glob
import argparse
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from moviepy.editor import ImageSequenceClip

def extract_pixel_location(image):
    global selected_pixel
    selected_pixel = None
    # Function to get the pixel value at a clicked position
    def get_pixel_location(event, x, y, flags, param):
        global selected_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixel = (x, y)
            print(f'Selected pixel location: ({selected_pixel})')
            cv2.destroyAllWindows()

    # Create a window and set a mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', get_pixel_location)
    while True:
        cv2.imshow('Image', image)
        if selected_pixel is not None:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    cv2.destroyAllWindows()
    return selected_pixel

def main(folder_name, img_name):
    root_dir = "/Users/li325/Documents/Publications/RotGS/figure/"
    method_list = ['2dgs', "colmap", "gt", "neus", "ours", "ours"]
    central_point=None
    save_path = "/Users/li325/Documents/Publications/RotGS/figure_out/"
    os.makedirs(save_path, exist_ok=True)
    for idx, m in enumerate(method_list):
        if idx == len(method_list) -1 : # rgb image
            img_path = root_dir + folder_name + "/" + m + "/0" + img_name + ".png"
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            save_image_path = save_path + folder_name + "_" + m + "_rgb_" + img_name + ".png"
        else:  # mesh image
            img_path = root_dir + folder_name + "/" + m + "/" + "render_" + img_name + ".jpg"
            image = cv2.imread(img_path)
            save_image_path = save_path + folder_name + "_" + m + "_" + img_name + ".png"

        h, w, _ = image.shape
        if central_point is None:
            central_point = extract_pixel_location(image)
            print(central_point)
        x_c, y_c = central_point

        offset = 1600
        if h == 4480: # scale = 8
            h_p = [y_c - offset, y_c + offset]
            x_p = [x_c - offset, x_c + offset]
            img_crop = image[h_p[0]:h_p[1], x_p[0]:x_p[1]]
            img_crop = cv2.resize(img_crop, (400, 400), interpolation=cv2.INTER_AREA)
        else:  # scale = 1
            offset1 = offset // 8
            y_c_t = y_c // 8
            x_c_t = x_c // 8
            h_p = [y_c_t - offset1, y_c_t + offset1]
            x_p = [x_c_t - offset1, x_c_t + offset1]
            img_crop = image[h_p[0]:h_p[1], x_p[0]:x_p[1]]
            img_crop = cv2.resize(img_crop, (400, 400), interpolation=cv2.INTER_AREA)

        cv2.imwrite(save_image_path, img_crop)


if __name__ == "__main__":
    # Parser
    folder_name = sys.argv[1]
    img_name = sys.argv[2]
    main(folder_name, img_name)

