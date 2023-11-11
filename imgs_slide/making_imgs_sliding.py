import numpy as np
from pathlib import Path
import cv2

W = 4000
H = 2600
NUM = 20
overall_img = np.zeros((H, W, 3), dtype=np.uint8)
h_int = int(H/NUM)
w_int = int(W/NUM)
break_b = 10

root_dir = Path("/home/li325/data/biomass/DJI_Raw_Data/DJI_202304031318_007_22Biomass-12m")
save_dir = Path("./imgs_0")
save_dir.mkdir(exist_ok=True, parents=True)
all_imgs = sorted(list(root_dir.glob("*.JPG")))

for idx, i_img_path in enumerate(all_imgs[::1]):
    i_img = cv2.imread(str(i_img_path))
    i_img_name = i_img_path.name
    img_small = cv2.resize(i_img, (w_int - break_b, h_int - break_b), interpolation=cv2.INTER_AREA)

    i_row = idx // NUM
    i_colum = idx % NUM

    overall_img[i_row*h_int:((i_row + 1)*h_int - break_b),  i_colum*w_int: ((i_colum + 1)*w_int - break_b), :] = img_small

    if idx >= NUM**2:
        break

cv2.imwrite(str(save_dir.joinpath(i_img_name)), overall_img)






# imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
