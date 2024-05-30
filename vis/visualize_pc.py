import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
import sys, os
import colorsys

def id2rgb(id, max_num_obj=100):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")
    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def main(points, colors):
    num_points = points.shape[0]
    # Set points for the original and offset point clouds
    pcd_points = o3d.geometry.PointCloud()

    pcd_points.points = o3d.utility.Vector3dVector(points)
    obj_color = visualize_obj(1 - colors)
    pcd_points.colors = o3d.utility.Vector3dVector(obj_color.squeeze())

    # Visualize using Open3D
    o3d.visualization.draw_geometries([pcd_points])


if __name__=="__main__":
    xyz_npy_path = sys.argv[1]
    print("xyz file:", xyz_npy_path)
    color_npy_path = xyz_npy_path.replace("_xyz.","_color.")
    points = np.load(xyz_npy_path)
    colors = np.load(color_npy_path)
    main(points, colors)

