"""
This is for pre-processing the point cloud!
1. Iterate each image W and H
2. Get the surface z_val value
3. Save the z_val to form a new image array
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
import open3d as o3d
import warnings
import numpy as np
import math
from argparse import ArgumentParser
from nerfvis import Scene  # pip install nerfvis
from scipy.spatial.transform import Rotation
from PIL import Image
from scipy.spatial.transform import Rotation
from numba import njit
import multiprocessing as mp

LARGE_VALUE = 65535


# @njit
def efficient_z_val(points, line_point, line_direction):
    perpendicular_vectors = np.cross(points - line_point, line_direction)
    # distances = np.linalg.norm(perpendicular_vectors, axis=1)
    distances = perpendicular_vectors[:,0]**2 +  perpendicular_vectors[:,1]**2 +  perpendicular_vectors[:,2]**2

    min_distance_index = np.argmin(distances)
    min_distance_point = points[min_distance_index]

    direction_vector = min_distance_point - line_point

    return np.dot(direction_vector, line_direction[0]), distances[min_distance_index]**(0.5)

def z_val_mp(inputs):
    points, line_point, line_direction, idx = inputs
    perpendicular_vectors = np.cross(points - line_point, line_direction)
    # distances = np.linalg.norm(perpendicular_vectors, axis=1)
    distances = perpendicular_vectors[:,0]**2 +  perpendicular_vectors[:,1]**2 +  perpendicular_vectors[:,2]**2
    min_distance_index = np.argmin(distances)
    min_distance_point = points[min_distance_index]

    direction_vector = min_distance_point - line_point

    return [idx, np.dot(direction_vector, line_direction[0]), distances[min_distance_index]**(0.5)]



def minimum_distance_to_line(points, line_point, line_direction):
    # line_direction = line_direction / np.linalg.norm(line_direction)  # Normalize line direction
    perpendicular_vectors = np.cross(points - line_point, line_direction)
    distances = np.linalg.norm(perpendicular_vectors, axis=1)

    min_distance_index = np.argmin(distances)
    min_distance_point = points[min_distance_index]
    min_distance = distances[min_distance_index]
    return min_distance_point, min_distance

def find_crossing_point(point, line_point, line_direction):
    line_direction = line_direction / np.linalg.norm(line_direction)  # Normalize line direction

    direction_vector = point - line_point
    dot_product = np.dot(direction_vector, line_direction)
    crossing_point = line_point + dot_product * line_direction
    return crossing_point


def look_for_dir(cands, required=False):
    for cand in cands:
        if path.isdir(path.join(args.data_dir, cand)):
            return path.join(args.data_dir, cand)
    if required:
        assert False, "None of " + str(cands) + " found in data directory"
    return ""

def sort_key(x):
    if len(x) > 2 and x[1] == "_":
        return x[2:]
    return x

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def read_pose_info(args, n_images):
    camera_dict = np.load(os.path.join(args.data_dir, "cameras_sphere.npz"))
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_np = []
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    intrinsics_all = []
    pose_all = []
    P_w2img  = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        P_w2img.append(P)
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
    P_w2img = np.stack(P_w2img)
    all_poses = np.stack(pose_all)
    intrinsics_all = np.stack(intrinsics_all)
    intrinsics_all_inv = np.linalg.inv(intrinsics_all)
    return P_w2img, all_poses, intrinsics_all, intrinsics_all_inv, scale_mats_np


def main(img_idx, args):
    mp.freeze_support()

    dataset_name = path.basename(path.abspath(args.data_dir))
    pose_dir = path.join(args.data_dir, "pose_colmap")
    pose_gt_dir = look_for_dir(["poses", "pose", "c2w", "cameras"])
    if not path.isdir(pose_dir):
        pose_dir, pose_gt_dir = pose_gt_dir, None
    images_dir = look_for_dir(["images", "image", "rgb", "color", "rgbs"])
    point_cloud_path = glob.glob(args.data_dir + "/*total.ply")[0]
    print("POSE_DIR", pose_dir)
    print("IMAGES_PATH", images_dir)
    print("POINT_CLOUD_PATH", point_cloud_path)
    image_files = sorted([x for x in os.listdir(images_dir) if x.lower().endswith('.png') or x.lower().endswith('.jpg')], key=sort_key)
    n_images = len(image_files)

    P_w2img, all_poses, intrinsics_all, intrinsics_all_inv, scale_mats_np = read_pose_info(args, n_images)

    R = all_poses[:, :3, :3]
    t = all_poses[:, :3, 3]
    intrins = intrinsics_all[0]
    focal = (intrins[0, 0] + intrins[1, 1]) * 0.5

    all_images = [ Image.open(images_dir + "/" + i) for i in image_files]
    all_images  = np.stack(all_images)

    # point cloud reading
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    point_cloud = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    pc_scale_mat = scale_mats_np[0]
    radius = pc_scale_mat[0,0]
    pc_translation = pc_scale_mat[:3,3]
    point_cloud -= pc_translation
    point_cloud /= radius
    point_cloud = point_cloud[::10,:]

    ## generate rays
    n_imgs, H_img, W_img, _ = all_images.shape

    pixels_x = np.arange(0, W_img, dtype=int)

    h_gap = args.column_gap
    for h_start in range(0, H_img, h_gap):
        pixels_y = np.arange(h_start, h_start + h_gap, dtype=int)
        pixels_xy = np.meshgrid(pixels_x, pixels_y, indexing='ij')

        all_pix_x = pixels_xy[0].reshape(-1)
        all_pix_y = pixels_xy[1].reshape(-1)

        # for img_idx in range(4):# range(n_imgs):
        z_val_array = np.zeros([h_gap, W_img])
        print("Process image: {}, {};".format(img_idx, h_start))
        t0 = time.time()
        p = np.stack([all_pix_x, all_pix_y, np.ones_like(all_pix_y)], axis=-1)  # batch_size, 3
        p = np.matmul(intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)    # batch_size, 3
        rays_v = np.matmul(all_poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = np.repeat(all_poses[img_idx,None,:3,3], rays_v.shape[0], axis=0)

        line_point = all_poses[img_idx, :3, 3]
        # print(all_pix_x.shape)
        with mp.Pool() as mp_pool:
            dist_new = mp_pool.map(z_val_mp, ((point_cloud, line_point, rays_v[[p_idx],:], p_idx) for p_idx in range(all_pix_x.shape[0])), chunksize=128)
        #  for p_idx in range(all_pix_x.shape[0]):
        #      print(p_idx)
        #      line_direction = rays_v[[p_idx],:]
        #      # min_point, min_dist = minimum_distance_to_line(point_cloud, line_point, line_direction)
        #      # direction_vector = min_point - line_point
        #      # p_z_val = np.dot(direction_vector, line_direction[0])
        #      p_z_val, min_dist = efficient_z_val(point_cloud, line_point, line_direction)
        #      z_val_array[all_pix_y[p_idx], all_pix_x[p_idx]] = p_z_val if min_dist < args.min_dist_threshold else LARGE_VALUE
        for idx_dist in dist_new:
            p_idx, p_z_val, min_dist = idx_dist
            z_val_array[all_pix_y[p_idx] - h_start, all_pix_x[p_idx]] = p_z_val if min_dist < args.min_dist_threshold else LARGE_VALUE

        np.save("z_val_npy/z_val_%03d_%04d_%d.npy"%(img_idx,h_start, h_gap), z_val_array)
        print(img_idx, h_start, "Processing time: ", time.time() - t0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, help="dataset root")
    parser.add_argument(
        "--semantic_mask", type=str, default=None, help= "the number of chosed camera poses"
    )
    parser.add_argument(
        "--show_outside_color_only", action="store_true", default=False, help= "Displaying the inside colors"
    )
    parser.add_argument(
        "--img_start", type=int, default=None
    )
    parser.add_argument(
        "--img_end", type=int, default=None
    )
    parser.add_argument(
        "--min_dist_threshold", type=float, default=0.001, help="min distance for valid point"
    )
    parser.add_argument("--column_gap", type=int, default=200, help="the starting column entry")
    args = parser.parse_args()
    for img_idx in range(args.img_start, args.img_end+1):
        main(img_idx, args)



