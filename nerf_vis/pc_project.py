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

LARGE_VALUE = 65535.

@njit
def efficient_z_val(points, line_point, line_direction):
    perpendicular_vectors = np.cross(points - line_point, line_direction)
    # distances = np.linalg.norm(perpendicular_vectors, axis=1)
    distances = perpendicular_vectors[:,0]**2 +  perpendicular_vectors[:,1]**2 +  perpendicular_vectors[:,2]**2

    min_distance_index = np.argmin(distances)
    min_distance_point = points[min_distance_index]

    direction_vector = min_distance_point - line_point

    return np.dot(direction_vector, line_direction[0]), distances[min_distance_index]**(0.5)

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

def main(args):
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
    ## generate rays
    n_imgs, H_img, W_img, _ = all_images.shape

    pixels_x = np.arange(0, W_img, dtype=int)
    pixels_y = np.arange(0, H_img, dtype=int)

    pixels_xy = np.meshgrid(pixels_x, pixels_y, indexing='ij')

    all_pix_x = pixels_xy[0].reshape(-1)
    all_pix_y = pixels_xy[1].reshape(-1)

    # point cloud reading
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    point_cloud = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals= np.array(pcd.normals)
    pc_scale_mat = scale_mats_np[0]
    radius = pc_scale_mat[0,0]
    pc_translation = pc_scale_mat[:3,3]
    point_cloud -= pc_translation
    point_cloud /= radius
    # import trimesh
    # cloud = trimesh.PointCloud(point_cloud)
    # cloud.convex_hull
    # cloud.export("test.ply")

    # point_cloud = point_cloud[::10,:]
    ## construct the mesh
    pcd_m = o3d.geometry.PointCloud()
    pcd_m.points = o3d.utility.Vector3dVector(point_cloud[::1,:])
    pcd_m.colors = o3d.utility.Vector3dVector(colors[::1,:])
    pcd_m.normals = o3d.utility.Vector3dVector(normals[::1,:])
    pcd = pcd_m
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2, radius * 4, radius * 8, radius * 16]))
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.01)
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9 , width=0, scale=1.1, linear_fit=False)[0]
    o3d.io.write_triangle_mesh("bpa_mesh.ply", bpa_mesh)



    pc_scale_mat = scale_mats_np[0]
    radius = pc_scale_mat[0,0]
    pc_translation = pc_scale_mat[:3,3]
    ## referring to the line 385 in exp_runner.py code
    point_cloud -= pc_translation
    point_cloud /= radius


    point_cloud4 = np.concatenate([point_cloud, np.ones_like(point_cloud[:,[0]])], axis=-1)  # batch_size, 3

    for img_idx in range(n_imgs):
        cur_pose =   all_poses[[img_idx], :3, 3]
        dist_pc_cam = point_cloud - cur_pose
        dist_pc_cam = (dist_pc_cam[:,0]**2 + dist_pc_cam[:,1]**2 + dist_pc_cam[:,2]**2)**(0.5)
        z_val_array = np.zeros([H_img, W_img])
        z_val_array[...] = LARGE_VALUE
        print("Process image: {};".format(img_idx))
        t0 = time.time()
        # p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1)  # batch_size, 3
        pts_img = np.matmul(P_w2img[[img_idx],:,:], point_cloud4[:,:, None])
        # pts_img = pts_img.reshape(batch_size*n_samples, 3).squeeze()
        pts_img = pts_img.reshape(-1, 3)
        u_img = np.rint((pts_img[:,0]/pts_img[:,2])).astype(int)
        v_img = np.rint((pts_img[:,1]/pts_img[:,2])).astype(int)
        projection_mask = (u_img > 0) & (u_img < W_img) & (v_img > 0) & (v_img < H_img)
        # u_img = u_img.clip(0, W_img - 1)
        # v_img = v_img.clip(0, H_img - 1)
        u_img = u_img[projection_mask]
        v_img = v_img[projection_mask]
        # pts_z = pts_img[projection_mask, 2]
        pts_z = dist_pc_cam[projection_mask]

        masked_point_cloud = point_cloud[projection_mask, :]


        uv_img_num = len(u_img)

        for i_uv in range(uv_img_num):
            z_val_array[v_img[i_uv], u_img[i_uv]] = min(z_val_array[v_img[i_uv], u_img[i_uv]],  pts_z[i_uv])

        z_val_array = np.where(z_val_array == LARGE_VALUE, 0., z_val_array)
        np.save("z_val_npy/%03d.npy"%img_idx, z_val_array)

        # change z_val_array into heat map
        rgb_img = all_images[img_idx]

        # z_val_array += 1.
        # z_val_array = z_val_array / 2.
        z_val_array = (z_val_array - z_val_array.min()) / (z_val_array.max() - z_val_array.min())
        z_val_array = (z_val_array * 255).clip(0,255).astype(np.uint8)
        z_val_heat_map = cv.applyColorMap(z_val_array, cv.COLORMAP_JET)

        full_img = np.concatenate([rgb_img, z_val_heat_map], axis=0)

        cv.imwrite("./z_val_img/"+ str(img_idx) + "depth.png" ,  full_img)

        print("Processing time: ", time.time() - t0)

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
        "--min_dist_threshold", type=float, default=0.001, help="min distance for valid point"
    )
    args = parser.parse_args()
    main(args)


