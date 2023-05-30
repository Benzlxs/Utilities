"""
Basic data visualization (using PlenOctree's volrend)
Usage: python view_data.py <data_root>
default output: data_vis.html. You can open this in your browser. (bash sensei/mkweb)
"""
# Copyright 2021 Alex Yu
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

import warnings
import numpy as np
import math
from argparse import ArgumentParser
from nerfvis import Scene  # pip install nerfvis
from scipy.spatial.transform import Rotation

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

# BEGIN BORROWED CODE
# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# All rights reserved.
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis. """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.  """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(((0.0,         -direction[2],  direction[1]),
                      (direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta

def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R

def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t

def align_procrustes_rt(t_a : np.ndarray, q_a : np.ndarray,
                        t_ref : np.ndarray,
                        use_first_k : int = 1000000,
                        want_transform : bool = False):
    """
    Align translation +  rotation
    :param t_a: camera translations to align (N, 3)
    :param q_a: camera rotations to align (xyz axis-angle, xyzw quaternion, or rotation matrix) (N, {3, 4, 9})
    :param t_ref: reference camera translations (N, 3)
    :param use_first_k: int, if set, uses only first k number of cameras to align
    :param want_transform: bool, if set, returns transform function instead of transformed points
    :return:
        if want_transform == False:
            t (N, 3), q (N, {3, 4, 9}) similarity-transformed version of cameraa poses, aligned to ref
        else: function which given points, applies the aligning transform
    """
    assert t_ref.shape[0] == t_a.shape[0]
    s, R, t = align_umeyama(t_ref[:use_first_k], t_a[:use_first_k])

    #  # Advanced alignment
    #  n_points = t_a.shape[0]
    #  z = np.zeros((n_points, 3))
    #  z[:, -1] = 0.05
    #  t_a_aug = t_a + quaternion_rotate_vector_np(q_a, z) / s
    #  t_ref_aug = t_ref + quaternion_rotate_vector_np(q_ref, z)
    #
    #  _, R, t = align_umeyama(np.concatenate([t_ref, t_ref_aug], axis=0), np.concatenate([t_a * s, t_a_aug * s], axis=0), known_scale=True)

    def transform(t_b : np.ndarray, q_b : np.ndarray):
        t_align = s * t_b @ R.T + t
        Ra = Rotation.from_matrix(R)
        q_align = (Ra * Rotation.from_matrix(q_b)).as_matrix()
        return t_align, q_align
    return transform if want_transform else transform(t_a, q_a)

# END BORROWED CODE
def get_image_size(path : str):
    """
    Get image size without loading it
    """
    from PIL import Image
    im = Image.open(path)
    return im.size # W, H

def sort_key(x):
    if len(x) > 2 and x[1] == "_":
        return x[2:]
    return x

from numba import njit
@njit
def efficient_dist_calc(A, B):
    Dist = A - B
    Dist = (Dist**2).sum(-1)
    return Dist

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    # return points[sample_inds]
    # return sample_inds.tolist()
    return sample_inds

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_colmap_sparse(sparse_path):
    points3D_idmap = {}
    points3D = []
    with open(sparse_path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            points3D_idmap[point3D_id] = i
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D.append(
                Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
            )
    return points3D

# This function is borrowed from IDR: https://github.com/lioryariv/idr
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

def main(args):
    batch_size = args.batch_size
    n_samples  = args.n_samples
    is_transform = args.is_transform
    dataset_name = path.basename(path.abspath(args.data_dir))
    near_n_cameras = args.near_n_cameras

    def look_for_dir(cands, required=False):
        for cand in cands:
            if path.isdir(path.join(args.data_dir, cand)):
                return path.join(args.data_dir, cand)
        if required:
            assert False, "None of " + str(cands) + " found in data directory"
        return ""

    pose_dir = path.join(args.data_dir, "pose_colmap")
    pose_gt_dir = look_for_dir(["poses", "pose", "c2w", "cameras"])
    if not path.isdir(pose_dir):
        pose_dir, pose_gt_dir = pose_gt_dir, None
    images_dir = look_for_dir(["images", "image", "rgb", "color", "rgbs"])
    # intrin_path = path.join(args.data_dir, "intrinsics.txt")

    #point_cloud_path = path.join(args.data_dir, "sparse/0/points3D.bin")
    point_cloud_path = glob.glob(args.data_dir + "/*total.ply")[0]

    print("POSE_DIR", pose_dir)
    print("IMAGES_PATH", images_dir)
    # print("INTRIN_PATH", intrin_path)
    print("POINT_CLOUD_PATH", point_cloud_path)
    # pose_files = sorted([x for x in os.listdir(pose_dir) if x.lower().endswith('.txt')], key=sort_key)
    image_files = sorted([x for x in os.listdir(images_dir) if x.lower().endswith('.png') or x.lower().endswith('.jpg')], key=sort_key)

    #TODO: generate new points
    # all_poses = []
    # for i, pose_file in enumerate(pose_files):
    #     pose = np.loadtxt(path.join(pose_dir, pose_file)).reshape(4, 4)
    #     #  splt = path.splitext(pose_file)[0].split('_')
    #     #  num = int(splt[1] if len(splt) > 1 else splt[0])
    #     all_poses.append(pose)
    # all_poses = np.stack(all_poses)
    n_images = len(image_files)
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

    is_transform = False

    def get_transform(c2w):
        t = c2w[:, :3, 3]
        R = c2w[:, :3, :3]

        # (1) Rotate the world so that z+ is the up axis
        # we estimate the up axis by averaging the camera up axes
        ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
        world_up = np.mean(ups, axis=0)
        world_up /= np.linalg.norm(world_up)

        up_camspace = np.array([0.0, -1.0, 0.0])
        c = (up_camspace * world_up).sum()
        cross = np.cross(world_up, up_camspace)
        skew = np.array([[0.0, -cross[2], cross[1]],
                         [cross[2], 0.0, -cross[0]],
                         [-cross[1], cross[0], 0.0]])
        R_align = np.eye(3)
        if c > -1:
            R_align = R_align + skew + (skew @ skew) * 1 / (1+c)
        else:
            # In the unlikely case the original data has y+ up axis,
            # rotate 180-deg about x axis
            R_align = np.array([[-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

        R = (R_align @ R)
        fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
        t = (R_align @ t[..., None])[..., 0]

        # (2) Recenter the scene using camera center rays
        # find the closest point to the origin for each camera's center ray
        dvec = t + (fwds * -t).sum(-1)[:, None] * fwds

        # Median for more robustness
        translate = -np.median(dvec, axis=0)

        transform = np.eye(4)
        transform[:3, 3] = translate
        transform[:3, :3] = R_align

        # (3) Rescale the scene using camera distances
        scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
        scale *= 0.95
        return transform, scale

    if is_transform:
        T, scale = get_transform(all_poses)
        all_poses = T @ all_poses # @ np.diag([1, -1, -1, 1])
        R = all_poses[:, :3, :3]
        t = all_poses[:, :3, 3] * scale
    else:
        R = all_poses[:, :3, :3]
        t = all_poses[:, :3, 3]

    # intrins = np.loadtxt(intrin_path)
    intrins = intrinsics_all[0]
    focal = (intrins[0, 0] + intrins[1, 1]) * 0.5

    image_wh = get_image_size(path.join(images_dir, image_files[0]))

    scene = Scene("colmap dataset: " + dataset_name)
    scene.set_opencv()

    # Try to pick a good frustum size
    avg_dist : float = np.mean(np.linalg.norm(t[1:] - t[:-1], axis=-1))
    cam_scale = avg_dist * 0.3

    # Infer world up direction from GT cams
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    # ups = np.sum(R * np.array([0, 1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    # Camera forward vector
    forwards = np.sum(R * np.array([0, 0, 1.0]), axis=-1)
    vforward = np.mean(forwards, axis=0)
    vforward /= np.linalg.norm(vforward)

    # Set camera center of rotation (origin) for orbit
    origin = np.mean(t, axis=0)

    # Set camera position
    center = origin - vforward * np.linalg.norm(t - origin, axis=-1).mean() * 0.7 * 3
    print('  camera center', center, 'vforward', vforward, 'world_up', world_up)

    R = R #  @ np.diag([1,-1,-1])

    scene.add_camera_frustum(name=f"traj_{n_images:04d}", focal_length=focal,
                             image_width=image_wh[0],
                             image_height=image_wh[1],
                             z=0.2,
                             # z=-1,
                             # opengl=True,
                             r=R,
                             t=t,
                             connect=args.seg,
                             color=[1.0, 0.0, 0.0])


    from PIL import Image
    from scipy.spatial.transform import Rotation
    all_images = [ Image.open(images_dir + "/" + i) for i in image_files]
    all_images  = np.stack(all_images)

    ## generate rays
    n_imgs, H_img, W_img, _ = all_images.shape

    if args.img_idx == -1:
        img_idx = np.random.randint(n_imgs)
    else:
        img_idx = args.img_idx

    # img_idx_neighbor = img_idx + 26

    #TODO find the neareast camera
    cur_loc = all_poses[[img_idx], :3, 3]
    distance = ((t - cur_loc)**2).sum(axis=-1)

    near_idx = distance.argsort()[1:(near_n_cameras + 1)]  # top 2
    img_idx_neighbor = near_idx

    pixels_x = np.random.randint(low=0, high=W_img, size=[batch_size])
    pixels_y = np.random.randint(low=0, high=H_img, size=[batch_size])

    print("Process image: {}; the nearest image is {}".format(img_idx, img_idx_neighbor))

    p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1)  # batch_size, 3
    p = np.matmul(intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
    if batch_size == 1:
        p = p[None,:]
    rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)    # batch_size, 3
    rays_v = np.matmul(all_poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
    if batch_size == 1:
        rays_v = rays_v[None,:]
    # rays_o = all_poses[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
    rays_o = np.repeat(all_poses[img_idx,None,:3,3], rays_v.shape[0], axis=0)
    a = np.sum(rays_v**2, axis=-1, keepdims=True)
    b = 2.0 * np.sum(rays_o * rays_v, axis=-1, keepdims=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    z_vals = np.linspace(0.0, 1.0, n_samples)
    z_vals = near + (far - near) * z_vals[None, :]
    pts = rays_o[:, None, :] + rays_v[:, None, :] * z_vals[..., :, None]

    one_mat = np.ones_like(pts[...,0])

    pts_ad = np.concatenate([pts, one_mat[:,:,None]], axis=-1)

    # plot the camera ray
    for bs in range(batch_size):
        scene.add_line("rays/{}".format(bs),
                       all_poses[img_idx,:3,3],
                       pts[bs,-1,:],
                       color=np.random.random(3),
                       )
    pts_plot = pts.reshape(-1,3)
    pts_flat = pts_ad.reshape(-1,4)
    # plot the sampled pts to neighoring cameras
    for _idx_neigh in img_idx_neighbor:
        scene.add_line("near_camera/{}".format(_idx_neigh),
                       all_poses[img_idx,:3,3],
                       all_poses[_idx_neigh,:3,3],
                       color=[0., 1., 0.])

        neighor_t = all_poses[[_idx_neigh],:3,3] # t[[img_idx+1], :3]
        neighor_t = np.repeat(neighor_t, batch_size*n_samples, axis=0)
        #                    pts_plot[k],
        #                    neighor_t[k],
        #                    color=np.random.random(3))
        all_line_pts = np.concatenate([pts_plot, neighor_t], axis=0)
        start_indice = np.arange(batch_size*n_samples)
        end_indice = np.arange(batch_size*n_samples) + batch_size*n_samples
        segs = np.stack([start_indice, end_indice],axis=1)
        if args.plot_neighbor:
            scene.add_lines("lines/{}".format(_idx_neigh),
                            all_line_pts,
                            segs = segs,
                            # r=R,
                            # t=t,
                            color=np.random.random(3),
                            # vert_color=np.random.random([batch_size*n_samples, 3])
                            )
        ## tranform from world coordinate to image planes
        pts_img = np.matmul(P_w2img[[_idx_neigh],:,:], pts_flat[:,:, None])
        pts_img = pts_img.reshape(batch_size*n_samples, 3).squeeze()
        u_img = np.rint((pts_img[:,0]/pts_img[:,2])).astype(int)
        v_img = np.rint((pts_img[:,1]/pts_img[:,2])).astype(int)
        print(u_img.max(), u_img.min(), v_img.max(), v_img.min())
        print(all_images.shape)

        for u, v in zip(u_img.tolist(), v_img.tolist()):
            cv.circle(all_images[_idx_neigh], (u, v), radius=8, color=np.random.random(3)*255, thickness=-1)


    for i_img in range(n_images):
        r = Rotation.from_matrix(np.asmatrix(R[i_img,...]))
        scene.add_image(
                    f"images/{i_img}",
                    # images_dir+ "/" + image_files[i_img],
                    all_images[i_img],
                    # r=R[[i_img],:3,:3],
                    r=r.as_rotvec(),
                    t=t[[i_img],:3],
                    focal_length=focal,
                    z=0.2,
                    #  z=-1,
                    # opengl=True,
                    image_size=512,
        )
    if pose_gt_dir is not None:
        print('Loading GT')
        pose_gt_files = sorted([x for x in os.listdir(pose_gt_dir) if x.endswith('.txt')], key=sort_key)
        all_gt_poses = []
        for pose_file in pose_gt_files:
            pose = np.loadtxt(path.join(pose_gt_dir, pose_file))
            all_gt_poses.append(pose)
        all_gt_poses = np.stack(all_gt_poses)
        R_gt = all_gt_poses[:, :3, :3]
        t_gt = all_gt_poses[:, :3, 3]
        pose_files_st = set(pose_files)
        pose_gt_inds = np.array([i for i, pose_gt_file in enumerate(pose_gt_files) if pose_gt_file in pose_files_st], dtype=np.int64)
        print(len(pose_gt_inds), 'of', len(pose_gt_files), 'registered')
        if len(pose_gt_inds) < len(pose_gt_files):
            warnings.warn("Not all frames registered")

        r = R.reshape(-1, 9)
        r_gt = R_gt.reshape(-1, 9)

        transform = align_procrustes_rt(
                t_gt[pose_gt_inds], r_gt[pose_gt_inds],
                t, r, use_first_k=args.n_cameras_for_procrustes, want_transform=True)

        t_gt, r_gt = transform(t_gt, r_gt)
        R_gt = r_gt.reshape(-1, 3, 3)
        scene.add_camera_frustum(name=f"traj_gt", focal_length=focal,
                                 image_width=image_wh[0],
                                 image_height=image_wh[1],
                                 z=0.1,
                                 r=R_gt,
                                 t=t_gt,
                                 connect=args.seg,
                                 color=[0.0, 0.0, 1.0])
        scene.add_sphere(name=f"start", translation=t_gt[0],
                         scale=avg_dist * 0.1,
                         color=[0.0, 1.0, 1.0])

    if path.isfile(point_cloud_path):
        #TODO recale the point clouds
        # point_cloud = np.load(point_cloud_path)
        # point3d = read_colmap_sparse(point_cloud_path)
        # point_cloud = np.stack([p.xyz for p in point3d])
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        point_cloud = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pc_scale_mat = scale_mats_np[0]
        radius = pc_scale_mat[0,0]
        pc_translation = pc_scale_mat[:3,3]
        point_cloud -= pc_translation
        point_cloud /= radius
        if is_transform:
            point_cloud = (T[:3, :3] @ point_cloud[:, :, None])[:, :, 0] + T[:3, 3]
            point_cloud *= scale
        scene.add_points("point_cloud", point_cloud, vert_color=colors, unlit=True)

    if args.process_point_cloud_for_mask:
        t0 = time.time()
        index_pc = fps(point_cloud, 40000)
        print("Finish FPS!")
        print("Processing time:", time.time() - t0)

        point_cloud = point_cloud[index_pc]
        # point_cloud = point_cloud[::128,:]
        print("Effecitve point number is: %d"%(point_cloud.shape[0]))
        # pts_pc_dist = [(pts_plot[:,[i]] - point_cloud[:,[i]])**2 for i in range(3)]
        # pts_pc_dist = (sum(pts_pc_dist))**(1/2)
        # pts_plot = pts_plot.reshape(batch_size, n_samples, 3)
        # pts_pc_dist = np.linalg.norm(pts_plot[:,:,None,:]-point_cloud[None,None,:,:], axis=3)
        # pts_pc_dist = np.linalg.norm(pts_plot[:,None,:] - point_cloud[None,:,:], axis=2)
        pts_pc_dist = efficient_dist_calc(pts_plot[:,None,:], point_cloud[None,:,:])

        pts_pc_dist = pts_pc_dist.reshape(batch_size, n_samples, -1)
        dist_min = pts_pc_dist.min(axis=2)
        min_dist_value = dist_min.min(axis=1)
        dist_mask = dist_min < min_dist_value[:,None] + 0.001
        dist_mask = (~dist_mask)*0.001 + dist_mask*(-10000.)
        dist_mask = np.cumsum(dist_mask, axis=1)
        dist_mask = dist_mask > 0
        dist_mask = dist_mask.reshape(-1,1)
        colors = dist_mask*np.array([[1.,0.,0.]]) + (~dist_mask)*np.array([[0.,0.,1.]])
        scene.add_points("dist_with_pc", pts_plot.reshape(-1,3), vert_color=colors, unlit=True)



    if  args.semantic_mask is not None:
        sem_pts  = np.load("middle_npy_matching/" +  args.semantic_mask + "_pts.npy")
        sem_mask = np.load("middle_npy_matching/" +  args.semantic_mask + "_sem_mask.npy")
        if len(sem_mask.shape) == 2:
            sem_mask = sem_mask[:,:,None]

        batchsize, sample_points, _ = sem_mask.shape
        sem_pts = sem_pts.reshape([batchsize, sample_points, -1])
        colors = sem_mask*np.array([[1., 0, 0]]) + (~sem_mask)*np.array([[0,0,1.]])
        colors = colors.reshape(-1, 3)
        sem_pts = sem_pts.reshape(-1,3)
        sem_mask = sem_mask.reshape(-1)
        if args.show_outside_color_only:
            scene.add_points("semantic_points", sem_pts[sem_mask], vert_color=colors[sem_mask], unlit=True)
        else:
            scene.add_points("semantic_points", sem_pts, vert_color=colors, unlit=True)


    out_dir = path.join(args.data_dir, "visual")
    scene.add_axes(length=1.0, visible=False)
    scene.add_sphere("Unit Sphere", visible=False)
    scene.add_wireframe_cube("Unit Cube", scale=2, visible=False)
    print('WRITING', out_dir)
    scene.display(out_dir, world_up=world_up, cam_origin=origin, cam_center=center, cam_forward=vforward)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, help="dataset root")
    parser.add_argument(
        "--seg",
        action="store_true",
        default=False,
        help="connect camera trajectories with lines, should be used e.g. in NeRF synthetic",
    )
    parser.add_argument(
        "--n_cameras_for_procrustes", '-P',
        type=int,
        default=100000,
        help="use at most first x cameras for procrustes. Useful if trajectory starts to diverge",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="type in the batchsize for processing the point rays"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="the number of smaple for each light ray"
    )
    parser.add_argument(
        "--is_transform", action="store_true", default=False, help= "Doing transformation on point cloud and camera poses"
    )
    parser.add_argument(
        "--plot_neighbor", action="store_true", default=False, help= "Doing transformation on point cloud and camera poses"
    )
    parser.add_argument(
        "--img_idx", type=int, default=-1, help= "the image index"
    )
    parser.add_argument(
        "--near_n_cameras", type=int, default=1, help= "the number of chosed camera poses"
    )
    parser.add_argument(
        "--semantic_mask", type=str, default=None, help= "the number of chosed camera poses"
    )
    parser.add_argument(
        "--show_outside_color_only", action="store_true", default=False, help= "Displaying the inside colors"
    )
    parser.add_argument(
        "--process_point_cloud_for_mask", action="store_true", default=False, help= "Processing the point cloud for semantic mask"
    )
    args = parser.parse_args()

    main(args)
