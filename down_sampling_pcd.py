import open3d as o3d
import time
import numpy as np


VOXEL_SIZE=0.1

# original_pcd_path = "/Users/li325/projects/biomass_dataset/LiDAR/point_cloud_ply/20230322.ply"
# pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR/20230322/"
# pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR2024/11_15_23_EarlyBio"
pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR2024/11_14_23_EarlyBio"

# all_names = ["2023_09_08_04_50_45Z", "2023_09_08_05_00_15Z", "2023_09_08_05_16_22Z", "2023_09_08_05_31_18Z"]
all_names = ["2023_09_07_05_51_26Z", "2023_09_07_05_55_49Z", "2023_09_07_06_02_10Z"]
for i in all_names:
    print(f"process {i}")
    original_pcd_path = pcd_path_root + "/{}/global_wildcat_velodyne.ply".format(i)
    pcd = o3d.io.read_point_cloud(original_pcd_path)

    t = time.time()

    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print('downsampling done in {}s'.format(time.time()-t))
    print('nb points: {}'.format(np.asarray(pcd.points).shape[0]))
    out_name = original_pcd_path.replace(".ply", "_downsample.ply")
    o3d.io.write_point_cloud(out_name, pcd)




