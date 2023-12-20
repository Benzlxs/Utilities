import open3d as o3d
import time
import numpy as np


VOXEL_SIZE=0.05

# original_pcd_path = "/Users/li325/projects/biomass_dataset/LiDAR/point_cloud_ply/20230322.ply"
# pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR/20230322/"
# pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR2024/11_15_23_EarlyBio"
pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR2024/11_14_23_EarlyBio"
# pcd_path_root = "/Users/li325/projects/biomass_dataset/LiDAR2024/12_8_23_Tests"
# pcd_path_root='/Users/li325/projects/biomass_dataset/LiDAR2024/12_18_23_Tests'

# all_names = ["2023_09_08_04_50_45Z", "2023_09_08_05_00_15Z", "2023_09_08_05_16_22Z", "2023_09_08_05_31_18Z"]
all_names = ["2023_09_07_05_51_26Z", "2023_09_07_05_55_49Z", "2023_09_07_06_02_10Z"]
# all_names=["10_12_8_23", "1_12_8_23", "2_12_8_23", "3_12_8_23",  "4_12_8_23", "5_12_8_23", "6_12_8_23","7_12_8_23", "8_12_8_23", "9_12_8_23"]
# all_names=["2023_10_11_09_05_19Z", "2023_10_11_09_08_34Z", "2023_10_11_09_10_57Z", "2023_10_11_09_18_55Z", "2023_10_11_09_21_57Z"]


for i in all_names:
    print(f"process {i}")
    # original_pcd_path = pcd_path_root + "/{}/global_wildcat_velodyne.ply".format(i)
    # pcd = o3d.io.read_point_cloud(original_pcd_path)

    # t = time.time()

    # pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # print('downsampling point cloud is  done in {}s'.format(time.time()-t))
    # print('nb points: {}'.format(np.asarray(pcd.points).shape[0]))
    # out_name = original_pcd_path.replace(".ply", "_downsample.ply")
    # o3d.io.write_point_cloud(out_name, pcd)

    # original_pcd_path = pcd_path_root + "/{}/paintcloud_coloured.ply".format(i)
    original_pcd_path = pcd_path_root + "/{}/global_wildcat_velodyne.ply".format(i)
    pcd = o3d.io.read_point_cloud(original_pcd_path)

    t = time.time()
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.)
    pcd = pcd.select_by_index(ind)
    print('downsampling colourized point cloud is done in {}s'.format(time.time()-t))
    print('nb points: {}'.format(np.asarray(pcd.points).shape[0]))
    out_name = original_pcd_path.replace(".ply", "_downsample.ply")
    o3d.io.write_point_cloud(out_name, pcd)



