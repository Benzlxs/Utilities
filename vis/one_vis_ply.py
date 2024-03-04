import open3d as o3d
import time
import numpy as np
import sys

if __name__=="__main__":
    #  original_pcd_path = "/Users/li325/projects/biomass_dataset/LiDAR/point_cloud_ply/20230322.ply"
    pcd_dir = sys.argv[1]

    pcd = o3d.io.read_point_cloud(pcd_dir)

    # pcd_pc = np.asarray(pcd.points)
    # pcd_colors = np.asarray(pcd.colors)

    # median_pc = np.median(pcd_pc, axis=0)
    # mask = (pcd_pc[:,0] > median_pc[0])
    # mask_2 = (pcd_pc[:,1] > median_pc[1])

    # pcd_pc = pcd_pc[mask & mask_2]
    # pcd_colors = pcd_colors[mask & mask_2]

    # pcd.points = o3d.utility.Vector3dVector(pcd_pc)
    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    o3d.visualization.draw_geometries([pcd])
