import open3d as o3d
import time
import numpy as np
import sys

if __name__=="__main__":
    #  original_pcd_path = "/Users/li325/projects/biomass_dataset/LiDAR/point_cloud_ply/20230322.ply"
    pcd_dir = sys.argv[1]

    points = np.load(pcd_dir)
    pcd_points = o3d.geometry.PointCloud()

    pcd_points.points = o3d.utility.Vector3dVector(points)

    # Optional: Assign colors to the point clouds for better visualization
    pcd_points.paint_uniform_color([0, 0, 1])  # Blue for original points


    o3d.visualization.draw_geometries([pcd_points])
