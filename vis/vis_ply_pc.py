
#ref: https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/visualizer/open3d_vis.py
import copy
import argparse
import os
import numpy as np

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lidarfile", default="0000000000.bin", help="Kitti lidar file"
    )#lidarxyzintensityright000150.bin
    parser.add_argument(
        "--points_size", default=1, help="point size"
    )#lidarxyzintensityright000150.bin

    flags = parser.parse_args()
    points_size = flags.points_size
    pcd = o3d.io.read_point_cloud(flags.lidarfile)

    """Load and parse a velodyne binary file."""
    # points = np.fromfile(lidarfile, dtype=np.float32) # vector points
    # points = np.load(lidarfile) # vector points
    points = np.asarray(pcd.points)
    z0 = points[:,2].min()
    z1 = points[:,2].max()
    x0 = points[:,0].min()
    x1 = points[:,0].max()
    y0 = points[:,1].min()
    y1 = points[:,1].max()
    #@ points = points[points[:,2] > z0 + 1.6, :]
    #@ points = points[points[:,2] < z1 - 6.6, :]
    #@ points = points[points[:,0] > x0 + 10.0, :]
    #@ points = points[points[:,0] < x1 - 20., :]
    #@ points = points[points[:,1] > y0 + 45.0, :]
    #@ points = points[points[:,1] < y1 - 45.0, :]

    # points = points[points[:,0] > -15.0, :]  # left``
    # points = points[points[:,0] < 25., :]
    # points = points[points[:,1] > -65.0, :] # ditch direction
    # points = points[points[:,1] < 15.0, :] #
    # points = points[points[:,2] > -0.9, :]
    # points = points[points[:,2] <  2.6, :]

    # pcd = geometry.PointCloud()
    # pcd = pcd.voxel_down_sample(voxel_size=0.4)
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    vis =  o3d.visualization.Visualizer()
    vis.create_window()

    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    vis.get_render_option().point_size = points_size  # set points size
    vis.add_geometry(pcd)
    vis.run()
    # o3d.visualization.draw_geometries([pcd])

