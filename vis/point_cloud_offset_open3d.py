import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
import sys

def main(points, offsets):
    num_points = points.shape[0]
    # Set points for the original and offset point clouds
    pcd_points = o3d.geometry.PointCloud()
    pcd_offsets = o3d.geometry.PointCloud()

    pcd_points.points = o3d.utility.Vector3dVector(points)
    pcd_offsets.points = o3d.utility.Vector3dVector(offsets)

    # Optional: Assign colors to the point clouds for better visualization
    pcd_points.paint_uniform_color([0, 0, 1])  # Blue for original points
    pcd_offsets.paint_uniform_color([1, 0, 0])  # Red for offset points

    # Function to create line and color data
    def create_line_data(i):
        line = [i, num_points + i]
        color = [0, 1, 0]  # Green for lines
        return line, color

    # Create lines connecting original points to their offsets using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(create_line_data, range(num_points)))

    # Separate the results into lines and colors
    lines, colors = zip(*results)

    # Convert to Open3D format
    lines = list(lines)
    colors = list(colors)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((points, offsets)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize using Open3D
    o3d.visualization.draw_geometries([pcd_points, pcd_offsets, line_set])


if __name__=="__main__":
    xyz_npy_path = sys.argv[1]
    d_offset_npy_path = xyz_npy_path.replace("_xyz.","_d_offset.")
    points = np.load(xyz_npy_path)
    offset_points = points + np.load(d_offset_npy_path)
    main(points, offset_points)

