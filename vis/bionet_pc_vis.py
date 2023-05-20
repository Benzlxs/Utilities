import copy

import numpy as np

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')



import argparse
import os
from spatialmath import SE3, SO3
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidarfile", type=str, default="0000000000.bin", help="Kitti lidar file",)
    parser.add_argument("--rx", type=int, default=0, help='',)
    parser.add_argument("--ry", type=int, default=0, help='',)
    parser.add_argument("--rz", type=int, default=0, help='', )#lidarxyzintensityright000150.bin
    parser.add_argument("--scale", type=float, default=1., help="")
    parser.add_argument("--file_list", type=str, default="/Users/li325/projects/biomass_dataset/vis_bionet/yanco_2019_test_list_early.txt", help="")
    parser.add_argument("--root_dir", type=str, default="/Users/li325/projects/biomass_dataset/vis_bionet/BioNet_Dataset", help="")
    parser.add_argument("--save_folder", type=str, default="/Users/li325/projects/biomass_dataset/vis_bionet/BioNet_Dataset/imgs", help="")
    flags = parser.parse_args()

    try:
        os.makedirs(flags.save_folder)
    except OSError:
        pass

    file_txt_list = open(flags.file_list, "r").readlines()

    num_file = len(file_txt_list)

    for idx in range(num_file):
        sample_pcd_data = o3d.data.PCDPointCloud()
        fn =  file_txt_list[idx][0:-1].split(' ')[0]
        fn2 = file_txt_list[idx][0:-1].split(' ')[1]
        pcd = o3d.io.read_point_cloud(flags.root_dir + fn+' '+fn2, 'pcd')
        points = np.asarray(pcd.points)

        m_x = SO3.Ry(flags.ry, 'deg')
        m_y = SO3.Rx(flags.rx, 'deg')
        m_z = SO3.Rz(flags.rz, 'deg')
        points = np.matmul(points, np.asarray(m_x))
        points = np.matmul(points, np.asarray(m_y))
        points = np.matmul(points, np.asarray(m_z))
        points = points * flags.scale

        center = points.mean(0)

        gt = file_txt_list[idx][0:-1].split(' ')[2]

        pcd_2 = geometry.PointCloud()

        pcd_2.points = o3d.utility.Vector3dVector(points[:, :3])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=center.tolist())  # create coordinate frame
        # vis.add_geometry(mesh_frame)

        box3d = geometry.OrientedBoundingBox(center.tolist(), geometry.get_rotation_matrix_from_xyz(np.zeros(3)) , [45, 45, 45])
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color([1,1,0])
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        vis.add_geometry(pcd_2)
        vis.run()
        name_img = flags.save_folder + "/" + flags.file_list.split('/')[-1][:-4] + "_%d_"%(idx) + str(int(float(gt))) + ".png"
        vis.capture_screen_image(name_img)
        vis.destroy_window()
    # """Load and parse a velodyne binary file."""
    # o3d.visualization.draw_geometries([pcd],)
    # o3d.visualization.capture_screen_image('test.png')

