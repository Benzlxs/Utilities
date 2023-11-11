import open3d as o3d
import time
import numpy as np
import os
from pathlib import Path

save_folder = Path("./")

weather_station_file = "/Users/li325/projects/Dynamic_NeRF/global_registration_dataset/registration_pipeline/07/scene_dense_ds_global.ply"
pcd = o3d.io.read_point_cloud(weather_station_file)

num_pts = 3

vis = o3d.visualization.VisualizerWithEditing()
print('[shift + left click]')
print('[shift + right click] to unselect')
print('q to close the window')
vis.create_window(window_name='Please select {} points'.format(num_pts), width=1400, height=1800, left=0, top=0, visible=True)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

vertices_indexes = vis.get_picked_points()
selected_points = []
for i in range(num_pts):
    point = pcd.points[vertices_indexes[i]]
    selected_points.append(point)
print(selected_points)
txt_save_dir = save_folder.joinpath("{}.txt".format("key_points"))
np.savetxt(str(txt_save_dir), selected_points)


