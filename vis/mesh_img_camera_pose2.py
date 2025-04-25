import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import json
root_path="/Users/li325/projects/public_dataset/active_percerption/1bf5f3d7a131657ca01094d9087d1cf485aee90d5d036a792815eaa7457c6495"
# Load the mesh
mesh = o3d.io.read_triangle_mesh(root_path + "/mesh.ply")
mesh.compute_vertex_normals()

# Load transforms.json
with open(root_path + "/transforms.json", 'r') as f:
    meta = json.load(f)

frames = meta["frames"]

# ---------- Make frustums ----------
def create_camera_frustum(transform_matrix, scale=0.2):
    cam_to_world = np.array(transform_matrix) # c2w

    # Blender -Z forward
    frustum = np.array([
        [0, 0, 0],        # Camera center
        [-0.5, -0.5, -1],
        [0.5, -0.5, -1],
        [0.5, 0.5, -1],
        [-0.5, 0.5, -1]
    ]) * scale

    frustum_world = (cam_to_world[:3, :3] @ frustum.T + cam_to_world[:3, 3:4]).T

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frustum_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([1, 0, 0])

    cam_center = frustum_world[0]
    return line_set, cam_center

# ---------- Build geometries ----------
geometries = [mesh]
label_data = []  # For camera centers and image names

for idx, frame in enumerate(frames):
    frustum, center = create_camera_frustum(frame["transform_matrix"], scale=0.1)
    geometries.append(frustum)
    label_data.append((center, frame["file_path"]))

# ---------- GUI window ----------
class FrustumVisualizer:
    def __init__(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Camera Frustums", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Add mesh and frustums
        for g in geometries:
            name = f"geom_{id(g)}"
            # self.scene.scene.add_geometry(name, g, rendering.MaterialRecord())
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [0.7, 0.7, 0.7, 1.0]  # light gray, opaque
            material.base_metallic = 0.0
            material.base_roughness = 0.6
            material.transmission = 0.0
            self.scene.scene.add_geometry(name, g, material)

        # Add labels at camera centers
        for center, label in label_data:
            # text = gui.Label(label)
            # text.frame = gui.Rect(0, 0, 100, 20)
            # self.scene.add_3d_label(center, text)
            self.scene.add_3d_label(np.array(center, dtype=np.float32), label)

        bbox = geometries[0].get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bbox, bbox.get_center())

    def run(self):
        gui.Application.instance.run()

# Run
FrustumVisualizer().run()
