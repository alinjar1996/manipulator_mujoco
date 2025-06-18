import mujoco
import numpy as np
import open3d as o3d
import os

# Load MuJoCo model
xml_path = "./sampling_based_planner/ur5e_hande_mjx/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Camera setup
cam_name = "camera1"
cam_id = model.camera(cam_name).id
height, width = 480, 640

renderer = mujoco.Renderer(model, height=height, width=width)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera=cam_name)

# Render RGB and depth
rgb = renderer.render()
renderer.enable_depth_rendering()
depth = renderer.render()

# Camera intrinsics
fovy = model.cam_fovy[cam_id]
f = height / (2 * np.tan(np.deg2rad(fovy / 2)))
cx, cy = width / 2, height / 2

# Camera pose
cam_pos = data.cam_xpos[cam_id]
cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

# Project to 3D
i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
z = depth
x = (i - cx) * z / f
y = (j - cy) * z / f
points_cam = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)

# Transform to world frame
points_world = (cam_mat @ points_cam.T).T + cam_pos
points = points_world

# Flatten RGB and filter valid points
rgb_flat = rgb.reshape(-1, 3)
valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
valid_mask &= np.abs(points[:, 2]) < 10.0

points = points[valid_mask]
colors = rgb_flat[valid_mask].astype(np.uint8)

# Combine into full array with unpacked r g b
pcd_data = np.hstack((points, colors))

# Convert to Open3D PointCloud
pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(points)
pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize to [0,1]

# Visualize
o3d.visualization.draw_geometries([pcd_o3d],
                                  zoom=0.7,
                                  front=[0.0, 0.0, 1.0],
                                  lookat=[0.0, 0.0, 0.0],
                                  up=[0.0, 1.0, 0.0])

# Save PCD file
os.makedirs("pcd_data", exist_ok=True)
output_path = "pcd_data/output_pointcloud_unpacked_rgb.pcd"
with open(output_path, "w") as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z r g b\n")
    f.write("SIZE 4 4 4 1 1 1\n")
    f.write("TYPE F F F U U U\n")
    f.write("COUNT 1 1 1 1 1 1\n")
    f.write(f"WIDTH {pcd_data.shape[0]}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {pcd_data.shape[0]}\n")
    f.write("DATA ascii\n")
    for row in pcd_data:
        f.write(f"{row[0]} {row[1]} {row[2]} {int(row[3])} {int(row[4])} {int(row[5])}\n")

print(f"Saved unpacked RGB point cloud to: {output_path}")
