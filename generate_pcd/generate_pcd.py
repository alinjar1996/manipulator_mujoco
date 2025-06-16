import mujoco
import numpy as np
import open3d as o3d
import os

print(os.getcwd())
# Paths
xml_path = "./sampling_based_planner/ur5e_hande_mjx/scene.xml"  # Update this to your XML file path

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set camera parameters (render from camera1)
cam_name = "camera1"
cam_id = model.camera(cam_name).id
height, width = 480, 640

# Create a renderer
renderer = mujoco.Renderer(model, height=height, width=width)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera=cam_name)

# Render RGB and depth images
rgb = renderer.render()
renderer.enable_depth_rendering()
depth = renderer.render()

# Alternative method if the above doesn't work:
# You can also try using mujoco.mj_render with depth buffer
# depth_buffer = np.zeros((height, width), dtype=np.float32)
# mujoco.mjr_render(mujoco.mjVIS_DEPTH, model, data, renderer.con)
# mujoco.mjr_readPixels(depth_buffer, None, mujoco.mjrRect(0, 0, width, height), renderer.con)
# depth = depth_buffer

# Get camera intrinsics
fovy = model.cam_fovy[cam_id] if hasattr(model, 'cam_fovy') else model.vis.global_.fovy

print("fovy", fovy)

aspect = width / height
f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
cx, cy = width / 2, height / 2

# Get camera pose in world coordinates
cam_pos = data.cam_xpos[cam_id]
cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

# Build point cloud in camera coordinates
i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
z = depth

# print("np.max(z)", np.max(z))
print("z", z.shape)
print("z_max", np.max(z))
print("z_min", np.min(z))

# # Handle depth values (MuJoCo depth might need conversion)
# # Sometimes depth is in range [0,1] and needs to be converted to actual distances
# if np.max(z) <= 1.0:
#     # If depth is normalized, you might need to scale it
#     # This depends on your scene setup - adjust znear and zfar accordingly
#     znear = 0.01  # Adjust based on your camera setup
#     zfar = 10.0   # Adjust based on your camera setup
#     z = znear / (1 - z * (1 - znear/zfar))

# Generate points in camera coordinates
x = (i - cx) * z / f
y = (j - cy) * z / f
# Note: MuJoCo camera frame: x=right, y=up, z=backward (towards camera)
points_cam = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)

# Transform points from camera coordinates to world coordinates
# Apply rotation and translation
points_world = (cam_mat @ points_cam.T).T + cam_pos
points = points_world

# Remove invalid points (NaN, inf, or very far points)
valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
valid_mask = valid_mask & (np.abs(points[:, 2]) < 10.0)  # Remove points too far away
points = points[valid_mask]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Assign colors from RGB image
if rgb is not None:
    rgb_flat = rgb.reshape(-1, 3) / 255.0
    rgb_valid = rgb_flat[valid_mask]
    if len(rgb_valid) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(rgb_valid)

# # Optional: Remove statistical outliers
# pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Optional: Set a better viewpoint for visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

# Set camera viewpoint for better 3D visualization
ctr = vis.get_view_control()
ctr.set_front([0, 0, 1])  # Look towards negative z
ctr.set_lookat([0, 0, 0])  # Look at origin
ctr.set_up([0, 1, 0])     # Y-axis points up
ctr.set_zoom(1.0)

vis.run()
vis.destroy_window()

# Optional: Save the point cloud
#o3d.io.write_point_cloud("output_pointcloud_.pcd", pcd)

o3d.io.write_point_cloud("output_pointcloud.pcd", pcd, write_ascii=True)
