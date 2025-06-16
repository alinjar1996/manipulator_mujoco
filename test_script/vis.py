import mujoco
from mujoco import viewer
import time

model_path = "./sampling_based_planner/ur5e_hande_mjx/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

cam_name = "camera1"
cam_id = model.camera(cam_name).id

# Optionally set camera pose (if you want to override)
data.cam_xpos[cam_id] = data.cam_xpos[cam_id]
data.cam_xmat[cam_id] = data.cam_xmat[cam_id]

# # Optionally set camera FOV
# model.cam_fovy[cam_id] = 0.0

# print(f"Using camera '{cam_name}' with FOV={model.cam_fovy[cam_id]} degrees")
print(f"Using camera '{cam_name}' ")

with viewer.launch_passive(model, data) as v:
    print("Viewer launched. Close the window to exit.")
    while v.is_running():
        mujoco.mj_step(model, data)  # step simulation (optional)
        # NO v.render() call here
        time.sleep(0.01)
