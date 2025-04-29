import mujoco
import mujoco.viewer
import os

import numpy as np
import time

# Path to your MJX XML file
xml_path = "sampling_based_planner/panda_mjx/singlearm_panda_tray.xml"

# Load and compile the XML into a MuJoCo model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# # Set a custom initial camera distance (zoom out)
# def set_camera(viewer):
#     viewer.cam.distance = 3.0
#     viewer.cam.azimuth = 0
#     viewer.cam.elevation = -90.0
#     viewer.cam.lookat[:] = np.array([0, 0, 0.5])

# # Launch the viewer with camera settings and timeout
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     set_camera(viewer)
#     print("Viewer launched. It will close automatically in 5 seconds.")
    
#     start_time = time.time()
#     while viewer.is_running() and (time.time() - start_time) < 5:  # 5 seconds
#         mujoco.mj_step(model, data)
#         viewer.sync()

print("Expected qpos size:", model.nq)
for i in range(model.njnt):
    print(f"Joint {i}: name={model.joint(i).name}, type={model.joint(i).type}, dof={model.dof_Madr[i+1] - model.dof_Madr[i]}")

print("Available bodies:", [model.body(i).name for i in range(model.nbody)])

# Path to your MJX XML file
