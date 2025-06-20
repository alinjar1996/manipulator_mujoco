import numpy as np
import os
import mujoco
from collision_free_ik.mink.lie import SE3, SO3
from collision_free_ik.mink.configuration import Configuration
from collision_free_ik.mink.limits.configuration_limit import ConfigurationLimit
from collision_free_ik.mink.solve_ik import solve_ik
from collision_free_ik.mink.tasks.frame_task import FrameTask

# Load MuJoCo model
# xml_path = os.path.join(os.path.dirname(__file__), "examples/universal_robots_ur5e/scene.xml")
xml_path = os.path.join(os.path.dirname(__file__), "./sampling_based_planner/ur5e_hande_mjx/scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)



current_robot_configuration = np.zeros(model.nq)

# Define current joint state
current_joint_positions = np.array([0.0, -0.5, 0.3, 0.0, 1.0, 0.0])

# current_robot_configuration[:6] = current_joint_positions

configuration = Configuration(model, current_joint_positions)

# Define target pose for end-effector
target_position = np.array([-0.3, -0.3, 0.5])  # [x, y, z] in meters

# Create target orientation (identity quaternion = no rotation)
target_quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]
rotation = SO3(wxyz=target_quaternion)

# Create SE3 transform
target_pose = SE3.from_rotation_and_translation(
    rotation=rotation,
    translation=target_position
)

# eef_rot = mjx_data.xquat[self.hande_id]	
# eef_pos = mjx_data.site_xpos[self.tcp_id]

# Create FrameTask
frame_task = FrameTask(
    frame_name="tcp",          # Replace with your end-effector body/site name
    frame_type="site",           # "body", "geom", or "site"
    position_cost=1.0,           # Weight for position control
    orientation_cost=0.5,        # Weight for orientation control
    gain=1.0,
    lm_damping=0.0
)
frame_task.set_target(target_pose)

# Define joint limits
limits = ConfigurationLimit(model)

# Solve IK
new_q = solve_ik(
    configuration=configuration,
    tasks=[frame_task],          # Pass as list
    dt=10.0,
    solver='daqp',
    damping=1e-4,
    safety_break=True,
    limits=[limits]              # Must be passed as list
)

print("New joint velocities:", new_q)