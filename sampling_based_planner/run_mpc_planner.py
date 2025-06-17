
import numpy as np
from mpc_planner import run_cem_planner
import os
import mujoco


inference = True
target_names=["target_1","target_2", "target_0", "home"]

#Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=6,
    num_batch=1000,  # Use More samples for better optimization
    num_steps=20,     # Use More steps for longer planning horizon
    num_elite=0.05,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=0.05,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=1,      # CEM iterations: Use More iterations for better convergence     
    maxiter_projection=5,   # Projection Filter iterations: Use More iterations for better Filtering
    w_pos=20.0,      # weight on position error
    w_rot=3.0,       # weight on rotation error
    w_col=80.0,      # weight on collision avoidance
    
    #Shower parameters
    show_viewer=True,
    show_contact_points=True,
    
    # Initial configuration
    initial_qpos=[1.5, -1.8, 1.75, -1.25, -1.6, 0],
    
    # Target sequence
    target_names=target_names,

    #Joint limits
    max_joint_pos= 180.0*np.pi/180.0,
    max_joint_vel= 1.0,
    max_joint_acc= 2.0,
    max_joint_jerk= 4.0,
    
    # Visualization
    cam_distance=4,  # View 
    
    # Convergence thresholds
    position_threshold=0.07,  # Stricter position convergence Better for more complex tasks
    rotation_threshold=0.1,   # Stricter rotation convergence Better for more complex tasks
    
    # Save Motion Related data
    save_data=True,
    data_dir=f'custom_data_{target_names[:-1]}_inference_{inference}',
    
    # Save Point Cloud data
    generate_pcd=True,
    accumulate_pcd=True,
    pcd_interval=10, # Save point cloud every 10 steps
    cam_name="camera1",

    #Inference MLP for lamda_init and s_init
    inference=inference 
,
    #rnn = 'GRU',
    rnn = 'LSTM',

    #Stop at final target
    stop_at_final_target=False 
)


