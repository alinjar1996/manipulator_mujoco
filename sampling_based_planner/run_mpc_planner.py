
from mpc_planner import run_cem_planner



#Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=6,
    num_batch=1000,  # Use More samples for better optimization
    num_steps=16,     # Use More steps for longer planning horizon
    num_elite=0.05,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=0.05,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=3,   # Use More iterations for better convergence
    w_pos=20.0,      # weight on position error
    w_rot=3.0,       # weight on rotation error
    w_col=80.0,      # weight on collision avoidance
    
    # Initial configuration
    initial_qpos=[1.5, -1.8, 1.75, -1.25, -1.6, 0],
    
    # Target sequence
    target_names=["target_0", "target_1", "target_2", "home"],
    
    # Visualization
    cam_distance=4,  # View 
    
    # Convergence thresholds
    position_threshold=0.05,  # Stricter position convergence Better for more complex tasks
    rotation_threshold=0.1,   # Stricter rotation convergence Better for more complex tasks
    
    # Save data
    save_data=True,
    data_dir='custom_data',
    stop_at_final_target=True #Stop at final target
)


