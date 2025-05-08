# Example 1: Basic usage
from mpc_planner import run_cem_planner

## Run with default parameters
#results = run_cem_planner()


# Example 2: Command line usage
# Run this from terminal:
# python3 mpc_planner.py --num_batch 2000 --w_pos 30.0 --w_rot 5.0 --save_data


# Example 3: Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=6,
    num_batch=1000,  # Use More samples for better optimization
    num_steps=8,     # Use More steps for longer planning horizon
    num_elite=0.05,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=0.05,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=3,   # Use More iterations for better convergence
    w_pos=20.0,      # weight on position error
    w_rot=3.0,       # weight on rotation error
    w_col=10.0,      # weight on collision avoidance
    
    # Initial configuration
    initial_qpos=[1.5, -1.8, 1.75, -1.25, -1.6, 0],
    
    # Target sequence
    target_names=["target_0", "target_1", "home"],
    
    # Visualization
    cam_distance=4,  # View 
    
    # Convergence thresholds
    position_threshold=0.04,  # Stricter position convergence Better for more complex tasks
    rotation_threshold=0.1,   # Stricter rotation convergence Better for more complex tasks
    
    # Save data
    save_data=True,
    data_dir='custom_data',
    stop_at_final_target=True #Stop at final target
)


# # Example 4: Running a quick test with fewer samples
# test_results = run_cem_planner(
#     num_batch=500,    # Fewer samples for faster computation
#     num_steps=12,     # Shorter horizon
#     maxiter_cem=1,    # Single iteration
#     position_threshold=0.06,  # Looser thresholds for testing
#     rotation_threshold=0.4,
#     save_data=False   # Don't save data during testing
# )


# # Example 5: Creating a custom target sequence with specific targets
# custom_results = run_cem_planner(
#     target_names=["target_0", "home", "target_1", "home"],
#     position_threshold=0.05,
#     rotation_threshold=0.25
# )