import numpy as np
from mjx_planner import cem_planner
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
import os
from mujoco import viewer
import matplotlib.pyplot as plt
from quat_math import rotation_quaternion, quaternion_multiply, quaternion_distance
import argparse
import signal
import sys

# Global variables to store data for saving on exit
global_data = {
    'xi_samples_list': [],
    'xi_filtered_list': [],
    'state_terms_list': [],
    'cost_list': [],
    'thetadot_list': [],
    'theta_list': [],
    'cost_g_list': [],
    'cost_r_list': [],
    'cost_c_list': [],
    'data_dir': None,
    'save_data': False
}

def save_data_on_exit(signum=None, frame=None):
    """Save all collected data when program is terminated"""
    if not global_data['save_data']:
        print("Data saving not enabled. Exiting without saving.")
        sys.exit(0)
        
    try:
        print("\nReceived termination signal. Saving data before exit...")
        data_dir = global_data['data_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save NPZ data
        npz_path = os.path.join(data_dir, 'sample_dataset_final.npz')
        np.savez(
            npz_path,
            xi_samples=np.array(global_data['xi_samples_list']),
            xi_filtered=np.array(global_data['xi_filtered_list']),
            state_terms=np.array(global_data['state_terms_list'])
        )
        print(f"Saved NPZ data to {npz_path}")
        
        # Save CSV data
        csv_files = {
            'costs_final.csv': global_data['cost_list'],
            'thetadot_final.csv': global_data['thetadot_list'],
            'theta_final.csv': global_data['theta_list'],
            'cost_g_final.csv': global_data['cost_g_list'],
            'cost_r_final.csv': global_data['cost_r_list'],
            'cost_c_final.csv': global_data['cost_c_list']
        }
        
        for filename, data_to_save in csv_files.items():
            filepath = os.path.join(data_dir, filename)
            np.savetxt(filepath, data_to_save, delimiter=",")
            print(f"Saved {filename}")
            
        print("All data saved successfully!")
    except Exception as e:
        print(f"Error during emergency save: {e}")
    finally:
        sys.exit(0)

def run_cem_planner(
    # CEM planner parameters
    num_dof=None,
    num_batch=None,
    num_steps=None,
    maxiter_cem=None,
    maxiter_projection=None,
    w_pos=None,
    w_rot=None,
    w_col=None,
    num_elite=None,
    timestep=None,
    # Robot initial configuration
    initial_qpos=None,
    # Target configuration
    target_names=None,
    # Visualization options
    show_viewer=None,
    cam_distance=None,
    show_contact_points=None,
    # Convergence criteria
    position_threshold=None,
    rotation_threshold=None,
    # Save data
    save_data=None,
    data_dir=None,
    save_interval=None,  # No longer used but kept for compatibility
    # Motion control
    stop_at_final_target=None
):
    """
    Run CEM planner with configurable parameters
    
    Parameters:
    -----------
    num_dof : int
        Number of degrees of freedom for the robot
    num_batch : int
        Number of samples in each CEM iteration
    num_steps : int
        Number of steps in the planning horizon
    maxiter_cem : int
        Maximum number of CEM iterations
    w_pos : float
        Weight for position error in the cost function
    w_rot : float
        Weight for rotation error in the cost function
    w_col : float
        Weight for collision penalty in the cost function
    num_elite : float
        Fraction of samples to use as elite samples
    timestep : float
        Time step for simulation
    initial_qpos : array-like or None
        Initial joint positions, if None uses [1.5, -1.8, 1.75, -1.25, -1.6, 0]
    target_names : list of str or None
        Names of targets to reach in sequence, if None uses ["target_0", "target_1", "home"]
    show_viewer : bool
        Whether to show the MuJoCo viewer
    cam_distance : float
        Camera distance in the viewer
    show_contact_points : bool
        Whether to show contact points in the viewer
    position_threshold : float
        Threshold for position convergence
    rotation_threshold : float
        Threshold for rotation convergence
    save_data : bool
        Whether to save data to CSV files
    data_dir : str
        Directory to save data
    save_interval : int
        Parameter kept for compatibility but no longer used
    stop_at_final_target : bool
        Whether to stop at the final target or loop back to the first target
    """
    
    # Update global data dictionary
    global_data['save_data'] = save_data
    global_data['data_dir'] = data_dir
    
    # Setup signal handlers for proper termination
    signal.signal(signal.SIGINT, save_data_on_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, save_data_on_exit)  # termination signal
    
    if save_data:
        print(f"Data will be saved to '{data_dir}' when the program is terminated.")
        print("Use Ctrl+C to gracefully exit and save data.")
    
    # Initialize the CEM planner
    start_time = time.time()
    cem = cem_planner(
        num_dof=num_dof, 
        num_batch=num_batch, 
        num_steps=num_steps, 
        maxiter_cem=maxiter_cem,
        w_pos=w_pos,
        w_rot=w_rot,
        w_col=w_col,
        num_elite=num_elite,
        timestep=timestep,
        maxiter_projection=maxiter_projection
    )
    print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

    # Get model and data
    model = cem.model
    data = cem.data
    
    # Set initial joint positions
    data.qpos[:num_dof] = jnp.array(initial_qpos)
    mujoco.mj_forward(model, data)

    # Initialize CEM mean
    xi_mean = jnp.zeros(cem.nvar)
    
    # Get initial end-effector position and orientation
    init_position = data.site_xpos[model.site(name="tcp").id].copy()
    init_rotation = data.xquat[model.body(name="hande").id].copy()

    # First target for test computation
    target_pos = model.body(name=target_names[0]).pos
    target_rot = model.body(name=target_names[0]).quat

    # Warm-up computation
    start_time = time.time()
    _ = cem.compute_cem(xi_mean, data.qpos[:num_dof], data.qvel[:num_dof], data.qacc[:num_dof], target_pos, target_rot)
    print(f"Compute CEM: {round(time.time()-start_time, 2)}s")

    # Initialize variables for data collection - we'll use the global variables now
    thetadot = np.array([0] * num_dof)
    
    # Current target index
    target_idx = 0
    current_target = target_names[target_idx]
    
    # Run the control loop
    if show_viewer:
        with viewer.launch_passive(model, data) as viewer_:
            viewer_.cam.distance = cam_distance
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            
            while viewer_.is_running():
                # Time the step
                start_time = time.time()
                
                # Determine target position and orientation
                if current_target != "home":
                    target_pos = model.body(name=current_target).pos
                    target_rot = model.body(name=current_target).quat
                else:
                    target_pos = init_position
                    target_rot = init_rotation

                # Compute CEM control
                cost, best_cost_g, best_cost_r, best_cost_c, best_vels, best_traj, xi_mean, state_terms, xi_samples, xi_filtered = cem.compute_cem(
                    xi_mean, data.qpos[:num_dof], data.qvel[:num_dof], 
                    data.qacc[:num_dof], target_pos, target_rot
                )
                
                # Apply the control (use average of planned velocities)
                thetadot = np.mean(best_vels[1:num_steps-2], axis=0)
                data.qvel[:num_dof] = thetadot
                mujoco.mj_step(model, data)

                # Calculate costs
                current_cost_g = np.linalg.norm(data.site_xpos[cem.tcp_id] - target_pos)   
                current_cost_r = quaternion_distance(data.xquat[cem.hande_id], target_rot)  
                current_cost = np.round(cost, 2)
                
                # Print status
                print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(current_cost_g))}'
                      f' | Cost r: {"%.2f"%(float(current_cost_r))} | Cost c: {"%.2f"%(float(best_cost_c))} | Cost: {current_cost}')
                print(f'eef_quat: {data.xquat[cem.hande_id]}')
                print(f'target: {current_target}')
                
                # Store data in global variables
                global_data['cost_g_list'].append(best_cost_g)
                global_data['cost_r_list'].append(best_cost_r)
                global_data['cost_c_list'].append(best_cost_c)
                global_data['thetadot_list'].append(thetadot.copy())  # Use copy() to avoid reference issues
                global_data['theta_list'].append(data.qpos[:num_dof].copy())
                global_data['cost_list'].append(current_cost[-1] if isinstance(current_cost, np.ndarray) else current_cost)

                # Store data for MLP in projection filter
                global_data['xi_samples_list'].append(np.array(xi_samples))
                global_data['xi_filtered_list'].append(np.array(xi_filtered))
                global_data['state_terms_list'].append(np.array(state_terms))
                
                # Update viewer
                viewer_.sync()

                # Check if target is reached based on thresholds
                if current_cost_g < position_threshold and current_cost_r < rotation_threshold:
                    # Check if this was the last target
                    if target_idx == len(target_names) - 1:
                        if stop_at_final_target:
                            print(f"Reached final target: {current_target}. Stopping motion.")
                            # Hold position by setting velocities to zero
                            thetadot = np.zeros(num_dof)
                            data.qvel[:num_dof] = thetadot
                        else:
                            # Loop back to first target
                            target_idx = 0
                            current_target = target_names[target_idx]
                            print(f"Reached final target. Looping back to first target: {current_target}")
                    else:
                        # Move to next target
                        target_idx = target_idx + 1
                        current_target = target_names[target_idx]
                        print(f"Moving to next target: {current_target}")
                    

                # Sleep to maintain simulation speed
                time_until_next_step = model.opt.timestep - (time.time() - start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        # Non-visualization mode would go here if needed
        print("Running without visualization is not implemented yet.")
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CEM planner with configurable parameters')
    
    # CEM planner parameters
    parser.add_argument('--num_dof', type=int, default=6, help='Number of degrees of freedom')
    parser.add_argument('--num_batch', type=int, default=1000, help='Number of samples in each CEM iteration')
    parser.add_argument('--num_steps', type=int, default=16, help='Number of steps in the planning horizon')
    parser.add_argument('--maxiter_cem', type=int, default=1, help='Maximum number of CEM iterations')
    parser.add_argument('--maxiter_projection', type=int, default=10, help='Maximum number of projection iterations')
    parser.add_argument('--w_pos', type=float, default=20.0, help='Weight for position error')
    parser.add_argument('--w_rot', type=float, default=3.0, help='Weight for rotation error')
    parser.add_argument('--w_col', type=float, default=10.0, help='Weight for collision penalty')
    parser.add_argument('--num_elite', type=float, default=0.05, help='Fraction of samples to use as elite samples')
    parser.add_argument('--timestep', type=float, default=0.05, help='Time step for simulation')
    
    # Initial configuration
    parser.add_argument('--initial_qpos', type=float, nargs='+', default=None, help='Initial joint positions')
    
    # Visualization options
    parser.add_argument('--no_viewer', action='store_true', help='Disable MuJoCo viewer')
    parser.add_argument('--cam_distance', type=float, default=4, help='Camera distance in the viewer')
    parser.add_argument('--no_contact_points', action='store_true', help='Disable contact point visualization')
    
    # Convergence criteria
    parser.add_argument('--position_threshold', type=float, default=0.04, help='Threshold for position convergence')
    parser.add_argument('--rotation_threshold', type=float, default=0.3, help='Threshold for rotation convergence')
    
    # Target sequence
    parser.add_argument('--targets', type=str, nargs='+', default=None, help='Target names in sequence')
    
    # Save data
    parser.add_argument('--save_data', action='store_true', help='Save data to CSV files')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save data')
    parser.add_argument('--save_interval', type=int, default=10, help='Parameter kept for compatibility but no longer used')
    parser.add_argument('--continue_after_final', action='store_true', help='Continue movement after reaching final target (loop)')
    
    args = parser.parse_args()
    
    try:
        # Run CEM planner with parsed arguments
        run_cem_planner(
            num_dof=args.num_dof,
            num_batch=args.num_batch,
            num_steps=args.num_steps,
            maxiter_cem=args.maxiter_cem,
            maxiter_projection=args.maxiter_projection,
            w_pos=args.w_pos,
            w_rot=args.w_rot,
            w_col=args.w_col,
            num_elite=args.num_elite,
            timestep=args.timestep,
            initial_qpos=args.initial_qpos,
            target_names=args.targets,
            show_viewer=not args.no_viewer,
            cam_distance=args.cam_distance,
            show_contact_points=not args.no_contact_points,
            position_threshold=args.position_threshold,
            rotation_threshold=args.rotation_threshold,
            save_data=args.save_data,
            data_dir=args.data_dir,
            save_interval=args.save_interval,
            stop_at_final_target=not args.continue_after_final
        )
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass
    finally:
        # Just in case the signal handler didn't trigger
        if args.save_data:
            save_data_on_exit()