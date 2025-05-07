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

start_time = time.time()
cem =  cem_planner(
    num_dof=6, 
    num_batch=1000,
    num_steps=40,
    maxiter_cem=1,
    w_pos=5,
    w_rot=1.5,
    w_col=10,
    num_elite=0.1,
    timestep=0.01,
    init_pos = jnp.array([0, 0, 0, 0, 0, 0])
    )

print(f"Initialized CEM Planner: {round(time.time()-start_time, 2)}s")

model = cem.model
data = cem.data
data.qpos[:6] = jnp.array([0, 0, 0, 0, 0, 0])
mujoco.mj_forward(model, data)

xi_mean = jnp.zeros(cem.nvar)
target_pos = model.body(name="target_0").pos
target_rot = model.body(name="target_0").quat


start_time = time.time()
_ = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6], target_pos, target_rot)
print(f"Compute CEM: {round(time.time()-start_time, 2)}s")


thetadot = np.array([0]*6)

cost_g_list = list()
cost_list = list()
cost_r_list = list()
cost_c_list = list()
thetadot_list = list()
theta_list = list()

init_position = data.xpos[model.body(name="hand").id].copy()
#init_position = data.site_xpos[model.site(name="gripper").id].copy()
init_rotation = data.xquat[model.body(name="hand").id].copy()


target_idx = 0

target = "target_0"

with viewer.launch_passive(model, data) as viewer_:
    viewer_.cam.distance = 4
    viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # viewer_.opt.sitegroup[:] = False  
    # viewer_.opt.sitegroup[1] = True 

    while viewer_.is_running():
        start_time = time.time()
        if target != "home":
            target_pos = model.body(name=target).pos
            target_rot = model.body(name=target).quat
        else:
            target_pos = init_position
            target_rot = init_rotation

        if target == "target_1":
            model.body(name="target_0").pos = data.site_xpos[cem.tcp_id]
            model.body(name="target_0").quat = data.xquat[cem.hande_id]

        cost, best_cost_g, best_cost_c, best_vels, best_traj, xi_mean = cem.compute_cem(xi_mean, data.qpos[:6], data.qvel[:6], data.qacc[:6], target_pos, target_rot)
        thetadot = np.mean(best_vels[1:30], axis=0)
        # thetadot = best_vels[1]
        print("thetadot_cem", thetadot)

       #thetadot = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        data.qvel[:6] = thetadot

        mujoco.mj_step(model, data)

        cost_g = np.linalg.norm(data.site_xpos[cem.tcp_id] - target_pos)   
        cost_r = quaternion_distance(data.xquat[cem.hande_id], target_rot)  
        cost = np.round(cost, 2)
        print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms | Cost g: {"%.2f"%(float(cost_g))} | Cost r: {"%.2f"%(float(cost_r))} | Cost c: {"%.2f"%(float(best_cost_c))} | Cost: {cost}')
        viewer_.sync()

        if cost_g<0.01 and cost_r<0.3:
            if target == "target_0":
                target = "target_1"
            elif target == "target_1":
                model.body(name="target_0").pos = data.site_xpos[cem.tcp_id].copy()
                model.body(name="target_0").quat = data.xquat[cem.hande_id].copy()
                target = "home"

            # model.body(name="target").pos = target_positions[target_idx]
            # model.body(name="target").quat = target_rotations[target_idx]
            # if target_idx<len(target_positions)-1:
            #     target_idx += 1


        cost_g_list.append(cost_g)
        cost_r_list.append(cost_r)
        cost_c_list.append(best_cost_c)
        thetadot_list.append(thetadot)
        theta_list.append(data.qpos[:6].copy())
        cost_list.append(cost[-1])

        time_until_next_step = model.opt.timestep - (time.time() - start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)  

np.savetxt('data/costs.csv',cost_list, delimiter=",")
np.savetxt('data/thetadot.csv',thetadot_list, delimiter=",")
np.savetxt('data/theta.csv',theta_list, delimiter=",")
np.savetxt('data/cost_g.csv',cost_g_list, delimiter=",")
np.savetxt('data/cost_r.csv',cost_r_list, delimiter=",")
np.savetxt('data/cost_c.csv',cost_c_list, delimiter=",")