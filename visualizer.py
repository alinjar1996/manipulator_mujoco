import os
import numpy as np
import time


import mujoco
from mujoco import viewer


class Visualizer():
    def __init__(self, ctrl: bool=False, traj: bool=False):
        self.init_joint_state = np.array([0, 0, 0, 0, 0, 0])

        if ctrl:
            model_path = f"{os.path.dirname(__file__)}/sampling_based_planner/franka_emika_panda/franka_emika_panda_customized_scene.xml"
        else:
            model_path = f"{os.path.dirname(__file__)}/sampling_based_planner/franka_emika_panda/franka_emika_panda_customized_scene.xml"

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.05
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = self.init_joint_state

        if traj:
            file_path = f"{os.path.dirname(__file__)}/sampling_based_planner/data/best_vels.csv"
            self.thetadot = np.genfromtxt(file_path, delimiter=',')


        


    def view_model(self):
        viewer.launch(self.model, self.data)

    def view_traj_mujoco(self):
        with viewer.launch_passive(self.model, self.data) as viewer_:
            viewer_.cam.distance = 4
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer_.opt.sitegroup[:] = False  
            viewer_.opt.sitegroup[1] = True 

            i = 0
            while viewer_.is_running():
                step_start = time.time()
                self.data.qvel[:6] = self.thetadot[i]

                mujoco.mj_step(self.model, self.data)
                viewer_.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)    

                if i < self.thetadot.shape[0]-1:
                    i+=1
                else:
                    self.data.qvel[:6] = np.zeros(6)
                    self.data.qpos[:6] = self.init_joint_state
                    i=0




def main():
    viz = Visualizer(ctrl=False, traj=True)
    # viz.view_model()
    viz.view_traj_mujoco()


if __name__=="__main__":
    main()