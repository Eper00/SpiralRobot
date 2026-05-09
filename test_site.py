import mujoco
from mujoco import viewer
import common.parameters as parameters
import numpy as np
from common.support import compute_spring_energy,_get_tip_position,_read_dataset,visualize_dataset

model = mujoco.MjModel.from_xml_path(parameters.path_xml_tentacle)
data = mujoco.MjData(model)
states,actions=_read_dataset("/home/tomi/karcsi/demonstration_dataset.npz")
visualize_dataset(states,actions,1)
"""
data.qvel[:] = 0
data.ctrl[:] = 0.19 
with viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            E = compute_spring_energy(model, data)

            print(_get_tip_position(model,data))
               

            viewer.sync()


"""