from il.expert import TentacleTargetFollowingExpert
config_path="/home/tomi/SpiralRobot/configs/default_rl_training.yaml"
expert=TentacleTargetFollowingExpert(config_path)
expert.one_rollout()
'''
import mujoco
import mujoco.viewer
import time

xml_file = "/home/tomi/SpiralRobot/assets/simulation/tentacle_2D.xml"

model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)   # szimuláció egy lépése
        viewer.sync()                 # képernyő frissítése
'''
