from il.expert import Expert
from common.base_class import TentacleBaseEnv 
from common.loaders import RLEnvironmentConfig
from common.support import _action_to_ctrl,_get_tip_position,_normalize_position
from mujoco import viewer
import numpy as np
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
# -------------------------
# CONFIG + ENV
# -------------------------
config = RLEnvironmentConfig()

env = TentacleBaseEnv(config=config, render_mode="human")
env._base_reset()
expert = Expert(env)



obs_list, act_list, target = expert.rollout_episode()

env.target_position = target.copy()

print("Trajectory length:", len(obs_list))
# -------------------------
# VISUAL COMPARISON
# -------------------------

model = env.model
data = env.data

k = 0

with viewer.launch_passive(model, data) as v:

    while v.is_running() and k < 8:
        expert_action = act_list[k]
         # current observation
        obs= env._get_current_raw_obs()
        
        print("obs diff:", np.linalg.norm((obs) - obs_list[k]))
        # apply SAME pipeline as env
        ok = env._simulate(expert_action)

        if not ok:
            print("Unstable - reset")
            env.reset()
            k += 1
            continue

       

        k += 1
       
        v.sync()
    print(obs,obs_list[-1])

