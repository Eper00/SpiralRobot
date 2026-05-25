from common.base_class import TentacleBaseEnv
from common.support import _get_tip_position, load_config
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import time
class TentacleTargetFollowingExpert(TentacleBaseEnv):
    def __init__(
        self,
        config: Dict[str, Any]=None,
        render_mode: str = None,
    ):
        super().__init__(config, render_mode)
        self.action = np.zeros(self.actuator_dim)
    def coil_policy(self,direction):
       
        if direction == 1:
            self.action[0] = -1.0
            self.action[1] = 1
        else:
            self.action[0] = 1
            self.action[1] = -1.0
        return np.clip(self.action, -1, 1)
    def uncoil_policy(self,direction,perturbation_input):
        if direction == 1:
            self.action[0] += perturbation_input
            self.action[1] = -0.1
        else:
            self.action[0] = -0.1
            self.action[1] += perturbation_input
        return np.clip(self.action, -1, 1)
    def relax_policy(self):
        self.action = self.action
        return self.action
    def one_rollout(self):

            self._base_reset()
            direction = np.random.choice([0,1])
            perturbation_input=np.random.uniform(0.001, 0.005)
            obs_list = []
            act_list = []
            self.render_mode = "human"
            coil_len = int(self._max_episode_steps * 0.1)
            uncoil_len=int(self._max_episode_steps * 0.15)
            for t in range(self._max_episode_steps):
                time.sleep(0.01)
                # ---- phase policy only ----
                if t < coil_len:
                    self.action = self.coil_policy(direction)
                elif  t> coil_len and t < coil_len+uncoil_len:
                     self.action = self.uncoil_policy(direction, perturbation_input)
                else:
                    self.action = self.relax_policy()
                self._base_step(self.action)
                self.render()
                
            

            final_tip = _get_tip_position(self.model, self.data)[1:3]

            # IMPORTANT: hindsight goal
            target = final_tip.copy()

            return (
                np.array(obs_list, dtype=np.float32),
                np.array(act_list, dtype=np.float32),
                target)