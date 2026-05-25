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
    def coil_policy(self,direction,coiling_bias=0.5):
       
        if direction == 1:
            self.action[0] = -1.0
            self.action[1] = coiling_bias
        else:
            self.action[0] = coiling_bias
            self.action[1] = -1.0
        return np.clip(self.action, -1, 1)
    def uncoil_policy(self,direction,uncoiling_bias=0.5,coiling_bias=0.5):
        if direction == 1:
            self.action[0] = uncoiling_bias
            self.action[1] = coiling_bias
        else:
            self.action[0] = coiling_bias
            self.action[1] = uncoiling_bias
        return np.clip(self.action, -0.8, 0.8)

    def one_rollout(self):

            self._base_reset()
            direction = np.random.choice([0,1])
            curling_bias=np.random.uniform(0, 1)
            uncoil_bias=np.random.uniform(curling_bias/2, 1)
            obs_list = []
            act_list = []
            self.render_mode = "human"
            coil_len = int(self._max_episode_steps * 0.1)

            for t in range(self._max_episode_steps):
                # ---- phase policy only ----
                if t < coil_len:
                    self.action = self.coil_policy(direction, curling_bias)
                else:
                     self.action = self.uncoil_policy(direction, uncoil_bias,curling_bias)
                self._base_step(self.action)
                self.render()
                
            

            final_tip = _get_tip_position(self.model, self.data)[1:3]

            # IMPORTANT: hindsight goal
            target = final_tip.copy()

            return (
                np.array(obs_list, dtype=np.float32),
                np.array(act_list, dtype=np.float32),
                target)