from matplotlib import pyplot as plt

from common.base_class import TentacleBaseEnv
from common.support import _get_sites_positions, load_config
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
    def workspace_edge_polcy(self,direction,coiling_bias):
       
        if direction == 1:
            self.action[0] = -0.9
            self.action[1] = coiling_bias
        else:
            self.action[0] = coiling_bias
            self.action[1] = -0.9
        return np.clip(self.action, -1, 1)
    def workspace_center_polcy(self,direction,base,small_gain):
        if direction == 1:
            self.action[0] = base
            self.action[1] = base+small_gain
        else:
            self.action[0] = base+small_gain
            self.action[1] = base
        return np.clip(self.action, -0.2, 0.2)

    def one_rollout(self):
        self._base_reset()

        direction = np.random.choice([0,1])
        policy_switch = np.random.choice([0,1])

        coiling_bias = np.random.uniform(-1, -0.5)
        base = np.random.uniform(-0.7,0.7)
        small_gain = np.random.uniform(-0.1,0.1)

        for t in range(self._max_episode_steps):
            if policy_switch == 0:
                self.workspace_edge_polcy(direction, coiling_bias)
                policy_name = "edge"
            else:
                self.workspace_center_polcy(direction, base, small_gain)
                policy_name = "center"

            self._base_step(self.action)

        final_tip = _get_sites_positions(
            self.model,
            self.data,
            self.marker_names[-1]
        )[0][1:]

        target = final_tip.copy()

        return target, policy_name
                
    def many_rollout(self, amount_of_demonstration):

        results = [self.one_rollout() for _ in range(amount_of_demonstration)]

        points = np.array([r[0] for r in results])
        labels = [r[1] for r in results]

        for label in set(labels):
            mask = np.array(labels) == label
            plt.scatter(
                points[mask,0],
                points[mask,1],
                label=label
            )

        plt.legend()
        plt.show()