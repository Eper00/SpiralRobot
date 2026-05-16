"""MuJoCo-based reinforcement learning environment for tentacle robots."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from collections import deque
from common.loaders import RLEnvironmentConfig
from typing import  Optional, Dict, Any
from common.support import _get_tip_position, _normalize_actuator_lengths
from common.base_class import TentacleBaseEnv

class TentacleTargetFollowingRL(TentacleBaseEnv):



    def __init__(
        self,
        config: Optional[RLEnvironmentConfig] = None,
        render_mode: str = None,
    ):
        super().__init__(config, render_mode)
       

    def _get_obs(self) -> np.ndarray:
        """Retrieves the stacked observation from the buffer."""
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)


    def step(self, action):
        if(self._simulate(action)):
            return self._compute_step(action)
        else:
            return self.fail_step()
    def _compute_step(self,action):
        # Smooth normalized reward
        tip=_get_tip_position(self.model,self.data)
        target=self.target_position
        actuator_lengths = self.data.actuator_length.copy()
        reward =self.reward_function(tip,target,actuator_lengths,action)
        truncated = (
            self._elapsed_steps >= self._max_episode_steps
        )

        obs = self._get_current_raw_obs()
        self.obs_buffer.append(obs)
        self.prev_action=action.copy()
        return (
            self._get_obs(),
            float(reward),
            False,
            truncated,
            self._get_info(),
        )
    
    def reward_function(self, tip, target, actuator_lengths, action):
        dist = np.linalg.norm(
                tip - target
            )

        normalized_dist = dist / self.max_distance

        normalized_dist = np.clip(
                normalized_dist,
                0.0,
                1.0
            )


        distance_reward =  np.exp(-self.reward_distance_scale * normalized_dist)
        if (self.prev_action is not None):
            action_change = action - self.prev_action
            input_change_penalty = self.action_change_penalty_scale * np.linalg.norm(action_change)/3
        else:
            input_change_penalty = 0.0
       
        return distance_reward - input_change_penalty
    def reset(self, *, seed=None, options=None):
        self.prev_action=None
        super().reset(seed=seed)

        self._base_reset()

        self.obs_buffer.clear()

        raw = self._get_current_raw_obs()

        for _ in range(self.num_frames):
            self.obs_buffer.append(raw)

        return self._get_obs(), self._get_info()
    


    
def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingRL:
    """Creator function for RLlib registration."""
    config = RLEnvironmentConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingRL(config=config, render_mode=render_mode)