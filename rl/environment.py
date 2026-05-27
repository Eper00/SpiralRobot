"""MuJoCo-based reinforcement learning environment for tentacle robots."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from collections import deque
from typing import  Optional, Dict, Any
from common.support import _get_sites_positions, load_config
from common.base_class import TentacleBaseEnv

class TentacleTargetFollowingRL(TentacleBaseEnv):
    def __init__(
        self,
        config: Dict[str, Any]=None,
        render_mode: str = None,
    ):
        super().__init__(config, render_mode)
        
    def _get_obs(self) -> np.ndarray:
        """Retrieves the stacked observation from the buffer."""
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)


    def step(self, action):
        if(self._base_step(action)):
            return self._compute_step(action)
        else:
            return self.fail_step()
    def _compute_step(self,action):
        # Smooth normalized reward
        tip = _get_sites_positions(self.model, self.data, self.marker_names[-1]).squeeze()[1:]
        target=self.target_position
        actuator_lengths = self.data.actuator_length.copy()
        reward =self.reward_function(tip,target,action)
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
    
    def reward_function(self, tip, target, action):

        dist = np.linalg.norm(tip - target)

        # 1. base reward
        r_dist = np.exp(-self.reward_distance_scale * dist)

        if self.prev_dist is not None:
            r_progress = self.prev_dist - dist
        else:
            r_progress = 0.0
            self.prev_dist = dist

        # 3. energy penalty
        r_energy = -0.01 * np.sum(action**2)

        reward = (
            2.0 * (self.prev_dist - dist)
            + 0.1 * np.exp(-self.reward_distance_scale * dist)
            - 0.01 * np.sum(action**2)
        )

        self.prev_dist = dist

        return reward
    def reset(self, *, seed=None, options=None):
        self.prev_dist=None
        super().reset(seed=seed)

        self._base_reset()

        self.obs_buffer.clear()

        raw = self._get_current_raw_obs()

        for _ in range(self.num_frames):
            self.obs_buffer.append(raw)

        return self._get_obs(), self._get_info()
    


    
def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingRL:
    """Creator function for RLlib registration."""
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingRL(config=config, render_mode=render_mode)