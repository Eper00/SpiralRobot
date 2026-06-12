"""MuJoCo-based reinforcement learning environment for tentacle robots."""


import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import os
from collections import deque
from typing import  Optional, Dict, Any
from common.support import _get_sites_positions, load_config
from common.base_class import TentacleBaseEnv






class TentacleRL(TentacleBaseEnv):
    def __init__(
        self,
        config: Dict[str, Any]=None,
        render_mode: str = None,
    ):  
        
        
        super().__init__(config, render_mode)
        self.warm_start=self.config['warm_start']
        if self.warm_start==True and  os.path.exists(self.config['prev_ppo_path']):
            self.prev_ppo_path=self.config['prev_ppo_path']
        else:
            self.prev_ppo_path=None
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
       
        
        
        obs = self._get_current_raw_obs()
        reward =self.reward_function(self.marker_positions,self.target_position,action)
        truncated = (
            self._elapsed_steps >= self._max_episode_steps
        )
        self.obs_buffer.append(obs)
        return (
            self._get_obs(),
            float(reward),
            False,
            truncated,
            self._get_info(),
        )
    '''
    def reward_function(self, markers_position, target_pos, action):

        R_target = self.model.geom_size[self.target_geom_id][0]

        # --- 1) Distance shaping (ugyanaz, mint az első rewardban) ---
        suface_dist = max(
            np.linalg.norm(markers_position[-3:] - target_pos) - R_target,
            0
        )
        distance_penalty = -suface_dist   # skálája kb. -0.0 ... -2.0


        # --- 2) Wrap-around contact reward (skálázva) ---
        contacts = []
        for i, m in enumerate(markers_position):
            R_seg = self.segment_effective_radius[i]
            dist = np.linalg.norm(m - target_pos)
            touching = dist < (R_target + R_seg)
            contacts.append(1.0 if touching else 0.0)

        # eredeti: 0 ... N
        contact_raw = float(np.sum(contacts))

        # skálázás: 0 ... 1
        contact_reward = contact_raw / len(markers_position)


        # --- 3) Összesített reward ---
        # distance shaping + wrap-around contact
        return distance_penalty + contact_reward

    '''
    def reward_function(self, markers_position, target_pos, action):

        R_target = self.model.geom_size[self.target_geom_id][0]

        # --- 1) Distance shaping: szegmens vastagság figyelembe véve ---
        # Utolsó marker (vagy bármelyik, amit használsz)
        m = markers_position[-3:]

        # Ehhez a markerhez tartozó szegmens sugara
        R_seg = self.segment_effective_radius[-1]

        # Felület–felület távolság
        surface_dist = np.linalg.norm(m - target_pos) - (R_target + R_seg)

        # Ha átfedés lenne, 0-ra vágjuk
        surface_dist = max(surface_dist, 0.0)

        distance_penalty = -surface_dist

        return distance_penalty

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._base_reset()

      
        self.obs_buffer.clear()
        raw = self._get_current_raw_obs()
        for _ in range(self.num_frames):
            self.obs_buffer.append(raw.copy())

        return self._get_obs(), self._get_info()

            

   


    
def env_creator(env_config: Dict[str, Any]) -> TentacleRL:
    """Creator function for RLlib registration."""
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    render_mode = env_config.get("render_mode", None)
    return TentacleRL(config=config, render_mode=render_mode)