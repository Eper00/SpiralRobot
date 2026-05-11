import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco

from common.loaders import RLTrainingConfig
from common.support import _get_tip_position,_action_to_ctrl,_read_dataset,_get_targets_tips,_normalize_position,_normalize_actuator_lengths


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from typing import Tuple, Dict, Any, Optional, Union
from imitation.data.types import Trajectory
class TentacleTargetFollowingIL(gym.Env):
    def __init__(
        self,
        config: Optional[RLTrainingConfig] = None,
        render_mode: str = None,
    ):
        self.config = config
        self.render_mode = render_mode

        # -------------------------
        # load xml
        # -------------------------
        xml_file = "../assets/simulation/tentacle.xml"
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            xml_file = os.path.join(script_dir, xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # config
        # -------------------------
        self.tip_site_name = self.config.tip_site_name
        self.include_actuator_lengths_in_obs = (
            self.config.include_actuator_lengths_in_obs
        )
        self.action_space = spaces.Box(
            low= -1,     
            high=1,   
            shape=(3,),
            dtype=np.float32,
        )
        self.num_frames = self.config.num_frames

        self.action_dim = 3
        self.demonstration_number = self.config.demonstration_number
        single_frame_obs_dim = 6
        if self.include_actuator_lengths_in_obs:
            single_frame_obs_dim += 3

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(single_frame_obs_dim,),
            dtype=np.float32,
        )
        # -------------------------
        # actuator range
        # -------------------------
        self.actuator_low = self.model.actuator_ctrlrange[:, 0]
        self.actuator_high = self.model.actuator_ctrlrange[:, 1]

        self.target_bounds_min = np.array(self.config.target_bounds_min)
        self.target_bounds_max = np.array(self.config.target_bounds_max)

        self.workspace_center = (self.target_bounds_min + self.target_bounds_max) / 2
        self.workspace_scale = (self.target_bounds_max - self.target_bounds_min) / 2

        
        # -------------------------
        # storage
        # -------------------------
        self.demonstration_states = []
        self.demonstration_actions = []

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name
        )
        self.simulation_length_seconds = self.config.simulation_length_seconds
        self.time_between_steps_seconds = self.config.time_between_steps_seconds
        self.timestep = self.model.opt.timestep

        # Calculate frame_skip based on desired time between steps
        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep

        # Calculate max_episode_steps based on simulation length and actual time per step
        self._max_episode_steps = int(
            self.simulation_length_seconds / self.time_per_step
        )
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._elapsed_steps = 0
        self.obs_buffer.clear()
        

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.19 
        
        
        

        mujoco.mj_forward(self.model, self.data)

        # Get_target
        
        return self._get_obs(), self._get_info()
    def _get_info(self) -> Dict[str, Any]:
        return {
            "elapsed_steps": self._elapsed_steps,
        }
    def _get_obs(self):

        tip = _get_tip_position(self.model, self.data)
        target = self.target_position

        tip = _normalize_position(tip, self.workspace_center, self.workspace_scale)
        target = _normalize_position(target, self.workspace_center, self.workspace_scale)

        actuator = self.data.actuator_length.copy()

        if self.include_actuator_lengths_in_obs:
            actuator = _normalize_actuator_lengths(
                actuator,
                self.actuator_low,
                self.actuator_high
            )

            return np.concatenate([tip, target, actuator]).astype(np.float32)

        return np.concatenate([tip, target]).astype(np.float32)

   
    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError(
                    "Renderer not initialized for rgb_array render mode."
                )
            self.renderer.update_scene(self.data, camera=self.camera_names[0])
            return self.renderer.render()
        elif self.render_mode == "human":
            self.data.site_xpos[self.target_site_id] = self.target_position.copy()
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()


def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingIL:
    """Creator function for RLlib registration."""
    config = RLTrainingConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingIL(config=config, render_mode=render_mode)