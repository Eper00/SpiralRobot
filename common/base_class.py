import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from collections import deque
from typing import  Optional, Dict, Any
from common.support import _get_tip_position, _action_to_ctrl,_normalize_position,_normalize_actuator_lengths,load_config

class TentacleBaseEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, config, render_mode=None):

        super().__init__()
        self.config = load_config(config) if isinstance(config, str) else config
        self.render_delay=self.config['rl_evaluation']['render_delay']
        self.config = self.config['rl_env']
        self.render_mode = render_mode
        self.num_frames = self.config['num_frames']
        self.obs_buffer = deque(maxlen=self.num_frames)
        # -------------------------
        # XML
        # -------------------------
        xml_file = self.config['xml_file']

        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            xml_file = os.path.join(script_dir, xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # -------------------------
        # Config
        # -------------------------
        self.tip_site_name = self.config['tip_site_name']

        self.include_actuator_lengths_in_obs = (
            self.config['include_actuator_lengths_in_obs']
        )

        self.num_frames = self.config['num_frames']

        self.target_bounds_min = np.array(self.config['target_bounds_min'])
        self.target_bounds_max = np.array(self.config['target_bounds_max'])

        self.workspace_center = (
            self.target_bounds_min + self.target_bounds_max
        ) / 2

        self.workspace_scale = (
            self.target_bounds_max - self.target_bounds_min
        ) / 2
        # -------------------------
        # Mujoco timing
        # -------------------------
        self.simulation_length_seconds = (
            self.config['simulation_length_seconds']
        )
        self.max_distance = np.linalg.norm(
        self.target_bounds_max - self.target_bounds_min
        )
        self.reward_distance_scale = (
        self.config['reward_distance_scale']
    )
        self.time_between_steps_seconds = (
            self.config['time_between_steps_seconds']
        )
 
        self.timestep = self.model.opt.timestep

        
        self.frame_skip = max(
            1,
            round(
                self.time_between_steps_seconds / self.timestep
            ),
        )

        self.time_per_step = (
            self.frame_skip * self.timestep
        )

        self._max_episode_steps = int(
            self.simulation_length_seconds
            / self.time_per_step
        )

        # -------------------------
        # Spaces
        # -------------------------
        self.actuator_dim = self.model.nu 
        self.target_dim= len(self.target_bounds_min)
        self.tip_dim=2
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.actuator_dim,),
            dtype=np.float32,
        )
        

        self.actuator_low = self.model.actuator_ctrlrange[:, 0]
        self.actuator_high = self.model.actuator_ctrlrange[:, 1]
        self.cable_min= np.array(self.config['actuator_limits'])[:, 0]
        self.cable_max= np.array(self.config['actuator_limits'])[:, 1]
        # -------------------------
        # Sites
        # -------------------------
        self.tip_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.tip_site_name,
        )

        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target",
        )

        # -------------------------
        # State
        # -------------------------
        self.target_position = np.zeros(self.target_dim)

        self._elapsed_steps = 0

        self.viewer = None
        self.renderer = None

        # -------------------------
        # Observation dims
        # -------------------------
        self.single_frame_obs_dim = self.tip_dim + self.target_dim
        self.prev_action=None
        if self.include_actuator_lengths_in_obs:
            self.single_frame_obs_dim += self.actuator_dim
        stacked_obs_shape = (self.num_frames * self.single_frame_obs_dim,)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=stacked_obs_shape,
            dtype=np.float32,
        )
    def _get_info(self):

        return {
                "elapsed_steps": self._elapsed_steps,
            }

    def _get_current_raw_obs(self):


        tip = _get_tip_position(self.model, self.data)[1:3]
        target = self.target_position.copy()
        tip = _normalize_position(
            tip,
            self.workspace_center,
            self.workspace_scale,
        )

        target = _normalize_position(
            target,
            self.workspace_center,
            self.workspace_scale,
        )
        
        obs_parts = [tip, target]
        if self.include_actuator_lengths_in_obs:

            actuator = self.data.actuator_length.copy()

            actuator = _normalize_actuator_lengths(
                actuator,
                self.cable_min,
                self.cable_max,
            )

            obs_parts.append(actuator)
   


        return np.concatenate(obs_parts).astype(np.float32)
    def _base_reset(self):

        mujoco.mj_resetData(self.model, self.data)

        self._elapsed_steps = 0

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0.

        self.target_position = np.random.uniform(
            self.target_bounds_min,
            self.target_bounds_max,
            self.target_dim,
        )

        self.data.site_xpos[self.target_site_id] = np.array(
        [0.0, self.target_position[0], self.target_position[1]]
    )

        mujoco.mj_forward(self.model, self.data)
    def _base_step(self, action):

        action = np.clip(action, -1, 1)

        ctrl = _action_to_ctrl(
            action,
            self.actuator_low,
            self.actuator_high,
        )

        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):

            mujoco.mj_step(self.model, self.data)

            if (
                self.is_unstable()
                or np.any(np.abs(self.data.qacc) > 1e9)
            ):
                return False

        self._elapsed_steps += 1

        mujoco.mj_forward(self.model, self.data)

        return True

    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError(
                    "Renderer not initialized for rgb_array render mode."
                )
            self.renderer.update_scene(self.data, camera=self.camera_names[0])
            return self.renderer.render()
        elif self.render_mode == "human":
            self.data.site_xpos[self.target_site_id] = [0.0, self.target_position[0], self.target_position[1]]
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            if self.render_delay is not None:
                time.sleep(self.render_delay)

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def is_unstable(self):
    
        return (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or not np.isfinite(self.data.qacc).all()
        )


    def fail_step(self):
            return (
                self._get_current_raw_obs(),
                -10.0,
                True,
                False,
                self._get_info()
            )
def env_creator(env_config: Dict[str, Any]) -> TentacleBaseEnv:
    """Creator function for RLlib registration."""
    config = load_config(env_config.get("config_path")) if "config_path" in env_config else env_config
    render_mode = env_config.get("render_mode", None)
    return TentacleBaseEnv(config=config, render_mode=render_mode)